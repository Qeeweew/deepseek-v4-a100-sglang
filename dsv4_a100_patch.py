import importlib.util
import contextlib
import logging
import os
from pathlib import Path

import torch

try:
    import transformers.models.deepseek_v4.modeling_deepseek_v4 as _HF_DSV4
except ModuleNotFoundError:
    _HF_DSV4 = None


logger = logging.getLogger(__name__)
_PATCH_APPLIED = False


def _env_enabled(name: str, default: str = "1") -> bool:
    return os.environ.get(name, default).strip().lower() not in {"0", "false", "no", "off"}


def _device_key(device: torch.device) -> int:
    return device.index if device.type == "cuda" else -1


def _get_tensor_cache(obj, attr: str) -> dict:
    cache = getattr(obj, attr, None)
    if cache is None:
        cache = {}
        setattr(obj, attr, cache)
    return cache


def _can_sync_cuda() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        return not torch.cuda.is_current_stream_capturing()
    except Exception:
        return True


@contextlib.contextmanager
def _record_function(name: str):
    with torch.profiler.record_function(f"dsv4_a100_patch::{name}"):
        yield
    if _env_enabled("SGLANG_DSV4_A100_DEBUG_SYNC", "0") and _can_sync_cuda():
        try:
            torch.cuda.synchronize()
        except Exception:
            logger.exception("CUDA sync failed after dsv4_a100_patch::%s", name)
            raise


def _apply_rotary_tail_torch(
    x: torch.Tensor,
    freqs_cis_or_real: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    freqs = (
        torch.view_as_real(freqs_cis_or_real).flatten(-2)
        if freqs_cis_or_real.is_complex()
        else freqs_cis_or_real
    )
    rope_dim = freqs.shape[-1]
    if rope_dim == 0:
        return x
    if rope_dim % 2 != 0 or rope_dim > x.shape[-1]:
        raise ValueError(f"invalid rotary dim {rope_dim} for q dim {x.shape[-1]}")

    pos = positions.to(torch.long).clamp_(0, freqs.shape[0] - 1)
    freq = freqs.index_select(0, pos).to(torch.float32)
    base_dim = x.shape[-1] - rope_dim
    tail = x[..., base_dim:].to(torch.float32).reshape(
        x.shape[0], x.shape[1], rope_dim // 2, 2
    )
    freq = freq.reshape(x.shape[0], 1, rope_dim // 2, 2)

    even = tail[..., 0]
    odd = tail[..., 1]
    freq_even = freq[..., 0]
    freq_odd = freq[..., 1]
    rotated = torch.stack(
        (even * freq_even - odd * freq_odd, even * freq_odd + odd * freq_even),
        dim=-1,
    ).reshape(x.shape[0], x.shape[1], rope_dim)
    x[..., base_dim:] = rotated.to(x.dtype)
    return x


def _hadamard_transform_torch(
    x: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    if x.shape[-1] & (x.shape[-1] - 1):
        raise ValueError(f"hadamard dim must be a power of 2, got {x.shape[-1]}")
    y = x.to(torch.float32)
    n = y.shape[-1]
    h = 1
    while h < n:
        y = y.reshape(*y.shape[:-1], -1, h * 2)
        left = y[..., :h].clone()
        right = y[..., h:].clone()
        y[..., :h] = left + right
        y[..., h:] = left - right
        y = y.reshape(*x.shape)
        h *= 2
    y = (y * (n**-0.5)).to(torch.bfloat16)
    if out is not None:
        out.copy_(y)
        return out
    return y


def _bf16_indexer_q_torch_fallback(
    q_input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: float,
    freqs_cis_or_real: torch.Tensor,
    positions: torch.Tensor,
    q_out: torch.Tensor | None = None,
    weights_out: torch.Tensor | None = None,
    allow_inplace_input: bool = False,
):
    if allow_inplace_input and q_input.dtype == torch.bfloat16 and q_input.is_contiguous():
        q = q_input
    else:
        q = q_input.contiguous().to(torch.bfloat16)
    _apply_rotary_tail_torch(q, freqs_cis_or_real, positions)
    q = _hadamard_transform_torch(q, q_out)
    weights = (weight.float() * float(weight_scale)).unsqueeze(-1)
    if weights_out is not None:
        weights_out.copy_(weights)
        weights = weights_out
    return q, weights


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_TRITON_COMMON = _load_module(
    "dsv4_triton_decode_common",
    Path("/workspace/sglang/python/sglang/srt/layers/attention/nsa/triton_decode/triton_mla_kernels_decode_common.py"),
)
def apply_patch() -> None:
    global _PATCH_APPLIED
    if _PATCH_APPLIED:
        return
    _PATCH_APPLIED = True
    _patch_deepseek_v4_hook()
    _patch_dsv4_pool_configurator()
    _patch_deepseek_v4_bf16_kv_pool()
    _patch_fused_rope_inplace()
    _patch_dsv4_indexer_torch_fallback()
    _patch_dsv4_core_compressor_bf16_store()
    _patch_dsv4_mxfp4_moe_a100()
    _patch_deepseek_v4_backend()
    logger.warning("ENABLE_SGLANG_DSV4_A100_PATCH=1: monkey patch applied")


def _patch_deepseek_v4_hook() -> None:
    from sglang.srt.arg_groups import deepseek_v4_hook
    from sglang.srt.server_args import ServerArgs

    def apply_deepseek_v4_defaults(server_args: "ServerArgs", model_arch: str) -> None:
        from sglang.srt.environ import envs

        server_args.attention_backend = "dsv4"
        server_args.page_size = 256
        if server_args.moe_runner_backend == "auto":
            server_args.moe_runner_backend = "marlin"
            logger.warning(
                "Monkey patch: setting MoE runner backend to %s for %s.",
                server_args.moe_runner_backend,
                model_arch,
            )
        if server_args.max_running_requests is None:
            server_args.max_running_requests = 256
        if server_args.kv_cache_dtype == "auto":
            server_args.kv_cache_dtype = "bf16"
            logger.warning(
                "Monkey patch: setting KV cache dtype to %s for %s.",
                server_args.kv_cache_dtype,
                model_arch,
            )
        assert server_args.kv_cache_dtype in {"bf16", "bfloat16"}
        if server_args.swa_full_tokens_ratio == ServerArgs.swa_full_tokens_ratio:
            server_args.swa_full_tokens_ratio = 0.1
        if server_args.speculative_algorithm is not None and not envs.SGLANG_ENABLE_SPEC_V2.get():
            envs.SGLANG_ENABLE_SPEC_V2.set(True)

    deepseek_v4_hook.apply_deepseek_v4_defaults = apply_deepseek_v4_defaults


def _patch_dsv4_pool_configurator() -> None:
    from sglang.srt.environ import envs
    from sglang.srt.model_executor import pool_configurator

    def _get_bytes_per_full_token(self) -> float:
        attn_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        bf16_size = torch.bfloat16.itemsize
        kv_bytes = attn_head_dim * bf16_size
        indexer_bytes = self.indexer_head_dim * bf16_size

        state_dtype_size = 4
        c4_state_bytes = 2 * 2 * attn_head_dim * state_dtype_size
        c128_online = envs.SGLANG_OPT_USE_ONLINE_COMPRESS.get()
        c128_state_bytes = (
            (3 if c128_online else 2) * attn_head_dim * state_dtype_size
        )
        c4_indexer_state_bytes = 2 * 2 * self.indexer_head_dim * state_dtype_size

        c4_state_ratio = self.c4_ring_size / self.swa_page_size
        c128_state_ratio = self.c128_ring_size / self.swa_page_size
        c4_frac = 1 / (4 * self.c4_shrink_factor)
        return (
            self.swa_ratio * kv_bytes * self.num_layers_total
            + c4_frac * kv_bytes * self.num_layers_ca4
            + 1 / 128 * kv_bytes * self.num_layers_ca128
            + 1 / 4 * indexer_bytes * self.num_layers_ca4
            + self.swa_ratio * c4_state_ratio * c4_state_bytes * self.num_layers_ca4
            + self.swa_ratio
            * c128_state_ratio
            * c128_state_bytes
            * self.num_layers_ca128
            + self.swa_ratio
            * c4_state_ratio
            * c4_indexer_state_bytes
            * self.num_layers_ca4
        )

    pool_configurator.DSV4PoolConfigurator._get_bytes_per_full_token = (
        _get_bytes_per_full_token
    )


def _patch_deepseek_v4_bf16_kv_pool() -> None:
    from sglang.srt.mem_cache import deepseek_v4_memory_pool as dsv4_pool
    from sglang.srt.models import deepseek_v4 as deepseek_v4_model
    from triton_kernels import scatter_bf16_rows

    original_single_kv_pool = dsv4_pool.DeepSeekV4SingleKVPool
    original_indexer_pool = dsv4_pool.DeepSeekV4IndexerPool
    original_token_to_kv_pool = dsv4_pool.DeepSeekV4TokenToKVPool

    class BF16DeepSeekV4SingleKVPool(original_single_kv_pool):
        def get_bytes_per_token(self) -> int:
            return (self.qk_nope_head_dim + self.qk_rope_head_dim) * self.dtype.itemsize

        def create_buffer(self, *, num_pages: int):
            self.kv_cache_total_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
            return torch.zeros(
                (num_pages, self.page_size, 1, self.kv_cache_total_dim),
                dtype=self.dtype,
                device=self.device,
            )

        def set_key_buffer(self, layer_id: int, loc: torch.Tensor, _pack) -> None:
            raise NotImplementedError("BF16 DSV4 path bypasses packed key-buffer writes")

        def set_key_buffer_fused(self, layer_id: int, loc: torch.Tensor, cache_k: torch.Tensor) -> None:
            with _record_function("bf16_single_kv_pool_set_key_buffer_fused"):
                kv = cache_k.to(self.dtype) if cache_k.dtype != self.dtype else cache_k
                if kv.ndim == 2:
                    kv = kv.unsqueeze(1)
                scatter_bf16_rows(self.kv_buffer[layer_id], loc, kv)

        def get_key_buffer(self, layer_id: int):
            return self.kv_buffer[layer_id]

    class BF16DeepSeekV4IndexerPool(original_indexer_pool):
        def _create_buffer(self):
            page_elems = self.page_size * self.index_head_dim
            with self.memory_saver_adapter.region(dsv4_pool.GPU_MEMORY_TYPE_KV_CACHE):
                ctx = (
                    torch.cuda.use_mem_pool(self.custom_mem_pool)
                    if self.custom_mem_pool
                    else dsv4_pool.nullcontext()
                )
                with ctx:
                    self.index_k_with_scale_buffer = [
                        torch.zeros(
                            (self.size + self.page_size + 1) // self.page_size,
                            page_elems,
                            dtype=torch.bfloat16,
                            device=self.device,
                        )
                        for _ in range(self.layer_num)
                    ]

        def set_index_k_scale_buffer(
            self,
            layer_id: int,
            loc: torch.Tensor,
            index_k: torch.Tensor,
            index_k_scale: torch.Tensor,
        ) -> None:
            raise NotImplementedError("BF16 DSV4 indexer path bypasses FP8 scale writes")

        def set_index_fused(
            self,
            layer_id: int,
            loc: torch.Tensor,
            cache_k: torch.Tensor,
        ) -> None:
            with _record_function("bf16_indexer_cache_store"):
                kv = cache_k.to(torch.bfloat16) if cache_k.dtype != torch.bfloat16 else cache_k
                scatter_bf16_rows(
                    self.index_k_with_scale_buffer[layer_id - self.start_layer],
                    loc,
                    kv.view(-1, self.index_head_dim),
                )

    class BF16DeepSeekV4TokenToKVPool(original_token_to_kv_pool):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            logger.warning("Monkey patch: DeepSeekV4TokenToKVPool using BF16 cache layout")

        def set_swa_key_buffer_radix_fused_norm_rope(
            self,
            layer_id: int,
            raw_loc: torch.Tensor,
            kv: torch.Tensor,
            kv_weight: torch.Tensor,
            eps: float,
            freqs_cis: torch.Tensor,
            positions: torch.Tensor,
        ) -> None:
            from sglang.srt.layers.rotary_embedding import fused_norm_rope_inplace

            with _record_function("bf16_swa_norm_rope_store"):
                swa_loc = self.translate_loc_from_full_to_swa(raw_loc)
                kv_bf16 = kv.contiguous().to(torch.bfloat16)
                fused_norm_rope_inplace(kv_bf16, kv_weight, eps, freqs_cis, positions)
                self.swa_kv_pool.set_key_buffer_fused(
                    self._swa_local_layer_id(layer_id), swa_loc, kv_bf16
                )

        def set_extra_key_buffer_fused(self, layer_id: int, loc: torch.Tensor, cache_k: torch.Tensor) -> None:
            _, compress_layer_id, compress_kv_pool = self.layer_mapping[layer_id]
            assert compress_kv_pool is not None
            with _record_function("bf16_extra_key_buffer_store"):
                return compress_kv_pool.set_key_buffer_fused(compress_layer_id, loc, cache_k)

    def _compute_kv_to_cache(self, x, positions, forward_batch, qkv_a=None) -> None:
        token_to_kv_pool = deepseek_v4_model.get_token_to_kv_pool()
        with _record_function("mqa_compute_kv_bf16_to_cache"):
            kv = self._compute_kv_bf16(x, positions, qkv_a=qkv_a)
            if kv.ndim == 2:
                kv = kv.unsqueeze(1)
            token_to_kv_pool.set_swa_key_buffer_radix_fused(
                layer_id=self.layer_id,
                raw_loc=forward_batch.out_cache_loc,
                cache_k=kv,
            )

    def _forward_prepare_multi_stream(self, x, positions, forward_batch, attn_backend, q_out=None, x_quant=None):
        with _record_function("mqa_forward_prepare_multistream_bf16"):
            q, _ = self._forward_prepare(
                x, positions, forward_batch, attn_backend, q_out=q_out, x_quant=x_quant
            )
            return q

    dsv4_pool.DeepSeekV4SingleKVPool = BF16DeepSeekV4SingleKVPool
    dsv4_pool.DeepSeekV4IndexerPool = BF16DeepSeekV4IndexerPool
    dsv4_pool.DeepSeekV4TokenToKVPool = BF16DeepSeekV4TokenToKVPool
    deepseek_v4_model.MQALayer._compute_kv_to_cache = _compute_kv_to_cache
    deepseek_v4_model.MQALayer._forward_prepare_multi_stream = _forward_prepare_multi_stream


def _patch_dsv4_mxfp4_moe_a100() -> None:
    from torch.nn import Module, Parameter

    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput
    from sglang.srt.layers.moe.topk import TopKOutputChecker
    from sglang.srt.layers.quantization import mxfp4_marlin_moe

    def process_weights_after_loading(self, layer: Module) -> None:
        with _record_function("mxfp4_moe_prepare_triton_a100"):
            self._fp8.process_weights_after_loading(layer)
            if getattr(layer, "_mega_moe_weights_built", False):
                return

            layer.w13_weight.data = layer.w13_weight.data.view(torch.int8)
            layer.w2_weight.data = layer.w2_weight.data.view(torch.int8)

            for scale_name in ("w13_weight_scale_inv", "w2_weight_scale_inv"):
                scale = getattr(layer, scale_name)
                if scale.dtype == torch.float32:
                    continue
                scale_data = scale.data
                if scale_data.dtype == torch.float8_e8m0fnu:
                    scale_data = scale_data.to(torch.float32)
                elif scale_data.dtype in (torch.uint8, torch.int8):
                    scale_data = scale_data.view(torch.uint8).view(torch.float8_e8m0fnu).to(torch.float32)
                else:
                    scale_data = scale_data.float()
                setattr(layer, scale_name, Parameter(scale_data, requires_grad=False))

            layer._dsv4_mxfp4_backend = "a100_triton"
            logger.warning(
                "Monkey patch: using Triton MXFP4 MoE fallback for %s.",
                self.prefix,
            )

    def apply(self, layer: Module, dispatch_output):
        topk_output = dispatch_output.topk_output
        if not TopKOutputChecker.format_is_standard(topk_output):
            raise ValueError(f"Unsupported topk output format: {topk_output.format}")

        if getattr(layer, "_dsv4_mxfp4_backend", None) == "a100_triton":
            from sglang.srt.layers.moe.fused_moe_triton.mxfp4_moe_sm120_triton import (
                mxfp4_moe_forward_triton,
            )

            hidden_states = dispatch_output.hidden_states
            w13 = layer.w13_weight.data
            w2 = layer.w2_weight.data
            w13_scale = layer.w13_weight_scale_inv.data
            w2_scale = layer.w2_weight_scale_inv.data
            intermediate_size = w13.shape[1] // 2
            hidden_size = w13.shape[2] * 2

            with _record_function("mxfp4_moe_forward_triton_a100"):
                output = mxfp4_moe_forward_triton(
                    hidden_states=hidden_states,
                    w13_packed=w13,
                    w2_packed=w2,
                    w13_scale=w13_scale,
                    w2_scale=w2_scale,
                    topk_ids=topk_output.topk_ids,
                    topk_weights=topk_output.topk_weights,
                    hidden_size=hidden_size,
                    intermediate_size=intermediate_size,
                    routed_scaling_factor=(
                        self.runner.config.routed_scaling_factor
                        if hasattr(self.runner, "config")
                        else None
                    ),
                    clamp_limit=(
                        self.runner.config.swiglu_limit
                        if hasattr(self.runner, "config")
                        else None
                    ),
                )
            return StandardCombineInput(hidden_states=output)

        return self._sglang_original_apply(layer, dispatch_output)

    if not hasattr(
        mxfp4_marlin_moe.Mxfp4MarlinMoEMethod, "_sglang_original_apply"
    ):
        mxfp4_marlin_moe.Mxfp4MarlinMoEMethod._sglang_original_apply = (
            mxfp4_marlin_moe.Mxfp4MarlinMoEMethod.apply
        )
    mxfp4_marlin_moe.Mxfp4MarlinMoEMethod.process_weights_after_loading = (
        process_weights_after_loading
    )
    mxfp4_marlin_moe.Mxfp4MarlinMoEMethod.apply = apply


def _patch_fused_rope_inplace() -> None:
    import sglang.jit_kernel.dsv4.elementwise as dsv4_elementwise
    import sglang.srt.models.deepseek_v4 as dsv4_model
    from triton_kernels import fused_rope_inplace as triton_fused_rope_inplace

    def fused_rope_inplace(
        q: torch.Tensor,
        k: torch.Tensor | None,
        freqs_cis: torch.Tensor,
        positions: torch.Tensor,
        inverse: bool = False,
    ) -> None:
        with _record_function("triton_fused_rope_inplace"):
            triton_fused_rope_inplace(q, k, freqs_cis, positions, inverse=inverse)

    dsv4_elementwise.fused_rope_inplace = fused_rope_inplace
    dsv4_model.fused_rope_inplace = fused_rope_inplace


def _patch_dsv4_indexer_torch_fallback() -> None:
    from sglang.srt.environ import envs
    import sglang.srt.layers.attention.dsv4.indexer as dsv4_indexer
    from sglang.srt.layers.attention.dsv4.metadata import PagedIndexerMetadata
    from sglang.srt.state_capturer.indexer_topk import get_global_indexer_capturer
    from triton_kernels import bf16_indexer_q, bf16_paged_mqa_logits

    def bf16_paged_mqa_logits_torch(
        q: torch.Tensor,
        kvcache_bf16: torch.Tensor,
        weight: torch.Tensor,
        seq_lens: torch.Tensor,
        page_table: torch.Tensor,
        deep_gemm_metadata,
        max_seq_len: int,
        clean_logits: bool = True,
    ) -> torch.Tensor:
        with _record_function("c4_bf16_paged_mqa_logits_triton"):
            return bf16_paged_mqa_logits(
                q,
                kvcache_bf16,
                weight,
                seq_lens,
                page_table,
                deep_gemm_metadata,
                max_seq_len,
                clean_logits,
            )

    def compute_q(
        self,
        q_lora: torch.Tensor,
        positions: torch.Tensor,
        weight: torch.Tensor,
    ):
        q, _ = self.wq_b(q_lora)
        q = q.view(-1, self.n_local_heads, self.head_dim)
        scratch_q = getattr(self, "_dsv4_bf16_indexer_scratch_q", None)
        freqs_real = getattr(self, "_dsv4_bf16_indexer_freqs_real", None)
        q_cache = _get_tensor_cache(self, "_dsv4_bf16_indexer_q_out_cache")
        weight_cache = _get_tensor_cache(self, "_dsv4_bf16_indexer_weights_out_cache")
        q_key = (tuple(q.shape[1:]), _device_key(q.device))
        w_shape = (*weight.shape, 1)
        w_key = (tuple(w_shape[1:]), _device_key(weight.device))
        q_entry = q_cache.get(q_key)
        q_out = None if q_entry is None else q_entry["tensor"]
        if q_out is None or q_out.shape[0] < q.shape[0]:
            if q_out is not None:
                q_entry.setdefault("retired", []).append(q_out)
            q_out = torch.empty(q.shape, device=q.device, dtype=torch.bfloat16)
            q_cache[q_key] = {"tensor": q_out, "retired": []}
        else:
            q_out = q_out[: q.shape[0]]
        w_entry = weight_cache.get(w_key)
        weights_out = None if w_entry is None else w_entry["tensor"]
        if weights_out is None or weights_out.shape[0] < w_shape[0]:
            if weights_out is not None:
                w_entry.setdefault("retired", []).append(weights_out)
            weights_out = torch.empty(w_shape, device=weight.device, dtype=torch.float32)
            weight_cache[w_key] = {"tensor": weights_out, "retired": []}
        else:
            weights_out = weights_out[: w_shape[0]]
        if freqs_real is None or freqs_real.device != self.freqs_cis.device:
            freqs_real = torch.view_as_real(self.freqs_cis).flatten(-2).contiguous()
            self._dsv4_bf16_indexer_freqs_real = freqs_real
        if _env_enabled("SGLANG_DSV4_A100_TORCH_INDEXER_Q", "1"):
            with _record_function("c4_bf16_indexer_q_torch_fallback"):
                return _bf16_indexer_q_torch_fallback(
                    q,
                    weight,
                    self.weight_scale,
                    freqs_real,
                    positions,
                    q_out=q_out,
                    weights_out=weights_out,
                    allow_inplace_input=True,
                )
        return bf16_indexer_q(
            q,
            weight,
            self.weight_scale,
            self.freqs_cis,
            positions,
            q_out=q_out,
            weights_out=weights_out,
            scratch_q=scratch_q,
            allow_inplace_input=True,
            freqs_real=freqs_real,
        )

    def forward_c4_indexer(
        self,
        x: torch.Tensor,
        q_lora: torch.Tensor,
        c4_indexer,
        forward_batch,
        alt_streams=None,
        enable_multi_stream: bool = False,
        q_lora_ready=None,
        skip_compressor: bool = False,
    ) -> None:
        if forward_batch.forward_mode.is_idle():
            return
        self._maybe_upgrade_forward_metadata()
        token_to_kv_pool = self.token_to_kv_pool
        metadata = self.forward_metadata
        indexer_metadata = metadata.indexer_metadata
        core_metadata = metadata.core_metadata
        assert isinstance(indexer_metadata, PagedIndexerMetadata)

        if enable_multi_stream:
            q_bf16, weights, c4_indexer_kv_cache = self._forward_prepare_multi_stream(
                x=x,
                q_lora=q_lora,
                c4_indexer=c4_indexer,
                positions=core_metadata.positions,
                forward_batch=forward_batch,
                token_to_kv_pool=token_to_kv_pool,
                alt_streams=alt_streams,
                q_lora_ready=q_lora_ready,
            )
        else:
            assert q_lora_ready is None
            q_bf16, weights, c4_indexer_kv_cache = self._forward_prepare_normal(
                x=x,
                q_lora=q_lora,
                c4_indexer=c4_indexer,
                positions=core_metadata.positions,
                forward_batch=forward_batch,
                token_to_kv_pool=token_to_kv_pool,
            )

        assert len(q_bf16.shape) == 3
        q_bf16 = q_bf16.unsqueeze(1)
        assert len(c4_indexer_kv_cache.shape) == 2
        assert c4_indexer_kv_cache.dtype == torch.bfloat16
        with _record_function("c4_indexer_bf16_cache_view"):
            c4_indexer_kv_cache = c4_indexer_kv_cache.view(
                c4_indexer_kv_cache.shape[0],
                64,
                1,
                c4_indexer.head_dim,
            )

        assert len(weights.shape) == 3
        weights = weights.squeeze(2)
        _c4sl = indexer_metadata.c4_seq_lens
        if _c4sl.dim() == 1:
            _c4sl = _c4sl.unsqueeze(-1)
        logits = bf16_paged_mqa_logits_torch(
            q_bf16,
            c4_indexer_kv_cache,
            weights,
            _c4sl,
            indexer_metadata.page_table,
            indexer_metadata.deep_gemm_metadata,
            indexer_metadata.max_c4_seq_len,
            False,
        )

        assert indexer_metadata.page_table is core_metadata.page_table
        if self.debug_use_external_c4_sparse_indices:
            return

        indexer_capturer = get_global_indexer_capturer()
        capture_enabled = indexer_capturer is not None

        raw_indices = None
        if capture_enabled:
            raw_indices = torch.empty_like(core_metadata.c4_sparse_page_indices)

        if envs.SGLANG_TOPK_TRANSFORM_512_TORCH.get():
            dsv4_indexer.topk_transform_512_pytorch_vectorized(
                logits,
                indexer_metadata.c4_seq_lens,
                core_metadata.page_table,
                core_metadata.c4_sparse_page_indices,
                indexer_metadata.c4_page_size,
                raw_indices,
            )
        elif envs.SGLANG_OPT_USE_TOPK_V2.get() and raw_indices is None:
            dsv4_indexer.topk_transform_512_v2(
                logits,
                indexer_metadata.c4_seq_lens,
                core_metadata.page_table,
                core_metadata.c4_sparse_page_indices,
                indexer_metadata.c4_page_size,
                indexer_metadata.topk_metadata,
            )
        else:
            dsv4_indexer.topk_transform_512(
                logits,
                indexer_metadata.c4_seq_lens,
                core_metadata.page_table,
                core_metadata.c4_sparse_page_indices,
                indexer_metadata.c4_page_size,
                raw_indices,
            )

        if capture_enabled:
            compress_layer_id = token_to_kv_pool.layer_mapping[
                c4_indexer.layer_id
            ].compress_layer_id
            indexer_capturer.capture(compress_layer_id, raw_indices)

    dsv4_indexer.bf16_paged_mqa_logits_torch = bf16_paged_mqa_logits_torch
    dsv4_indexer.fp8_paged_mqa_logits_torch = bf16_paged_mqa_logits_torch
    dsv4_indexer.C4Indexer.compute_q = compute_q
    dsv4_indexer.C4IndexerBackendMixin.forward_c4_indexer = forward_c4_indexer


def _extract_compressor_positions(plan, compress_ratio: int) -> torch.Tensor:
    from triton_kernels import compressor_positions_from_plan

    return compressor_positions_from_plan(plan[1].view(torch.int32).reshape(plan[1].shape[0], 4), compress_ratio)


def _patch_dsv4_core_compressor_bf16_store() -> None:
    import sglang.srt.layers.attention.dsv4.compressor_v2 as compressor_v2
    from sglang.jit_kernel.dsv4 import compress_forward
    from sglang.srt.layers.deepseek_v4_rope import fused_norm_rope_inplace_triton
    from triton_kernels import (
        compressor_decode_mask_positions,
        compressor_prefill_metadata,
        compressor_positions_from_plan,
    )

    original_forward_unified = compressor_v2.CompressorBackendMixin.forward_unified
    is_overlap_compress = compressor_v2.is_overlap_compress
    _use_online_compress = compressor_v2._use_online_compress

    def _compress_to_bf16(self, x, forward_batch, compressor):
        with _record_function("compressor_compute_kv_score"):
            kv_score_input = compressor.compute_kv_score(x, forward_batch)
        state_pool = compressor.get_state_pool(self)
        compress_ratio = compressor.ratio
        head_dim = compressor.head_dim
        plan = self._get_paged_compress_metadata(compress_ratio)
        is_online = _use_online_compress(compress_ratio)
        kv_score_buffer = state_pool.kv_score_buffer.kv_score
        if is_online:
            kv_score_buffer = kv_score_buffer.view(-1, 1, head_dim * 3)
        else:
            coff = 2 if is_overlap_compress(compress_ratio) else 1
            kv_score_buffer = kv_score_buffer.view(
                -1, compress_ratio, 2 * head_dim * coff
            )

        with _record_function("compressor_compress_forward"):
            kv_compressed = compress_forward(
                kv_score_buffer=kv_score_buffer,
                kv_score_input=kv_score_input,
                ape=compressor.ape.view(-1, head_dim),
                plan=plan,
                compress_ratio=compress_ratio,
                head_dim=head_dim,
                is_online=is_online,
            )
        if kv_compressed.shape[0] == 0:
            return kv_compressed, plan, None

        if plan.is_decode:
            with _record_function("compressor_decode_boundary_mask"):
                positions = compressor_decode_mask_positions(
                    kv_compressed,
                    plan[1].view(torch.int32).reshape(plan[1].shape[0], 4),
                    compress_ratio,
                )
        else:
            positions = compressor_positions_from_plan(
                plan[1].view(torch.int32).reshape(plan[1].shape[0], 4), compress_ratio
            )
        with _record_function("compressor_norm_rope_triton"):
            fused_norm_rope_inplace_triton(
                kv_compressed,
                compressor.norm.weight,
                compressor.norm.variance_epsilon,
                compressor.freqs_cis,
                positions=positions,
            )
        return kv_compressed, plan, positions

    def _compressed_out_loc(out_loc: torch.Tensor, plan):
        if plan.is_decode:
            return out_loc
        with _record_function("compressor_prefill_out_loc_select"):
            _, out_loc_to_store = compressor_prefill_metadata(
                plan[1].view(torch.int32).reshape(plan[1].shape[0], 4),
                out_loc,
                plan.compress_ratio,
            )
            return out_loc_to_store

    def forward_unified(self, x, forward_batch, layer_id: int, compressor) -> None:
        if forward_batch.forward_mode.is_idle():
            return

        token_to_kv_pool = self.token_to_kv_pool
        use_bf16_kv = getattr(token_to_kv_pool.swa_kv_pool, "dtype", None) == torch.bfloat16
        if compressor.is_in_indexer or not use_bf16_kv:
            return original_forward_unified(
                self, x, forward_batch, layer_id, compressor
            )

        with _record_function(f"compressor_bf16_forward_unified_r{compressor.ratio}"):
            self._maybe_upgrade_forward_metadata()
            with _record_function("compressor_compute_kv_score"):
                kv_score_input = compressor.compute_kv_score(x, forward_batch)
            state_pool = compressor.get_state_pool(self)
            out_loc = self._get_out_loc(compressor.ratio)

            _, _, compress_kv_pool = token_to_kv_pool.layer_mapping[layer_id]
            assert compress_kv_pool is not None

            compress_ratio = compressor.ratio
            head_dim = compressor.head_dim
            plan = self._get_paged_compress_metadata(compress_ratio)
            is_online = _use_online_compress(compress_ratio)
            kv_score_buffer = state_pool.kv_score_buffer.kv_score
            if is_online:
                kv_score_buffer = kv_score_buffer.view(-1, 1, head_dim * 3)
            else:
                coff = 2 if is_overlap_compress(compress_ratio) else 1
                kv_score_buffer = kv_score_buffer.view(
                    -1, compress_ratio, 2 * head_dim * coff
                )

            with _record_function("compressor_compress_forward"):
                kv_compressed = compress_forward(
                    kv_score_buffer=kv_score_buffer,
                    kv_score_input=kv_score_input,
                    ape=compressor.ape.view(-1, head_dim),
                    plan=plan,
                    compress_ratio=compress_ratio,
                    head_dim=head_dim,
                    is_online=is_online,
                )
            if kv_compressed.shape[0] == 0:
                return

            if plan.is_decode:
                with _record_function("compressor_decode_boundary_mask"):
                    positions = compressor_decode_mask_positions(
                        kv_compressed,
                        plan[1].view(torch.int32).reshape(plan[1].shape[0], 4),
                        compress_ratio,
                    )
                out_loc_to_store = out_loc
            else:
                with _record_function("compressor_prefill_metadata"):
                    positions, out_loc_to_store = compressor_prefill_metadata(
                        plan[1].view(torch.int32).reshape(plan[1].shape[0], 4),
                        out_loc,
                        compress_ratio,
                    )
            with _record_function("compressor_norm_rope_triton"):
                fused_norm_rope_inplace_triton(
                    kv_compressed,
                    compressor.norm.weight,
                    compressor.norm.variance_epsilon,
                    compressor.freqs_cis,
                    positions=positions,
                )

            token_to_kv_pool.set_extra_key_buffer_fused(
                layer_id=layer_id,
                loc=out_loc_to_store,
                cache_k=kv_compressed,
            )

    def forward_indexer_compressor(self, x, forward_batch, layer_id: int, compressor) -> None:
        if forward_batch.forward_mode.is_idle():
            return
        assert compressor.is_in_indexer
        assert compressor.ratio == 4

        with _record_function("compressor_bf16_forward_indexer_r4"):
            self._maybe_upgrade_forward_metadata()
            kv_compressed, plan, _ = _compress_to_bf16(
                self, x, forward_batch, compressor
            )
            if kv_compressed.shape[0] == 0:
                return
            out_loc_to_store = _compressed_out_loc(
                self.forward_metadata.core_metadata.c4_out_loc, plan
            )
            self.token_to_kv_pool.set_index_k_fused(
                layer_id=layer_id,
                loc=out_loc_to_store,
                cache_k=kv_compressed,
            )

    compressor_v2.CompressorBackendMixin.forward_unified = forward_unified
    compressor_v2.CompressorBackendMixin.forward_core_compressor = forward_unified
    compressor_v2.CompressorBackendMixin.forward_indexer_compressor = forward_indexer_compressor


def _trim_rows(page_indices: torch.Tensor, lengths: torch.Tensor, q_tokens: int):
    with _record_function("trim_sparse_index_rows"):
        return trim_and_pad_rows(page_indices, lengths, q_tokens)


def _gather_bf16_kv(buffer: torch.Tensor, indices: torch.Tensor, lengths: torch.Tensor, total_topk: int):
    from triton_kernels import gather_bf16_kv

    with _record_function("gather_bf16_kv"):
        return gather_bf16_kv(buffer, indices, lengths, total_topk)


def _prepare_sparse_metadata(
    page_indices: torch.Tensor,
    lengths: torch.Tensor,
    q_tokens: int,
):
    if page_indices.ndim == 3:
        page_indices = page_indices.squeeze(1)
    if page_indices.shape[0] != q_tokens or lengths.shape[0] != q_tokens:
        page_indices, lengths = _trim_rows(page_indices, lengths, q_tokens)
    else:
        lengths = lengths.to(torch.int32)
    lengths = torch.clamp(lengths, min=0, max=page_indices.shape[-1])
    return page_indices, lengths


def _patch_deepseek_v4_backend() -> None:
    from sglang.srt.layers.attention import deepseek_v4_backend as dsv4_backend
    from triton_kernels import (
        direct_dual_sparse_attention,
        direct_sparse_attention,
        gather_bf16_kv_into,
        gather_bf16_kv_torch,
    )

    original_forward = dsv4_backend.DeepseekV4AttnBackend.forward
    _pad_tensor_to_size = dsv4_backend._pad_tensor_to_size
    run_unified_attention = _TRITON_COMMON.run_unified_attention

    def _get_reusable_sparse_buffers(
        self,
        q_tokens: int,
        total_topk: int,
        head_dim: int,
        device: torch.device,
    ):
        cache = _get_tensor_cache(self, "_dsv4_sparse_buffers")
        needed = (head_dim, _device_key(device))
        cached = cache.get(needed)
        if (
            cached is not None
            and cached["gathered"].shape[0] >= q_tokens
            and cached["gathered"].shape[1] >= total_topk
        ):
            return (
                cached["gathered"][:q_tokens, :total_topk, :],
                cached["invalid_mask"][:q_tokens, :total_topk],
            )
        retired = [] if cached is None else cached.setdefault("retired", [])
        if cached is not None:
            retired.extend([cached["gathered"], cached["invalid_mask"]])
        gathered = torch.empty((q_tokens, total_topk, head_dim), dtype=torch.bfloat16, device=device)
        invalid_mask = torch.empty((q_tokens, total_topk), dtype=torch.bool, device=device)
        cache[needed] = {
            "gathered": gathered,
            "invalid_mask": invalid_mask,
            "retired": retired,
        }
        return gathered, invalid_mask

    def _get_reusable_attention_outputs(
        self,
        q_tokens: int,
        q_heads: int,
        head_dim: int,
        device: torch.device,
    ):
        cache = _get_tensor_cache(self, "_dsv4_attention_outputs")
        needed = (q_heads, head_dim, _device_key(device))
        cached = cache.get(needed)
        if cached is not None and cached["output"].shape[0] >= q_tokens:
            return cached["output"][:q_tokens], cached["lse"][:q_tokens]
        retired = [] if cached is None else cached.setdefault("retired", [])
        if cached is not None:
            retired.extend([cached["output"], cached["lse"]])
        output = torch.empty((q_tokens, q_heads, head_dim), dtype=torch.bfloat16, device=device)
        lse = torch.empty((q_tokens, q_heads), dtype=torch.float32, device=device)
        cache[needed] = {
            "output": output,
            "lse": lse,
            "retired": retired,
        }
        return output, lse

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer,
        forward_batch,
        compress_ratio,
        save_kv_cache=True,
        attn_sink=None,
        **kwargs,
    ):
        token_to_kv_pool = self.token_to_kv_pool
        use_bf16_kv = getattr(token_to_kv_pool.swa_kv_pool, "dtype", None) == torch.bfloat16
        if not use_bf16_kv:
            return original_forward(
                self, q, k, v, layer, forward_batch, compress_ratio,
                save_kv_cache=save_kv_cache, attn_sink=attn_sink, **kwargs
            )

        with _record_function(f"dsv4_bf16_attention_forward_r{compress_ratio}"):
            self._maybe_upgrade_forward_metadata()
            if self.mtp_enabled and forward_batch.forward_mode.is_idle():
                return q.new_empty(q.shape[0], q.shape[1], layer.v_head_dim)

            assert k is v
            if save_kv_cache:
                with _record_function("dsv4_bf16_attention_store_cache"):
                    self.store_cache(layer.layer_id, k, forward_batch)

            core = self.forward_metadata.core_attn_metadata
            q3 = q.squeeze(1) if q.ndim == 4 else q
            q_tokens = q3.shape[0]

            with _record_function("dsv4_bf16_attention_gather_swa"):
                swa_idx = core.swa_page_indices.squeeze(1) if core.swa_page_indices.ndim == 3 else core.swa_page_indices
                swa_len = core.swa_topk_lengths
                swa_idx, swa_len = _prepare_sparse_metadata(swa_idx, swa_len, q_tokens)
                swa_topk = swa_idx.shape[-1]
                total_topk = swa_topk
                swa_buf = token_to_kv_pool.get_swa_key_buffer_radix(layer.layer_id).squeeze(2)
                head_dim = swa_buf.shape[-1]

            extra_idx = extra_len = extra_buf = None

            if compress_ratio in (4, 128):
                with _record_function(f"dsv4_bf16_attention_prepare_extra_r{compress_ratio}"):
                    if compress_ratio == 4:
                        extra_idx = (
                            core.c4_sparse_page_indices.squeeze(1)
                            if core.c4_sparse_page_indices.ndim == 3
                            else core.c4_sparse_page_indices
                        )
                        extra_len = core.c4_sparse_topk_lengths
                    else:
                        extra_idx = (
                            core.c128_page_indices.squeeze(1)
                            if core.c128_page_indices.ndim == 3
                            else core.c128_page_indices
                        )
                        extra_len = core.c128_topk_lengths_clamp1
                    extra_idx, extra_len = _prepare_sparse_metadata(
                        extra_idx, extra_len, q_tokens
                    )
                    extra_buf = token_to_kv_pool.get_extra_key_buffer(layer.layer_id)
                    assert extra_buf is not None
                    extra_buf = extra_buf.squeeze(2)
                    total_topk = swa_topk + extra_idx.shape[-1]

            qk_head_dim = q3.shape[-1]
            q_head_num = q3.shape[1]
            if _env_enabled("SGLANG_DSV4_A100_DIRECT_ATTENTION", "1"):
                with _record_function("dsv4_bf16_attention_direct_triton"):
                    if extra_idx is None:
                        out, _lse = direct_sparse_attention(
                            q3.contiguous(),
                            swa_buf,
                            swa_idx,
                            swa_len,
                            self.softmax_scale,
                            attn_sink=attn_sink,
                        )
                    else:
                        out, _lse = direct_dual_sparse_attention(
                            q3.contiguous(),
                            swa_buf,
                            swa_idx,
                            swa_len,
                            extra_buf,
                            extra_idx,
                            extra_len,
                            self.softmax_scale,
                            attn_sink=attn_sink,
                        )
                return out

            with _record_function("dsv4_bf16_attention_gather_triton"):
                gathered, invalid_mask = _get_reusable_sparse_buffers(
                    self, q_tokens, total_topk, head_dim, q3.device
                )
                if _env_enabled("SGLANG_DSV4_A100_TORCH_GATHER", "0"):
                    with _record_function("dsv4_bf16_attention_gather_torch"):
                        swa_gathered, swa_invalid = gather_bf16_kv_torch(
                            swa_buf, swa_idx, swa_len, swa_topk
                        )
                        gathered[:, :swa_topk].copy_(swa_gathered)
                        invalid_mask[:, :swa_topk].copy_(swa_invalid)
                        if extra_idx is not None:
                            extra_topk = extra_idx.shape[-1]
                            extra_gathered, extra_invalid = gather_bf16_kv_torch(
                                extra_buf, extra_idx, extra_len, extra_topk
                            )
                            gathered[:, swa_topk:total_topk].copy_(extra_gathered)
                            invalid_mask[:, swa_topk:total_topk].copy_(extra_invalid)
                else:
                    gather_bf16_kv_into(
                        swa_buf,
                        swa_idx,
                        swa_len,
                        swa_topk,
                        gathered,
                        invalid_mask,
                        0,
                    )
                    if extra_idx is not None:
                        gather_bf16_kv_into(
                            extra_buf,
                            extra_idx,
                            extra_len,
                            extra_idx.shape[-1],
                            gathered,
                            invalid_mask,
                            swa_topk,
                        )

            with _record_function("dsv4_bf16_attention_run_unified_attention"):
                out, _lse = run_unified_attention(
                    q3.contiguous(),
                    gathered.contiguous(),
                    invalid_mask.contiguous(),
                    qk_head_dim,
                    self.softmax_scale,
                    q_tokens,
                    q_head_num,
                    total_topk,
                    qk_head_dim,
                    attn_sink=attn_sink,
                )
            return out

    dsv4_backend.DeepseekV4AttnBackend.forward = forward
