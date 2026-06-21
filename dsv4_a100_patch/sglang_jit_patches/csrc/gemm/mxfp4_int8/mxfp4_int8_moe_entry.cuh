#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>
#include <sgl_kernel/utils.cuh>

#include "cutlass/cutlass.h"
#include "cutlass/device_kernel.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/layout/matrix.h"
#include "mxfp4_moe_cutlass_core.cuh"

#include <algorithm>
#include <cstdint>

namespace {

using namespace mxfp4_int8::cutlass_core;

constexpr int kMxfp4Int8MoeReduceBlock = 256;

template <int BlockM, int BlockN, bool SourceRowsAreSlots>
struct Mxfp4Int8MoeKernelSelector;

#define SGL_MXFP4_INT8_MOE_SELECT_KERNEL(TM, TN)                                  \
  template <>                                                                     \
  struct Mxfp4Int8MoeKernelSelector<TM, TN, false> {                              \
    using Kernel = Mxfp4PackedBGroupedScaledBf16CutlassKernel_##TM##x##TN##x128##_W13; \
  };                                                                              \
  template <>                                                                     \
  struct Mxfp4Int8MoeKernelSelector<TM, TN, true> {                               \
    using Kernel = Mxfp4PackedBGroupedScaledBf16CutlassKernel_##TM##x##TN##x128##_W2;  \
  }

SGL_MXFP4_INT8_MOE_SELECT_KERNEL(16, 32);
SGL_MXFP4_INT8_MOE_SELECT_KERNEL(16, 64);
SGL_MXFP4_INT8_MOE_SELECT_KERNEL(16, 128);
SGL_MXFP4_INT8_MOE_SELECT_KERNEL(32, 32);
SGL_MXFP4_INT8_MOE_SELECT_KERNEL(32, 64);
SGL_MXFP4_INT8_MOE_SELECT_KERNEL(32, 128);
SGL_MXFP4_INT8_MOE_SELECT_KERNEL(64, 32);
SGL_MXFP4_INT8_MOE_SELECT_KERNEL(64, 64);
SGL_MXFP4_INT8_MOE_SELECT_KERNEL(64, 128);
SGL_MXFP4_INT8_MOE_SELECT_KERNEL(128, 32);
SGL_MXFP4_INT8_MOE_SELECT_KERNEL(128, 64);
SGL_MXFP4_INT8_MOE_SELECT_KERNEL(128, 128);

#undef SGL_MXFP4_INT8_MOE_SELECT_KERNEL

template <typename Kernel>
void set_max_dynamic_smem_if_needed() {
  constexpr int smem_size = int(sizeof(typename Kernel::SharedStorage));
  if constexpr (smem_size > (48 << 10)) {
    cudaError_t status = cudaFuncSetAttribute(
        cutlass::Kernel<Kernel>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_size);
    host::RuntimeCheck(
        status == cudaSuccess,
        "failed to set MXFP4-int8 MoE CUTLASS dynamic shared memory attribute: ",
        cudaGetErrorString(status));
  }
}

template <int TopK>
__global__ void reduce_moe_slots_bf16_x2_kernel(
    const __nv_bfloat16* __restrict__ slots,
    const float* __restrict__ topk_weights,
    __nv_bfloat16* __restrict__ out,
    int M,
    int N,
    int total_valid_slots) {
  int pair_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int n_pairs = N >> 1;
  int total_pairs = M * n_pairs;
  if (pair_idx >= total_pairs) {
    return;
  }

  int token = pair_idx / n_pairs;
  int n = (pair_idx - token * n_pairs) << 1;
  int slot_base = token * TopK;
  float2 acc{0.0f, 0.0f};

  CUTLASS_PRAGMA_UNROLL
  for (int k = 0; k < TopK; ++k) {
    int slot = slot_base + k;
    if (slot < total_valid_slots) {
      __nv_bfloat162 packed =
          *reinterpret_cast<const __nv_bfloat162*>(
              slots + static_cast<int64_t>(slot) * N + n);
      float2 value = __bfloat1622float2(packed);
      float weight = topk_weights[slot];
      acc.x += value.x * weight;
      acc.y += value.y * weight;
    }
  }

  *reinterpret_cast<__nv_bfloat162*>(out + static_cast<int64_t>(token) * N + n) =
      __floats2bfloat162_rn(acc.x, acc.y);
}

template <int TopK>
__global__ void reduce_moe_slots_bf16_kernel(
    const __nv_bfloat16* __restrict__ slots,
    const float* __restrict__ topk_weights,
    __nv_bfloat16* __restrict__ out,
    int M,
    int N,
    int total_valid_slots) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = M * N;
  if (idx >= total) {
    return;
  }
  int token = idx / N;
  int n = idx - token * N;
  int slot_base = token * TopK;
  float acc = 0.0f;

  CUTLASS_PRAGMA_UNROLL
  for (int k = 0; k < TopK; ++k) {
    int slot = slot_base + k;
    if (slot < total_valid_slots) {
      float value = __bfloat162float(slots[static_cast<int64_t>(slot) * N + n]);
      acc += value * topk_weights[slot];
    }
  }
  out[idx] = __float2bfloat16(acc);
}

template <int HiddenSize, int IntermediateSize, int TopK, int BlockM, int BlockN, bool SourceRowsAreSlots>
struct Mxfp4Int8MoeGemm {
  using BaseKernel = typename Mxfp4Int8MoeKernelSelector<BlockM, BlockN, SourceRowsAreSlots>::Kernel;
  static constexpr int kN = SourceRowsAreSlots ? HiddenSize : (IntermediateSize * 2);
  static constexpr int kK = SourceRowsAreSlots ? IntermediateSize : HiddenSize;
  using Kernel = Mxfp4PackedBGroupedGemmKernel<
      typename BaseKernel::Mma,
      typename BaseKernel::Epilogue,
      SourceRowsAreSlots,
      kK,
      kN>;
  static constexpr int kBlockM = Kernel::Mma::Shape::kM;
  static constexpr int kBlockN = Kernel::Mma::Shape::kN;

  static void init() {
    set_max_dynamic_smem_if_needed<Kernel>();
  }

  static void run(
      tvm::ffi::TensorView a_q,
      tvm::ffi::TensorView a_scale,
      tvm::ffi::TensorView b_mxfp4,
      tvm::ffi::TensorView b_shift2,
      tvm::ffi::TensorView b_channel_scale,
      tvm::ffi::TensorView out,
      tvm::ffi::TensorView routed_out,
      tvm::ffi::TensorView topk_weights,
      tvm::ffi::TensorView sorted_token_ids,
      tvm::ffi::TensorView expert_ids,
      tvm::ffi::TensorView num_tokens_post_padded,
      int64_t num_valid_tokens,
      int64_t multi_processor_count) {
    using namespace host;

    auto device = SymbolicDevice{};
    auto MIn = SymbolicSize{"m_in"};
    auto E = SymbolicSize{"num_experts"};
    auto KBlocks = SymbolicSize{"k_blocks"};
    auto NGroups8 = SymbolicSize{"n_groups8"};
    auto ShiftStride = SymbolicSize{"shift_stride"};
    auto NumSorted = SymbolicSize{"sorted_token_capacity"};
    auto NumExpertBlocks = SymbolicSize{"expert_blocks"};
    auto TopKRows = SymbolicSize{"topk_weight_rows"};

    device.set_options<kDLCUDA>();

    TensorMatcher({MIn, kK}).with_dtype<int8_t>().with_device(device).verify(a_q);
    TensorMatcher({MIn}).with_dtype<float>().with_device(device).verify(a_scale);
    TensorMatcher({E, KBlocks, NGroups8, 128}).with_dtype<uint8_t>().with_device(device).verify(b_mxfp4);
    TensorMatcher({E, ShiftStride, NGroups8, 8}).with_dtype<uint8_t>().with_device(device).verify(b_shift2);
    TensorMatcher({E, kN}).with_dtype<float>().with_device(device).verify(b_channel_scale);
    TensorMatcher({NumSorted}).with_dtype<int32_t>().with_device(device).verify(sorted_token_ids);
    TensorMatcher({NumExpertBlocks}).with_dtype<int32_t>().with_device(device).verify(expert_ids);
    TensorMatcher({1}).with_dtype<int32_t>().with_device(device).verify(num_tokens_post_padded);
    TensorMatcher({TopKRows, TopK}).with_dtype<float>().with_device(device).verify(topk_weights);

    if constexpr (SourceRowsAreSlots) {
      auto MOut = SymbolicSize{"m_out"};
      TensorMatcher({MOut, kN}).with_dtype<bf16_t>().with_device(device).verify(out);
      TensorMatcher({num_valid_tokens, kN}).with_dtype<bf16_t>().with_device(device).verify(routed_out);
      RuntimeCheck(MOut.unwrap() * TopK >= num_valid_tokens, "W2 output rows do not cover routed slots");
      RuntimeCheck(MIn.unwrap() >= num_valid_tokens, "W2 input rows do not cover routed slots");
    } else {
      TensorMatcher({num_valid_tokens, kN}).with_dtype<bf16_t>().with_device(device).verify(out);
      TensorMatcher({num_valid_tokens, kN}).with_dtype<bf16_t>().with_device(device).verify(routed_out);
    }

    RuntimeCheck(kK % 128 == 0, "K must be divisible by 128");
    RuntimeCheck(KBlocks.unwrap() == kK / 32, "b_mxfp4 K blocks mismatch");
    RuntimeCheck(ShiftStride.unwrap() >= (KBlocks.unwrap() + 3) / 4, "b_shift2 stride mismatch");
    RuntimeCheck(NGroups8.unwrap() == (kN + 7) / 8, "packed N group mismatch");
    RuntimeCheck(num_valid_tokens >= 0, "num_valid_tokens must be non-negative");
    RuntimeCheck(multi_processor_count > 0, "multi_processor_count must be positive");
    RuntimeCheck(TopKRows.unwrap() * TopK >= num_valid_tokens, "topk_weights does not cover routed slots");

    const int64_t num_align_experts = E.unwrap() + 1;
    const int64_t max_m = (num_valid_tokens < num_align_experts)
        ? num_valid_tokens * kBlockM
        : num_valid_tokens + num_align_experts * (kBlockM - 1);
    const int grid_m_tiles = static_cast<int>(div_ceil(max_m, static_cast<int64_t>(kBlockM)));
    const int grid_n_tiles = static_cast<int>(div_ceil(static_cast<int64_t>(kN), static_cast<int64_t>(kBlockN)));
    RuntimeCheck(
        NumSorted.unwrap() >= max_m,
        "sorted_token_ids capacity is smaller than maximum aligned token count");
    RuntimeCheck(
        NumExpertBlocks.unwrap() >= grid_m_tiles,
        "expert_ids capacity is smaller than maximum M tile count");

    const DLDevice dl_device = device.unwrap();
    const cudaStream_t stream = LaunchKernel::resolve_device(dl_device);

    const bool persistent =
        kBlockM >= 128 &&
        kBlockN >= 64 &&
        grid_m_tiles * grid_n_tiles > multi_processor_count * 2;

    cutlass::gemm::GemmCoord problem_size(
        static_cast<int>(max_m),
        kN,
        kK);
    typename Kernel::Params params{
        problem_size,
        {static_cast<const int8_t*>(a_q.data_ptr()), kK},
        static_cast<const uint8_t*>(b_mxfp4.data_ptr()),
        static_cast<const uint8_t*>(b_shift2.data_ptr()),
        static_cast<int>(b_shift2.shape()[2]),
        static_cast<int64_t>(b_mxfp4.shape()[1]) * b_mxfp4.shape()[2] * b_mxfp4.shape()[3],
        static_cast<int64_t>(b_shift2.shape()[1]) * b_shift2.shape()[2] * b_shift2.shape()[3],
        static_cast<const int32_t*>(expert_ids.data_ptr()),
        static_cast<const int32_t*>(sorted_token_ids.data_ptr()),
        static_cast<const int32_t*>(num_tokens_post_padded.data_ptr()),
        static_cast<int>(num_valid_tokens),
        TopK,
        grid_m_tiles,
        grid_n_tiles,
        persistent,
        typename Kernel::Epilogue::OutputTileIterator::Params(cutlass::layout::RowMajor(kN)),
        {nullptr, cutlass::layout::RowMajor(kN)},
        typename Kernel::Epilogue::OutputTileIterator::Params(
            cutlass::layout::RowMajor(kN),
            static_cast<const float*>(a_scale.data_ptr()),
            static_cast<const float*>(b_channel_scale.data_ptr()),
            static_cast<const int32_t*>(sorted_token_ids.data_ptr()),
            static_cast<const int32_t*>(expert_ids.data_ptr()),
            static_cast<int>(num_valid_tokens),
            kBlockM,
            TopK,
            kN),
        {reinterpret_cast<__nv_bfloat16*>(routed_out.data_ptr()), cutlass::layout::RowMajor(kN)},
        {}};

    dim3 grid;
    if (persistent) {
      int blocks = multi_processor_count * 2;
      int total_tiles = grid_m_tiles * grid_n_tiles;
      grid = dim3(static_cast<unsigned>(std::min(blocks, total_tiles)), 1, 1);
    } else {
      grid = dim3(static_cast<unsigned>(grid_m_tiles), static_cast<unsigned>(grid_n_tiles), 1);
    }

    dim3 block(Kernel::kThreadCount, 1, 1);
    constexpr int smem_size = int(sizeof(typename Kernel::SharedStorage));
    cutlass::Kernel<Kernel><<<grid, block, smem_size, stream>>>(params);
    host::RuntimeDeviceCheck();

    if constexpr (SourceRowsAreSlots) {
      int m_out = static_cast<int>(out.shape()[0]);
      if constexpr ((kN & 1) == 0 && TopK <= 8) {
        int total_pairs = (m_out * kN) >> 1;
        int reduce_grid = static_cast<int>(div_ceil(total_pairs, kMxfp4Int8MoeReduceBlock));
        LaunchKernel(
            dim3(static_cast<unsigned>(reduce_grid), 1, 1),
            dim3(kMxfp4Int8MoeReduceBlock, 1, 1),
            stream)(
            reduce_moe_slots_bf16_x2_kernel<TopK>,
            reinterpret_cast<const __nv_bfloat16*>(routed_out.data_ptr()),
            static_cast<const float*>(topk_weights.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
            m_out,
            kN,
            static_cast<int>(num_valid_tokens));
      } else {
        int total = m_out * kN;
        int reduce_grid = static_cast<int>(div_ceil(total, kMxfp4Int8MoeReduceBlock));
        LaunchKernel(
            dim3(static_cast<unsigned>(reduce_grid), 1, 1),
            dim3(kMxfp4Int8MoeReduceBlock, 1, 1),
            stream)(
            reduce_moe_slots_bf16_kernel<TopK>,
            reinterpret_cast<const __nv_bfloat16*>(routed_out.data_ptr()),
            static_cast<const float*>(topk_weights.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
            m_out,
            kN,
            static_cast<int>(num_valid_tokens));
      }
    }
  }
};

}  // namespace
