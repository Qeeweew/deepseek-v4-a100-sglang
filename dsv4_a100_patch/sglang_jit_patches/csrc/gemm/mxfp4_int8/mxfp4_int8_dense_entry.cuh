#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>
#include <sgl_kernel/utils.cuh>

#include "cutlass/cutlass.h"
#include "cutlass/device_kernel.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/layout/matrix.h"
#include "mxfp4_cutlass_core.cuh"

#include <algorithm>
#include <cstdint>

namespace {

using namespace mxfp4_int8::cutlass_core;

constexpr int kMxfp4Int8DenseReduceBlock = 256;
constexpr int kMxfp4Int8DenseMaxSplitK = 4;

#define MXFP4_DENSE_LAUNCH_SPLITK_PARTIAL(KernelType)                 \
  launch_dense_mxfp4_int8_splitk_partial<KernelType>(                  \
      a_q, b_mxfp4, b_shift2, partial, m, N, K, kAccLd, split_k_slices, stream)

#define MXFP4_DENSE_LAUNCH_GEMM(KernelType)                            \
  launch_dense_mxfp4_int8_gemm<KernelType>(                            \
      a_q,                                                             \
      a_scale,                                                         \
      b_mxfp4,                                                         \
      b_shift2,                                                        \
      b_channel_scale,                                                 \
      out,                                                             \
      m,                                                               \
      N,                                                               \
      K,                                                               \
      static_cast<int>(multi_processor_count),                         \
      stream)

template <typename Kernel>
void set_dense_max_dynamic_smem_if_needed() {
  constexpr int smem_size = int(sizeof(typename Kernel::SharedStorage));
  if constexpr (smem_size > (48 << 10)) {
    cudaError_t status = cudaFuncSetAttribute(
        cutlass::Kernel<Kernel>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_size);
    host::RuntimeCheck(
        status == cudaSuccess,
        "failed to set dense MXFP4-int8 CUTLASS dynamic shared memory attribute: ",
        cudaGetErrorString(status));
  }
}

__global__ void reduce_splitk_scale_to_bf16_kernel(
    const int32_t* __restrict__ partial,
    const float* __restrict__ a_scale,
    const float* __restrict__ b_channel_scale,
    __nv_bfloat16* __restrict__ out,
    int M,
    int N,
    int acc_ld,
    int split_k_slices) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = M * N;
  if (idx >= total) {
    return;
  }
  int m = idx / N;
  int n = idx - m * N;
  int32_t acc = 0;
  int64_t slice_stride = static_cast<int64_t>(M) * acc_ld;
  for (int s = 0; s < split_k_slices; ++s) {
    acc += partial[static_cast<int64_t>(s) * slice_stride + m * acc_ld + n];
  }
  float value = static_cast<float>(acc) * a_scale[m] * b_channel_scale[n];
  out[idx] = __float2bfloat16(value);
}

template <int SplitK>
__global__ void reduce_splitk_scale_to_bf16_x2_kernel(
    const int32_t* __restrict__ partial,
    const float* __restrict__ a_scale,
    const float* __restrict__ b_channel_scale,
    __nv_bfloat16* __restrict__ out,
    int M,
    int N,
    int acc_ld) {
  int pair_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total_pairs = (M * N) >> 1;
  if (pair_idx >= total_pairs) {
    return;
  }
  int idx = pair_idx << 1;
  int m = idx / N;
  int n = idx - m * N;
  int32_t acc0 = 0;
  int32_t acc1 = 0;
  int64_t slice_stride = static_cast<int64_t>(M) * acc_ld;
  CUTLASS_PRAGMA_UNROLL
  for (int s = 0; s < SplitK; ++s) {
    int64_t base = static_cast<int64_t>(s) * slice_stride + m * acc_ld + n;
    int2 p = *reinterpret_cast<const int2*>(partial + base);
    acc0 += p.x;
    acc1 += p.y;
  }
  float a = a_scale[m];
  float v0 = static_cast<float>(acc0) * a * b_channel_scale[n];
  float v1 = static_cast<float>(acc1) * a * b_channel_scale[n + 1];
  *reinterpret_cast<__nv_bfloat162*>(out + idx) =
      __floats2bfloat162_rn(v0, v1);
}

template <int SplitK>
void launch_reduce_splitk_scale_to_bf16_x2(
    const int32_t* partial,
    const float* a_scale,
    const float* b_channel_scale,
    __nv_bfloat16* out,
    int M,
    int N,
    int acc_ld,
    int grid,
    cudaStream_t stream) {
  reduce_splitk_scale_to_bf16_x2_kernel<SplitK><<<
      grid,
      kMxfp4Int8DenseReduceBlock,
      0,
      stream>>>(partial, a_scale, b_channel_scale, out, M, N, acc_ld);
}

void dispatch_reduce_splitk_scale_to_bf16(
    const int32_t* partial,
    const float* a_scale,
    const float* b_channel_scale,
    __nv_bfloat16* out,
    int M,
    int N,
    int acc_ld,
    int split_k_slices,
    cudaStream_t stream) {
  if ((N & 1) == 0) {
    int total_pairs = (M * N) >> 1;
    int grid = static_cast<int>(host::div_ceil(
        static_cast<int64_t>(total_pairs),
        static_cast<int64_t>(kMxfp4Int8DenseReduceBlock)));
    if (split_k_slices == 2) {
      launch_reduce_splitk_scale_to_bf16_x2<2>(
          partial, a_scale, b_channel_scale, out, M, N, acc_ld, grid, stream);
    } else if (split_k_slices == 4) {
      launch_reduce_splitk_scale_to_bf16_x2<4>(
          partial, a_scale, b_channel_scale, out, M, N, acc_ld, grid, stream);
    } else {
      launch_reduce_splitk_scale_to_bf16_x2<8>(
          partial, a_scale, b_channel_scale, out, M, N, acc_ld, grid, stream);
    }
  } else {
    int total = M * N;
    int grid = static_cast<int>(host::div_ceil(
        static_cast<int64_t>(total),
        static_cast<int64_t>(kMxfp4Int8DenseReduceBlock)));
    reduce_splitk_scale_to_bf16_kernel<<<
        grid,
        kMxfp4Int8DenseReduceBlock,
        0,
        stream>>>(
        partial,
        a_scale,
        b_channel_scale,
        out,
        M,
        N,
        acc_ld,
        split_k_slices);
  }
  host::RuntimeDeviceCheck();
}

template <typename Kernel>
void launch_dense_mxfp4_int8_gemm(
    tvm::ffi::TensorView a_q,
    tvm::ffi::TensorView a_scale,
    tvm::ffi::TensorView b_mxfp4,
    tvm::ffi::TensorView b_shift2,
    tvm::ffi::TensorView b_channel_scale,
    tvm::ffi::TensorView out,
    int m,
    int n,
    int k,
    int multi_processor_count,
    cudaStream_t stream) {
  cutlass::gemm::GemmCoord problem_size(m, n, k);
  Mxfp4ThreadblockSwizzle swizzle;
  cutlass::gemm::GemmCoord grid_shape = swizzle.get_tiled_shape(
      problem_size,
      {Kernel::Mma::Shape::kM, Kernel::Mma::Shape::kN, Kernel::Mma::Shape::kK},
      1);
  int grid_m_tiles = static_cast<int>(grid_shape.m());
  int grid_n_tiles = static_cast<int>(grid_shape.n());

  typename Kernel::Params params{
      problem_size,
      grid_shape,
      grid_m_tiles,
      grid_n_tiles,
      {static_cast<const int8_t*>(a_q.data_ptr()), k},
      static_cast<const uint8_t*>(b_mxfp4.data_ptr()),
      static_cast<const uint8_t*>(b_shift2.data_ptr()),
      static_cast<int>(b_shift2.shape()[1]),
      typename Kernel::Epilogue::OutputTileIterator::Params(
          cutlass::layout::RowMajor(n)),
      {nullptr, cutlass::layout::RowMajor(n)},
      typename Kernel::Epilogue::OutputTileIterator::Params(
          cutlass::layout::RowMajor(n),
          static_cast<const float*>(a_scale.data_ptr()),
          static_cast<const float*>(b_channel_scale.data_ptr())),
      {reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
       cutlass::layout::RowMajor(n)},
      {},
      1,
      0};

  dim3 grid;
  if constexpr (Kernel::kPersistent) {
    int blocks = multi_processor_count * 2;
    int total_tiles = grid_m_tiles * grid_n_tiles;
    grid = dim3(static_cast<unsigned>(std::min(blocks, total_tiles)), 1, 1);
  } else {
    grid = swizzle.get_grid_shape(grid_shape);
  }

  dim3 block(Kernel::kThreadCount, 1, 1);
  constexpr int smem_size = int(sizeof(typename Kernel::SharedStorage));
  cutlass::Kernel<Kernel><<<grid, block, smem_size, stream>>>(params);
  host::RuntimeDeviceCheck();
}

template <typename Kernel>
void launch_dense_mxfp4_int8_splitk_partial(
    tvm::ffi::TensorView a_q,
    tvm::ffi::TensorView b_mxfp4,
    tvm::ffi::TensorView b_shift2,
    tvm::ffi::TensorView partial,
    int m,
    int n,
    int k,
    int acc_ld,
    int split_k_slices,
    cudaStream_t stream) {
  cutlass::gemm::GemmCoord problem_size(m, n, k);
  Mxfp4ThreadblockSwizzle swizzle;
  cutlass::gemm::GemmCoord grid_shape = swizzle.get_tiled_shape(
      problem_size,
      {Kernel::Mma::Shape::kM, Kernel::Mma::Shape::kN, Kernel::Mma::Shape::kK},
      split_k_slices);

  int32_t* partial_ptr = static_cast<int32_t*>(partial.data_ptr());
  typename Kernel::Params params{
      problem_size,
      grid_shape,
      static_cast<int>(grid_shape.m()),
      static_cast<int>(grid_shape.n()),
      {static_cast<const int8_t*>(a_q.data_ptr()), k},
      static_cast<const uint8_t*>(b_mxfp4.data_ptr()),
      static_cast<const uint8_t*>(b_shift2.data_ptr()),
      static_cast<int>(b_shift2.shape()[1]),
      typename Kernel::Epilogue::OutputTileIterator::Params(
          cutlass::layout::RowMajor(acc_ld)),
      {partial_ptr, cutlass::layout::RowMajor(acc_ld)},
      typename Kernel::Epilogue::OutputTileIterator::Params(
          cutlass::layout::RowMajor(acc_ld)),
      {partial_ptr, cutlass::layout::RowMajor(acc_ld)},
      {1, 0},
      split_k_slices,
      static_cast<int>(static_cast<int64_t>(m) * acc_ld)};

  dim3 grid = swizzle.get_grid_shape(grid_shape);
  grid.z = static_cast<unsigned>(split_k_slices);
  dim3 block(Kernel::kThreadCount, 1, 1);
  constexpr int smem_size = int(sizeof(typename Kernel::SharedStorage));
  cutlass::Kernel<Kernel><<<grid, block, smem_size, stream>>>(params);
  host::RuntimeDeviceCheck();
}

int choose_dense_split_k_slices(
    int m,
    int n,
    int k,
    int multi_processor_count) {
  if (m > 128 || k % Mxfp4ThreadblockShape::kK != 0) {
    return 1;
  }

  int64_t tile_m = 128;
  int64_t tile_n = 64;
  if (m <= 16) {
    tile_m = 16;
  } else if (m <= 32) {
    tile_m = 32;
  } else if (m <= 64) {
    tile_m = 64;
  }

  int64_t grid_m = host::div_ceil(static_cast<int64_t>(m), tile_m);
  int64_t grid_n = host::div_ceil(static_cast<int64_t>(n), tile_n);
  int64_t base_tiles = grid_m * grid_n;
  int64_t target_tiles = static_cast<int64_t>(multi_processor_count) * 4;
  if (kMxfp4Int8DenseMaxSplitK >= 8 && base_tiles * 4 < target_tiles) {
    return 8;
  }
  if (kMxfp4Int8DenseMaxSplitK >= 4 && base_tiles * 2 < target_tiles) {
    return 4;
  }
  if (kMxfp4Int8DenseMaxSplitK >= 2 && base_tiles < target_tiles) {
    return 2;
  }
  return 2;
}

template <int K, int N>
struct Mxfp4Int8DenseGemm {
  static void init() {
    set_dense_max_dynamic_smem_if_needed<Mxfp4PackedBOnDemandCutlassKernel16N32K128>();
    set_dense_max_dynamic_smem_if_needed<Mxfp4PackedBOnDemandCutlassKernel32N64K128>();
    set_dense_max_dynamic_smem_if_needed<Mxfp4PackedBOnDemandCutlassKernel64N64K128>();
    set_dense_max_dynamic_smem_if_needed<Mxfp4PackedBCutlassKernel128N64K128>();
    set_dense_max_dynamic_smem_if_needed<Mxfp4PackedBScaledBf16CutlassKernel>();
    set_dense_max_dynamic_smem_if_needed<Mxfp4PackedBScaledBf16PersistentCutlassKernel>();
    set_dense_max_dynamic_smem_if_needed<Mxfp4PackedBScaledBf16_256x64x64_CutlassKernel>();
    set_dense_max_dynamic_smem_if_needed<Mxfp4PackedBScaledBf16Persistent_256x64x64_CutlassKernel>();
  }

  static void run(
      tvm::ffi::TensorView a_q,
      tvm::ffi::TensorView a_scale,
      tvm::ffi::TensorView b_mxfp4,
      tvm::ffi::TensorView b_shift2,
      tvm::ffi::TensorView b_channel_scale,
      tvm::ffi::TensorView out,
      tvm::ffi::TensorView partial,
      int64_t multi_processor_count) {
    using namespace host;

    auto device = SymbolicDevice{};
    auto M = SymbolicSize{"m"};
    auto KBlocks = SymbolicSize{"k_blocks"};
    auto NGroups8 = SymbolicSize{"n_groups8"};
    auto ShiftStride = SymbolicSize{"shift_stride"};

    device.set_options<kDLCUDA>();

    TensorMatcher({M, K}).with_dtype<int8_t>().with_device(device).verify(a_q);
    TensorMatcher({M}).with_dtype<float>().with_device(device).verify(a_scale);
    TensorMatcher({KBlocks, NGroups8, 128}).with_dtype<uint8_t>().with_device(device).verify(b_mxfp4);
    TensorMatcher({ShiftStride, NGroups8, 8}).with_dtype<uint8_t>().with_device(device).verify(b_shift2);
    TensorMatcher({N}).with_dtype<float>().with_device(device).verify(b_channel_scale);
    TensorMatcher({M, N}).with_dtype<bf16_t>().with_device(device).verify(out);

    RuntimeCheck(K % 32 == 0, "K must be divisible by 32");
    RuntimeCheck(
        K <= Mxfp4ThreadblockShape::kK || K % Mxfp4ThreadblockShape::kK == 0,
        "K must be <= 64 or divisible by 64 for current direct MXFP4 path");
    RuntimeCheck(KBlocks.unwrap() == K / 32, "b_mxfp4 K blocks mismatch");
    RuntimeCheck(NGroups8.unwrap() == (N + 7) / 8, "packed N group mismatch");
    RuntimeCheck(ShiftStride.unwrap() >= (KBlocks.unwrap() + 3) / 4, "b_shift2 stride mismatch");
    RuntimeCheck(multi_processor_count > 0, "multi_processor_count must be positive");

    const int m = static_cast<int>(M.unwrap());
    const DLDevice dl_device = device.unwrap();
    const cudaStream_t stream = LaunchKernel::resolve_device(dl_device);
    const int split_k_slices = choose_dense_split_k_slices(
        m, N, K, static_cast<int>(multi_processor_count));
    if (split_k_slices > 1) {
      constexpr int kAccLd = ((N + 31) / 32) * 32;
      auto PartialSplit = SymbolicSize{"partial_split"};
      TensorMatcher({PartialSplit, M, kAccLd})
          .with_dtype<int32_t>()
          .with_device(device)
          .verify(partial);
      RuntimeCheck(
          PartialSplit.unwrap() >= split_k_slices,
          "partial split dimension is smaller than selected split-k slices");
      if (m <= 16) {
        MXFP4_DENSE_LAUNCH_SPLITK_PARTIAL(
            Mxfp4PackedBOnDemandCutlassKernel16N32K128);
      } else if (m <= 32) {
        MXFP4_DENSE_LAUNCH_SPLITK_PARTIAL(
            Mxfp4PackedBOnDemandCutlassKernel32N64K128);
      } else if (m <= 64) {
        MXFP4_DENSE_LAUNCH_SPLITK_PARTIAL(
            Mxfp4PackedBOnDemandCutlassKernel64N64K128);
      } else {
        MXFP4_DENSE_LAUNCH_SPLITK_PARTIAL(
            Mxfp4PackedBCutlassKernel128N64K128);
      }
      dispatch_reduce_splitk_scale_to_bf16(
          static_cast<const int32_t*>(partial.data_ptr()),
          static_cast<const float*>(a_scale.data_ptr()),
          static_cast<const float*>(b_channel_scale.data_ptr()),
          reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
          m,
          N,
          kAccLd,
          split_k_slices,
          stream);
      return;
    }

    const bool use_n64_tile = m >= 256;
    const int grid_m_tiles = static_cast<int>(
        div_ceil(static_cast<int64_t>(m), static_cast<int64_t>(Mxfp4PackedBThreadblockMma::Shape::kM)));
    const int grid_n_tiles = static_cast<int>(
        div_ceil(static_cast<int64_t>(N), static_cast<int64_t>(Mxfp4PackedBThreadblockMma::Shape::kN)));
    const bool use_persistent =
        m >= 4096 &&
        N >= 8192 &&
        K >= 8192 &&
        grid_m_tiles * grid_n_tiles > multi_processor_count * 2;

    if (use_persistent && use_n64_tile) {
      MXFP4_DENSE_LAUNCH_GEMM(
          Mxfp4PackedBScaledBf16Persistent_256x64x64_CutlassKernel);
    } else if (use_n64_tile) {
      MXFP4_DENSE_LAUNCH_GEMM(
          Mxfp4PackedBScaledBf16_256x64x64_CutlassKernel);
    } else if (use_persistent) {
      MXFP4_DENSE_LAUNCH_GEMM(
          Mxfp4PackedBScaledBf16PersistentCutlassKernel);
    } else {
      MXFP4_DENSE_LAUNCH_GEMM(Mxfp4PackedBScaledBf16CutlassKernel);
    }
  }
};

#undef MXFP4_DENSE_LAUNCH_GEMM
#undef MXFP4_DENSE_LAUNCH_SPLITK_PARTIAL

}  // namespace
