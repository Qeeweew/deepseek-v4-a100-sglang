#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "cutlass/cutlass.h"
#include "cutlass/arch/memory_sm80.h"
#include "cutlass/arch/mma_sm80.h"
#include "cutlass/device_kernel.h"
#include "cutlass/epilogue/thread/linear_combination_clamp.h"
#include "cutlass/epilogue/threadblock/default_epilogue_tensor_op.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/threadblock/default_mma_core_sm80.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/transform/threadblock/predicated_tile_iterator.h"

#include <cstdint>
#include <type_traits>

#ifndef MXFP4_SMALL_M_THREADBLOCK_N
#define MXFP4_SMALL_M_THREADBLOCK_N 64
#endif

#ifndef MXFP4_SMALL_M_WARP_N
#define MXFP4_SMALL_M_WARP_N 16
#endif

#ifndef MXFP4_SMALL_M_THREADBLOCK_K
#define MXFP4_SMALL_M_THREADBLOCK_K 128
#endif

namespace mxfp4_int8 {
namespace cutlass_core {

__device__ __forceinline__ uint32_t negate_i8x4_small_magnitude(uint32_t magnitude) {
  return (0x80808080u - magnitude) ^ 0x80808080u;
}

__device__ __forceinline__ uint32_t lookup_mxfp4_x1_byte_perm(
    uint32_t q4,
    uint8_t shift) {
  uint32_t magnitude = __byte_perm(0x03020100u, 0x0c080604u, q4) << shift;
  uint32_t negative = negate_i8x4_small_magnitude(magnitude);
  uint32_t select = 0x3210u | ((q4 & 0x8888u) >> 1);
  return __byte_perm(magnitude, negative, select);
}

__device__ __forceinline__ uint32_t decode_mxfp4_4_linear(uint32_t q4, uint8_t shift) {
  return lookup_mxfp4_x1_byte_perm(q4, shift);
}

__device__ __forceinline__ void lookup_mxfp4_x2_byte_perm(
    uint32_t q4_pair,
    uint8_t shift,
    uint32_t& out_lo,
    uint32_t& out_hi) {
  uint32_t magnitude_lo = __byte_perm(0x03020100u, 0x0c080604u, q4_pair) << shift;
  uint32_t q4_hi = q4_pair >> 16;
  uint32_t magnitude_hi = __byte_perm(0x03020100u, 0x0c080604u, q4_hi) << shift;
  uint32_t select_pair = 0x32103210u | ((q4_pair & 0x88888888u) >> 1);
  out_lo = __byte_perm(
      magnitude_lo,
      negate_i8x4_small_magnitude(magnitude_lo),
      select_pair);
  out_hi = __byte_perm(
      magnitude_hi,
      negate_i8x4_small_magnitude(magnitude_hi),
      select_pair >> 16);
}

using Mxfp4ThreadblockShape = cutlass::gemm::GemmShape<256, 128, 64>;
using Mxfp4WarpShape = cutlass::gemm::GemmShape<128, 32, 64>;
using Mxfp4InstructionShape = cutlass::gemm::GemmShape<16, 8, 32>;
using Mxfp4EpilogueOp = cutlass::epilogue::thread::LinearCombinationClamp<
    int32_t,
    128 / cutlass::sizeof_bits<int32_t>::value,
    int32_t,
    int32_t>;

template <int Count>
struct IdentityInt32OutputOp {
  using ElementOutput = int32_t;
  using ElementAccumulator = int32_t;
  using ElementCompute = int32_t;
  using ElementScalar = int32_t;
  using ElementC = int32_t;
  using ElementD = int32_t;

  static int const kCount = Count;
  using FragmentOutput = cutlass::Array<int32_t, kCount>;
  using FragmentSource = cutlass::Array<int32_t, kCount>;
  using FragmentAccumulator = cutlass::Array<int32_t, kCount>;
  using FragmentCompute = cutlass::Array<int32_t, kCount>;

  struct Params {};

  CUTLASS_HOST_DEVICE
  explicit IdentityInt32OutputOp(Params const& = Params()) {}

  CUTLASS_HOST_DEVICE
  bool is_source_needed() const {
    return false;
  }

  CUTLASS_HOST_DEVICE
  void set_k_partition(int, int) {}

  CUTLASS_HOST_DEVICE
  FragmentOutput operator()(FragmentAccumulator const& accumulator) const {
    return accumulator;
  }

  CUTLASS_HOST_DEVICE
  FragmentOutput operator()(
      FragmentAccumulator const& accumulator,
      FragmentSource const&) const {
    return accumulator;
  }
};

template <typename ThreadMap_>
class ScaledBf16OutputTileIterator {
 public:
  using ThreadMap = ThreadMap_;
  using Shape = typename ThreadMap::Shape;
  using Element = int32_t;
  using Layout = cutlass::layout::RowMajor;
  using TensorRef = cutlass::TensorRef<__nv_bfloat16, Layout>;
  using ConstTensorRef = typename TensorRef::ConstTensorRef;
  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;
  using TensorCoord = cutlass::MatrixCoord;

  static int const kElementsPerAccess = ThreadMap::kElementsPerAccess;
  static int const kThreads = ThreadMap::kThreads;
  static int const kIterations = ThreadMap::Count::kTile;

  using Fragment = cutlass::Array<
      int32_t,
      ThreadMap::Iterations::kColumn * ThreadMap::Iterations::kRow *
          ThreadMap::Iterations::kGroup * ThreadMap::Iterations::kCluster *
          ThreadMap::kElementsPerAccess>;
  using AccessType = cutlass::AlignedArray<int32_t, ThreadMap::kElementsPerAccess>;

  struct Params : cutlass::epilogue::threadblock::PredicatedTileIteratorParams {
    using Base = cutlass::epilogue::threadblock::PredicatedTileIteratorParams;

    float const* a_scale;
    float const* b_channel_scale;
    int output_stride;

    CUTLASS_HOST_DEVICE
    Params()
        : Base(),
          a_scale(nullptr),
          b_channel_scale(nullptr),
          output_stride(0) {}

    CUTLASS_HOST_DEVICE
    explicit Params(Layout const& layout)
        : Base(
              layout.stride(0) * int(sizeof(__nv_bfloat16)),
              cutlass::epilogue::threadblock::make_OutputTileThreadMapDesc<ThreadMap>()),
          a_scale(nullptr),
          b_channel_scale(nullptr),
          output_stride(layout.stride(0)) {}

    CUTLASS_HOST_DEVICE
    Params(
        Layout const& layout,
        float const* a_scale_,
        float const* b_channel_scale_)
        : Base(
              layout.stride(0) * int(sizeof(__nv_bfloat16)),
              cutlass::epilogue::threadblock::make_OutputTileThreadMapDesc<ThreadMap>()),
          a_scale(a_scale_),
          b_channel_scale(b_channel_scale_),
          output_stride(layout.stride(0)) {}
  };

  struct Mask {
    static int const kCount = ThreadMap::Iterations::kColumn;
    bool predicates[kCount];

    CUTLASS_HOST_DEVICE
    Mask() {
      enable();
    }

    CUTLASS_HOST_DEVICE
    void clear() {
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < kCount; ++i) {
        predicates[i] = false;
      }
    }

    CUTLASS_DEVICE
    void enable() {
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < kCount; ++i) {
        predicates[i] = true;
      }
    }
  };

 private:
  Params params_;
  uint8_t* byte_pointer_;
  Mask mask_;
  Index extent_row_;
  Index extent_column_;
  Index thread_start_row_;
  Index thread_start_column_;
  bool no_bounds_check_;
  int state_[3];

 public:
  CUTLASS_DEVICE
  ScaledBf16OutputTileIterator(
      Params const& params,
      __nv_bfloat16* pointer,
      TensorCoord extent,
      int thread_idx,
      TensorCoord threadblock_offset = TensorCoord())
      : params_(params) {
    TensorCoord thread_offset = ThreadMap::initial_offset(thread_idx) + threadblock_offset;
    extent_row_ = extent.row();
    extent_column_ = extent.column();
    thread_start_row_ = thread_offset.row();
    thread_start_column_ = thread_offset.column();
    no_bounds_check_ =
        pointer && ((extent.row() & 255) == 0) && ((extent.column() & 127) == 0) &&
        ((params_.output_stride & 1) == 0);

    CUTLASS_PRAGMA_UNROLL
    for (int c = 0; c < ThreadMap::Iterations::kColumn; ++c) {
      mask_.predicates[c] =
          ((thread_offset.column() + ThreadMap::Delta::kColumn * c) < extent.column());
    }

    byte_pointer_ = reinterpret_cast<uint8_t*>(pointer) +
        LongIndex(thread_offset.row()) * LongIndex(params_.stride) +
        LongIndex(thread_offset.column()) * int(sizeof(__nv_bfloat16));

    if (!pointer) {
      mask_.clear();
    }

    state_[0] = state_[1] = state_[2] = 0;
  }

  CUTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {
    byte_pointer_ += pointer_offset * int(sizeof(__nv_bfloat16));
  }

  CUTLASS_DEVICE
  void clear_mask() {
    mask_.clear();
  }

  CUTLASS_DEVICE
  void load(Fragment& frag) const {
    frag.clear();
  }

  CUTLASS_DEVICE
  void store(Fragment const& frag) const {
    uint8_t* byte_pointer = byte_pointer_;
    AccessType const* frag_ptr = reinterpret_cast<AccessType const*>(&frag);

    if constexpr (ThreadMap::kElementsPerAccess == 4) {
      if (no_bounds_check_) {
        CUTLASS_PRAGMA_UNROLL
        for (int cluster = 0; cluster < ThreadMap::Iterations::kCluster; ++cluster) {
          CUTLASS_PRAGMA_UNROLL
          for (int group = 0; group < ThreadMap::Iterations::kGroup; ++group) {
            CUTLASS_PRAGMA_UNROLL
            for (int row = 0; row < ThreadMap::Iterations::kRow; ++row) {
              int frag_row_idx =
                  row + ThreadMap::Iterations::kRow *
                            (group + ThreadMap::Iterations::kGroup * cluster);
              int row_offset = row * ThreadMap::Delta::kRow +
                  group * ThreadMap::Delta::kGroup +
                  cluster * ThreadMap::Delta::kCluster;
              int global_m = row_offset + thread_start_row_;
              float a = params_.a_scale[global_m];
              uint8_t* row_pointer = byte_pointer;

              CUTLASS_PRAGMA_UNROLL
              for (int column = 0; column < ThreadMap::Iterations::kColumn; ++column) {
                int column_offset = column * ThreadMap::Delta::kColumn;
                int global_n = column_offset + thread_start_column_;
                AccessType const& access =
                    frag_ptr[frag_row_idx * ThreadMap::Iterations::kColumn + column];
                float4 b = *reinterpret_cast<float4 const*>(
                    params_.b_channel_scale + global_n);
                float v0 = static_cast<float>(access[0]) * a * b.x;
                float v1 = static_cast<float>(access[1]) * a * b.y;
                float v2 = static_cast<float>(access[2]) * a * b.z;
                float v3 = static_cast<float>(access[3]) * a * b.w;
                __nv_bfloat16* dst = reinterpret_cast<__nv_bfloat16*>(
                    row_pointer + column_offset * int(sizeof(__nv_bfloat16)));
                *reinterpret_cast<__nv_bfloat162*>(dst + 0) =
                    __floats2bfloat162_rn(v0, v1);
                *reinterpret_cast<__nv_bfloat162*>(dst + 2) =
                    __floats2bfloat162_rn(v2, v3);
              }

              if (row + 1 < ThreadMap::Iterations::kRow) {
                byte_pointer += params_.increment_row;
              }
            }

            if (group + 1 < ThreadMap::Iterations::kGroup) {
              byte_pointer += params_.increment_group;
            }
          }

          if (cluster + 1 < ThreadMap::Iterations::kCluster) {
            byte_pointer += params_.increment_cluster;
          }
        }
        return;
      }
    }

    CUTLASS_PRAGMA_UNROLL
    for (int cluster = 0; cluster < ThreadMap::Iterations::kCluster; ++cluster) {
      CUTLASS_PRAGMA_UNROLL
      for (int group = 0; group < ThreadMap::Iterations::kGroup; ++group) {
        CUTLASS_PRAGMA_UNROLL
        for (int row = 0; row < ThreadMap::Iterations::kRow; ++row) {
          int frag_row_idx =
              row + ThreadMap::Iterations::kRow *
                        (group + ThreadMap::Iterations::kGroup * cluster);
          int row_offset = row * ThreadMap::Delta::kRow +
              group * ThreadMap::Delta::kGroup +
              cluster * ThreadMap::Delta::kCluster;
          int global_m = row_offset + thread_start_row_;
          bool row_guard = global_m < extent_row_;
          float a = row_guard ? params_.a_scale[global_m] : 0.0f;
          uint8_t* row_pointer = byte_pointer;

          CUTLASS_PRAGMA_UNROLL
          for (int column = 0; column < ThreadMap::Iterations::kColumn; ++column) {
            int column_offset = column * ThreadMap::Delta::kColumn;
            int global_n = column_offset + thread_start_column_;
            bool guard = row_guard && mask_.predicates[column];
            AccessType const& access =
                frag_ptr[frag_row_idx * ThreadMap::Iterations::kColumn + column];

            if constexpr (ThreadMap::kElementsPerAccess == 4) {
              bool vector_store_aligned =
                  ((params_.output_stride | global_n) & 1) == 0;
              bool scale_load_aligned = (global_n & 3) == 0;
              if (guard && vector_store_aligned && global_n + 3 < extent_column_) {
                float4 b;
                if (scale_load_aligned) {
                  b = *reinterpret_cast<float4 const*>(
                      params_.b_channel_scale + global_n);
                } else {
                  b.x = params_.b_channel_scale[global_n + 0];
                  b.y = params_.b_channel_scale[global_n + 1];
                  b.z = params_.b_channel_scale[global_n + 2];
                  b.w = params_.b_channel_scale[global_n + 3];
                }
                float v0 = static_cast<float>(access[0]) * a * b.x;
                float v1 = static_cast<float>(access[1]) * a * b.y;
                float v2 = static_cast<float>(access[2]) * a * b.z;
                float v3 = static_cast<float>(access[3]) * a * b.w;
                __nv_bfloat16* dst = reinterpret_cast<__nv_bfloat16*>(
                    row_pointer + column_offset * int(sizeof(__nv_bfloat16)));
                *reinterpret_cast<__nv_bfloat162*>(dst + 0) =
                    __floats2bfloat162_rn(v0, v1);
                *reinterpret_cast<__nv_bfloat162*>(dst + 2) =
                    __floats2bfloat162_rn(v2, v3);
              } else {
                CUTLASS_PRAGMA_UNROLL
                for (int e = 0; e < ThreadMap::kElementsPerAccess; ++e) {
                  int n = global_n + e;
                  if (guard && n < extent_column_) {
                    float value = static_cast<float>(access[e]) *
                        a * params_.b_channel_scale[n];
                    reinterpret_cast<__nv_bfloat16*>(
                        row_pointer + (column_offset + e) *
                            int(sizeof(__nv_bfloat16)))[0] =
                            __float2bfloat16(value);
                  }
                }
              }
            } else {
              CUTLASS_PRAGMA_UNROLL
              for (int e = 0; e < ThreadMap::kElementsPerAccess; ++e) {
                int n = global_n + e;
                if (guard && n < extent_column_) {
                  float value = static_cast<float>(access[e]) *
                      a * params_.b_channel_scale[n];
                  reinterpret_cast<__nv_bfloat16*>(
                      row_pointer + (column_offset + e) *
                          int(sizeof(__nv_bfloat16)))[0] =
                          __float2bfloat16(value);
                }
              }
            }
          }

          if (row + 1 < ThreadMap::Iterations::kRow) {
            byte_pointer += params_.increment_row;
          }
        }

        if (group + 1 < ThreadMap::Iterations::kGroup) {
          byte_pointer += params_.increment_group;
        }
      }

      if (cluster + 1 < ThreadMap::Iterations::kCluster) {
        byte_pointer += params_.increment_cluster;
      }
    }
  }

  CUTLASS_DEVICE
  ScaledBf16OutputTileIterator& operator++() {
    ++state_[0];
    byte_pointer_ += params_.advance_row;
    thread_start_row_ += ThreadMap::Shape::kRow;

    if (state_[0] == ThreadMap::Count::kRow) {
      state_[0] = 0;
      ++state_[1];
      byte_pointer_ += params_.advance_group;
      thread_start_row_ += (ThreadMap::Shape::kGroup - 1) *
          ThreadMap::Shape::kRow * ThreadMap::Count::kRow;

      if (state_[1] == ThreadMap::Count::kGroup) {
        state_[1] = 0;
        ++state_[2];
        byte_pointer_ += params_.advance_cluster;
        thread_start_row_ += ThreadMap::Count::kGroup *
            ThreadMap::Shape::kGroup * ThreadMap::Count::kRow *
            ThreadMap::Shape::kRow;

        if (state_[2] == ThreadMap::Count::kCluster) {
          state_[2] = 0;
          byte_pointer_ += params_.advance_tile;
          thread_start_row_ += ThreadMap::Shape::kGroup *
              ThreadMap::Shape::kRow * ThreadMap::Shape::kCluster *
              ThreadMap::Shape::kTile;
        }
      }
    }

    return *this;
  }

  CUTLASS_DEVICE
  ScaledBf16OutputTileIterator& operator+=(int increment) {
    state_[0] += increment;
    int increment_row = state_[0] / ThreadMap::Count::kRow;
    state_[0] = state_[0] % ThreadMap::Count::kRow;
    byte_pointer_ += params_.advance_row * increment;
    thread_start_row_ += ThreadMap::Shape::kRow * increment;

    state_[1] += increment_row;
    int increment_group = state_[1] / ThreadMap::Count::kGroup;
    state_[1] = state_[1] % ThreadMap::Count::kGroup;
    byte_pointer_ += params_.advance_group * increment_row;
    thread_start_row_ += (ThreadMap::Shape::kGroup - 1) *
        ThreadMap::Shape::kRow * ThreadMap::Count::kRow * increment_row;

    state_[2] += increment_group;
    int increment_cluster = state_[2] / ThreadMap::Count::kCluster;
    state_[2] = state_[2] % ThreadMap::Count::kCluster;
    byte_pointer_ += params_.advance_cluster * increment_group;
    thread_start_row_ += ThreadMap::Count::kGroup *
        ThreadMap::Shape::kGroup * ThreadMap::Count::kRow *
        ThreadMap::Shape::kRow * increment_group;

    byte_pointer_ += params_.advance_tile * increment_cluster;
    thread_start_row_ += ThreadMap::Shape::kGroup *
        ThreadMap::Shape::kRow * ThreadMap::Shape::kCluster *
        ThreadMap::Shape::kTile * increment_cluster;
    return *this;
  }
};
using Mxfp4ThreadblockSwizzle =
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
using Mxfp4MmaCore = cutlass::gemm::threadblock::DefaultMmaCore<
    Mxfp4ThreadblockShape,
    Mxfp4WarpShape,
    Mxfp4InstructionShape,
    int8_t,
    cutlass::layout::RowMajor,
    int8_t,
    cutlass::layout::ColumnMajor,
    int32_t,
    cutlass::layout::RowMajor,
    cutlass::arch::OpClassTensorOp,
    2,
    cutlass::arch::OpMultiplyAddSaturate>;
template <typename MmaCore_, bool EnableRoutedRows_ = false>
struct Mxfp4PackedBThreadblockMmaSkeleton {
  using MmaCore = MmaCore_;
  static bool const kEnableRoutedRows = EnableRoutedRows_;
  using Shape = typename MmaCore::Shape;
  using Policy = typename MmaCore::MmaPolicy;
  using Operator = typename Policy::Operator;
  using WarpCount = typename MmaCore::WarpCount;
  using FragmentC = typename Operator::FragmentC;
  using SmemAccessIteratorA = cutlass::transform::threadblock::RegularTileAccessIterator<
      cutlass::MatrixShape<Shape::kM, Shape::kK>,
      int8_t,
      typename MmaCore::SmemLayoutA,
      0,
      typename MmaCore::IteratorThreadMapA,
      16>;
  using SmemIteratorA = typename MmaCore::SmemIteratorA;
  using WarpFragmentA = typename Operator::FragmentA;
  using WarpFragmentB = typename Operator::FragmentB;

  static int const kStages = 3;
  static int const kWarpGemmIterations =
      Operator::Shape::kK / Operator::Policy::MmaShape::kK;
  static int const kThreadCount = 32 * WarpCount::kCount;
  static int const kPackedBKBlocks = Shape::kK / 32;
  static int const kPackedBNGroups8 = Shape::kN / 8;
  static int const kPackedBBlockBytes = 128;
  static int const kPackedBStageBytes =
      kPackedBKBlocks * kPackedBNGroups8 * kPackedBBlockBytes;
  static int const kShiftStageBytes = kPackedBNGroups8 * 8;

  struct SharedStorage {
    using ShapeA = cutlass::MatrixShape<Shape::kM, Shape::kK * kStages>;

    cutlass::AlignedBuffer<int8_t, ShapeA::kCount> operand_A;
    cutlass::AlignedBuffer<uint8_t, kPackedBStageBytes * kStages> operand_B_packed;
    cutlass::AlignedBuffer<uint8_t, kShiftStageBytes * kStages> operand_B_shift;
    int routed_source_rows[kEnableRoutedRows ? Shape::kM : 1];

    CUTLASS_DEVICE
    static typename Operator::LayoutA LayoutA() {
      return Operator::LayoutA::packed({ShapeA::kRow, ShapeA::kColumn});
    }

    CUTLASS_DEVICE
    cutlass::TensorRef<typename Operator::ElementA, typename Operator::LayoutA>
    operand_A_ref() {
      return {operand_A.data(), LayoutA()};
    }

    CUTLASS_DEVICE
        cutlass::TensorRef<typename Operator::ElementA, typename MmaCore::SmemLayoutA>
    operand_A_access_ref() {
      return {operand_A.data(), typename MmaCore::SmemLayoutA(ShapeA::kColumn)};
    }

    CUTLASS_DEVICE
    uint8_t* operand_B_stage(int stage) {
      return operand_B_packed.data() + stage * kPackedBStageBytes;
    }

    CUTLASS_DEVICE
    uint8_t* operand_B_shift_stage(int stage) {
      return operand_B_shift.data() + stage * kShiftStageBytes;
    }
  };

  Operator warp_mma;
  typename Operator::IteratorA warp_tile_iterator_A_;
  SmemAccessIteratorA smem_access_iterator_A_;
  SharedStorage& shared_storage_;
  int thread_idx_;
  int warp_idx_;
  int lane_idx_;
  int warp_idx_m_;
  int warp_idx_n_;
  int warp_idx_k_;
  int smem_read_stage_idx_;

  CUTLASS_DEVICE
  Mxfp4PackedBThreadblockMmaSkeleton(
      SharedStorage& shared_storage,
      int thread_idx,
      int warp_idx,
      int lane_idx)
      : warp_tile_iterator_A_(shared_storage.operand_A_ref(), lane_idx),
        smem_access_iterator_A_(shared_storage.operand_A_access_ref(), thread_idx),
        shared_storage_(shared_storage),
        thread_idx_(thread_idx),
        warp_idx_(warp_idx),
        lane_idx_(lane_idx),
        warp_idx_m_(0),
        warp_idx_n_(0),
        warp_idx_k_(0),
        smem_read_stage_idx_(0) {
    int warp_idx_mn = warp_idx % (WarpCount::kM * WarpCount::kN);
    warp_idx_k_ = warp_idx / (WarpCount::kM * WarpCount::kN);
    warp_idx_m_ = warp_idx_mn % WarpCount::kM;
    warp_idx_n_ = warp_idx_mn / WarpCount::kM;
    this->warp_tile_iterator_A_.add_tile_offset(
        {warp_idx_m_, kWarpGemmIterations * warp_idx_k_});
  }

  CUTLASS_DEVICE
  void advance_smem_read_stage() {
    ++smem_read_stage_idx_;
    if (smem_read_stage_idx_ == kStages) {
      this->warp_tile_iterator_A_.add_tile_offset(
          {0, -kStages * Policy::kPartitionsK * kWarpGemmIterations});
      smem_read_stage_idx_ = 0;
    }
  }

  CUTLASS_DEVICE
  void wait_a_stage_steady() {
    cutlass::arch::cp_async_wait<kStages - 2>();
    __syncthreads();
  }

  CUTLASS_DEVICE
  void wait_a_stage_tail() {
    cutlass::arch::cp_async_wait<0>();
    __syncthreads();
  }

  template <bool SourceRowsAreSlots>
  CUTLASS_DEVICE
  void cache_routed_source_rows(
      int threadblock_m_offset,
      int problem_m,
      int32_t const* sorted_token_ids,
      int total_valid_slots,
      int top_k) {
    for (int local_m = thread_idx_; local_m < Shape::kM; local_m += kThreadCount) {
      int global_m = threadblock_m_offset + local_m;
      int source_row = -1;
      if (global_m < problem_m) {
        int slot = sorted_token_ids[global_m];
        if (slot < total_valid_slots) {
          source_row = SourceRowsAreSlots ? slot : (slot / top_k);
        }
      }
      shared_storage_.routed_source_rows[local_m] = source_row;
    }
    __syncthreads();
  }

  template <bool NoBoundsCheck>
  CUTLASS_DEVICE
  void copy_a_stage(
      int8_t const* ptr_A,
      int lda,
      int32_t const* sorted_token_ids,
      int stage,
      int k_offset,
      int m_offset,
      int problem_m,
      int problem_k) {
    using ThreadMapA = typename MmaCore::IteratorThreadMapA;
    using AccessType = typename SmemAccessIteratorA::AccessType;
    static_assert(sizeof(AccessType) == 16, "A cp.async path expects 16-byte accesses");

    auto thread_offset = ThreadMapA::initial_offset(thread_idx_);
    SmemAccessIteratorA smem_iterator = this->smem_access_iterator_A_;
    smem_iterator.add_tile_offset({0, stage});

    CUTLASS_PRAGMA_UNROLL
    for (int s = 0; s < ThreadMapA::Iterations::kStrided; ++s) {
      CUTLASS_PRAGMA_UNROLL
      for (int c = 0; c < ThreadMapA::Iterations::kContiguous; ++c) {
        int idx = c + s * ThreadMapA::Iterations::kContiguous;
        int local_k =
            thread_offset.contiguous() + c * ThreadMapA::Delta::kContiguous;
        int local_m =
            thread_offset.strided() + s * ThreadMapA::Delta::kStrided;
        int global_m = m_offset + local_m;
        int global_k = k_offset + local_k;
        int source_row = global_m;
        bool valid = NoBoundsCheck ||
            (global_m < problem_m &&
             global_k + ThreadMapA::kElementsPerAccess <= problem_k);
        if constexpr (kEnableRoutedRows) {
          if (sorted_token_ids) {
            source_row = shared_storage_.routed_source_rows[local_m];
            valid = (source_row >= 0) &&
                (global_k + ThreadMapA::kElementsPerAccess <= problem_k);
            if (source_row < 0) {
              source_row = 0;
            }
          }
        }

        smem_iterator.set_iteration_index(idx);
        cutlass::arch::cp_async_zfill<16, cutlass::arch::CacheOperation::Global>(
            smem_iterator.get(),
            ptr_A + source_row * lda + global_k,
            valid);
      }
    }
  }

  template <bool NoNBoundsCheck>
  CUTLASS_DEVICE
  void copy_b_stage(
      uint8_t const* packed_b,
      int packed_n_groups8,
      uint8_t const* shift2,
      int shift_ld,
      int stage,
      int k_offset,
      int threadblock_n_offset,
      int problem_k,
      int problem_n) const {
    static_assert((kPackedBStageBytes % 16) == 0, "packed B stage must be 16B aligned");
    static_assert((kShiftStageBytes % 16) == 0, "shift stage must be 16B aligned");

    uint8_t* smem_b = shared_storage_.operand_B_stage(stage);
    int packed_k_stride_bytes = packed_n_groups8 << 7;
    int packed_k_blocks = (problem_k + 31) >> 5;
    int base_k_block = k_offset >> 5;
    int base_n_group8 = threadblock_n_offset >> 3;

    for (int byte_offset = thread_idx_ * 16;
         byte_offset < kPackedBStageBytes;
         byte_offset += kThreadCount * 16) {
      int k_block_rel = byte_offset / (kPackedBNGroups8 * kPackedBBlockBytes);
      int rem = byte_offset - k_block_rel * (kPackedBNGroups8 * kPackedBBlockBytes);
      int n_group_rel = rem / kPackedBBlockBytes;
      int group_byte = rem - n_group_rel * kPackedBBlockBytes;
      int global_k_block = base_k_block + k_block_rel;
      int global_n_group8 = base_n_group8 + n_group_rel;
      bool valid = NoNBoundsCheck ||
          (global_k_block < packed_k_blocks && global_n_group8 < packed_n_groups8);
      uint8_t const* src = packed_b +
          global_k_block * packed_k_stride_bytes +
          (valid ? global_n_group8 : 0) * kPackedBBlockBytes +
          group_byte;
      cutlass::arch::cp_async_zfill<16, cutlass::arch::CacheOperation::Global>(
          smem_b + byte_offset,
          src,
          valid);
    }

    uint8_t* smem_shift = shared_storage_.operand_B_shift_stage(stage);
    int shift_row = k_offset >> 7;
    uint8_t const* shift_row_ptr = shift2 + shift_row * shift_ld * 8;
    for (int byte_offset = thread_idx_ * 8;
         byte_offset < kShiftStageBytes;
         byte_offset += kThreadCount * 8) {
      int n_group_rel = byte_offset >> 3;
      int global_n_group8 = base_n_group8 + n_group_rel;
      bool valid = NoNBoundsCheck || (global_n_group8 < packed_n_groups8);
      uint8_t const* src = shift_row_ptr + (valid ? global_n_group8 : 0) * 8;
      cutlass::arch::cp_async_zfill<8, cutlass::arch::CacheOperation::Always>(
          smem_shift + byte_offset,
          src,
          valid);
    }
  }

  template <bool NoNBoundsCheck>
  CUTLASS_DEVICE
  void load_packed_b_warp_fragment_shared(
      WarpFragmentB& frag,
      int threadblock_n_offset,
      int warp_n_offset,
      int warp_k_offset,
      int problem_n) const {
    int group_id = lane_idx_ >> 2;
    int shift_select = (warp_k_offset >> 5) & 3;
    int k_block_rel = (warp_k_offset >> 5) % kPackedBKBlocks;
    uint8_t const* smem_b = shared_storage_.operand_B_packed.data() +
        smem_read_stage_idx_ * kPackedBStageBytes +
        k_block_rel * kPackedBNGroups8 * kPackedBBlockBytes;
    uint8_t const* smem_shift =
        shared_storage_.operand_B_shift_stage(smem_read_stage_idx_);
    int lane_byte_offset = lane_idx_ << 2;
    uint32_t* frag_words = reinterpret_cast<uint32_t*>(&frag);

    CUTLASS_PRAGMA_UNROLL
    for (int inst_n = 0; inst_n < (Operator::Shape::kN / 8); ++inst_n) {
      int n_group_rel = (warp_n_offset >> 3) + inst_n;
      uint8_t shift_byte = smem_shift[n_group_rel * 8 + group_id];
      uint8_t shift = (shift_byte >> (shift_select * 2)) & 0x03;
      if (NoNBoundsCheck) {
        uint8_t const* q4_ptr =
            smem_b + n_group_rel * kPackedBBlockBytes + lane_byte_offset;
        uint32_t q4_pair = *reinterpret_cast<uint32_t const*>(q4_ptr);
        lookup_mxfp4_x2_byte_perm(
            q4_pair,
            shift,
            frag_words[inst_n * 2 + 0],
            frag_words[inst_n * 2 + 1]);
      } else {
        int local_n = warp_n_offset + inst_n * 8 + group_id;
        int global_n = threadblock_n_offset + local_n;
        bool global_n_valid = global_n < problem_n;
        uint8_t const* q4_ptr =
            smem_b + n_group_rel * kPackedBBlockBytes + lane_byte_offset;
        uint32_t q4_pair = *reinterpret_cast<uint32_t const*>(q4_ptr);
        uint32_t decoded_lo = 0;
        uint32_t decoded_hi = 0;
        if (global_n_valid) {
          lookup_mxfp4_x2_byte_perm(q4_pair, shift, decoded_lo, decoded_hi);
        }
        frag_words[inst_n * 2 + 0] = decoded_lo;
        frag_words[inst_n * 2 + 1] = decoded_hi;
      }
    }
  }

  template <bool NoNBoundsCheck>
  CUTLASS_DEVICE
  void load_packed_b_warp_tile_shared(
      WarpFragmentB (&frags)[kWarpGemmIterations],
      int threadblock_n_offset,
      int warp_n_offset,
      int tile_k_offset,
      int problem_n) const {
    CUTLASS_PRAGMA_UNROLL
    for (int warp_mma_k = 0; warp_mma_k < kWarpGemmIterations; ++warp_mma_k) {
      int warp_k_offset =
          tile_k_offset + warp_mma_k * Operator::Policy::MmaShape::kK;
      load_packed_b_warp_fragment_shared<NoNBoundsCheck>(
          frags[warp_mma_k],
          threadblock_n_offset,
          warp_n_offset,
          warp_k_offset,
          problem_n);
    }
  }

  template <bool NoBoundsCheck>
  CUTLASS_DEVICE
  void copy_stage(
      int8_t const* ptr_A,
      int lda,
      int32_t const* sorted_token_ids,
      uint8_t const* packed_b,
      int packed_n_groups8,
      uint8_t const* shift2,
      int shift_ld,
      int stage,
      int k_offset,
      int m_offset,
      int n_offset,
      int problem_m,
      int problem_k,
      int problem_n) {
    copy_a_stage<NoBoundsCheck>(
        ptr_A,
        lda,
        sorted_token_ids,
        stage,
        k_offset,
        m_offset,
        problem_m,
        problem_k);
    copy_b_stage<NoBoundsCheck>(
        packed_b,
        packed_n_groups8,
        shift2,
        shift_ld,
        stage,
        k_offset,
        n_offset,
        problem_k,
        problem_n);
    cutlass::arch::cp_async_fence();
  }

  template <bool NoNBoundsCheck>
  CUTLASS_DEVICE
  int prologue(
      int8_t const* ptr_A,
      int lda,
      int32_t const* sorted_token_ids,
      uint8_t const* packed_b,
      int packed_n_groups8,
      uint8_t const* shift2,
      int shift_ld,
      int threadblock_k_offset,
      int threadblock_m_offset,
      int threadblock_n_offset,
      int problem_m,
      int problem_k,
      int problem_n) {
    int stages_issued = 0;
    CUTLASS_PRAGMA_UNROLL
    for (int stage = 0; stage < kStages - 1; ++stage) {
      int stage_k_offset = threadblock_k_offset + stage * Shape::kK;
      if (stage_k_offset < problem_k) {
        if (NoNBoundsCheck &&
            threadblock_m_offset + Shape::kM <= problem_m &&
            stage_k_offset + Shape::kK <= problem_k) {
          copy_stage<true>(
              ptr_A,
              lda,
              sorted_token_ids,
              packed_b,
              packed_n_groups8,
              shift2,
              shift_ld,
              stage,
              stage_k_offset,
              threadblock_m_offset,
              threadblock_n_offset,
              problem_m,
              problem_k,
              problem_n);
        } else {
          copy_stage<false>(
              ptr_A,
              lda,
              sorted_token_ids,
              packed_b,
              packed_n_groups8,
              shift2,
              shift_ld,
              stage,
              stage_k_offset,
              threadblock_m_offset,
              threadblock_n_offset,
              problem_m,
              problem_k,
              problem_n);
        }
        ++stages_issued;
      }
    }
    return stages_issued;
  }

  template <bool NoNBoundsCheck, bool SourceRowsAreSlots = false>
  CUTLASS_DEVICE
  void operator()(
      int gemm_k_iterations,
      FragmentC& accum,
      int8_t const* ptr_A,
      int lda,
      int32_t const* sorted_token_ids,
      int total_valid_slots,
      int top_k,
      uint8_t const* packed_b,
      int packed_n_groups8,
      uint8_t const* shift2,
      int shift_ld,
      int threadblock_m_offset,
      int threadblock_k_offset,
      int threadblock_n_offset,
      int problem_m,
      int problem_k,
      int problem_n,
      FragmentC const& src_accum) {
    if constexpr (kEnableRoutedRows) {
      if (sorted_token_ids) {
        cache_routed_source_rows<SourceRowsAreSlots>(
            threadblock_m_offset,
            problem_m,
            sorted_token_ids,
            total_valid_slots,
            top_k);
      }
    }

    int a_stages_issued = prologue<NoNBoundsCheck>(
        ptr_A,
        lda,
        sorted_token_ids,
        packed_b,
        packed_n_groups8,
        shift2,
        shift_ld,
        threadblock_k_offset,
        threadblock_m_offset,
        threadblock_n_offset,
        problem_m,
        problem_k,
        problem_n);

    WarpFragmentA warp_frag_A;
    WarpFragmentB warp_frag_B[kWarpGemmIterations];

    int warp_n_offset = warp_idx_n_ * Operator::Shape::kN;

    if (a_stages_issued >= kStages - 1) {
      wait_a_stage_steady();
    } else {
      wait_a_stage_tail();
    }
    accum = src_accum;

    CUTLASS_GEMM_LOOP
    for (int gemm_k = 0; gemm_k < gemm_k_iterations; ++gemm_k) {
      int current_k_offset = threadblock_k_offset + gemm_k * Shape::kK;
      if (NoNBoundsCheck) {
        load_packed_b_warp_tile_shared<true>(
            warp_frag_B,
            threadblock_n_offset,
            warp_n_offset,
            current_k_offset,
            problem_n);
      } else {
        load_packed_b_warp_tile_shared<false>(
            warp_frag_B,
            threadblock_n_offset,
            warp_n_offset,
            current_k_offset,
            problem_n);
      }

      int prefetch_gemm_k = gemm_k + kStages - 1;
      if (prefetch_gemm_k < gemm_k_iterations) {
        int prefetch_k_offset =
            threadblock_k_offset + prefetch_gemm_k * Shape::kK;
        int smem_write_stage_idx = prefetch_gemm_k % kStages;
        if (NoNBoundsCheck &&
            threadblock_m_offset + Shape::kM <= problem_m &&
            prefetch_k_offset + Shape::kK <= problem_k) {
          copy_stage<true>(
              ptr_A,
              lda,
              sorted_token_ids,
              packed_b,
              packed_n_groups8,
              shift2,
              shift_ld,
              smem_write_stage_idx,
              prefetch_k_offset,
              threadblock_m_offset,
              threadblock_n_offset,
              problem_m,
              problem_k,
              problem_n);
        } else {
          copy_stage<false>(
              ptr_A,
              lda,
              sorted_token_ids,
              packed_b,
              packed_n_groups8,
              shift2,
              shift_ld,
              smem_write_stage_idx,
              prefetch_k_offset,
              threadblock_m_offset,
              threadblock_n_offset,
              problem_m,
              problem_k,
              problem_n);
        }
        ++a_stages_issued;
      }

      CUTLASS_PRAGMA_UNROLL
      for (int warp_mma_k = 0; warp_mma_k < kWarpGemmIterations; ++warp_mma_k) {
        this->warp_tile_iterator_A_.set_kgroup_index(warp_mma_k);
        this->warp_tile_iterator_A_.load(warp_frag_A);
        ++this->warp_tile_iterator_A_;

        warp_mma(
            accum,
            warp_frag_A,
            warp_frag_B[warp_mma_k],
            accum);
      }

      if (gemm_k + 1 < gemm_k_iterations) {
        if (gemm_k + kStages < gemm_k_iterations) {
          wait_a_stage_steady();
        } else {
          wait_a_stage_tail();
        }
        advance_smem_read_stage();
      }
    }
  }
};

template <typename MmaCore_>
struct Mxfp4PackedBOnDemandThreadblockMmaSkeleton
    : public Mxfp4PackedBThreadblockMmaSkeleton<MmaCore_> {
  using Base = Mxfp4PackedBThreadblockMmaSkeleton<MmaCore_>;
  using Shape = typename Base::Shape;
  using Operator = typename Base::Operator;
  using FragmentC = typename Base::FragmentC;
  using WarpFragmentA = typename Base::WarpFragmentA;
  using WarpFragmentB = typename Base::WarpFragmentB;
  using SharedStorage = typename Base::SharedStorage;

  static int const kStages = Base::kStages;
  static int const kWarpGemmIterations = Base::kWarpGemmIterations;
  static int const kThreadCount = Base::kThreadCount;

  CUTLASS_DEVICE
  Mxfp4PackedBOnDemandThreadblockMmaSkeleton(
      SharedStorage& shared_storage,
      int thread_idx,
      int warp_idx,
      int lane_idx)
      : Base(shared_storage, thread_idx, warp_idx, lane_idx) {}

  template <bool NoNBoundsCheck, bool SourceRowsAreSlots = false>
  CUTLASS_DEVICE
  void operator()(
      int gemm_k_iterations,
      FragmentC& accum,
      int8_t const* ptr_A,
      int lda,
      int32_t const* sorted_token_ids,
      int total_valid_slots,
      int top_k,
      uint8_t const* packed_b,
      int packed_n_groups8,
      uint8_t const* shift2,
      int shift_ld,
      int threadblock_m_offset,
      int threadblock_k_offset,
      int threadblock_n_offset,
      int problem_m,
      int problem_k,
      int problem_n,
      FragmentC const& src_accum) {
    if constexpr (Base::kEnableRoutedRows) {
      if (sorted_token_ids) {
        this->template cache_routed_source_rows<SourceRowsAreSlots>(
            threadblock_m_offset,
            problem_m,
            sorted_token_ids,
            total_valid_slots,
            top_k);
      }
    }

    int a_stages_issued = this->template prologue<NoNBoundsCheck>(
        ptr_A,
        lda,
        sorted_token_ids,
        packed_b,
        packed_n_groups8,
        shift2,
        shift_ld,
        threadblock_k_offset,
        threadblock_m_offset,
        threadblock_n_offset,
        problem_m,
        problem_k,
        problem_n);

    WarpFragmentA warp_frag_A;
    WarpFragmentB warp_frag_B;
    int warp_n_offset = this->warp_idx_n_ * Operator::Shape::kN;

    if (a_stages_issued >= kStages - 1) {
      this->wait_a_stage_steady();
    } else {
      this->wait_a_stage_tail();
    }
    accum = src_accum;

    CUTLASS_GEMM_LOOP
    for (int gemm_k = 0; gemm_k < gemm_k_iterations; ++gemm_k) {
      int current_k_offset = threadblock_k_offset + gemm_k * Shape::kK;

      int prefetch_gemm_k = gemm_k + kStages - 1;
      if (prefetch_gemm_k < gemm_k_iterations) {
        int prefetch_k_offset =
            threadblock_k_offset + prefetch_gemm_k * Shape::kK;
        int smem_write_stage_idx = prefetch_gemm_k % kStages;
        if (NoNBoundsCheck &&
            threadblock_m_offset + Shape::kM <= problem_m &&
            prefetch_k_offset + Shape::kK <= problem_k) {
          this->template copy_stage<true>(
              ptr_A,
              lda,
              sorted_token_ids,
              packed_b,
              packed_n_groups8,
              shift2,
              shift_ld,
              smem_write_stage_idx,
              prefetch_k_offset,
              threadblock_m_offset,
              threadblock_n_offset,
              problem_m,
              problem_k,
              problem_n);
        } else {
          this->template copy_stage<false>(
              ptr_A,
              lda,
              sorted_token_ids,
              packed_b,
              packed_n_groups8,
              shift2,
              shift_ld,
              smem_write_stage_idx,
              prefetch_k_offset,
              threadblock_m_offset,
              threadblock_n_offset,
              problem_m,
              problem_k,
              problem_n);
        }
        ++a_stages_issued;
      }

      CUTLASS_PRAGMA_UNROLL
      for (int warp_mma_k = 0; warp_mma_k < kWarpGemmIterations; ++warp_mma_k) {
        int warp_k_offset =
            current_k_offset + warp_mma_k * Operator::Policy::MmaShape::kK;
        if (NoNBoundsCheck) {
          this->template load_packed_b_warp_fragment_shared<true>(
              warp_frag_B,
              threadblock_n_offset,
              warp_n_offset,
              warp_k_offset,
              problem_n);
        } else {
          this->template load_packed_b_warp_fragment_shared<false>(
              warp_frag_B,
              threadblock_n_offset,
              warp_n_offset,
              warp_k_offset,
              problem_n);
        }
        this->warp_tile_iterator_A_.set_kgroup_index(warp_mma_k);
        this->warp_tile_iterator_A_.load(warp_frag_A);
        ++this->warp_tile_iterator_A_;

        this->warp_mma(
            accum,
            warp_frag_A,
            warp_frag_B,
            accum);
      }

      if (gemm_k + 1 < gemm_k_iterations) {
        if (gemm_k + kStages < gemm_k_iterations) {
          this->wait_a_stage_steady();
        } else {
          this->wait_a_stage_tail();
        }
        this->advance_smem_read_stage();
      }
    }
  }
};
using Mxfp4PackedBThreadblockMma =
    Mxfp4PackedBThreadblockMmaSkeleton<Mxfp4MmaCore>;
template <int ThreadblockM, int WarpM, int ThreadblockN = 128, int WarpN = 64>
using Mxfp4PackedBMmaCoreForM = cutlass::gemm::threadblock::DefaultMmaCore<
    cutlass::gemm::GemmShape<ThreadblockM, ThreadblockN, 64>,
    cutlass::gemm::GemmShape<WarpM, WarpN, 64>,
    Mxfp4InstructionShape,
    int8_t,
    cutlass::layout::RowMajor,
    int8_t,
    cutlass::layout::ColumnMajor,
    int32_t,
    cutlass::layout::RowMajor,
    cutlass::arch::OpClassTensorOp,
    2,
    cutlass::arch::OpMultiplyAddSaturate>;

template <
    int ThreadblockM,
    int WarpM,
    int ThreadblockN,
    int WarpN,
    int ThreadblockK>
using Mxfp4PackedBMmaCoreForShape = cutlass::gemm::threadblock::DefaultMmaCore<
    cutlass::gemm::GemmShape<ThreadblockM, ThreadblockN, ThreadblockK>,
    cutlass::gemm::GemmShape<WarpM, WarpN, ThreadblockK>,
    Mxfp4InstructionShape,
    int8_t,
    cutlass::layout::RowMajor,
    int8_t,
    cutlass::layout::ColumnMajor,
    int32_t,
    cutlass::layout::RowMajor,
    cutlass::arch::OpClassTensorOp,
    2,
    cutlass::arch::OpMultiplyAddSaturate>;

using Mxfp4PackedBThreadblockMma16N32K128 =
    Mxfp4PackedBThreadblockMmaSkeleton<
        Mxfp4PackedBMmaCoreForShape<
            16,
            16,
            MXFP4_SMALL_M_THREADBLOCK_N,
            MXFP4_SMALL_M_WARP_N,
            MXFP4_SMALL_M_THREADBLOCK_K>>;
using Mxfp4PackedBOnDemandThreadblockMma16N32K128 =
    Mxfp4PackedBOnDemandThreadblockMmaSkeleton<
        Mxfp4PackedBMmaCoreForShape<
            16,
            16,
            MXFP4_SMALL_M_THREADBLOCK_N,
            MXFP4_SMALL_M_WARP_N,
            MXFP4_SMALL_M_THREADBLOCK_K>>;
using Mxfp4PackedBThreadblockMma32 =
    Mxfp4PackedBThreadblockMmaSkeleton<Mxfp4PackedBMmaCoreForM<32, 16>>;
using Mxfp4PackedBThreadblockMma32K128 =
    Mxfp4PackedBThreadblockMmaSkeleton<
        Mxfp4PackedBMmaCoreForShape<32, 16, 128, 64, 128>>;
using Mxfp4PackedBThreadblockMma32N64K128 =
    Mxfp4PackedBThreadblockMmaSkeleton<
        Mxfp4PackedBMmaCoreForShape<32, 16, 64, 32, 128>>;
using Mxfp4PackedBOnDemandThreadblockMma32N64K128 =
    Mxfp4PackedBOnDemandThreadblockMmaSkeleton<
        Mxfp4PackedBMmaCoreForShape<32, 16, 64, 32, 128>>;
using Mxfp4PackedBThreadblockMma64 =
    Mxfp4PackedBThreadblockMmaSkeleton<Mxfp4PackedBMmaCoreForM<64, 32>>;
using Mxfp4PackedBThreadblockMma64K128 =
    Mxfp4PackedBThreadblockMmaSkeleton<
        Mxfp4PackedBMmaCoreForShape<64, 32, 128, 64, 128>>;
using Mxfp4PackedBOnDemandThreadblockMma64N64K128 =
    Mxfp4PackedBOnDemandThreadblockMmaSkeleton<
        Mxfp4PackedBMmaCoreForShape<64, 32, 64, 32, 128>>;
using Mxfp4PackedBThreadblockMma128 =
    Mxfp4PackedBThreadblockMmaSkeleton<Mxfp4PackedBMmaCoreForM<128, 64>>;
using Mxfp4PackedBThreadblockMma128K128 =
    Mxfp4PackedBThreadblockMmaSkeleton<
        Mxfp4PackedBMmaCoreForShape<128, 64, 128, 64, 128>>;
using Mxfp4PackedBThreadblockMma128N64K128 =
    Mxfp4PackedBThreadblockMmaSkeleton<
        Mxfp4PackedBMmaCoreForShape<128, 64, 64, 32, 128>>;
using Mxfp4PackedBThreadblockMma256N64K64 =
    Mxfp4PackedBThreadblockMmaSkeleton<
        Mxfp4PackedBMmaCoreForShape<256, 128, 64, 32, 64>>;
using Mxfp4Epilogue = typename cutlass::epilogue::threadblock::DefaultEpilogueTensorOp<
    Mxfp4ThreadblockShape,
    typename Mxfp4PackedBThreadblockMma::Operator,
    Mxfp4ThreadblockShape::kK / Mxfp4WarpShape::kK,
    Mxfp4EpilogueOp,
    Mxfp4EpilogueOp::kCount>::Epilogue;
template <typename Mma_>
using Mxfp4EpilogueForMma =
    typename cutlass::epilogue::threadblock::DefaultEpilogueTensorOp<
        typename Mma_::Shape,
        typename Mma_::Operator,
        Mma_::Shape::kK / Mma_::Operator::Shape::kK,
        Mxfp4EpilogueOp,
        Mxfp4EpilogueOp::kCount>::Epilogue;

template <typename Mma_>
struct Mxfp4ScaledBf16EpilogueForMma {
  using Mma = Mma_;
  using OutputOp = IdentityInt32OutputOp<Mxfp4EpilogueOp::kCount>;
  using OutputTileThreadMap =
      typename cutlass::epilogue::threadblock::DefaultThreadMapTensorOp<
          typename Mma::Shape,
          typename Mma::Operator::Shape,
          Mma::Shape::kK / Mma::Operator::Shape::kK,
          int32_t,
          Mxfp4EpilogueOp::kCount>::Type;
  using OutputTileIterator =
      ScaledBf16OutputTileIterator<OutputTileThreadMap>;
  using AccumulatorFragmentIterator =
      cutlass::epilogue::warp::FragmentIteratorTensorOp<
          typename Mma::Operator::Shape,
          typename Mma::Operator::Policy::Operator::Shape,
          typename Mma::Operator::Policy::Operator::ElementC,
          typename Mma::Operator::Policy::Operator::FragmentC,
          typename Mma::Operator::LayoutC>;
  using DefaultIterators =
      cutlass::epilogue::threadblock::detail::DefaultIteratorsTensorOp<
          int32_t,
          int32_t,
          Mxfp4EpilogueOp::kCount,
          typename Mma::Shape,
          typename Mma::Operator::Shape,
          typename Mma::Operator::Policy::Operator::Shape,
          typename OutputTileThreadMap::CompactedThreadMap>;
  using WarpTileIterator = typename DefaultIterators::WarpTileIterator;
  using SharedLoadIterator = typename DefaultIterators::SharedLoadIterator;
  using Padding =
      cutlass::MatrixShape<0, 64 / cutlass::sizeof_bits<int32_t>::value * 4>;
  static int const kFragmentsPerIteration =
      (Mma::Shape::kK / Mma::Operator::Shape::kK) == 1
      ? DefaultIterators::kFragmentsPerIteration
      : 1;
  using Epilogue = cutlass::epilogue::threadblock::Epilogue<
      typename Mma::Shape,
      typename Mma::Operator,
      Mma::Shape::kK / Mma::Operator::Shape::kK,
      OutputTileIterator,
      AccumulatorFragmentIterator,
      WarpTileIterator,
      SharedLoadIterator,
      OutputOp,
      Padding,
      kFragmentsPerIteration>;
};

template <
    typename Mma_,
    typename Epilogue_,
    typename ThreadblockSwizzle_,
    bool Persistent_ = false>
struct Mxfp4PackedBGemmKernel {
  using Mma = Mma_;
  using Epilogue = Epilogue_;
  using OutputOp = typename Epilogue::OutputOp;
  using ThreadblockSwizzle = ThreadblockSwizzle_;
  using WarpCount = typename Mma::WarpCount;
  static bool const kPersistent = Persistent_;
  static int const kThreadCount = 32 * WarpCount::kCount;

  struct Params {
    cutlass::gemm::GemmCoord problem_size;
    cutlass::gemm::GemmCoord grid_tiled_shape;
    int swizzle_log_tile;
    int grid_m_tiles;
    int grid_n_tiles;
    int8_t* ptr_A;
    int lda;
    uint8_t const* ptr_B_packed;
    uint8_t const* ptr_B_shift2;
    int b_shift_ld;
    typename Epilogue::OutputTileIterator::Params params_C;
    typename Epilogue::OutputTileIterator::TensorRef ref_C;
    typename Epilogue::OutputTileIterator::Params params_D;
    typename Epilogue::OutputTileIterator::TensorRef ref_D;
    typename OutputOp::Params output_op;
    int gemm_k_size;
    int batch_stride;

    CUTLASS_HOST_DEVICE
    Params()
        : swizzle_log_tile(0),
          grid_m_tiles(0),
          grid_n_tiles(0),
          ptr_A(nullptr),
          lda(0),
          ptr_B_packed(nullptr),
          ptr_B_shift2(nullptr),
          b_shift_ld(0),
          gemm_k_size(0),
          batch_stride(0) {}

    CUTLASS_HOST_DEVICE
    Params(
        cutlass::gemm::GemmCoord const& problem_size_,
        cutlass::gemm::GemmCoord const& grid_tiled_shape_,
        int grid_m_tiles_,
        int grid_n_tiles_,
        cutlass::TensorRef<int8_t const, cutlass::layout::RowMajor> ref_A,
        uint8_t const* ptr_B_packed_,
        uint8_t const* ptr_B_shift2_,
        int b_shift_ld_,
        typename Epilogue::OutputTileIterator::Params params_C_,
        typename Epilogue::OutputTileIterator::TensorRef ref_C_,
        typename Epilogue::OutputTileIterator::Params params_D_,
        typename Epilogue::OutputTileIterator::TensorRef ref_D_,
        typename OutputOp::Params output_op_,
        int split_k_slices = 1,
        int batch_stride_ = 0)
        : problem_size(problem_size_),
          grid_tiled_shape(grid_tiled_shape_),
          swizzle_log_tile(ThreadblockSwizzle().get_log_tile(grid_tiled_shape_)),
          grid_m_tiles(grid_m_tiles_),
          grid_n_tiles(grid_n_tiles_),
          ptr_A(const_cast<int8_t*>(ref_A.data())),
          lda(ref_A.layout().stride(0)),
          ptr_B_packed(ptr_B_packed_),
          ptr_B_shift2(ptr_B_shift2_),
          b_shift_ld(b_shift_ld_),
          params_C(params_C_),
          ref_C(ref_C_),
          params_D(params_D_),
          ref_D(ref_D_),
          output_op(output_op_),
          batch_stride(batch_stride_) {
      int total_gemm_k_iterations =
          (problem_size.k() + Mma::Shape::kK - 1) / Mma::Shape::kK;
      int iterations_per_slice =
          (total_gemm_k_iterations + split_k_slices - 1) / split_k_slices;
      gemm_k_size = iterations_per_slice * Mma::Shape::kK;
    }
  };

  union SharedStorage {
    typename Mma::SharedStorage main_loop;
    typename Epilogue::SharedStorage epilogue;
  };

  CUTLASS_DEVICE
  void run_tile(
      Params const& params,
      SharedStorage& shared_storage,
      cutlass::gemm::GemmCoord threadblock_tile_offset) {

    if (params.grid_tiled_shape.m() <= threadblock_tile_offset.m() ||
        params.grid_tiled_shape.n() <= threadblock_tile_offset.n()) {
      return;
    }

    cutlass::MatrixCoord tb_offset_A{
        threadblock_tile_offset.m() * Mma::Shape::kM,
        threadblock_tile_offset.k() * params.gemm_k_size};

    int threadblock_k_offset = threadblock_tile_offset.k() * params.gemm_k_size;
    int threadblock_n_offset = threadblock_tile_offset.n() * Mma::Shape::kN;
    int problem_size_k = min(
        params.problem_size.k(),
        (threadblock_tile_offset.k() + 1) * params.gemm_k_size);
    int gemm_k_iterations =
        (problem_size_k - tb_offset_A.column() + Mma::Shape::kK - 1) /
        Mma::Shape::kK;

    int thread_idx = threadIdx.x;
    int warp_idx = cutlass::canonical_warp_idx_sync();
    int lane_idx = threadIdx.x % 32;

    Mma mma(shared_storage.main_loop, thread_idx, warp_idx, lane_idx);
    typename Mma::FragmentC accumulators;
    accumulators.clear();
    if (gemm_k_iterations > 0) {
      if ((params.problem_size.n() % Mma::Shape::kN) == 0) {
        mma.template operator()<true>(
            gemm_k_iterations,
            accumulators,
            params.ptr_A,
            params.lda,
            nullptr,
            0,
            1,
            params.ptr_B_packed,
            (params.problem_size.n() + 7) / 8,
            params.ptr_B_shift2,
            params.b_shift_ld,
            tb_offset_A.row(),
            threadblock_k_offset,
            threadblock_n_offset,
            params.problem_size.m(),
            problem_size_k,
            params.problem_size.n(),
            accumulators);
      } else {
        mma.template operator()<false>(
            gemm_k_iterations,
            accumulators,
            params.ptr_A,
            params.lda,
            nullptr,
            0,
            1,
            params.ptr_B_packed,
            (params.problem_size.n() + 7) / 8,
            params.ptr_B_shift2,
            params.b_shift_ld,
            tb_offset_A.row(),
            threadblock_k_offset,
            threadblock_n_offset,
            params.problem_size.m(),
            problem_size_k,
            params.problem_size.n(),
            accumulators);
      }
    }

    OutputOp output_op(params.output_op);
    cutlass::MatrixCoord threadblock_offset(
        threadblock_tile_offset.m() * Mma::Shape::kM,
        threadblock_tile_offset.n() * Mma::Shape::kN);

    typename Epilogue::OutputTileIterator iterator_C(
        params.params_C,
        params.ref_C.data() + threadblock_tile_offset.k() * params.batch_stride,
        params.problem_size.mn(),
        thread_idx,
        threadblock_offset);
    typename Epilogue::OutputTileIterator iterator_D(
        params.params_D,
        params.ref_D.data() + threadblock_tile_offset.k() * params.batch_stride,
        params.problem_size.mn(),
        thread_idx,
        threadblock_offset);

    Epilogue epilogue(shared_storage.epilogue, thread_idx, warp_idx, lane_idx);
    epilogue(output_op, iterator_D, accumulators, iterator_C);
  }

  CUTLASS_DEVICE
  void operator()(Params const& params, SharedStorage& shared_storage) {
    if (kPersistent) {
      int stripe_id = blockIdx.x % params.grid_n_tiles;
      int stripe_worker = blockIdx.x / params.grid_n_tiles;
      int stripe_workers = (gridDim.x + params.grid_n_tiles - 1) / params.grid_n_tiles;
      for (int m_tile = stripe_worker;
           m_tile < params.grid_m_tiles;
           m_tile += stripe_workers) {
        run_tile(params, shared_storage, {m_tile, stripe_id, 0});
      }
    } else {
      ThreadblockSwizzle threadblock_swizzle;
      cutlass::gemm::GemmCoord threadblock_tile_offset =
          threadblock_swizzle.get_tile_offset(params.swizzle_log_tile);
      run_tile(params, shared_storage, threadblock_tile_offset);
    }
  }
};

using Mxfp4PackedBCutlassKernel =
    Mxfp4PackedBGemmKernel<Mxfp4PackedBThreadblockMma, Mxfp4Epilogue, Mxfp4ThreadblockSwizzle>;
using Mxfp4PackedBCutlassKernel16N32K128 = Mxfp4PackedBGemmKernel<
    Mxfp4PackedBThreadblockMma16N32K128,
    Mxfp4EpilogueForMma<Mxfp4PackedBThreadblockMma16N32K128>,
    Mxfp4ThreadblockSwizzle>;
using Mxfp4PackedBOnDemandCutlassKernel16N32K128 = Mxfp4PackedBGemmKernel<
    Mxfp4PackedBOnDemandThreadblockMma16N32K128,
    Mxfp4EpilogueForMma<Mxfp4PackedBOnDemandThreadblockMma16N32K128>,
    Mxfp4ThreadblockSwizzle>;
using Mxfp4PackedBCutlassKernel32 = Mxfp4PackedBGemmKernel<
    Mxfp4PackedBThreadblockMma32,
    Mxfp4EpilogueForMma<Mxfp4PackedBThreadblockMma32>,
    Mxfp4ThreadblockSwizzle>;
using Mxfp4PackedBCutlassKernel32K128 = Mxfp4PackedBGemmKernel<
    Mxfp4PackedBThreadblockMma32K128,
    Mxfp4EpilogueForMma<Mxfp4PackedBThreadblockMma32K128>,
    Mxfp4ThreadblockSwizzle>;
using Mxfp4PackedBCutlassKernel32N64K128 = Mxfp4PackedBGemmKernel<
    Mxfp4PackedBThreadblockMma32N64K128,
    Mxfp4EpilogueForMma<Mxfp4PackedBThreadblockMma32N64K128>,
    Mxfp4ThreadblockSwizzle>;
using Mxfp4PackedBOnDemandCutlassKernel32N64K128 = Mxfp4PackedBGemmKernel<
    Mxfp4PackedBOnDemandThreadblockMma32N64K128,
    Mxfp4EpilogueForMma<Mxfp4PackedBOnDemandThreadblockMma32N64K128>,
    Mxfp4ThreadblockSwizzle>;
using Mxfp4PackedBCutlassKernel64 = Mxfp4PackedBGemmKernel<
    Mxfp4PackedBThreadblockMma64,
    Mxfp4EpilogueForMma<Mxfp4PackedBThreadblockMma64>,
    Mxfp4ThreadblockSwizzle>;
using Mxfp4PackedBCutlassKernel64K128 = Mxfp4PackedBGemmKernel<
    Mxfp4PackedBThreadblockMma64K128,
    Mxfp4EpilogueForMma<Mxfp4PackedBThreadblockMma64K128>,
    Mxfp4ThreadblockSwizzle>;
using Mxfp4PackedBOnDemandCutlassKernel64N64K128 = Mxfp4PackedBGemmKernel<
    Mxfp4PackedBOnDemandThreadblockMma64N64K128,
    Mxfp4EpilogueForMma<Mxfp4PackedBOnDemandThreadblockMma64N64K128>,
    Mxfp4ThreadblockSwizzle>;
using Mxfp4PackedBCutlassKernel128 = Mxfp4PackedBGemmKernel<
    Mxfp4PackedBThreadblockMma128,
    Mxfp4EpilogueForMma<Mxfp4PackedBThreadblockMma128>,
    Mxfp4ThreadblockSwizzle>;
using Mxfp4PackedBCutlassKernel128K128 = Mxfp4PackedBGemmKernel<
    Mxfp4PackedBThreadblockMma128K128,
    Mxfp4EpilogueForMma<Mxfp4PackedBThreadblockMma128K128>,
    Mxfp4ThreadblockSwizzle>;
using Mxfp4PackedBCutlassKernel128N64K128 = Mxfp4PackedBGemmKernel<
    Mxfp4PackedBThreadblockMma128N64K128,
    Mxfp4EpilogueForMma<Mxfp4PackedBThreadblockMma128N64K128>,
    Mxfp4ThreadblockSwizzle>;
using Mxfp4PackedBScaledBf16CutlassKernel = Mxfp4PackedBGemmKernel<
    Mxfp4PackedBThreadblockMma,
    typename Mxfp4ScaledBf16EpilogueForMma<Mxfp4PackedBThreadblockMma>::Epilogue,
    Mxfp4ThreadblockSwizzle>;
using Mxfp4PackedBScaledBf16PersistentCutlassKernel = Mxfp4PackedBGemmKernel<
    Mxfp4PackedBThreadblockMma,
    typename Mxfp4ScaledBf16EpilogueForMma<Mxfp4PackedBThreadblockMma>::Epilogue,
    Mxfp4ThreadblockSwizzle,
    true>;
using Mxfp4PackedBScaledBf16_256x64x64_CutlassKernel = Mxfp4PackedBGemmKernel<
    Mxfp4PackedBThreadblockMma256N64K64,
    typename Mxfp4ScaledBf16EpilogueForMma<Mxfp4PackedBThreadblockMma256N64K64>::Epilogue,
    Mxfp4ThreadblockSwizzle>;
using Mxfp4PackedBScaledBf16Persistent_256x64x64_CutlassKernel = Mxfp4PackedBGemmKernel<
    Mxfp4PackedBThreadblockMma256N64K64,
    typename Mxfp4ScaledBf16EpilogueForMma<Mxfp4PackedBThreadblockMma256N64K64>::Epilogue,
    Mxfp4ThreadblockSwizzle,
    true>;

}  // namespace cutlass_core
}  // namespace mxfp4_int8
