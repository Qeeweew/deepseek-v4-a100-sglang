#pragma once

#include "mxfp4_cutlass_core.cuh"

namespace mxfp4_int8 {
namespace cutlass_core {

template <typename ThreadMap_, bool SourceRowsAreSlots_>
class GroupedScaledBf16OutputTileIterator {
 public:
  using ThreadMap = ThreadMap_;
  static bool const kSourceRowsAreSlots = SourceRowsAreSlots_;
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
    int32_t const* sorted_token_ids;
    int32_t const* expert_ids;
    int total_valid_slots;
    int block_size_m;
    int top_k;
    int output_stride;
    int b_channel_scale_stride;

    CUTLASS_HOST_DEVICE
    Params()
        : Base(),
          a_scale(nullptr),
          b_channel_scale(nullptr),
          sorted_token_ids(nullptr),
          expert_ids(nullptr),
          total_valid_slots(0),
          block_size_m(1),
          top_k(1),
          output_stride(0),
          b_channel_scale_stride(0) {}

    CUTLASS_HOST_DEVICE
    explicit Params(Layout const& layout)
        : Base(
              layout.stride(0) * int(sizeof(__nv_bfloat16)),
              cutlass::epilogue::threadblock::make_OutputTileThreadMapDesc<ThreadMap>()),
          a_scale(nullptr),
          b_channel_scale(nullptr),
          sorted_token_ids(nullptr),
          expert_ids(nullptr),
          total_valid_slots(0),
          block_size_m(1),
          top_k(1),
          output_stride(layout.stride(0)),
          b_channel_scale_stride(layout.stride(0)) {}

    CUTLASS_HOST_DEVICE
    Params(
        Layout const& layout,
        float const* a_scale_,
        float const* b_channel_scale_,
        int32_t const* sorted_token_ids_,
        int32_t const* expert_ids_,
        int total_valid_slots_,
        int block_size_m_,
        int top_k_,
        int b_channel_scale_stride_)
        : Base(
              layout.stride(0) * int(sizeof(__nv_bfloat16)),
              cutlass::epilogue::threadblock::make_OutputTileThreadMapDesc<ThreadMap>()),
          a_scale(a_scale_),
          b_channel_scale(b_channel_scale_),
          sorted_token_ids(sorted_token_ids_),
          expert_ids(expert_ids_),
          total_valid_slots(total_valid_slots_),
          block_size_m(block_size_m_),
          top_k(top_k_),
          output_stride(layout.stride(0)),
          b_channel_scale_stride(b_channel_scale_stride_) {}
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
  __nv_bfloat16* pointer_;
  Mask mask_;
  Index extent_row_;
  Index extent_column_;
  Index thread_start_row_;
  Index thread_start_column_;
  int state_[3];

 public:
  CUTLASS_DEVICE
  GroupedScaledBf16OutputTileIterator(
      Params const& params,
      __nv_bfloat16* pointer,
      TensorCoord extent,
      int thread_idx,
      TensorCoord threadblock_offset = TensorCoord())
      : params_(params), pointer_(pointer) {
    TensorCoord thread_offset = ThreadMap::initial_offset(thread_idx) + threadblock_offset;
    extent_row_ = extent.row();
    extent_column_ = extent.column();
    thread_start_row_ = thread_offset.row();
    thread_start_column_ = thread_offset.column();

    CUTLASS_PRAGMA_UNROLL
    for (int c = 0; c < ThreadMap::Iterations::kColumn; ++c) {
      mask_.predicates[c] =
          ((thread_offset.column() + ThreadMap::Delta::kColumn * c) < extent.column());
    }

    if (!pointer) {
      mask_.clear();
    }

    state_[0] = state_[1] = state_[2] = 0;
  }

  CUTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {
    pointer_ += pointer_offset;
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
    AccessType const* frag_ptr = reinterpret_cast<AccessType const*>(&frag);

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
          int slot = row_guard ? params_.sorted_token_ids[global_m] :
              params_.total_valid_slots;
          bool slot_guard = slot < params_.total_valid_slots;
          int source_row = kSourceRowsAreSlots ? slot : (slot / params_.top_k);
          int expert = row_guard ? params_.expert_ids[global_m / params_.block_size_m] : 0;
          float a = slot_guard ? params_.a_scale[source_row] : 0.0f;
          float const* b_scale = params_.b_channel_scale +
              static_cast<int64_t>(expert) * params_.b_channel_scale_stride;
          __nv_bfloat16* row_pointer = pointer_ +
              static_cast<int64_t>(slot) * params_.output_stride;

          CUTLASS_PRAGMA_UNROLL
          for (int column = 0; column < ThreadMap::Iterations::kColumn; ++column) {
            int column_offset = column * ThreadMap::Delta::kColumn;
            int global_n = column_offset + thread_start_column_;
            bool guard = slot_guard && mask_.predicates[column];
            AccessType const& access =
                frag_ptr[frag_row_idx * ThreadMap::Iterations::kColumn + column];

            if constexpr (ThreadMap::kElementsPerAccess == 4) {
              bool vector_store_aligned =
                  ((params_.output_stride | global_n) & 1) == 0;
              bool scale_load_aligned = (global_n & 3) == 0;
              if (guard && vector_store_aligned && global_n + 3 < extent_column_) {
                float4 b;
                if (scale_load_aligned) {
                  b = *reinterpret_cast<float4 const*>(b_scale + global_n);
                } else {
                  b.x = b_scale[global_n + 0];
                  b.y = b_scale[global_n + 1];
                  b.z = b_scale[global_n + 2];
                  b.w = b_scale[global_n + 3];
                }
                float v0 = static_cast<float>(access[0]) * a * b.x;
                float v1 = static_cast<float>(access[1]) * a * b.y;
                float v2 = static_cast<float>(access[2]) * a * b.z;
                float v3 = static_cast<float>(access[3]) * a * b.w;
                __nv_bfloat16* dst = row_pointer + global_n;
                *reinterpret_cast<__nv_bfloat162*>(dst + 0) =
                    __floats2bfloat162_rn(v0, v1);
                *reinterpret_cast<__nv_bfloat162*>(dst + 2) =
                    __floats2bfloat162_rn(v2, v3);
              } else {
                CUTLASS_PRAGMA_UNROLL
                for (int e = 0; e < ThreadMap::kElementsPerAccess; ++e) {
                  int n = global_n + e;
                  if (guard && n < extent_column_) {
                    float value =
                        static_cast<float>(access[e]) * a * b_scale[n];
                    row_pointer[n] = __float2bfloat16(value);
                  }
                }
              }
            } else {
              CUTLASS_PRAGMA_UNROLL
              for (int e = 0; e < ThreadMap::kElementsPerAccess; ++e) {
                int n = global_n + e;
                if (guard && n < extent_column_) {
                  float value =
                      static_cast<float>(access[e]) * a * b_scale[n];
                  row_pointer[n] = __float2bfloat16(value);
                }
              }
            }
          }
        }
      }
    }
  }

  CUTLASS_DEVICE
  GroupedScaledBf16OutputTileIterator& operator++() {
    ++state_[0];
    thread_start_row_ += ThreadMap::Shape::kRow;

    if (state_[0] == ThreadMap::Count::kRow) {
      state_[0] = 0;
      ++state_[1];
      thread_start_row_ += (ThreadMap::Shape::kGroup - 1) *
          ThreadMap::Shape::kRow * ThreadMap::Count::kRow;

      if (state_[1] == ThreadMap::Count::kGroup) {
        state_[1] = 0;
        ++state_[2];
        thread_start_row_ += ThreadMap::Count::kGroup *
            ThreadMap::Shape::kGroup * ThreadMap::Count::kRow *
            ThreadMap::Shape::kRow;

        if (state_[2] == ThreadMap::Count::kCluster) {
          state_[2] = 0;
          thread_start_row_ += ThreadMap::Shape::kGroup *
              ThreadMap::Shape::kRow * ThreadMap::Shape::kCluster *
              ThreadMap::Shape::kTile;
        }
      }
    }

    return *this;
  }

  CUTLASS_DEVICE
  GroupedScaledBf16OutputTileIterator& operator+=(int increment) {
    state_[0] += increment;
    int increment_row = state_[0] / ThreadMap::Count::kRow;
    state_[0] = state_[0] % ThreadMap::Count::kRow;
    thread_start_row_ += ThreadMap::Shape::kRow * increment;

    state_[1] += increment_row;
    int increment_group = state_[1] / ThreadMap::Count::kGroup;
    state_[1] = state_[1] % ThreadMap::Count::kGroup;
    thread_start_row_ += (ThreadMap::Shape::kGroup - 1) *
        ThreadMap::Shape::kRow * ThreadMap::Count::kRow * increment_row;

    state_[2] += increment_group;
    int increment_cluster = state_[2] / ThreadMap::Count::kCluster;
    state_[2] = state_[2] % ThreadMap::Count::kCluster;
    thread_start_row_ += ThreadMap::Count::kGroup *
        ThreadMap::Shape::kGroup * ThreadMap::Count::kRow *
        ThreadMap::Shape::kRow * increment_group;

    thread_start_row_ += ThreadMap::Shape::kGroup *
        ThreadMap::Shape::kRow * ThreadMap::Shape::kCluster *
        ThreadMap::Shape::kTile * increment_cluster;
    return *this;
  }
};

template <typename Mma_, bool SourceRowsAreSlots_>
struct Mxfp4GroupedScaledBf16EpilogueForMma {
  using Mma = Mma_;
  static bool const kSourceRowsAreSlots = SourceRowsAreSlots_;
  using OutputOp = IdentityInt32OutputOp<Mxfp4EpilogueOp::kCount>;
  using OutputTileThreadMap =
      typename cutlass::epilogue::threadblock::DefaultThreadMapTensorOp<
          typename Mma::Shape,
          typename Mma::Operator::Shape,
          Mma::Shape::kK / Mma::Operator::Shape::kK,
          int32_t,
          Mxfp4EpilogueOp::kCount>::Type;
  using OutputTileIterator =
      GroupedScaledBf16OutputTileIterator<OutputTileThreadMap, kSourceRowsAreSlots>;
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
    bool SourceRowsAreSlots_,
    int ProblemK_ = 0,
    int ProblemN_ = 0>
struct Mxfp4PackedBGroupedGemmKernel {
  using Mma = Mma_;
  using Epilogue = Epilogue_;
  static bool const kSourceRowsAreSlots = SourceRowsAreSlots_;
  static int const kProblemK = ProblemK_;
  static int const kProblemN = ProblemN_;
  using OutputOp = typename Epilogue::OutputOp;
  using WarpCount = typename Mma::WarpCount;
  static int const kThreadCount = 32 * WarpCount::kCount;

  struct Params {
    cutlass::gemm::GemmCoord problem_size;
    int8_t const* ptr_A;
    int lda;
    uint8_t const* ptr_B_packed;
    uint8_t const* ptr_B_shift2;
    int b_shift_ld;
    int64_t b_expert_stride;
    int64_t b_shift_expert_stride;
    int32_t const* expert_ids;
    int32_t const* sorted_token_ids;
    int32_t const* num_tokens_post_padded;
    int total_valid_slots;
    int top_k;
    int grid_m_tiles;
    int grid_n_tiles;
    bool persistent;
    typename Epilogue::OutputTileIterator::Params params_C;
    typename Epilogue::OutputTileIterator::TensorRef ref_C;
    typename Epilogue::OutputTileIterator::Params params_D;
    typename Epilogue::OutputTileIterator::TensorRef ref_D;
    typename OutputOp::Params output_op;

    CUTLASS_HOST_DEVICE
    Params()
        : ptr_A(nullptr),
          lda(0),
          ptr_B_packed(nullptr),
          ptr_B_shift2(nullptr),
          b_shift_ld(0),
          b_expert_stride(0),
          b_shift_expert_stride(0),
          expert_ids(nullptr),
          sorted_token_ids(nullptr),
          num_tokens_post_padded(nullptr),
          total_valid_slots(0),
          top_k(1),
          grid_m_tiles(0),
          grid_n_tiles(0),
          persistent(false) {}

    CUTLASS_HOST_DEVICE
    Params(
        cutlass::gemm::GemmCoord const& problem_size_,
        cutlass::TensorRef<int8_t const, cutlass::layout::RowMajor> ref_A,
        uint8_t const* ptr_B_packed_,
        uint8_t const* ptr_B_shift2_,
        int b_shift_ld_,
        int64_t b_expert_stride_,
        int64_t b_shift_expert_stride_,
        int32_t const* expert_ids_,
        int32_t const* sorted_token_ids_,
        int32_t const* num_tokens_post_padded_,
        int total_valid_slots_,
        int top_k_,
        int grid_m_tiles_,
        int grid_n_tiles_,
        bool persistent_,
        typename Epilogue::OutputTileIterator::Params params_C_,
        typename Epilogue::OutputTileIterator::TensorRef ref_C_,
        typename Epilogue::OutputTileIterator::Params params_D_,
        typename Epilogue::OutputTileIterator::TensorRef ref_D_,
        typename OutputOp::Params output_op_)
        : problem_size(problem_size_),
          ptr_A(ref_A.data()),
          lda(ref_A.layout().stride(0)),
          ptr_B_packed(ptr_B_packed_),
          ptr_B_shift2(ptr_B_shift2_),
          b_shift_ld(b_shift_ld_),
          b_expert_stride(b_expert_stride_),
          b_shift_expert_stride(b_shift_expert_stride_),
          expert_ids(expert_ids_),
          sorted_token_ids(sorted_token_ids_),
          num_tokens_post_padded(num_tokens_post_padded_),
          total_valid_slots(total_valid_slots_),
          top_k(top_k_),
          grid_m_tiles(grid_m_tiles_),
          grid_n_tiles(grid_n_tiles_),
          persistent(persistent_),
          params_C(params_C_),
          ref_C(ref_C_),
          params_D(params_D_),
          ref_D(ref_D_),
          output_op(output_op_) {}
  };

  union SharedStorage {
    typename Mma::SharedStorage main_loop;
    typename Epilogue::SharedStorage epilogue;
  };

  struct ProblemShape {
    int m;
    int n;
    int k;
    int padded_m;

    CUTLASS_DEVICE
    bool contains_tile(int m_offset, int n_offset) const {
      return m_offset < m && m_offset < padded_m && n_offset < n;
    }

    CUTLASS_DEVICE
    int m_tiles() const {
      return (padded_m + Mma::Shape::kM - 1) / Mma::Shape::kM;
    }

    CUTLASS_DEVICE
    int k_iterations() const {
      return (k + Mma::Shape::kK - 1) / Mma::Shape::kK;
    }

    CUTLASS_DEVICE
    int packed_n_groups8() const {
      return (n + 7) / 8;
    }

    CUTLASS_DEVICE
    cutlass::MatrixCoord mn() const {
      return {m, n};
    }
  };

  CUTLASS_DEVICE
  ProblemShape problem_shape(Params const& params) const {
    ProblemShape shape{
        params.problem_size.m(),
        params.problem_size.n(),
        params.problem_size.k(),
        params.num_tokens_post_padded ?
            params.num_tokens_post_padded[0] : params.problem_size.m()};
    if constexpr (kProblemN > 0) {
      shape.n = kProblemN;
    }
    if constexpr (kProblemK > 0) {
      shape.k = kProblemK;
    }
    return shape;
  }

  template <bool FullNTile>
  CUTLASS_DEVICE
  void run_mma(
      Params const& params,
      Mma& mma,
      typename Mma::FragmentC& accumulators,
      ProblemShape const& shape,
      uint8_t const* b_packed,
      uint8_t const* b_shift,
      int threadblock_m_offset,
      int threadblock_n_offset) const {
    mma.template operator()<FullNTile, kSourceRowsAreSlots>(
        shape.k_iterations(),
        accumulators,
        params.ptr_A,
        params.lda,
        params.sorted_token_ids,
        params.total_valid_slots,
        params.top_k,
        b_packed,
        shape.packed_n_groups8(),
        b_shift,
        params.b_shift_ld,
        threadblock_m_offset,
        0,
        threadblock_n_offset,
        shape.m,
        shape.k,
        shape.n,
        accumulators);
  }

  CUTLASS_DEVICE
  void run_tile(
      Params const& params,
      SharedStorage& shared_storage,
      int m_tile,
      int n_tile) {
    int threadblock_m_offset = m_tile * Mma::Shape::kM;
    int threadblock_n_offset = n_tile * Mma::Shape::kN;
    ProblemShape shape = problem_shape(params);
    if (!shape.contains_tile(threadblock_m_offset, threadblock_n_offset)) {
      return;
    }
    int expert = params.expert_ids[m_tile];

    int thread_idx = threadIdx.x;
    int warp_idx = cutlass::canonical_warp_idx_sync();
    int lane_idx = threadIdx.x % 32;

    Mma mma(shared_storage.main_loop, thread_idx, warp_idx, lane_idx);
    typename Mma::FragmentC accumulators;
    accumulators.clear();

    uint8_t const* b_packed =
        params.ptr_B_packed + static_cast<int64_t>(expert) * params.b_expert_stride;
    uint8_t const* b_shift =
        params.ptr_B_shift2 + static_cast<int64_t>(expert) * params.b_shift_expert_stride;

    if ((shape.n % Mma::Shape::kN) == 0) {
      run_mma<true>(
          params,
          mma,
          accumulators,
          shape,
          b_packed,
          b_shift,
          threadblock_m_offset,
          threadblock_n_offset);
    } else {
      run_mma<false>(
          params,
          mma,
          accumulators,
          shape,
          b_packed,
          b_shift,
          threadblock_m_offset,
          threadblock_n_offset);
    }

    OutputOp output_op(params.output_op);
    cutlass::MatrixCoord threadblock_offset(threadblock_m_offset, threadblock_n_offset);
    typename Epilogue::OutputTileIterator iterator_C(
        params.params_C,
        params.ref_C.data(),
        shape.mn(),
        thread_idx,
        threadblock_offset);
    typename Epilogue::OutputTileIterator iterator_D(
        params.params_D,
        params.ref_D.data(),
        shape.mn(),
        thread_idx,
        threadblock_offset);

    Epilogue epilogue(shared_storage.epilogue, thread_idx, warp_idx, lane_idx);
    epilogue(output_op, iterator_D, accumulators, iterator_C);
  }

  CUTLASS_DEVICE
  void operator()(Params const& params, SharedStorage& shared_storage) {
    if (params.persistent) {
      int actual_m_tiles = problem_shape(params).m_tiles();
      actual_m_tiles =
          actual_m_tiles < params.grid_m_tiles ? actual_m_tiles : params.grid_m_tiles;
      int total_tiles = actual_m_tiles * params.grid_n_tiles;
      for (int tile = blockIdx.x; tile < total_tiles; tile += gridDim.x) {
        int m_tile = tile / params.grid_n_tiles;
        int n_tile = tile - m_tile * params.grid_n_tiles;
        run_tile(params, shared_storage, m_tile, n_tile);
      }
    } else {
      run_tile(params, shared_storage, blockIdx.x, blockIdx.y);
    }
  }
};

#define MXFP4_DEFINE_GROUPED_SCALED_BF16_KERNEL(TM, WM, TN, WN)       \
  using Mxfp4GroupedMma_##TM##x##TN##x128 =                          \
      Mxfp4PackedBThreadblockMmaSkeleton<                             \
          Mxfp4PackedBMmaCoreForShape<TM, WM, TN, WN, 128>, true>;    \
  using Mxfp4PackedBGroupedScaledBf16CutlassKernel_##TM##x##TN##x128##_W13 = \
      Mxfp4PackedBGroupedGemmKernel<                                  \
          Mxfp4GroupedMma_##TM##x##TN##x128,                          \
          typename Mxfp4GroupedScaledBf16EpilogueForMma<              \
              Mxfp4GroupedMma_##TM##x##TN##x128, false>::Epilogue,    \
          false>;                                                     \
  using Mxfp4PackedBGroupedScaledBf16CutlassKernel_##TM##x##TN##x128##_W2 = \
      Mxfp4PackedBGroupedGemmKernel<                                  \
          Mxfp4GroupedMma_##TM##x##TN##x128,                          \
          typename Mxfp4GroupedScaledBf16EpilogueForMma<              \
              Mxfp4GroupedMma_##TM##x##TN##x128, true>::Epilogue,     \
          true>;                                                      \
  using Mxfp4PackedBGroupedScaledBf16CutlassKernel_##TM##x##TN##x128 = \
      Mxfp4PackedBGroupedScaledBf16CutlassKernel_##TM##x##TN##x128##_W13

MXFP4_DEFINE_GROUPED_SCALED_BF16_KERNEL(16, 16, 32, 16);
MXFP4_DEFINE_GROUPED_SCALED_BF16_KERNEL(16, 16, 64, 16);
MXFP4_DEFINE_GROUPED_SCALED_BF16_KERNEL(16, 16, 128, 32);

MXFP4_DEFINE_GROUPED_SCALED_BF16_KERNEL(32, 16, 32, 16);
MXFP4_DEFINE_GROUPED_SCALED_BF16_KERNEL(32, 16, 64, 32);
MXFP4_DEFINE_GROUPED_SCALED_BF16_KERNEL(32, 16, 128, 64);

MXFP4_DEFINE_GROUPED_SCALED_BF16_KERNEL(64, 32, 32, 16);
MXFP4_DEFINE_GROUPED_SCALED_BF16_KERNEL(64, 32, 64, 32);
MXFP4_DEFINE_GROUPED_SCALED_BF16_KERNEL(64, 32, 128, 64);

MXFP4_DEFINE_GROUPED_SCALED_BF16_KERNEL(128, 64, 32, 16);
MXFP4_DEFINE_GROUPED_SCALED_BF16_KERNEL(128, 64, 64, 32);
MXFP4_DEFINE_GROUPED_SCALED_BF16_KERNEL(128, 64, 128, 64);

#undef MXFP4_DEFINE_GROUPED_SCALED_BF16_KERNEL

}  // namespace cutlass_core
}  // namespace mxfp4_int8
