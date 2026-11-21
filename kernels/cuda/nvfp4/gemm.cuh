// Adapted from FlashInfer fp4_gemm_template_sm100.h, Apache-2.0.
// Copyright (c) 2020-2023, NVIDIA CORPORATION. All rights reserved.
#pragma once
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include <stdexcept>

namespace thorustinfer_nvfp4 {
using namespace cute;
using Shape = cute::Shape<_256, _128, _256>;
using Cluster = cute::Shape<int, int, _1>;
using Arch = cutlass::arch::Sm100;
using Row = cutlass::layout::RowMajor;
using Col = cutlass::layout::ColumnMajor;
using Weight = cutlass::float_e2m1_t;
using Scale = cutlass::float_ue4m3_t;
using Output = cutlass::half_t;
using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    Arch, cutlass::arch::OpClassTensorOp, Shape, Cluster,
    cutlass::epilogue::collective::EpilogueTileAuto, float, float, void, Row, 8,
    Output, Row, 8, cutlass::epilogue::TmaWarpSpecialized2Sm,
    cutlass::epilogue::fusion::PerRowLinCombPerRowBiasEltAct<
        cutlass::epilogue::thread::Identity, Output, float, float, void,
        float>>::CollectiveOp;
using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    Arch, cutlass::arch::OpClassBlockScaledTensorOp, cute::tuple<Weight, Scale>,
    Row, 32, cute::tuple<Weight, Scale>, Col, 32, float, Shape, Cluster,
    cutlass::gemm::collective::StageCountAutoCarveout<int(
        sizeof(typename Epilogue::SharedStorage))>,
    cutlass::gemm::KernelTmaWarpSpecialized2SmNvf4Sm100>::CollectiveOp;
using Kernel =
    cutlass::gemm::kernel::GemmUniversal<cute::Shape<int, int, int, int>,
                                         Mainloop, Epilogue,
                                         cutlass::gemm::PersistentScheduler>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
} // namespace thorustinfer_nvfp4
inline void nvfp4_gemm(int m, int n, int k, void *out, const void *a,
                       const void *b, const void *as, const void *bs,
                       const float *alpha, void *work, size_t bytes,
                       cudaStream_t stream) {
  using namespace thorustinfer_nvfp4;
  Gemm::Arguments args{};
  args.mode = cutlass::gemm::GemmUniversalMode::kGemm;
  args.problem_shape = cute::make_shape(m, n, k, 1);
  args.epilogue.thread.alpha_ptr = alpha;
  args.mainloop.ptr_A = static_cast<const Weight *>(a);
  args.mainloop.ptr_B = static_cast<const Weight *>(b);
  args.mainloop.ptr_SFA = static_cast<const Scale *>(as);
  args.mainloop.ptr_SFB = static_cast<const Scale *>(bs);
  args.epilogue.ptr_C = out;
  args.epilogue.ptr_D = static_cast<Output *>(out);
  args.mainloop.dA = cute::make_int_tuple_from<Kernel::StrideA>(k, 0);
  args.mainloop.dB = cute::make_int_tuple_from<Kernel::StrideB>(k, 0);
  args.epilogue.dC = cute::make_int_tuple_from<Kernel::StrideC>(n, 0);
  args.epilogue.dD = args.epilogue.dC;
  using SF = Mainloop::Sm1xxBlkScaledConfig;
  args.mainloop.layout_SFA = SF::tile_atom_to_shape_SFA(args.problem_shape);
  args.mainloop.layout_SFB = SF::tile_atom_to_shape_SFB(args.problem_shape);
  args.hw_info.cluster_shape = dim3(2, 4, 1);
  args.hw_info.cluster_shape_fallback = dim3(2, 1, 1);
  args.scheduler.max_swizzle_size = 1;
  Gemm gemm;
  if (gemm.get_workspace_size(args) > bytes)
    throw std::runtime_error("insufficient NVFP4 workspace");
  auto status = gemm.can_implement(args);
  if (status == cutlass::Status::kSuccess)
    status = gemm.run(args, work, stream, nullptr, true);
  if (status != cutlass::Status::kSuccess)
    throw std::runtime_error(cutlassGetStatusString(status));
}
