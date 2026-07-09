// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Numerical 3D Wrw test for ConvHipImplicitGemm3DGroupWrwXdlops on a shape
// with element-strides exceeding INT_MAX (ROCM-23997). NOTE: as of this
// backport, the container CK does not ship a Large_Tensor grouped 3D WRW
// instance (MIOPEN_CK_LARGE_TENSOR_BWD_WRW==0), so this solver is expected to
// report inapplicable/skip for this shape until CK is rebuilt with
// ROCm/rocm-libraries PR #7734.
//
// Shape: x = (1, 96, 512, 512, 88), w = (16, 96, 1, 1, 1), group=1, pad=0,
// stride=1.

#include <algorithm>
#include <cstddef>

#include "unit_conv_solver.hpp"
#include "get_handle.hpp"

namespace {

using miopen::unit_tests::ConvolutionDescriptorParams;
using miopen::unit_tests::ConvTestCase;
using miopen::unit_tests::TensorDescriptorParams;

std::vector<ConvTestCase> GetLargeStrideWrwTestCases(miopenDataType_t dt)
{
    std::vector<ConvTestCase> cases;
    for(auto layout : {miopenTensorNDHWC, miopenTensorNCDHW})
    {
        cases.push_back(
            ConvTestCase{TensorDescriptorParams{dt, layout, {1, 96, 512, 512, 88}},
                        TensorDescriptorParams{dt, layout, {16, 96, 1, 1, 1}},
                        dt,
                        ConvolutionDescriptorParams{{0, 0, 0}, {1, 1, 1}, {1, 1, 1}, 1}});
    }
    return cases;
}

const miopen::unit_tests::UnitTestConvSolverParams& GetTestParams()
{
    static const auto params = [] {
#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
        Gpu supportedDevices = Gpu::gfx90A | Gpu::gfx94X | Gpu::gfx950;
#else
        Gpu supportedDevices = Gpu::None;
#endif
        auto p = miopen::unit_tests::UnitTestConvSolverParams(supportedDevices);
        p.Tunable(5);
        return p;
    }();
    return params;
}

struct MemoryEstimate
{
    std::size_t required;
    std::size_t available;
    bool applicable;
};

MemoryEstimate EstimateRequiredMemoryWrw(const ConvTestCase& conv_config,
                                         const miopen::solver::conv::ConvSolverInterface& solver)
{
    auto&& handle = get_handle();

    const auto x_desc    = conv_config.GetXTensorDescriptor();
    const auto w_desc    = conv_config.GetWTensorDescriptor();
    const auto conv_desc = conv_config.GetConv();
    const auto y_desc = conv_desc.GetForwardOutputTensor(x_desc, w_desc, conv_config.GetYDataType());

    const auto problem = miopen::conv::ProblemDescription(
        y_desc, w_desc, x_desc, conv_desc, miopen::conv::Direction::BackwardWeights);
    auto ctx = miopen::ExecutionContext{&handle};
    problem.SetupFloats(ctx);

    // The grouped CK bwd/wrw large-tensor (>INT_MAX) path is compiled out until
    // the container CK ships the int64 MakeArgumentPointer overloads for grouped
    // bwd/wrw (MIOPEN_CK_LARGE_TENSOR_BWD_WRW==0, pending rocm-libraries PR
    // #7734). For this >INT_MAX shape the solver is therefore inapplicable, and
    // RunTest() treats an inapplicable config as a hard failure -- so report it
    // here and let the caller GTEST_SKIP instead. Self-heals (becomes
    // applicable) once the macro is flipped to 1 and CK is rebuilt.
    if(!solver.IsApplicable(ctx, problem))
        return {0, 0, false};

    const std::size_t ws_size =
        solver.MayNeedWorkspace() ? solver.GetWorkspaceSize(ctx, problem) : 0;
    const std::size_t x_bytes = x_desc.GetNumBytes();
    const std::size_t y_bytes = y_desc.GetNumBytes();
    const std::size_t w_bytes = w_desc.GetNumBytes();
    const std::size_t h_bytes = std::max(x_bytes, y_bytes);

    const std::size_t raw_mem      = ws_size + x_bytes + y_bytes + w_bytes + 4 * h_bytes;
    const std::size_t headroom     = std::max<std::size_t>(1ULL << 30, raw_mem / 10);
    const std::size_t required_mem = raw_mem + headroom;
    const std::size_t device_mem   = handle.GetGlobalMemorySize();

    return {required_mem, device_mem, true};
}

} // namespace

#define SKIP_IF_INSUFFICIENT_DEVICE_MEMORY(solver_expr)                           \
    do                                                                            \
    {                                                                             \
        miopen::unit_tests::UnitTestConvSolverParams _params;                     \
        miopenConvAlgorithm_t _algo;                                              \
        ConvTestCase _tc;                                                         \
        std::tie(_params, _algo, _tc) = this->GetParam();                         \
        const auto _mem = EstimateRequiredMemoryWrw(_tc, (solver_expr));          \
        if(!_mem.applicable)                                                      \
        {                                                                         \
            GTEST_SKIP() << "grouped CK bwd/wrw large-tensor path not yet enabled "\
                            "(MIOPEN_CK_LARGE_TENSOR_BWD_WRW=0, pending "         \
                            "rocm-libraries PR #7734)";                           \
        }                                                                         \
        if(_mem.available < _mem.required)                                        \
        {                                                                         \
            GTEST_SKIP() << "Insufficient device memory: need " << _mem.required  \
                         << " bytes, device has " << _mem.available;              \
        }                                                                         \
    } while(0)

using GPU_UnitTestConvSolverImplicitGemm3DGroupWrwXdlops_LargeStride_FP16 =
    GPU_UnitTestConvSolverWrw_FP16;
using GPU_UnitTestConvSolverImplicitGemm3DGroupWrwXdlops_LargeStride_BFP16 =
    GPU_UnitTestConvSolverWrw_BFP16;
using GPU_UnitTestConvSolverImplicitGemm3DGroupWrwXdlops_LargeStride_FP32 =
    GPU_UnitTestConvSolverWrw_FP32;

TEST_P(GPU_UnitTestConvSolverImplicitGemm3DGroupWrwXdlops_LargeStride_FP16,
       ConvHipImplicitGemm3DGroupWrwXdlops)
{
    const auto solver = miopen::solver::conv::ConvHipImplicitGemm3DGroupWrwXdlops{};
    SKIP_IF_INSUFFICIENT_DEVICE_MEMORY(solver);
    this->RunTest(solver);
};

TEST_P(GPU_UnitTestConvSolverImplicitGemm3DGroupWrwXdlops_LargeStride_BFP16,
       ConvHipImplicitGemm3DGroupWrwXdlops)
{
    const auto solver = miopen::solver::conv::ConvHipImplicitGemm3DGroupWrwXdlops{};
    SKIP_IF_INSUFFICIENT_DEVICE_MEMORY(solver);
    this->RunTest(solver);
};

TEST_P(GPU_UnitTestConvSolverImplicitGemm3DGroupWrwXdlops_LargeStride_FP32,
       ConvHipImplicitGemm3DGroupWrwXdlops)
{
    const auto solver = miopen::solver::conv::ConvHipImplicitGemm3DGroupWrwXdlops{};
    SKIP_IF_INSUFFICIENT_DEVICE_MEMORY(solver);
    this->RunTest(solver);
};

INSTANTIATE_TEST_SUITE_P(
    Full,
    GPU_UnitTestConvSolverImplicitGemm3DGroupWrwXdlops_LargeStride_FP16,
    testing::Combine(testing::Values(GetTestParams()),
                     testing::Values(miopenConvolutionAlgoImplicitGEMM),
                     testing::ValuesIn(GetLargeStrideWrwTestCases(miopenHalf))));

INSTANTIATE_TEST_SUITE_P(
    Full,
    GPU_UnitTestConvSolverImplicitGemm3DGroupWrwXdlops_LargeStride_BFP16,
    testing::Combine(testing::Values(GetTestParams()),
                     testing::Values(miopenConvolutionAlgoImplicitGEMM),
                     testing::ValuesIn(GetLargeStrideWrwTestCases(miopenBFloat16))));

INSTANTIATE_TEST_SUITE_P(
    Full,
    GPU_UnitTestConvSolverImplicitGemm3DGroupWrwXdlops_LargeStride_FP32,
    testing::Combine(testing::Values(GetTestParams()),
                     testing::Values(miopenConvolutionAlgoImplicitGEMM),
                     testing::ValuesIn(GetLargeStrideWrwTestCases(miopenFloat))));
