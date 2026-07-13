// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "pipeline_tests_helper.hpp"

#include "ck_tile/core/arch/arch.hpp"
#include "ck_tile/core/arch/mma/mma.hpp"
#include "ck_tile/core/arch/mma/mma_wavewise.hpp"

#include <gtest/gtest.h>

using namespace ck_tile;
using namespace ck_tile::core::arch;
using namespace ck_tile::core::arch::mma;

namespace {

template <typename AType,
          typename BType,
          typename CType,
          uint32_t WaveTileM,
          uint32_t WaveTileN,
          uint32_t WaveTileK>
struct CabOperatorKernel
{
    static constexpr int kBlockSize = mma_pipeline_test::getCMakeWaveSize();

    __device__ void
    operator()(const void* a_per_lane, const void* b_per_lane, void* c_per_lane) const
    {
        using CompilerTarget = decltype(get_compiler_target());
        using Pipeline       = WaveWiseMmaPipeline<AType,
                                                   BType,
                                                   CType,
                                                   WaveTileM,
                                                   WaveTileN,
                                                   WaveTileK,
                                                   MmaAccumPolicy::ROW_MAJOR,
                                                   false,
                                                   1,
                                                   1,
                                                   1,
                                                   CompilerTarget>;
        using ATensor        = typename Pipeline::AWarpTensor;
        using BTensor        = typename Pipeline::BWarpTensor;
        using CTensor        = typename Pipeline::CWarpTensor;

        const uint32_t lane = threadIdx.x;

        ATensor a;
        BTensor b;
        CTensor c;
        __builtin_memcpy(
            &a, static_cast<const uint8_t*>(a_per_lane) + lane * sizeof(ATensor), sizeof(ATensor));
        __builtin_memcpy(
            &b, static_cast<const uint8_t*>(b_per_lane) + lane * sizeof(BTensor), sizeof(BTensor));
        __builtin_memset(&c, 0, sizeof(CTensor));

        if constexpr(MmaOpTraits<typename Pipeline::MmaOp>::IsSupported)
        {
            Pipeline pipeline{};
            pipeline(c, a, b);
            __builtin_memcpy(
                static_cast<uint8_t*>(c_per_lane) + lane * sizeof(CTensor), &c, sizeof(CTensor));
        }
    }
};

struct CabPipelineFactory
{
    template <typename Target>
    struct Create
    {
        using type = WaveWiseMmaPipeline<fp16_t,
                                         fp16_t,
                                         fp32_t,
                                         16u,
                                         16u,
                                         32u,
                                         MmaAccumPolicy::ROW_MAJOR,
                                         false,
                                         1,
                                         1,
                                         1,
                                         Target>;
    };
};

const auto should_skip = [](amdgcn_target_id currentArchId) {
    bool isSupportedMfma =
        (currentArchId >= amdgcn_target_id::GFX942) && (currentArchId <= amdgcn_target_id::GFX950);
    return ((currentArchId == amdgcn_target_id::HOST) || !isSupportedMfma);
};

} // namespace

TEST(MmaPipelineOperatorOverloads, Dense_CAB)
{
    using Kernel = CabOperatorKernel<fp16_t, fp16_t, fp32_t, 16u, 16u, 32u>;
    mma_pipeline_test::run_pipeline_matrix_test<CabPipelineFactory::template Create,
                                                Kernel,
                                                fp16_t,
                                                fp16_t,
                                                fp32_t>(16, 16, 32, should_skip, Kernel{});
}
