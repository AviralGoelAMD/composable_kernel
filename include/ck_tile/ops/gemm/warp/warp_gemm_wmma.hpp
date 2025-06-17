// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm_impl.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm_attribute_wmma.hpp"

namespace ck_tile {

// fp16

using WarpGemmWmma_f32_16x16x16_f16 =
    WarpGemmImpl<WarpGemmAttributeWmma<WarpGemmAttributeWmmaImpl_f32_16x16x16_f16>>;

using WarpGemmWmma_f32_16x16x16_f16_CTransposed = WarpGemmImpl<
    WarpGemmAttributeWmmaTransposedCDistribution<WarpGemmAttributeWmmaImpl_f32_16x16x16_f16>>;

// bf16

using WarpGemmWmma_f32_16x16x16_bf16 =
    WarpGemmImpl<WarpGemmAttributeWmma<WarpGemmAttributeWmmaImpl_f32_16x16x16_bf16>>;

using WarpGemmWmma_f32_16x16x16_bf16_CTransposed = WarpGemmImpl<
    WarpGemmAttributeWmmaTransposedCDistribution<WarpGemmAttributeWmmaImpl_f32_16x16x16_bf16>>;

// fp8 and bf8

using WarpGemmWmma_f32_16x16x16_fp8_fp8 =
    WarpGemmImpl<WarpGemmAttributeWmma<WarpGemmAttributeWmmaImpl_f32_16x16x16_fp8_fp8>>;

using WarpGemmWmma_f32_16x16x16_fp8_bf8 =
    WarpGemmImpl<WarpGemmAttributeWmma<WarpGemmAttributeWmmaImpl_f32_16x16x16_fp8_bf8>>;

using WarpGemmWmma_f32_16x16x16_bf8_fp8 =
    WarpGemmImpl<WarpGemmAttributeWmma<WarpGemmAttributeWmmaImpl_f32_16x16x16_bf8_fp8>>;

using WarpGemmWmma_f32_16x16x16_bf8_bf8 =
    WarpGemmImpl<WarpGemmAttributeWmma<WarpGemmAttributeWmmaImpl_f32_16x16x16_bf8_bf8>>;

using WarpGemmWmma_f32_16x16x16_fp8_fp8_CTransposed = WarpGemmImpl<
    WarpGemmAttributeWmmaTransposedCDistribution<WarpGemmAttributeWmmaImpl_f32_16x16x16_fp8_fp8>>;

using WarpGemmWmma_f32_16x16x16_fp8_bf8_CTransposed = WarpGemmImpl<
    WarpGemmAttributeWmmaTransposedCDistribution<WarpGemmAttributeWmmaImpl_f32_16x16x16_fp8_bf8>>;

using WarpGemmWmma_f32_16x16x16_bf8_fp8_CTransposed = WarpGemmImpl<
    WarpGemmAttributeWmmaTransposedCDistribution<WarpGemmAttributeWmmaImpl_f32_16x16x16_bf8_fp8>>;

using WarpGemmWmma_f32_16x16x16_bf8_bf8_CTransposed = WarpGemmImpl<
    WarpGemmAttributeWmmaTransposedCDistribution<WarpGemmAttributeWmmaImpl_f32_16x16x16_bf8_bf8>>;

} // namespace ck_tile
