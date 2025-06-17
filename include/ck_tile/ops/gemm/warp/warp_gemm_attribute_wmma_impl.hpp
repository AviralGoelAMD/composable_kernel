// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

struct WarpGemmAttributeWmmaImpl_f32_16x16x16_f16
{
    using ADataType = fp16_t;
    using BDataType = fp16_t;
    using CDataType = float;

    using AVecType = ext_vector_t<ADataType, 8>;
    using BVecType = ext_vector_t<BDataType, 8>;
    using CVecType = ext_vector_t<CDataType, 8>;

    static constexpr index_t kM = 16;
    static constexpr index_t kN = 16;
    static constexpr index_t kK = 16;

    static constexpr index_t kAMBlock = 1;
    static constexpr index_t kBNBlock = 1;

    static constexpr index_t kAMLane     = 16;
    static constexpr index_t kBNLane     = 16;
    static constexpr index_t kABKLane    = 2;
    static constexpr index_t kABKPerLane = 8;

    static constexpr index_t kCMLane     = 2;
    static constexpr index_t kCNLane     = 16;
    static constexpr index_t kCM0PerLane = 1;
    static constexpr index_t kCM1PerLane = 8;

    // c_vec += a_vec * b_vec
    CK_TILE_DEVICE void
    operator()(CVecType& c_vec, const AVecType& a_vec, const BVecType& b_vec) const
    {
#if defined(__gfx12__)
        c_vec = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_vec, b_vec, c_vec);
#else
        ck_tile::ignore = c_vec;
        ck_tile::ignore = a_vec;
        ck_tile::ignore = b_vec;
#endif
    }

    // c_vec = a_vec * b_vec
    CK_TILE_DEVICE CVecType operator()(const AVecType& a_vec, const BVecType& b_vec) const
    {
#if defined(__gfx12__)
        return bit_cast<CVecType>(
            __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_vec, b_vec, CVecType{0.f}));
#else
        ck_tile::ignore = a_vec;
        ck_tile::ignore = b_vec;
        return CVecType{0.f};
#endif
    }
};

struct WarpGemmAttributeWmmaImpl_f32_16x16x16_bf16
{
    using ADataType = bf16_t;
    using BDataType = bf16_t;
    using CDataType = float;

    using AVecType = ext_vector_t<ADataType, 8>;
    using BVecType = ext_vector_t<BDataType, 8>;
    using CVecType = ext_vector_t<CDataType, 8>;

    static constexpr index_t kM = 16;
    static constexpr index_t kN = 16;
    static constexpr index_t kK = 16;

    static constexpr index_t kAMBlock = 1;
    static constexpr index_t kBNBlock = 1;

    static constexpr index_t kAMLane     = 16;
    static constexpr index_t kBNLane     = 16;
    static constexpr index_t kABKLane    = 2;
    static constexpr index_t kABKPerLane = 8;

    static constexpr index_t kCMLane     = 2;
    static constexpr index_t kCNLane     = 16;
    static constexpr index_t kCM0PerLane = 1;
    static constexpr index_t kCM1PerLane = 8;

    // c_vec += a_vec * b_vec
    CK_TILE_DEVICE void
    operator()(CVecType& c_vec, const AVecType& a_vec, const BVecType& b_vec) const
    {
#if defined(__gfx12__)
        c_vec = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a_vec, b_vec, c_vec);
#else
        ck_tile::ignore = c_vec;
        ck_tile::ignore = a_vec;
        ck_tile::ignore = b_vec;
#endif
    }

    // c_vec = a_vec * b_vec
    CK_TILE_DEVICE CVecType operator()(const AVecType& a_vec, const BVecType& b_vec) const
    {
#if defined(__gfx12__)
        return bit_cast<CVecType>(
            __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a_vec, b_vec, CVecType{0.f}));
#else
        ck_tile::ignore = a_vec;
        ck_tile::ignore = b_vec;
        return CVecType{0.f};
#endif
    }
};

template <typename AType_, typename BType_>
struct WarpGemmAttributeWmmaImpl_f32_16x16x16_f8_base
{
    using ADataType = AType_;
    using BDataType = BType_;
    using CDataType = float;

    using AVecType = ext_vector_t<ADataType, 8>;
    using BVecType = ext_vector_t<BDataType, 8>;
    using CVecType = ext_vector_t<CDataType, 8>;

    static constexpr index_t kM = 16;
    static constexpr index_t kN = 16;
    static constexpr index_t kK = 16;

    static constexpr index_t kAMBlock = 1;
    static constexpr index_t kBNBlock = 1;

    static constexpr index_t kAMLane     = 16;
    static constexpr index_t kBNLane     = 16;
    static constexpr index_t kABKLane    = 2;
    static constexpr index_t kABKPerLane = 8;

    static constexpr index_t kCMLane     = 2;
    static constexpr index_t kCNLane     = 16;
    static constexpr index_t kCM0PerLane = 1;
    static constexpr index_t kCM1PerLane = 8;

    // c_vec += a_vec * b_vec
    CK_TILE_DEVICE void
    operator()(CVecType& c_vec, const AVecType& a_vec, const BVecType& b_vec) const
    {
#if defined(__gfx12__)
        if constexpr(std::is_same_v<ADataType, fp8_t> && std::is_same_v<BDataType, fp8_t>)
            c_vec = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(
                bit_cast<int32x2_t>(a_vec), bit_cast<int32x2_t>(b_vec), c_vec);
        else if constexpr(std::is_same_v<ADataType, fp8_t> && std::is_same_v<BDataType, bf8_t>)
            c_vec = __builtin_amdgcn_wmma_f32_16x16x16_fp8_bf8_w32_gfx12(
                bit_cast<int32x2_t>(a_vec), bit_cast<int32x2_t>(b_vec), c_vec);
        else if constexpr(std::is_same_v<ADataType, bf8_t> && std::is_same_v<BDataType, fp8_t>)
            c_vec = __builtin_amdgcn_wmma_f32_16x16x16_bf8_fp8_w32_gfx12(
                bit_cast<int32x2_t>(a_vec), bit_cast<int32x2_t>(b_vec), c_vec);
        else if constexpr(std::is_same_v<ADataType, bf8_t> && std::is_same_v<BDataType, bf8_t>)
            c_vec = __builtin_amdgcn_wmma_f32_16x16x16_bf8_bf8_w32_gfx12(
                bit_cast<int32x2_t>(a_vec), bit_cast<int32x2_t>(b_vec), c_vec);
#else
        ck_tile::ignore = c_vec;
        ck_tile::ignore = a_vec;
        ck_tile::ignore = b_vec;
#endif
    }

    // c_vec = a_vec * b_vec
    CK_TILE_DEVICE CVecType operator()(const AVecType& a_vec, const BVecType& b_vec) const
    {
#if defined(__gfx12__)
        if constexpr(std::is_same_v<ADataType, fp8_t> && std::is_same_v<BDataType, fp8_t>)
            return __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(
                bit_cast<int32x2_t>(a_vec), bit_cast<int32x2_t>(b_vec), CVecType{0.f});
        else if constexpr(std::is_same_v<ADataType, fp8_t> && std::is_same_v<BDataType, bf8_t>)
            return __builtin_amdgcn_wmma_f32_16x16x16_fp8_bf8_w32_gfx12(
                bit_cast<int32x2_t>(a_vec), bit_cast<int32x2_t>(b_vec), CVecType{0.f});
        else if constexpr(std::is_same_v<ADataType, bf8_t> && std::is_same_v<BDataType, fp8_t>)
            return __builtin_amdgcn_wmma_f32_16x16x16_bf8_fp8_w32_gfx12(
                bit_cast<int32x2_t>(a_vec), bit_cast<int32x2_t>(b_vec), CVecType{0.f});
        else if constexpr(std::is_same_v<ADataType, bf8_t> && std::is_same_v<BDataType, bf8_t>)
            return __builtin_amdgcn_wmma_f32_16x16x16_bf8_bf8_w32_gfx12(
                bit_cast<int32x2_t>(a_vec), bit_cast<int32x2_t>(b_vec), CVecType{0.f});
#else
        ck_tile::ignore = a_vec;
        ck_tile::ignore = b_vec;
        return CVecType{0.f};
#endif
    }
};

using WarpGemmAttributeWmmaImpl_f32_16x16x16_fp8_fp8 =
    WarpGemmAttributeWmmaImpl_f32_16x16x16_f8_base<fp8_t, fp8_t>;

using WarpGemmAttributeWmmaImpl_f32_16x16x16_fp8_bf8 =
    WarpGemmAttributeWmmaImpl_f32_16x16x16_f8_base<fp8_t, bf8_t>;

using WarpGemmAttributeWmmaImpl_f32_16x16x16_bf8_fp8 =
    WarpGemmAttributeWmmaImpl_f32_16x16x16_f8_base<bf8_t, fp8_t>;

using WarpGemmAttributeWmmaImpl_f32_16x16x16_bf8_bf8 =
    WarpGemmAttributeWmmaImpl_f32_16x16x16_f8_base<bf8_t, bf8_t>;

} // namespace ck_tile
