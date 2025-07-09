// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core/config.hpp"

namespace ck_tile {

// Philox-4x32-7 generates four 32-bit values using 7 rounds.
// It has a period of 2^128 and supports fast jumping by 2^64 (subsequence).
// Reference: https://github.com/Dao-AILab/flash-attention/blob/main/csrc/flash_attn/src/philox.cuh
class philox
{
    public:
    CK_TILE_HOST_DEVICE philox(unsigned long long seed_, unsigned long long offset_)
        : seed(seed_), offset(offset_)
    {
    }

    CK_TILE_HOST_DEVICE uint4 get_philox_4x32(const unsigned long long subsequence) const
    {
        uint4 counter_{static_cast<unsigned int>(offset),
                       static_cast<unsigned int>(offset >> 32),
                       static_cast<unsigned int>(subsequence),
                       static_cast<unsigned int>(subsequence >> 32)};

        uint2 key_{static_cast<unsigned int>(seed), static_cast<unsigned int>(seed >> 32)};
#pragma unroll
        for(int i = 0; i < kPhiloxRounds - 1; i++)
        {
            counter_ = philox_single_round(counter_, key_);
            key_.x += kPhiloxWA;
            key_.y += kPhiloxWB;
        }
        return philox_single_round(counter_, key_);
    }

    CK_TILE_HOST_DEVICE void get_random_16x8(uint8_t* out,
                                             const unsigned long long subsequence) const
    {
        uint4 tmp_ph;
        tmp_ph = get_philox_4x32(subsequence);

        uint32_t* out_tmp = reinterpret_cast<uint32_t*>(&out[0]);

        out_tmp[0] = tmp_ph.x;
        out_tmp[1] = tmp_ph.y;
        out_tmp[2] = tmp_ph.z;
        out_tmp[3] = tmp_ph.w;
    }

    CK_TILE_HOST_DEVICE void get_random_8x8(uint8_t* out,
                                            const unsigned long long subsequence,
                                            const index_t idx0,
                                            const index_t idx1) const
    {
        uint4 tmp_ph;
        tmp_ph = get_philox_4x32(subsequence);

        uint32x4_t tmp;
        tmp[0]            = tmp_ph.x;
        tmp[1]            = tmp_ph.y;
        tmp[2]            = tmp_ph.z;
        tmp[3]            = tmp_ph.w;
        uint32_t* out_tmp = reinterpret_cast<uint32_t*>(&out[0]);
        out_tmp[0]        = tmp[idx0];
        out_tmp[1]        = tmp[idx1];
    }

    CK_TILE_HOST_DEVICE void
    get_random_4x8(uint8_t* out, const unsigned long long subsequence, const index_t idx) const
    {
        uint4 tmp_ph;
        tmp_ph = get_philox_4x32(subsequence);

        uint32x4_t tmp;
        tmp[0]            = tmp_ph.x;
        tmp[1]            = tmp_ph.y;
        tmp[2]            = tmp_ph.z;
        tmp[3]            = tmp_ph.w;
        uint32_t* out_tmp = reinterpret_cast<uint32_t*>(&out[0]);
        out_tmp[0]        = tmp[idx];
    }

    private:
    const unsigned long long seed;
    const unsigned long long offset;

    CK_TILE_HOST_DEVICE uint2 mulhilo32(const unsigned int a, const unsigned int b) const
    {
        unsigned long long res = static_cast<unsigned long long>(a) * b;
        return uint2{static_cast<unsigned int>(res), static_cast<unsigned int>(res >> 32)};
    }

    CK_TILE_HOST_DEVICE uint4 philox_single_round(const uint4 ctr, const uint2 key) const
    {
        uint2 res0 = mulhilo32(kPhiloxMA, ctr.x);
        uint2 res1 = mulhilo32(kPhiloxMB, ctr.z);
        return uint4{res1.y ^ ctr.y ^ key.x, res1.x, res0.y ^ ctr.w ^ key.y, res0.x};
    }

    static constexpr unsigned int kPhiloxWA = 0x9E3779B9;
    static constexpr unsigned int kPhiloxWB = 0xBB67AE85;
    static constexpr unsigned int kPhiloxMA = 0xD2511F53;
    static constexpr unsigned int kPhiloxMB = 0xCD9E8D57;

    static constexpr int kPhiloxRounds = 7;
};

} // namespace ck_tile
