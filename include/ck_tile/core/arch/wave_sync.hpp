// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core/config.hpp"
#include "ck_tile/core/arch/arch.hpp"

#include <stdint.h>

namespace ck_tile {

struct wave_barrier
{
    CK_TILE_DEVICE wave_barrier(uint32_t* ptr) : barrier_ptr_(ptr) {}

    CK_TILE_DEVICE uint32_t ld(uint32_t offset) const
    {
        return __atomic_load_n(barrier_ptr_ + offset, __ATOMIC_RELAXED);
    }

    CK_TILE_DEVICE void st(uint32_t offset, uint32_t value)
    {
        __atomic_store_n(barrier_ptr_ + offset, value, __ATOMIC_RELEASE);
    }

    CK_TILE_DEVICE void wait_eq(uint32_t offset, uint32_t value)
    {
        if(get_lane_id() == 0)
        {
            while(ld(offset) != value) {}
        }
    }

    CK_TILE_DEVICE void wait_lt(uint32_t offset, uint32_t value)
    {
        if(get_lane_id() == 0)
        {
            while(ld(offset) < value) {}
        }
    }

    CK_TILE_DEVICE void wait_set(uint32_t offset, uint32_t compare, uint32_t value)
    {
        if(get_lane_id() == 0)
        {
            while(atomicCAS(barrier_ptr_ + offset, compare, value) != compare) {}
        }
    }

    // enter critical zone, assume buffer is zero at the kernel launch
    CK_TILE_DEVICE void aquire(uint32_t offset) { wait_set(offset, 0, 1); }

    // exit critical zone, assume buffer is zero at the kernel launch
    CK_TILE_DEVICE void release(uint32_t offset) { wait_set(offset, 1, 0); }

    CK_TILE_DEVICE void inc(uint32_t offset)
    {
        if(get_lane_id() == 0)
        {
            atomicInc(barrier_ptr_ + offset, 1);
        }
    }

    CK_TILE_DEVICE void reset(uint32_t offset)
    {
        if(get_lane_id() == 0)
        {
            st(offset, 0);
        }
    }

    uint32_t* barrier_ptr_;
};

} // namespace ck_tile
