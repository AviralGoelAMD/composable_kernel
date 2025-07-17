// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common.hpp"
#include "ck_tile/ops/common/tensor_layout.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/host/kernel_launch.hpp"

namespace ck_tile {

template <typename BlockWaves, // num warps along seq<M, N>
          typename BlockTile,  // block size, seq<M, N>
          typename WaveTile,   // warp size, seq<M, N>
          typename Vector>     // contiguous elements(vector size) along seq<M, N>
struct TileCopyShape
{
    // We split Workgroup waves into two specialized groups.
    // One for reading data from global -> LDS, the other is doing reduction
    static constexpr index_t WaveGroups = 1;
    static constexpr index_t MWarps     = BlockWaves::at(number<0>{});
    static constexpr index_t NWarps     = BlockWaves::at(number<0>{});

    static constexpr index_t Block_M = BlockTile::at(number<0>{});
    static constexpr index_t Block_N = BlockTile::at(number<1>{});

    static constexpr index_t Warp_M = WaveTile::at(number<0>{});
    static constexpr index_t Warp_N = WaveTile::at(number<1>{});

    static constexpr index_t Vector_M = Vector::at(number<0>{});
    static constexpr index_t Vector_N = Vector::at(number<1>{});

    static constexpr index_t ThreadPerWarp_M = Warp_M / Vector_M;
    static constexpr index_t ThreadPerWarp_N = Warp_N / Vector_N;

    static constexpr index_t WarpPerBlock_M =
        integer_divide_ceil(BlockWaves::at(number<0>{}), WaveGroups);
    static constexpr index_t WarpPerBlock_N =
        integer_divide_ceil(BlockWaves::at(number<1>{}), WaveGroups);

    static constexpr index_t Repeat_M = Block_M / (WarpPerBlock_M * Warp_M);
    static constexpr index_t Repeat_N = Block_N / (WarpPerBlock_N * Warp_N);

    static constexpr index_t WaveNum = reduce_on_sequence(BlockWaves{}, multiplies{}, number<1>{});

    static constexpr index_t BlockSize     = get_warp_size() * WaveNum;
};

template <typename XDataType_, typename BlockShape_>
struct TileCopyProblem
{
    using XDataType  = remove_cvref_t<XDataType_>;
    using BlockShape = remove_cvref_t<BlockShape_>;
};

template <typename Problem_>
struct TileCopy
{
    using Problem   = ck_tile::remove_cvref_t<Problem_>;
    using XDataType = typename Problem::XDataType;

    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto MakeDRAMDistribution()
    {
        using S = typename Problem::BlockShape;

        constexpr index_t warp_size = get_warp_size();
        constexpr index_t X0 = S::ThreadPerWarp_N; // threads needed along N dimension, fastest
                                                   // changing with given vector size.
        constexpr index_t X1 =
            S::Vector_N; // no. of elements along N dimensions to be read by each thread.

        constexpr index_t Y0 =
            S::WaveNum / S::WaveGroups;        // no. of active warps working in this thread block.
        constexpr index_t Y1 = warp_size / X0; // no. of threads in a warp needed along M dimension.
        constexpr index_t Y2 =
            S::Warp_M /
            (Y1 *
             Y0); // no. of iterations each warp needs to perform to cover the entire tile window.

        constexpr auto outer_encoding = 
            tile_distribution_encoding<sequence<>,
                                        tuple<sequence<Y0, Y1, Y2>, sequence<X0, X1>>, 
                                        tuple<sequence<1>, sequence<1, 2>>, 
                                        tuple<sequence<0>, sequence<1, 0>>, 
                                        sequence<1, 2>, 
                                        sequence<2, 1>>{};

        return make_static_tile_distribution(outer_encoding);
    }

    CK_TILE_DEVICE void
    operator()(XDataType* p_y, index_t M, index_t N) const
    {
        using S = typename Problem::BlockShape;

        // Input tensor
        const auto iM    = get_block_id() * S::Block_M;

        // Output tensor
        const auto y_m = make_naive_tensor_view<address_space_enum::global, memory_operation_enum::atomic_add>(
            p_y, make_tuple(M, N), make_tuple(N, 1), number<S::Vector_N>{}, number<1>{});

        auto y_block_window =
            make_tile_window(y_m, make_tuple(number<S::Block_M>{}, number<S::Block_N>{}), {iM, 0});

        // Programming logic
        index_t num_n_tile_iteration =
            __builtin_amdgcn_readfirstlane(integer_divide_ceil(N, S::Block_N));

        //auto DramTileDist   = x_block_window.get_tile_distribution();
        using dram_reg_tile = decltype(make_static_distributed_tensor<XDataType>(MakeDRAMDistribution<Problem>()));

        for(int iN = __builtin_amdgcn_readfirstlane(0); iN < num_n_tile_iteration; ++iN)
        {
            dram_reg_tile dram_tile;

            // Add 1 to the input matrix. 
            tile_elementwise_inout([](auto& c) { c = 1; }, dram_tile);

            // store from registers to DRAM
            update_tile(y_block_window, dram_tile);

            __syncthreads();
            move_tile_window(y_block_window, {0, S::Block_N});
        }
    }
};

} // namespace ck_tile
