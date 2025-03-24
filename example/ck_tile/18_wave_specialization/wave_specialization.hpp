// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common.hpp"
#include "ck_tile/ops/reduce/block/block_reduce.hpp"
#include "ck_tile/ops/reduce/block/block_reduce2d_default_policy.hpp"

namespace ck_tile {

template <index_t... Ids>
CK_TILE_DEVICE static bool is_thread_id_idx()
{
    const auto tid = get_thread_id();
    return ((tid == Ids) || ...);
}

template <typename BlockWaves, // num warps along seq<M, N>
          typename BlockTile,  // block size, seq<M, N>
          typename WaveTile,   // warp size, seq<M, N>
          typename Vector>     // contiguous elements(vector size) along seq<M, N>
struct Reduce2dShape
{
    // We split Workgroup waves into two specialized groups.
    // One for reading data from global -> LDS, the other is doing reduction
    static constexpr index_t WaveGroups = 2;

    static constexpr index_t Block_M = BlockTile::at(number<0>{});
    static constexpr index_t Block_N = BlockTile::at(number<1>{});

    static constexpr index_t Warp_M = WaveTile::at(number<0>{});
    static constexpr index_t Warp_N = WaveTile::at(number<1>{});

    static constexpr index_t Vector_M = Vector::at(number<0>{});
    static constexpr index_t Vector_N = Vector::at(number<1>{});

    static constexpr index_t WarpPerBlock_M = integer_divide_ceil(BlockWaves::at(number<0>{}), WaveGroups);
    static constexpr index_t WarpPerBlock_N = integer_divide_ceil(BlockWaves::at(number<1>{}), WaveGroups);

    static constexpr index_t ThreadPerWarp_M = Warp_M / Vector_M;
    static constexpr index_t ThreadPerWarp_N = Warp_N / Vector_N;

    static constexpr index_t Repeat_M = Block_M / (WarpPerBlock_M * Warp_M);
    static constexpr index_t Repeat_N = Block_N / (WarpPerBlock_N * Warp_N);

    static constexpr index_t WaveNum = reduce_on_sequence(BlockWaves{}, multiplies{}, number<1>{});

    static constexpr index_t BlockSize = get_warp_size() * WaveNum;
    static constexpr index_t WaveGroupSize = WaveNum / WaveGroups;
    static_assert(WaveGroupSize == WarpPerBlock_M * WarpPerBlock_N, "Inconsisten wave group size!");
};

template <typename XDataType_,
          typename ComputeDataType_,
          typename YDataType_,
          typename BlockShape_,
          typename ReduceOp_>
struct Reduce2dProblem
{
    using XDataType       = remove_cvref_t<XDataType_>;
    using ComputeDataType = remove_cvref_t<ComputeDataType_>;
    using YDataType       = remove_cvref_t<YDataType_>;
    using BlockShape      = remove_cvref_t<BlockShape_>;
    using ReduceOp        = ReduceOp_;

    static constexpr bool kNeedCrossLaneSync = BlockShape::ThreadPerWarp_N > 1;
    static constexpr bool kNeedCrossWarpSync = BlockShape::WarpPerBlock_N > 1;
};

template <typename Problem_, typename Policy_ = BlockReduce2dDefaultPolicy>
struct Reduce
{
    using Problem = ck_tile::remove_cvref_t<Problem_>;
    using Policy  = ck_tile::remove_cvref_t<Policy_>;

    using XDataType       = ck_tile::remove_cvref_t<typename Problem::XDataType>;
    using ComputeDataType = ck_tile::remove_cvref_t<typename Problem::ComputeDataType>;
    using YDataType       = ck_tile::remove_cvref_t<typename Problem::YDataType>;

    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto MakeXWaveTileDistribution()
    {
        using S = typename Problem::BlockShape;
        return make_static_tile_distribution(
            tile_distribution_encoding<
                sequence<>,
                tuple<sequence<S::ThreadPerWarp_M, S::Vector_M>,
                      sequence<S::ThreadPerWarp_N, S::Vector_N>>,
                tuple<sequence<1, 2>>,
                tuple<sequence<0, 0>>,
                sequence<1, 2>,
                sequence<1, 1>>{});
    }

    using XWaveDstr = remove_cvref_t<decltype(MakeXWaveTileDistribution<Problem>())>;
    using XWaveTensor = static_distributed_tensor<XDataType, XWaveDstr>;

    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto MakeXBlockTileDistribution()
    {
        using S = typename Problem::BlockShape;
        constexpr auto x_block_outer_distr_enc = tile_distribution_encoding<
                sequence<>,
                tuple<sequence<S::Repeat_M, S::WarpPerBlock_M>,
                      sequence<S::Repeat_N, S::WarpPerBlock_N>>,
                tuple<sequence<1, 2>>,
                tuple<sequence<1, 1>>,
                sequence<1, 2>,
                sequence<0, 0>>{};
        
        constexpr auto x_block_dstr_enc = detail::make_embed_tile_distribution_encoding(
            x_block_outer_distr_enc, XWaveDstr::get_static_tile_distribution_encoding());
        constexpr auto x_block_dstr = make_static_tile_distribution(x_block_dstr_enc);

        return x_block_dstr;
    }

    static constexpr auto x_wave_y_lengths =
        to_sequence(XWaveDstr{}.get_ys_to_d_descriptor().get_lengths());
    static constexpr auto x_wave_y_index_zeros = uniform_sequence_gen_t<XWaveDstr::NDimY, 0>{};

    using XTileDstr = remove_cvref_t<decltype(MakeXBlockTileDistribution<Problem>())>;
    // static constexpr auto x_wave_y_lengths =
    //     to_sequence(XTileDstr{}.get_ys_to_d_descriptor().get_lengths());
    // static constexpr auto x_tile_hs_lengths = XTileDstr::get_lengths();
    // static constexpr auto x_wave_hs_lengths = XWaveDstr::get_lengths();

    // 2
    // static_assert(XTileDstr::get_num_of_dimension_x() == 3, "XTileDstr::Xs ndim");

    // <1, 1, 8, 8>
    // static_assert(x_tile_y_lengths == make_tuple(number<0>{}), "x_tile_y_lengths");
    // <8, 8>
    // static_assert(x_wave_y_lengths == make_tuple(number<0>{}), "x_wave_y_lengths");
    // <32, 128>
    // static_assert(x_tile_hs_lengths == make_tuple(number<0>{}), "x_tile_hs_lengths");
    // <32, 128>
    // static_assert(x_wave_hs_lengths == make_tuple(number<0>{}), "x_wave_hs_lengths");


    CK_TILE_DEVICE void operator()(const XDataType* p_x, YDataType* p_y, index_t M, index_t N) const
    {
        using S = typename Problem::BlockShape;

        __shared__ XDataType x_lds[number<S::Block_M>{} * number<S::Block_N>{}];

        const auto x_m_n = make_naive_tensor_view<address_space_enum::global>(
            p_x, make_tuple(M, N), make_tuple(N, 1), number<S::Vector_N>{}, number<1>{});

        const auto x_lds_view = make_naive_tensor_view<address_space_enum::lds>(
                x_lds, 
                make_tuple(number<S::Block_M>{}, number<S::Block_N>{}),
                make_tuple(number<S::Block_M>{}, 1),
                number<S::Vector_N>{},
                number<1>{});
    
        const auto y_m = make_naive_tensor_view_packed<address_space_enum::global>(
            p_y, make_tuple(M), number<1>{});

        const auto iM = get_block_id() * S::Block_M;

        auto x_block_window = make_tile_window(x_m_n,
                                               make_tuple(number<S::Block_M>{}, number<S::Block_N>{}),
                                               {iM, 0},
                                               MakeXBlockTileDistribution<Problem>());

        auto x_block_lds_window = make_tile_window(x_lds_view,
                                                   make_tuple(number<S::Block_M>{}, number<S::Block_N>{}),
                                                   {0, 0},
                                                   MakeXBlockTileDistribution<Problem>());

        auto y_window = make_tile_window(y_m, make_tuple(number<S::Block_M>{}), {iM});

        // __shared__ char smem[Policy::template GetSmemSize<Problem>()];
        // We need sync flags
        __shared__ uint32_t smem_flags[S::WaveNum];
        wave_barrier wave_group_sync(smem_flags);
        
        const auto wave_id = get_warp_id();
        auto wave_peer_id = wave_id + S::WaveGroupSize;
        if (wave_id < S::WaveGroupSize)
        {
            wave_group_sync.reset(wave_id);
        }
        else
        {
            wave_peer_id = wave_id - S::WaveGroupSize;
            wave_group_sync.st(wave_id, 1);
        }

        if (get_lane_id() == 0)
        {
            const auto flag = wave_group_sync.ld(wave_id);
            printf("[tid: %d], flag: %d\n",
                    static_cast<index_t>(threadIdx.x),
                    flag
                  );
        }

        index_t num_n_tile_iteration =
            __builtin_amdgcn_readfirstlane(integer_divide_ceil(N, S::Block_N));

        auto reduce_func         = typename Problem::ReduceOp{};
        auto block_reduce2d      = Policy::template GetBlockReduce2d<Problem>();
        auto block_reduce2d_sync = Policy::template GetBlockReduce2dSync<Problem>();

        using XTensorType = decltype(load_tile(x_block_window));
        auto y_wgp_compute    = block_reduce2d.template MakeYBlockTile<XTensorType>();
        set_tile(y_wgp_compute, reduce_func.template GetIdentityValue<ComputeDataType>());

        // thr vec 8x8
        // M: 32, N: 128
        // 

        // X WGP Tile
        // Ps ({1, 1}, {4, 16})
        // Ys ({1, 8, 1, 8})
        // Xs( Rs ({}), Hs({1, 1, 4, 8}, {1, 1, 16, 8}) )

        // Y Wgp Tile - reduced on second Xs axis
        // Ps ({1, 1}, {4, 16})
        // Ys ({1, 8})
        // Rs ({1, 16})
        // Hs ({1, 1, 4, 8})

        // make_reduce_tile_distribution_encoding:
        // <1, 16>
        // static_assert(rs_lengths == sequence<0, 0>{}, "rs_lengths");
        // const tuple<ck_tile::sequence<1, 1, 4, 8>>
        // static_assert(hs_lengthss == make_tuple(number<0>{}), "hs_lengthss");
        // const tuple<ck_tile::sequence<1, 0>, ck_tile::sequence<1, 0>>
        // static_assert(ps_to_rhss_major == make_tuple(number<0>{}), "ps_to_rhss_major");
        // const tuple<ck_tile::sequence<0, 0>, ck_tile::sequence<2, 1>>
        // static_assert(ps_to_rhss_minor == make_tuple(number<0>{}), "ps_to_rhss_minor");
        // 2
        // static_assert(ndim_y == false, "ndim_y");
        // const ck_tile::sequence<1, 1>{}
        // static_assert(decltype(ys_to_rhs_major){} == sequence<3, 3>{}, "ys_to_rhs_major");
        // const ck_tile::sequence<1, 3>{}
        // static_assert(decltype(ys_to_rhs_minor){} == sequence<3, 3>{}, "ys_to_rhs_minor");

        // auto y_wave_compute    = block_reduce2d.template MakeYBlockTile<XWaveTensor>();
        // set_tile(y_wave_compute, reduce_func.template GetIdentityValue<ComputeDataType>());

        // using YTileDstr = remove_cvref_t<decltype(y_wave_compute.get_tile_distribution())>;
        // static constexpr auto y_wave_compute_tile_y_lengths =
        //     to_sequence(YTileDstr{}.get_ys_to_d_descriptor().get_lengths());
        // static constexpr auto y_wave_compute_hs_lengths = YTileDstr::get_lengths();

        // <1, 8> - y_comp on XTensorType
        // <8> - y_wave_comp on XWaveTensor
        // static_assert(y_wave_compute_tile_y_lengths == make_tuple(number<0>{}), "y_wave_compute_tile_y_lengths");
        // <32>
        // <32>
        // static_assert(y_wave_compute_hs_lengths == make_tuple(number<0>{}), "y_wave_compute_hs_lengths");

        for(int iN = __builtin_amdgcn_readfirstlane(0); iN < num_n_tile_iteration; ++iN)
        {
            // Data Load Waves (Producer)
            if (wave_id < S::WaveGroupSize)
            {
                if (get_lane_id() == 0)
                {
                    const auto flag = wave_group_sync.ld(wave_id);
                    const auto peer_flag = wave_group_sync.ld(wave_peer_id);
                    printf("[tid: %d, wave id: %d], <data_wave> wait for data flag: %d, peer_flag: %d\n",
                        static_cast<index_t>(threadIdx.x),
                        wave_id,
                        flag,
                        peer_flag
                    );
                }
                const auto x = load_tile(x_block_window);
                move_tile_window(x_block_window, {0, S::Block_N});
                if (get_lane_id() == 0)
                {
                    const auto flag = wave_group_sync.ld(wave_id);
                    const auto peer_flag = wave_group_sync.ld(wave_peer_id);
                    printf("[tid: %d, wave id: %d], <data_wave> wait for comp: %d, peer_flag: %d\n",
                        static_cast<index_t>(threadIdx.x),
                        wave_id,
                        flag,
                        peer_flag
                    );
                    printf("[tid: %d, wave id: %d], <data_wave> data loaded, [%f, %f, %f, %f]\n",
                        static_cast<index_t>(threadIdx.x),
                        wave_id,
                        static_cast<float>(x.get_thread_buffer().at(number<0>{})),
                        static_cast<float>(x.get_thread_buffer().at(number<1>{})),
                        static_cast<float>(x.get_thread_buffer().at(number<2>{})),
                        static_cast<float>(x.get_thread_buffer().at(number<3>{}))
                    );
                }
                wave_group_sync.wait_eq(wave_peer_id, 1);
                store_tile(x_block_lds_window, x);
                // sync just LDS
                __builtin_amdgcn_s_waitcnt(0xc07f);
                wave_group_sync.reset(wave_peer_id);
                wave_group_sync.st(wave_id, 1);
                if (get_lane_id() == 0)
                {
                    const auto flag = wave_group_sync.ld(wave_id);
                    const auto peer_flag = wave_group_sync.ld(wave_peer_id);
                    printf("[tid: %d, wave id: %d], <data_wave> stored to lds flag: %d, peer_flag: %d\n",
                        static_cast<index_t>(threadIdx.x),
                        wave_id,
                        flag,
                        peer_flag
                    );
                }
            }
            // Compute Waves (Consumer)
            else
            {
                if (get_lane_id() == 0)
                {
                    const auto flag = wave_group_sync.ld(wave_id);
                    const auto peer_flag = wave_group_sync.ld(wave_peer_id);
                    printf("[tid: %d, wave id: %d], <comp_wave> wait for data flag: %d, peer_flag: %d\n",
                        static_cast<index_t>(threadIdx.x),
                        wave_id,
                        flag,
                        peer_flag
                    );
                }
                wave_group_sync.wait_eq(wave_peer_id, 1);
                const auto x = load_tile(x_block_lds_window);
                // sync just LDS
                __builtin_amdgcn_s_waitcnt(0xc07f);
                wave_group_sync.reset(wave_peer_id);
                wave_group_sync.st(wave_id, 1);
                if (get_lane_id() == 0)
                {
                    const auto flag = wave_group_sync.ld(wave_id);
                    const auto peer_flag = wave_group_sync.ld(wave_peer_id);
                    printf("[tid: %d, wave id: %d], <comp_wave> start reduce flag: %d, peer_flag: %d\n",
                        static_cast<index_t>(threadIdx.x),
                        wave_id,
                        flag,
                        peer_flag
                    );
                    printf("[tid: %d, wave id: %d], <comp_wave> data loaded, [%f, %f, %f, %f]\n",
                        static_cast<index_t>(threadIdx.x),
                        wave_id,
                        static_cast<float>(x.get_thread_buffer().at(number<0>{})),
                        static_cast<float>(x.get_thread_buffer().at(number<1>{})),
                        static_cast<float>(x.get_thread_buffer().at(number<2>{})),
                        static_cast<float>(x.get_thread_buffer().at(number<3>{}))
                    );
                }
                
                static_for<0, S::Repeat_M, 1>{}([&](auto m_iter) {
                    static_for<0, S::Repeat_N, 1>{}([&](auto n_iter) {

                        XWaveTensor x_wave_tensor;
                        x_wave_tensor.get_thread_buffer() = x.get_y_sliced_thread_data(
                            merge_sequences(sequence<m_iter, n_iter>{}, x_wave_y_index_zeros),
                            merge_sequences(sequence<1, 1>{}, x_wave_y_lengths));

                        // y_wave_tensor (reduced)
                        // ys_to_d: <8>
                        // hs: <32>
                        auto y_wave_tensor = block_reduce2d.template MakeYBlockTile<XWaveTensor>();
                        
                        using YWaveDstr = remove_cvref_t<decltype(y_wave_tensor.get_tile_distribution())>;
                        static constexpr auto y_wave_y_lengths = to_sequence(YWaveDstr{}.get_ys_to_d_descriptor().get_lengths());
                        static constexpr auto y_wave_y_index_zeros = uniform_sequence_gen_t<YWaveDstr::NDimY, 0>{};
                        
                        y_wave_tensor.get_thread_buffer() = y_wgp_compute.get_y_sliced_thread_data(
                            merge_sequences(sequence<m_iter>{}, y_wave_y_index_zeros),
                            merge_sequences(sequence<1>{}, y_wave_y_lengths));

                        block_reduce2d(x_wave_tensor, y_wave_tensor, reduce_func);

                        if (get_lane_id() == 0)
                        {
                            printf("[tid: %d, wave id: %d], <comp_wave> data reduce loop (partial), [%f, %f, %f, %f]\n",
                                static_cast<index_t>(threadIdx.x),
                                wave_id,
                                static_cast<float>(y_wave_tensor.get_thread_buffer().at(number<0>{})),
                                static_cast<float>(y_wave_tensor.get_thread_buffer().at(number<1>{})),
                                static_cast<float>(y_wave_tensor.get_thread_buffer().at(number<2>{})),
                                static_cast<float>(y_wave_tensor.get_thread_buffer().at(number<3>{}))
                            );
                        }
                    });
                });

                if (get_lane_id() == 0)
                {
                    printf("[tid: %d, wave id: %d], <comp_wave> data reduce (partial), [%f, %f, %f, %f]\n",
                        static_cast<index_t>(threadIdx.x),
                        wave_id,
                        static_cast<float>(y_wgp_compute.get_thread_buffer().at(number<0>{})),
                        static_cast<float>(y_wgp_compute.get_thread_buffer().at(number<1>{})),
                        static_cast<float>(y_wgp_compute.get_thread_buffer().at(number<2>{})),
                        static_cast<float>(y_wgp_compute.get_thread_buffer().at(number<3>{}))
                    );
                }
            }
        }
        
        __syncthreads();
        
        if (wave_id >= S::WaveGroupSize)
        {
            // TODO: doesn't this assume whole WGP to do reduce? 
            block_reduce2d_sync(y_wgp_compute, reduce_func);
         
            if (get_lane_id() == 0)
            {
                const auto flag = wave_group_sync.ld(wave_id);
                const auto peer_flag = wave_group_sync.ld(wave_peer_id);
                printf("[tid: %d, wave id: %d], <comp_wave> data reduced flag: %d, peer_flag: %d\n",
                    static_cast<index_t>(threadIdx.x),
                    wave_id,
                    flag,
                    peer_flag
                );
                printf("[tid: %d, wave id: %d], <comp_wave> data reduced, [%f, %f, %f, %f]\n",
                    static_cast<index_t>(threadIdx.x),
                    wave_id,
                    static_cast<float>(y_wgp_compute.get_thread_buffer().at(number<0>{})),
                    static_cast<float>(y_wgp_compute.get_thread_buffer().at(number<1>{})),
                    static_cast<float>(y_wgp_compute.get_thread_buffer().at(number<2>{})),
                    static_cast<float>(y_wgp_compute.get_thread_buffer().at(number<3>{}))
                );
            }

            store_tile(y_window, cast_tile<YDataType>(y_wgp_compute));
        }
    }
};

} // namespace ck_tile
