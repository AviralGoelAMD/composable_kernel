// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

#include <hip/hip_runtime.h>

#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <tuple>

#include "ck_tile/host.hpp"
#include "gemm_utils.hpp"
#include "run_gemm_example.inc"


struct RunLoadGlobalStoreLDSLoadLDS
{
    static constexpr auto NPerBlock = 256;
    static constexpr auto NPerXdl   = 32;
    static constexpr auto KPerBlock = 64;
    static constexpr auto BlockSize = 256;
    static constexpr auto VecLoadSize = 8;
    static constexpr auto AccessPattern = ck_tile::tile_distribution_pattern::thread_raked;

    using TileEncodingPattern  = ck_tile::TileDistributionEncodingPattern2D<BlockSize,
                                                                        KPerBlock,
                                                                        NPerBlock,
                                                                        VecLoadSize,
                                                                        AccessPattern>;

    CK_TILE_DEVICE void operator()(const ck_tile::half_t* x, ck_tile::half_t* out)
    {
        using namespace ck_tile;
        // ADAPTED OLD CK LDS TILE DESCRIPTOR
        // BK1
        constexpr auto BK1 = number<TileEncodingPattern::Y2>{};
        constexpr auto BK0 = number<KPerBlock / BK1>{};

        // How threads access data on N dim
        constexpr auto N0 = TileEncodingPattern::X0;
        constexpr auto N1 = TileEncodingPattern::X1;

        // How many elements we can write by single thread to LDS
        constexpr auto KThreadWrite     = TileEncodingPattern::X1;
        constexpr auto K0PerThreadWrite = BK0 / KThreadWrite;
        
        constexpr auto KThreadRead     = get_warp_size() / NPerXdl;
        // constexpr auto K0PerThreadRead = BK0 / KThreadRead;

        // check if we exceed all 32banks width - (32x4B)
        constexpr auto LdsBanksWidth = 128;
        constexpr auto kfold = (BK1 * N0 * sizeof(half_t) > LdsBanksWidth) 
                                ? 1
                                : LdsBanksWidth / (BK1 * N0 * sizeof(half_t));
        constexpr auto KThreadReadPerm = KThreadRead;
        // 1<=npair<=n0
        constexpr auto npair = (BK1 * NPerXdl * sizeof(half_t) > LdsBanksWidth)
                ? 1
                : ((LdsBanksWidth / (BK1 * NPerXdl * sizeof(half_t))) > N0
                        ? N0
                        : LdsBanksWidth / (BK1 * NPerXdl * sizeof(half_t)));

        constexpr auto b_lds_block_desc = make_naive_tensor_descriptor_packed(
            make_tuple(number<KThreadWrite / kfold / KThreadReadPerm>{},
                    number<K0PerThreadWrite>{},
                    number<KThreadReadPerm * N1>{},
                    number<kfold * N0 / npair>{},
                    number<npair>{},
                    BK1));

        constexpr auto b_lds_block_desc_permuted = transform_tensor_descriptor(
            b_lds_block_desc,
            make_tuple(
                make_pass_through_transform(number<KThreadWrite / kfold / KThreadReadPerm>{}),
                make_pass_through_transform(number<K0PerThreadWrite>{}),
                make_xor_transform(
                    make_tuple(number<KThreadReadPerm * N1>{}, number<kfold * N0 / npair>{})),
                make_pass_through_transform(number<npair>{}),
                make_pass_through_transform(BK1)),
            make_tuple(
                sequence<0>{}, sequence<1>{}, sequence<2, 3>{}, sequence<4>{}, sequence<5>{}),
            make_tuple(
                sequence<0>{}, sequence<1>{}, sequence<2, 3>{}, sequence<4>{}, sequence<5>{}));

        constexpr auto b_lds_block_desc_unmerged = transform_tensor_descriptor(
            b_lds_block_desc_permuted,
            make_tuple(
                make_pass_through_transform(number<KThreadWrite / kfold / KThreadReadPerm>{}),
                make_pass_through_transform(number<K0PerThreadWrite>{}),
                make_unmerge_transform(make_tuple(number<KThreadReadPerm>{}, number<N1>{})),
                make_unmerge_transform(make_tuple(number<kfold>{}, number<N0 / npair>{})),
                make_pass_through_transform(number<npair>{}),
                make_pass_through_transform(BK1)),
            make_tuple(sequence<0>{},
                    sequence<1>{},
                    sequence<2>{},
                    sequence<3>{},
                    sequence<4>{},
                    sequence<5>{}),
            make_tuple(sequence<1>{},
                    sequence<2>{},
                    sequence<0, 3>{},
                    sequence<4, 5>{},
                    sequence<6>{},
                    sequence<7>{}));

        constexpr auto b_lds_block_desc_nk = transform_tensor_descriptor(
            b_lds_block_desc_unmerged,
            make_tuple(make_merge_transform_v3_division_mod(
                        make_tuple(number<KThreadReadPerm>{},
                                    number<KThreadWrite / kfold / KThreadReadPerm>{},
                                    number<kfold>{},
                                    number<K0PerThreadWrite>{},
                                    BK1)),
                    make_merge_transform_v3_division_mod(
                        make_tuple(number<N0 / npair>{}, number<npair>{}, number<N1>{}))),
            make_tuple(sequence<0, 1, 4, 2, 7>{}, sequence<5, 6, 3>{}),
            make_tuple(sequence<1>{}, sequence<0>{}));
        static_assert(b_lds_block_desc_nk.is_known_at_compile_time(), "not constexpr!");

        // Load from global
        const auto global_tensor_view = make_naive_tensor_view<address_space_enum::global>(
            x,
            make_tuple(KPerBlock, NPerBlock),  // whole tensor shape
            make_tuple(NPerBlock, 1),          // stride
            number<8>{},                       // last dim size at least this
            number<1>{}                        // last dim stride
        );
        const auto global_tile_window = make_tile_window(
            global_tensor_view,
            make_tuple(KPerBlock, NPerBlock),  // window shape, we only have one block for simplicity
            {0, 0},                            // origin
            TileEncodingPattern::Make2DStaticTileDistribution() // tile distribution
        );
        // This does the same as GlobalPrefetch in gemm pipeline
        // using BlockTileDstr = decltype(global_tile_window.get_tile_distribution());
        // using BlockTile = decltype(make_static_distributed_tensor<half_t>(BlockTileDstr{}));
        // BlockTile local_tile;
        auto local_tile = make_static_distributed_tensor<half_t>(TileEncodingPattern::Make2DStaticTileDistribution());
        load_tile(local_tile, global_tile_window);
        
        // Setup LDS tile
        __shared__ half_t smem[b_lds_block_desc_nk.get_element_space_size()];
        half_t* __restrict__ smem_ptr = static_cast<half_t*>(smem);
        auto lds_tensor_view = make_tensor_view<address_space_enum::lds>(smem_ptr, b_lds_block_desc_nk);
        auto lds_window = make_tile_window(lds_tensor_view, make_tuple(NPerBlock, KPerBlock), {0, 0});
        // This does the same as transpose+LocalPrefill in gemm pipeline
        {
            auto shuffled_tile = make_static_distributed_tensor<half_t>(TileEncodingPattern::MakeShuffled2DStaticTileDistribution());
            transpose_tile2d(shuffled_tile, local_tile);
            store_tile(lds_window, shuffled_tile);
        }

        // Do something based on lds data
        out[get_thread_id()] = smem_ptr[get_thread_id()];
    }
};

template <typename ADataType,
          typename BDataType,
          typename AccDataType,
          typename CDataType,
          typename ALayout,
          typename BLayout,
          typename CLayout,
          bool Persistent>
float gemm_calc(const ck_tile::GemmHostArgs& args, const ck_tile::stream_config& s)
{
    using GemmShape = ck_tile::TileGemmShape<
        ck_tile::sequence<GemmConfig::M_Tile, GemmConfig::N_Tile, GemmConfig::K_Tile>,
        ck_tile::sequence<GemmConfig::M_Warp, GemmConfig::N_Warp, GemmConfig::K_Warp>,
        ck_tile::
            sequence<GemmConfig::M_Warp_Tile, GemmConfig::N_Warp_Tile, GemmConfig::K_Warp_Tile>,
        GemmConfig::PermuteA,
        GemmConfig::PermuteB>;
    using TilePartitioner =
        ck_tile::GemmSpatiallyLocalTilePartitioner<GemmShape,
                                                   GemmConfig::TileParitionerGroupNum,
                                                   GemmConfig::TileParitionerM01>;

    using Traits              = ck_tile::TileGemmTraits<GemmConfig::kPadM,
                                           GemmConfig::kPadN,
                                           GemmConfig::kPadK,
                                           ALayout,
                                           BLayout,
                                           CLayout>;
    using GemmUniversalTraits = ck_tile::TileGemmUniversalTraits<GemmConfig::kPadM,
                                                                 GemmConfig::kPadN,
                                                                 GemmConfig::kPadK,
                                                                 GemmConfig::DoubleSmemBuffer,
                                                                 ALayout,
                                                                 BLayout,
                                                                 CLayout,
                                                                 GemmConfig::TransposeC,
                                                                 GemmConfig::UseStructuredSparsity,
                                                                 Persistent>;
    using GemmPipelineProblem =
        ck_tile::GemmPipelineProblem<ADataType, BDataType, AccDataType, GemmShape, Traits>;

    using BaseGemmPipeline = UNIVERSAL_GEMM_PIPELINE<GemmPipelineProblem>;

    const ck_tile::index_t k_grain     = args.k_batch * GemmConfig::K_Tile;
    const ck_tile::index_t K_split     = (args.K + k_grain - 1) / k_grain * GemmConfig::K_Tile;
    const ck_tile::index_t num_loop    = TilePartitioner::GetLoopNum(K_split);
    const bool has_hot_loop            = BaseGemmPipeline::BlockHasHotloop(num_loop);
    const ck_tile::TailNumber tail_num = BaseGemmPipeline::GetBlockLoopTailNum(num_loop);

    float ave_time{0};

    const auto Run =
        [&](const auto has_hot_loop_, const auto tail_number_, const auto memory_operation_) {
            constexpr bool has_hot_loop_v   = has_hot_loop_.value;
            constexpr auto tail_number_v    = tail_number_.value;
            constexpr auto scheduler        = GEMM_PIPELINE_SCHEDULER;
            constexpr auto memory_operation = memory_operation_.value;

            using UniversalGemmProblem = ck_tile::UniversalGemmPipelineProblem<ADataType,
                                                                               BDataType,
                                                                               AccDataType,
                                                                               GemmShape,
                                                                               GemmUniversalTraits,
                                                                               scheduler,
                                                                               has_hot_loop_v,
                                                                               tail_number_v>;

            using GemmPipeline = GEMM_PIPELINE<UniversalGemmProblem>;
            using GemmEpilogue = ck_tile::CShuffleEpilogue<
                ck_tile::CShuffleEpilogueProblem<ADataType,
                                                 BDataType,
                                                 AccDataType,
                                                 CDataType,
                                                 CLayout,
                                                 GemmPipelineProblem::kBlockSize,
                                                 TilePartitioner::MPerBlock,
                                                 TilePartitioner::NPerBlock,
                                                 GemmConfig::M_Warp,
                                                 GemmConfig::N_Warp,
                                                 GemmConfig::M_Warp_Tile,
                                                 GemmConfig::N_Warp_Tile,
                                                 GemmConfig::K_Warp_Tile,
                                                 UniversalGemmProblem::TransposeC,
                                                 memory_operation>>;
            using Kernel = ck_tile::GemmKernel<TilePartitioner, GemmPipeline, GemmEpilogue>;
            auto kargs   = Kernel::MakeKernelArgs(args);
            using namespace ck_tile;
            using Policy = ck_tile::UniversalGemmPipelineAgBgCrPolicy;
            constexpr auto BlockSize   = GemmPipelineProblem::kBlockSize;
            constexpr auto VecLoadSize = Policy::template GetVectorSizeB<UniversalGemmProblem>();
            constexpr index_t NPerBlock = GemmPipelineProblem::BlockGemmShape::kN;
            constexpr index_t KPerBlock = GemmPipelineProblem::BlockGemmShape::kK;
            using TileEncodingPattern  = ck_tile::TileDistributionEncodingPattern2D<BlockSize,
                                                                        KPerBlock,
                                                                        NPerBlock,
                                                                        VecLoadSize,
                                                                        Policy::BTileAccessPattern>;
            {
                // DEFAULT LDS DESCRIPTOR
                TileEncodingPattern::print();
                constexpr index_t KPack     = Policy::template GetSmemPackB<GemmPipelineProblem>();
                constexpr auto BK0          = number<KPerBlock / KPack>{};
                constexpr auto DataTypeSize = 2;
                constexpr auto NLdsLayer =
                    (32 * 4 / KPerBlock / DataTypeSize) < 1 ? 1 : (32 * 4 / KPerBlock / DataTypeSize);
                
                std::cout << "KPack: " << KPack << ", BK0: " << BK0 << ", NLdsLayer: " << NLdsLayer
                        << std::endl;
                constexpr auto b_lds_block_desc_0 = make_naive_tensor_descriptor(
                    make_tuple(
                        BK0 * number<NLdsLayer>{}, number<NPerBlock / NLdsLayer>{}, number<KPack>{}),
                    make_tuple(number<KPack>{}, number<KPerBlock * NLdsLayer>{}, number<1>{}),
                    number<KPack>{},
                    number<1>{});
                b_lds_block_desc_0.print();

                constexpr auto b_lds_block_desc_permuted = transform_tensor_descriptor(
                    b_lds_block_desc_0,
                    make_tuple(make_xor_transform(make_tuple(number<NPerBlock / NLdsLayer>{},
                                                            BK0 * number<NLdsLayer>{})),
                            make_pass_through_transform(number<KPack>{})),
                    make_tuple(sequence<1, 0>{}, sequence<2>{}),
                    make_tuple(sequence<1, 0>{}, sequence<2>{}));
                b_lds_block_desc_permuted.print();

                constexpr auto b_lds_block_desc_bk0_nldslayer_n_bk1 = transform_tensor_descriptor(
                    b_lds_block_desc_permuted,
                    make_tuple(make_unmerge_transform(make_tuple(number<NLdsLayer>{}, BK0)),
                            make_pass_through_transform(number<NPerBlock / NLdsLayer>{}),
                            make_pass_through_transform(number<KPack>{})),
                    make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}),
                    make_tuple(sequence<0, 2>{}, sequence<1>{}, sequence<3>{}));
                b_lds_block_desc_bk0_nldslayer_n_bk1.print();

                constexpr auto b_lds_block_desc = transform_tensor_descriptor(
                    b_lds_block_desc_bk0_nldslayer_n_bk1,
                    make_tuple(make_merge_transform_v3_division_mod(
                                make_tuple(number<NPerBlock / NLdsLayer>{}, number<NLdsLayer>{})),
                            make_merge_transform_v3_division_mod(make_tuple(BK0, number<KPack>{}))),
                    make_tuple(sequence<1, 0>{}, sequence<2, 3>{}),
                    make_tuple(sequence<0>{}, sequence<1>{}));
                b_lds_block_desc.print();
            }

            {
                // ADAPTED OLD CK LDS TILE DESCRIPTOR
                std::cout << std::endl << "OLD CK LDS TILE:" << std::endl;
                TileEncodingPattern::print();
                // BK1
                constexpr auto BK1 = number<TileEncodingPattern::Y2>{};
                constexpr auto BK0 = number<KPerBlock / BK1>{};
                printf("BK1: %d, BK0: %d\n", BK1.value, BK0.value);

                // How threads access data on N dim
                constexpr auto N0 = TileEncodingPattern::X0;
                constexpr auto N1 = TileEncodingPattern::X1;
                printf("N0: %d, N1: %d", N0, N1);

                using WarpTile         = typename GemmPipelineProblem::BlockGemmShape::WarpTile;
                constexpr auto NPerXdl = number<WarpTile::at(number<1>{})>{};
                printf(", NPerXdl: %d\n", NPerXdl.value);

                // How many elements we can write by single thread to LDS
                constexpr auto KThreadWrite     = TileEncodingPattern::X1;
                constexpr auto K0PerThreadWrite = BK0 / KThreadWrite;
                
                constexpr auto KThreadRead     = get_warp_size() / NPerXdl;
                constexpr auto K0PerThreadRead = BK0 / KThreadRead;
                printf("KThreadWrite: %d, K0PerThreadWrite: %d, KThreadRead: %d, K0PerThreadRead: %d\n",
                       KThreadWrite,
                       K0PerThreadWrite,
                       KThreadRead,
                       K0PerThreadRead);

                // check if we exceed all 32banks width - (32x4B)
                constexpr auto LdsBanksWidth = 128;
                constexpr auto kfold = (BK1 * N0 * sizeof(BDataType) > LdsBanksWidth) 
                                        ? 1
                                        : LdsBanksWidth / (BK1 * N0 * sizeof(BDataType));
                // constexpr auto KThreadReadPerm =
                //         (kfold * K0PerThreadWrite / K0PerThreadRead) > 1
                //             ? KThreadRead / (kfold * K0PerThreadWrite / K0PerThreadRead)
                //             : KThreadRead;
                constexpr auto KThreadReadPerm = KThreadRead;
                printf("LdsBanksWidth: %d, kfold: %lu, KThreadReadPerm: %d\n",
                       LdsBanksWidth, kfold, KThreadReadPerm);
                // 1<=npair<=n0
                constexpr auto npair = (BK1 * NPerXdl * sizeof(BDataType) > LdsBanksWidth)
                        ? 1
                        : ((LdsBanksWidth / (BK1 * NPerXdl * sizeof(BDataType))) > N0
                                ? N0
                                : LdsBanksWidth / (BK1 * NPerXdl * sizeof(BDataType)));
                printf("npair: %lu\n", npair);

                constexpr auto b_lds_block_desc = make_naive_tensor_descriptor_packed(
                    make_tuple(number<KThreadWrite / kfold / KThreadReadPerm>{},
                            number<K0PerThreadWrite>{},
                            number<KThreadReadPerm * N1>{},
                            number<kfold * N0 / npair>{},
                            number<npair>{},
                            BK1));
                printf("b_lds_block_desc:\n");
                b_lds_block_desc.print();

                constexpr auto b_lds_block_desc_permuted = transform_tensor_descriptor(
                    b_lds_block_desc,
                    make_tuple(
                        make_pass_through_transform(number<KThreadWrite / kfold / KThreadReadPerm>{}),
                        make_pass_through_transform(number<K0PerThreadWrite>{}),
                        make_xor_transform(
                            make_tuple(number<KThreadReadPerm * N1>{}, number<kfold * N0 / npair>{})),
                        make_pass_through_transform(number<npair>{}),
                        make_pass_through_transform(BK1)),
                    make_tuple(
                        sequence<0>{}, sequence<1>{}, sequence<2, 3>{}, sequence<4>{}, sequence<5>{}),
                    make_tuple(
                        sequence<0>{}, sequence<1>{}, sequence<2, 3>{}, sequence<4>{}, sequence<5>{}));
                printf("b_lds_block_desc_permuted:\n");
                b_lds_block_desc_permuted.print();

                constexpr auto b_lds_block_desc_unmerged = transform_tensor_descriptor(
                    b_lds_block_desc_permuted,
                    make_tuple(
                        make_pass_through_transform(number<KThreadWrite / kfold / KThreadReadPerm>{}),
                        make_pass_through_transform(number<K0PerThreadWrite>{}),
                        make_unmerge_transform(make_tuple(number<KThreadReadPerm>{}, number<N1>{})),
                        make_unmerge_transform(make_tuple(number<kfold>{}, number<N0 / npair>{})),
                        make_pass_through_transform(number<npair>{}),
                        make_pass_through_transform(BK1)),
                    make_tuple(sequence<0>{},
                            sequence<1>{},
                            sequence<2>{},
                            sequence<3>{},
                            sequence<4>{},
                            sequence<5>{}),
                    make_tuple(sequence<1>{},
                            sequence<2>{},
                            sequence<0, 3>{},
                            sequence<4, 5>{},
                            sequence<6>{},
                           sequence<7>{}));
                printf("b_lds_block_desc_unmerged:\n");
                b_lds_block_desc_unmerged.print();

                constexpr auto b_lds_block_desc_nk = transform_tensor_descriptor(
                    b_lds_block_desc_unmerged,
                    make_tuple(make_merge_transform_v3_division_mod(
                                make_tuple(number<KThreadReadPerm>{},
                                            number<KThreadWrite / kfold / KThreadReadPerm>{},
                                            number<kfold>{},
                                            number<K0PerThreadWrite>{},
                                            BK1)),
                            make_merge_transform_v3_division_mod(
                                make_tuple(number<N0 / npair>{}, number<npair>{}, number<N1>{}))),
                    make_tuple(sequence<0, 1, 4, 2, 7>{}, sequence<5, 6, 3>{}),
                    make_tuple(sequence<1>{}, sequence<0>{}));
                static_assert(b_lds_block_desc_nk.is_known_at_compile_time(), "not constexpr!");
                printf("b_lds_block_desc_nk:\n");
                b_lds_block_desc_nk.print();

                printf("2D static tile distribution:\n");
                TileEncodingPattern::Make2DStaticTileDistribution().print();
                printf("2D static shuffled tile distribution:\n");
                TileEncodingPattern::MakeShuffled2DStaticTileDistribution().print();
            }
            dim3 grids;
            if constexpr(Persistent)
            {
                grids = Kernel::MaxOccupancyGridSize(s);
            }
            else
            {
                grids = Kernel::GridSize(args.M, args.N, args.k_batch);
            }
            constexpr dim3 blocks = Kernel::BlockSize();

            if(!Kernel::IsSupportedArgument(kargs))
            {
                throw std::runtime_error("Wrong! Arguments not supported! Skipping gemm!\n");
            }

            if(s.log_level_ > 0)
            {
                std::cout << "Launching kernel with args: " << Kernel::GetName() << '\n'
                          << "shape: " << GemmShape::GetName() << '\n'
                          << "problem: " << GemmPipelineProblem::GetName() << '\n'
                          << "pipeline: " << GemmPipeline::GetName() << '\n'
                          << "grid: {" << grids.x << ", " << grids.y << ", " << grids.z << "}"
                          << ", blocks: {" << blocks.x << ", " << blocks.y << ", " << blocks.z
                          << "}" << std::endl;
            }
            if(s.flush_cache_)
            {
                std::cout << "Flushing cache..." << std::endl;
                static constexpr ck_tile::index_t APackedSize =
                    std::is_same_v<BDataType, ck_tile::pk_int4_t> ? 2 : 1;
                static constexpr ck_tile::index_t BPackedSize =
                    std::is_same_v<BDataType, ck_tile::pk_int4_t> ? 2 : 1;

                ck_tile::HostTensor<ADataType> a_m(ck_tile::host_tensor_descriptor(
                    args.M, args.K, args.stride_A, is_row_major(ALayout{})));
                ck_tile::HostTensor<BDataType> b_n(ck_tile::host_tensor_descriptor(
                    args.K, args.N, args.stride_B, is_row_major(BLayout{})));

                auto size_a_buffer = a_m.get_element_space_size_in_bytes() / APackedSize;
                auto size_b_buffer = b_n.get_element_space_size_in_bytes() / BPackedSize;

                ck_tile::RotatingMemWrapper<ADataType, BDataType> rotating_mem(
                    kargs.a_ptr, kargs.b_ptr, s.rotating_count_, size_a_buffer, size_b_buffer);
                rotating_mem.Print();

                auto run_flush_cache = [&]() {
                    // flush icache
                    ck_tile::flush_icache();
                    // rotating mem
                    rotating_mem.Next();
                    // clear c mem
                    if(args.k_batch > 1)
                        hipGetErrorString(hipMemsetAsync(
                            args.c_ptr, 0, args.M * args.N * sizeof(CDataType), s.stream_id_));
                };
                ave_time = ck_tile::launch_kernel_preprocess(
                    s,
                    run_flush_cache,
                    ck_tile::make_kernel<blocks.x, GemmConfig::kBlockPerCu>(
                        Kernel{}, grids, blocks, 0, kargs));
            }
            else
            {
                ave_time =
                    ck_tile::launch_kernel(s,
                                           ck_tile::make_kernel<blocks.x, GemmConfig::kBlockPerCu>(
                                               Kernel{}, grids, blocks, 0, kargs));
            }

            // Run toy load/store case
            ck_tile::ignore = ck_tile::launch_kernel(
                s,
                ck_tile::make_kernel<256, 1>(RunLoadGlobalStoreLDSLoadLDS{}, dim3(1), dim3(256), 0, static_cast<half_t*>(kargs.c_ptr), static_cast<half_t*>(kargs.c_ptr))
            );

            return ave_time;
        };

    const auto RunSplitk = [&](const auto has_hot_loop_, const auto tail_number_) {
        if(args.k_batch == 1)
        {
            Run(has_hot_loop_,
                tail_number_,
                ck_tile::integral_constant<ck_tile::memory_operation_enum,
                                           ck_tile::memory_operation_enum::set>{});
        }
        // else
        // {
        //     Run(has_hot_loop_,
        //         tail_number_,
        //         ck_tile::integral_constant<ck_tile::memory_operation_enum,
        //                                    ck_tile::memory_operation_enum::atomic_add>{});
        // }
    };

    BaseGemmPipeline::TailHandler(RunSplitk, has_hot_loop, tail_num);

    return ave_time;
}

template <typename APrecType, typename BPrecType = APrecType, typename CPrecType = APrecType>
int run_gemm_example_prec_type(std::string a_layout, std::string b_layout, int argc, char* argv[])
{
    using Row = ck_tile::tensor_layout::gemm::RowMajor;
    // using Col = ck_tile::tensor_layout::gemm::ColumnMajor;

    // if constexpr(std::is_same_v<BPrecType, ck_tile::pk_int4_t>)
    // {
    //     if(a_layout == "R" && b_layout == "C")
    //     {
    //         return run_gemm_example_with_layouts<APrecType, BPrecType, CPrecType>(
    //             argc, argv, Row{}, Col{}, Row{});
    //     }
    //     else if(a_layout == "C" && b_layout == "C")
    //     {
    //         return run_gemm_example_with_layouts<APrecType, BPrecType, CPrecType>(
    //             argc, argv, Col{}, Col{}, Row{});
    //     }
    //     else
    //     {
    //         throw std::runtime_error("Unsupported memory layout for the input matrices when "
    //                                  "BPrecType is ck_tile::pk_int4_t!");
    //     }
    // }
    // else
    // {
    if(a_layout == "R" && b_layout == "R")
    {
        return run_gemm_example_with_layouts<APrecType, BPrecType, CPrecType>(
            argc, argv, Row{}, Row{}, Row{});
    }
    // else if(a_layout == "R" && b_layout == "C")
    // {
    //     return run_gemm_example_with_layouts<APrecType, BPrecType, CPrecType>(
    //         argc, argv, Row{}, Col{}, Row{});
    // }
    // else if(a_layout == "C" && b_layout == "R")
    // {
    //     return run_gemm_example_with_layouts<APrecType, BPrecType, CPrecType>(
    //         argc, argv, Col{}, Row{}, Row{});
    // }
    // else if(a_layout == "C" && b_layout == "C")
    // {
    //     return run_gemm_example_with_layouts<APrecType, BPrecType, CPrecType>(
    //         argc, argv, Col{}, Col{}, Row{});
    // }
    else
    {
        throw std::runtime_error("Unsupported memory layout for the input matrices!");
    }
    // }
}

int run_gemm_example(int argc, char* argv[])
{
    auto [result, arg_parser] = create_args(argc, argv);
    if(!result)
        return -1;

    std::string data_type = arg_parser.get_str("prec");
    std::string a_layout  = arg_parser.get_str("a_layout");
    std::string b_layout  = arg_parser.get_str("b_layout");

    if(data_type == "fp16")
    {
        return run_gemm_example_prec_type<ck_tile::half_t>(a_layout, b_layout, argc, argv);
    }
    //     else if(data_type == "bf16")
    //     {
    //         return run_gemm_example_prec_type<ck_tile::bf16_t>(a_layout, b_layout, argc, argv);
    //     }
    //     else if(data_type == "fp8")
    //     {
    //         return run_gemm_example_prec_type<ck_tile::fp8_t, ck_tile::fp8_t, ck_tile::half_t>(
    //             a_layout, b_layout, argc, argv);
    //     }
    //     else if(data_type == "bf8")
    //     {
    //         return run_gemm_example_prec_type<ck_tile::bf8_t, ck_tile::bf8_t, ck_tile::half_t>(
    //             a_layout, b_layout, argc, argv);
    //     }

    // #if(CK_TILE_PIPELINE_DEFAULT == CK_TILE_PIPELINE_COMPUTE_V3)
    //     else if(data_type == "pk_int4_t")
    //     {
    //         // TODO: Add support for bhalf_t ADataType
    //         return run_gemm_example_prec_type<ck_tile::half_t, ck_tile::pk_int4_t,
    //         ck_tile::half_t>(
    //             a_layout, b_layout, argc, argv);
    //     }
    // #endif
    else
    {
        throw std::runtime_error("Unsupported data type for this operation !!!");
    }
}

int main(int argc, char* argv[])
{
    try
    {
        return !run_gemm_example(argc, argv);
    }
    catch(const std::runtime_error& e)
    {
        std::cerr << "Caught runtime error: " << e.what() << '\n';
        // Return a non-zero code to indicate failure
        return EXIT_FAILURE;
    }
}
