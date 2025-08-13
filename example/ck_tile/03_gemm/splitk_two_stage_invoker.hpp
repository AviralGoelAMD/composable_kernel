// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include "gemm_utils.hpp"
#include "ck_tile/ops/elementwise.hpp"

template <typename WorkspaceType_>
struct GemmConfigTwoStage : public GemmConfigBase
{
    using WorkspaceType = ck_tile::remove_cvref_t<WorkspaceType_>;
};

struct SplitKTwoStageInvoker
{
    template <typename GemmConfig,
              typename ADataType,
              typename BDataType,
              typename DsDataType,
              typename AccDataType,
              typename CDataType,
              typename ALayout,
              typename BLayout,
              typename DsLayout,
              typename CLayout,
              bool Persistent,
              typename CDEElementWise>
    static float gemm(const ck_tile::GemmHostArgs& args, const ck_tile::stream_config& s)

    {
        if constexpr(Persistent)
            std::cout << "WARNING: Ignoring persistent kernel option for basic gemm." << std::endl;

        // This part comes from the Codegen
        constexpr ck_tile::index_t M_Tile = 256;
        constexpr ck_tile::index_t N_Tile = 256;
        constexpr ck_tile::index_t K_Tile = 64;

        constexpr ck_tile::index_t M_Warp = 2;
        constexpr ck_tile::index_t N_Warp = 2;
        constexpr ck_tile::index_t K_Warp = 1;

        constexpr ck_tile::index_t M_Warp_Tile = 32;
        constexpr ck_tile::index_t N_Warp_Tile = 32;
        constexpr ck_tile::index_t K_Warp_Tile = 16;

        using CodegenGemmShape =
            ck_tile::TileGemmShape<ck_tile::sequence<M_Tile, N_Tile, K_Tile>,
                                   ck_tile::sequence<M_Warp, N_Warp, K_Warp>,
                                   ck_tile::sequence<M_Warp_Tile, N_Warp_Tile, K_Warp_Tile>>;

        using TilePartitioner = ck_tile::GemmTile1DPartitioner<CodegenGemmShape>;

        using CodegenGemmTraits = ck_tile::TileGemmTraits<GemmConfig::kPadM,
                                                          GemmConfig::kPadN,
                                                          GemmConfig::kPadK,
                                                          ALayout,
                                                          BLayout,
                                                          CLayout>;

        using CodegenPipelineProblem = ck_tile::GemmPipelineProblem<ADataType,
                                                                    BDataType,
                                                                    AccDataType,
                                                                    CodegenGemmShape,
                                                                    CodegenGemmTraits>;

        using CodegenGemmPipeline = ck_tile::GemmPipelineAGmemBGmemCRegV1<CodegenPipelineProblem>;
        using WorkspaceType       = ck_tile::remove_cvref_t<typename GemmConfig::WorkspaceType>;

        ck_tile::DeviceMem ws_m_n_dev_buf(args.M * args.N * sizeof(WorkspaceType));
        ws_m_n_dev_buf.SetZero();

        const auto Run = [&](const auto memory_operation_) {
            constexpr auto memory_operation = memory_operation_.value;

            using GemmEpilogue = ck_tile::CShuffleEpilogue<
                ck_tile::CShuffleEpilogueProblem<ADataType,
                                                 BDataType,
                                                 ck_tile::tuple<>,
                                                 AccDataType,
                                                 WorkspaceType,
                                                 ck_tile::tuple<>,
                                                 CLayout,
                                                 ck_tile::element_wise::PassThrough,
                                                 CodegenPipelineProblem::kBlockSize,
                                                 TilePartitioner::MPerBlock,
                                                 TilePartitioner::NPerBlock,
                                                 M_Warp,
                                                 N_Warp,
                                                 M_Warp_Tile,
                                                 N_Warp_Tile,
                                                 K_Warp_Tile,
                                                 CodegenPipelineProblem::TransposeC,
                                                 memory_operation>>;

            // ToDo: Will add the codegen part to test different pipeline policies in GEMM.
            // Now we only use the BlockGemmASmemBSmemCRegV1DefaultPolicy.
            using GemmKernel =
                ck_tile::GemmKernel<TilePartitioner, CodegenGemmPipeline, GemmEpilogue>;
            ck_tile::GemmHostArgs ws_args = ck_tile::GemmHostArgs(args);
            auto c_ptr                    = ws_args.c_ptr;
            ws_args.c_ptr                 = ws_m_n_dev_buf.GetDeviceBuffer();
            auto gemm_kargs               = GemmKernel::MakeKernelArgs(ws_args);

            if(!GemmKernel::IsSupportedArgument(gemm_kargs))
            {
                throw std::runtime_error("Wrong! Gemm arguments not supported! Skipping gemm!\n");
            }

            using XElementwiseOperation = ck_tile::element_wise::UnaryConvert;
            using BlockTile             = ck_tile::sequence<2048>;
            using BlockWarps            = ck_tile::sequence<8>;
            using WarpTile              = ck_tile::sequence<64>;

            using ElementwiseShape =
                ck_tile::ElementWiseShape<BlockWarps, BlockTile, WarpTile, WorkspaceType>;
            using Problem = ck_tile::ElementWisePipelineProblem<WorkspaceType,
                                                                WorkspaceType,
                                                                CDataType,
                                                                ElementwiseShape,
                                                                XElementwiseOperation>;
            using ElementwiseKernel =
                ck_tile::ElementWiseKernel<Problem, ck_tile::ElementWiseDefaultPolicy>;

            ck_tile::index_t total_elements     = 1;
            std::vector<ck_tile::index_t> shape = {args.M, args.N};

            for(auto d : shape)
                total_elements *= d;

            constexpr ck_tile::index_t kBlockSize =
                ck_tile::get_warp_size() * BlockWarps::at(ck_tile::number<0>{});
            constexpr ck_tile::index_t kBlockPerCu = 1;

            constexpr ck_tile::index_t elements_per_block = BlockTile::at(ck_tile::number<0>{});
            ck_tile::index_t kGridSize =
                (total_elements + elements_per_block - 1) / elements_per_block;

            auto input_tensors = ck_tile::make_tuple(static_cast<WorkspaceType*>(ws_args.c_ptr));
            auto input_size    = ck_tile::make_tuple(args.M, args.N);

            // Check if the kernel configuration is supported
            if(!ElementwiseKernel::IsSupportedArgument(input_size))
            {
                throw std::runtime_error(
                    "Wrong! Elementwise arguments not supported! Skipping gemm!\n");
            }

            const dim3 grids      = GemmKernel::GridSize(args.M, args.N, args.k_batch);
            constexpr dim3 blocks = GemmKernel::BlockSize();

            if(s.log_level_ > 0)
            {
                std::cout << "Launching kernel with args: " << GemmKernel::GetName() << '\n'
                          << "shape: " << CodegenGemmShape::GetName() << '\n'
                          << "problem: " << CodegenPipelineProblem::GetName() << '\n'
                          << "pipeline: " << CodegenGemmPipeline::GetName() << '\n'
                          << "grid: {" << grids.x << ", " << grids.y << ", " << grids.z << "}"
                          << ", blocks: {" << blocks.x << ", " << blocks.y << ", " << blocks.z
                          << "}" << std::endl;
            }

            float ave_time =
                ck_tile::launch_kernel(s,
                                       ck_tile::make_kernel<blocks.x, GemmConfig::kBlockPerCu>(
                                           GemmKernel{}, grids, blocks, 0, gemm_kargs),
                                       ck_tile::make_kernel<kBlockSize, kBlockPerCu>(
                                           ElementwiseKernel{},
                                           kGridSize,
                                           kBlockSize,
                                           0,
                                           input_size,
                                           ck_tile::make_tuple(args.N, 1), // Input Stride
                                           ck_tile::make_tuple(args.N, 1), // Output Stride
                                           input_tensors,
                                           static_cast<CDataType*>(c_ptr)));

            return ave_time;
        };

        if(args.k_batch == 1)
        {
            return Run(MemoryOpSet{});
        }
        else
        {
            return Run(MemoryOpAtomicAdd{});
        }
    }
};
