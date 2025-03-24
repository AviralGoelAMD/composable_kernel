// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include "ck_tile/host.hpp"
#include "wave_specialization.hpp"
#include <cstring>

auto create_args(int argc, char* argv[])
{
    ck_tile::ArgParser arg_parser;
    arg_parser.insert("m", "3328", "m dimension")
        .insert("n", "4096", "n dimension")
        .insert("v", "1", "cpu validation or not")
        .insert("prec", "fp16", "precision")
        .insert("warmup", "50", "cold iter")
        .insert("repeat", "100", "hot iter");

    bool result = arg_parser.parse(argc, argv);
    return std::make_tuple(result, arg_parser);
}

template <typename DataType>
bool run(const ck_tile::ArgParser& arg_parser)
{
    using XDataType       = DataType;
    using ComputeDataType = float;
    using YDataType       = DataType;

    ck_tile::index_t m = arg_parser.get_int("m");
    ck_tile::index_t n = arg_parser.get_int("n");
    int do_validation  = arg_parser.get_int("v");
    int warmup         = arg_parser.get_int("warmup");
    int repeat         = arg_parser.get_int("repeat");

    ck_tile::HostTensor<XDataType> x_host({m, n});
    ck_tile::HostTensor<YDataType> y_host_ref({m});
    ck_tile::HostTensor<YDataType> y_host_dev({m});

    // ck_tile::FillUniformDistribution<XDataType>{-5.f, 5.f}(x_host);
    ck_tile::FillConstant<XDataType>{1.f}(x_host);

    ck_tile::DeviceMem x_buf(x_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem y_buf(y_host_dev.get_element_space_size_in_bytes());

    x_buf.ToDevice(x_host.data());

    using ReduceOp   = ck_tile::ReduceOp::Add;
    // using BlockWaves = ck_tile::sequence<4, 1>;
    using BlockWaves = ck_tile::sequence<2, 1>;
    // using BlockTile  = ck_tile::sequence<64, 128>;
    using BlockTile  = ck_tile::sequence<32, 128>;
    using WaveTile   = ck_tile::sequence<32, 128>;
    using Vector     = ck_tile::sequence<8, 8>;

    // cross warp-reduce
    // using BlockWaves = ck_tile::sequence<2, 2>;
    // using BlockTile  = ck_tile::sequence<2, 1024>;
    // using WaveTile   = ck_tile::sequence<1, 512>;
    // using Vector = ck_tile::sequence<1, 8>;

    ck_tile::index_t kGridSize             = (m / BlockTile::at(ck_tile::number<0>{}));
    std::cout << "grid size " << kGridSize << std::endl;
    
    using Shape = ck_tile::Reduce2dShape<BlockWaves, BlockTile, WaveTile, Vector>;
    using Porblem =
    ck_tile::Reduce2dProblem<XDataType, ComputeDataType, YDataType, Shape, ReduceOp>;
    
    constexpr ck_tile::index_t kBlockSize  = Shape::BlockSize;
    constexpr ck_tile::index_t kBlockPerCu = 1;
    
    using Kernel = ck_tile::Reduce<Porblem>;

    float ave_time = launch_kernel(ck_tile::stream_config{nullptr, true, 0, warmup, repeat},
                                   ck_tile::make_kernel<kBlockSize, kBlockPerCu>(
                                       Kernel{},
                                       kGridSize,
                                       kBlockSize,
                                       0,
                                       static_cast<XDataType*>(x_buf.GetDeviceBuffer()),
                                       static_cast<YDataType*>(y_buf.GetDeviceBuffer()),
                                       m,
                                       n));

    std::size_t num_btype = sizeof(XDataType) * m * n + sizeof(YDataType) * m;

    float gb_per_sec = num_btype / 1.E6 / ave_time;
    std::cout << "Perf: " << ave_time << " ms, " << gb_per_sec << " GB/s" << std::endl;

    bool pass = true;

    if(do_validation)
    {
        // reference
        ck_tile::reference_reduce<XDataType, ComputeDataType, YDataType>(
            x_host, y_host_ref, ReduceOp{});
        y_buf.FromDevice(y_host_dev.mData.data());
        pass = ck_tile::check_err(y_host_dev, y_host_ref);

        std::cout << "valid:" << (pass ? "y" : "n") << std::flush << std::endl;
    }

    return pass;
}

int main(int argc, char* argv[])
{
    auto [result, arg_parser] = create_args(argc, argv);
    if(!result)
        return -1;

    const std::string data_type = arg_parser.get_str("prec");

    if(data_type == "fp16")
    {
        return run<ck_tile::half_t>(arg_parser) ? 0 : -2;
    }
    // else if(data_type == "bf16")
    // {
    //     return run<ck_tile::bf16_t>(arg_parser) ? 0 : -2;
    // }
}
