// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include "ck_tile/host.hpp"
#include <cstring>
#include "test_atomic.hpp"

auto create_args(int argc, char* argv[])
{
    ck_tile::ArgParser arg_parser;
    arg_parser.insert("m", "64", "m dimension")
        .insert("n", "8", "n dimension")
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
    using XDataType = DataType;
    using YDataType = DataType;

    ck_tile::index_t m       = arg_parser.get_int("m");
    ck_tile::index_t n       = arg_parser.get_int("n");
    int do_validation        = arg_parser.get_int("v");
    int warmup               = arg_parser.get_int("warmup");
    int repeat               = arg_parser.get_int("repeat");

    constexpr auto dword_bytes = 4;

        if(n % (dword_bytes / sizeof(DataType)) != 0)
        {
            std::cerr << "n size should be multiple of dword_bytes" << std::endl;
        }

    ck_tile::HostTensor<YDataType> y_host_ref({m, n});
    ck_tile::HostTensor<YDataType> y_host_dev({m, n});

    /*
        build the expected output result for this operation. 

        We are performing the following operation 
        0 + 1 = 1

        using atomic_add as the memory_operation of the input tensor. 

        As of now, input precision is set to fp16_t or half_t.

    */

    ck_tile::half_t value = 1;
    for(int i = 0; i < m; i++)
    {
        value = 1;
        for(int j = 0; j < n; j++)
        {
            y_host_ref(i, j) = value;
        }
    }

    /*
        Set the initial value of the input tensor to be all 0's
    */
    ck_tile::DeviceMem y_buf(y_host_dev.get_element_space_size_in_bytes());
    //y_buf.ToDevice(y_host_ref.data());
    y_buf.SetZero();

    using BlockWaves = ck_tile::sequence<2, 1>;
    using BlockTile  = ck_tile::sequence<64, 8>;
    using WaveTile   = ck_tile::sequence<64, 8>;
    using Vector     = ck_tile::sequence<1, dword_bytes / sizeof(DataType)>;

    ck_tile::index_t kGridSize = (m / BlockTile::at(ck_tile::number<0>{}));
    std::cout << "grid size " << kGridSize << std::endl;

    using Shape   = ck_tile::TileCopyShape<BlockWaves, BlockTile, WaveTile, Vector>;
    using Problem = ck_tile::TileCopyProblem<XDataType, Shape>;
    using Kernel  = ck_tile::TileCopy<Problem>;

    constexpr ck_tile::index_t kBlockSize  = 128;
    constexpr ck_tile::index_t kBlockPerCu = 1;
    std::cout << "block size " << kBlockSize << std::endl;
    std::cout << "warp SIze " << ck_tile::get_warp_size() << std::endl;
    std::cout << "warps per block _M " << Shape::WarpPerBlock_M << " " << Shape::WarpPerBlock_N
              << std::endl;
    std::cout << "Block waves: " << BlockWaves::at(ck_tile::number<0>{}) << " "
              << BlockWaves::at(ck_tile::number<1>{}) << std::endl;
    std::cout << " Wave Groups: " << Shape::WaveGroups << std::endl;

    float ave_time = launch_kernel(ck_tile::stream_config{nullptr, true, 0, warmup, repeat},
                                   ck_tile::make_kernel<kBlockSize, kBlockPerCu>(
                                       Kernel{},
                                       kGridSize,
                                       kBlockSize,
                                       0,
                                       static_cast<YDataType*>(y_buf.GetDeviceBuffer()),
                                       m,
                                       n));

    y_buf.SetZero();

    launch_kernel(ck_tile::stream_config{nullptr, false, 0, 0, 1},
              ck_tile::make_kernel<kBlockSize, kBlockPerCu>(
                  Kernel{}, kGridSize, kBlockSize, 0,
                  static_cast<YDataType*>(y_buf.GetDeviceBuffer()), m, n));

    std::size_t num_btype = sizeof(XDataType) * m * n + sizeof(YDataType) * m;

    float gb_per_sec = num_btype / 1.E6 / ave_time;
    std::cout << "Perf: " << ave_time << " ms, " << gb_per_sec << " GB/s" << std::endl;

    bool pass = true;

    if(do_validation)
    {
        // reference
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
    return run<ck_tile::half_t>(arg_parser) ? 0 : -2;
}
