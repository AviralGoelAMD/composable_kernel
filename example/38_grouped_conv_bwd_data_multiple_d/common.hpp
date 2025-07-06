// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <cstdlib>
#include <initializer_list>
#include <iostream>
#include <numeric>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/convolution_backward_data_specialization.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/reference_tensor_operation/cpu/reference_conv_bwd_data.hpp"
#include "ck/library/utility/algorithm.hpp"
#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/convolution_host_tensor_descriptor_helper.hpp"
#include "ck/library/utility/convolution_parameter.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;

static inline constexpr ck::index_t NDimSpatial = 2;

static constexpr auto ConvBwdDataDefault =
    ck::tensor_operation::device::ConvolutionBackwardDataSpecialization::Filter1x1Stride1Pad0;

using FP16 = ck::half_t;
using FP32 = float;
using FP8  = ck::f8_t;
using BF8  = ck::bf8_t;

struct ExecutionConfig final
{
    bool do_verification = true;
    int init_method      = 1;
    bool time_kernel     = false;
};

#define DefaultConvParams                                                                \
    ck::utils::conv::ConvParam                                                           \
    {                                                                                    \
        NDimSpatial, 32, 4, 192, 192, {3, 3}, {28, 28}, {1, 1}, {1, 1}, {1, 1}, { 1, 1 } \
    }

inline void print_help_msg()
{
    std::cerr << "arg1: verification (0=no, 1=yes)\n"
              << "arg2: initialization (0=no init, 1=integer value, 2=decimal value)\n"
              << "arg3: time kernel (0=no, 1=yes)\n"
              << ck::utils::conv::get_conv_param_parser_helper_msg() << std::endl;
}

inline bool parse_cmd_args(int argc,
                           char* argv[],
                           ExecutionConfig& config,
                           ck::utils::conv::ConvParam& conv_params)
{
    constexpr int num_execution_config_args =
        3; // arguments for do_verification, init_method, time_kernel
    constexpr int num_conv_param_leading_args = 5; // arguments for num_dim_spatial_, G_, N_, K_, C_

    constexpr int threshold_to_catch_partial_args = 1 + num_execution_config_args;
    constexpr int threshold_to_catch_all_args =
        threshold_to_catch_partial_args + num_conv_param_leading_args;

    if(argc == 1)
    {
        // use default
        config = ExecutionConfig{};
    }
    // catch only ExecutionConfig arguments
    else if(argc == threshold_to_catch_partial_args)
    {
        config.do_verification = std::stoi(argv[1]);
        config.init_method     = std::stoi(argv[2]);
        config.time_kernel     = std::stoi(argv[3]);
    }
    // catch both ExecutionConfig & ConvParam arguments
    else if(threshold_to_catch_all_args < argc && ((argc - threshold_to_catch_all_args) % 3 == 0))
    {
        config.do_verification = std::stoi(argv[1]);
        config.init_method     = std::stoi(argv[2]);
        config.time_kernel     = std::stoi(argv[3]);

        const ck::index_t num_dim_spatial = std::stoi(argv[4]);
        conv_params                       = ck::utils::conv::parse_conv_param(
            num_dim_spatial, threshold_to_catch_partial_args + 1, argv);
    }
    else
    {
        print_help_msg();
        return false;
    }

    return true;
}


template <typename DataType>
void dump_tensor(const Tensor<DataType>& tensor, const char* str)
{
    //if(config.dump_tensor == false)
     //  return;
    assert(tensor.GetNumOfDimension() >= 4 && tensor.GetNumOfDimension() <= 6);
    auto lengths = tensor.GetLengths();
    auto strides = tensor.GetStrides();

    auto numDim = tensor.GetNumOfDimension() - 3;
    if(numDim == 1)
    {
        std::cout << str << "  [ " << std::endl;
        for(uint32_t i0 = 0; i0 < lengths[0]; i0++)
        {
            if(lengths[1] > 1)
            {
                std::cout << "  [";
            }
            if(i0 > 0 && strides[0] == 0)
            {
                continue;
            }
            for(uint32_t i1 = 0; i1 < lengths[1]; i1++)
            {
                if(lengths[2] > 1)
                {
                    std::cout << "  [";
                }
                if(i1 > 0 && strides[1] == 0)
                {
                    continue;
                }
                for(uint32_t i2 = 0; i2 < lengths[2]; i2++)
                {
                    if(lengths[3] > 1)
                    {
                        std::cout << "  [";
                    }
                    if(i2 > 0 && strides[2] == 0)
                    {
                        continue;
                    }
                    for(uint32_t i3 = 0; i3 < lengths[3]; i3++)
                    {
                        if(i3 > 0 && strides[3] == 0)
                        {
                            continue;
                        }
                        std::vector<std::size_t> idx({i0, i1, i2, i3});
                        std::cout << ck::type_convert<float>(tensor(idx)) << ", ";
                    }
                    if(lengths[3] > 1)
                    {
                        std::cout << "]" << std::endl;
                    }
                    if(lengths[3] > 3)
                    {
                        std::cout << std::endl;
                    }
                }
                if(lengths[2] > 1)
                {
                    std::cout << "]" << std::endl;
                }
            }
            if(lengths[1] > 1)
            {
                std::cout << "]" << std::endl;
            }
        }
        std::cout << "]" << std::endl;
    }
    else if(numDim == 2)
    {
        std::cout << str << "  [ " << std::endl;
        for(uint32_t i0 = 0; i0 < lengths[0]; i0++)
        {
            if(lengths[1] > 1)
            {
                std::cout << "  [";
            }
            if(i0 > 0 && strides[0] == 0)
            {
                continue;
            }
            for(uint32_t i1 = 0; i1 < lengths[1]; i1++)
            {
                if(lengths[2] > 1)
                {
                    std::cout << "  [";
                }
                if(i1 > 0 && strides[1] == 0)
                {
                    continue;
                }
                for(uint32_t i2 = 0; i2 < lengths[2]; i2++)
                {
                    if(lengths[3] > 1)
                    {
                        std::cout << "  [";
                    }
                    if(i2 > 0 && strides[2] == 0)
                    {
                        continue;
                    }
                    for(uint32_t i3 = 0; i3 < lengths[3]; i3++)
                    {
                        if(lengths[4] > 1)
                        {
                            std::cout << "  [";
                        }
                        if(i3 > 0 && strides[3] == 0)
                        {
                            continue;
                        }
                        for(uint32_t i4 = 0; i4 < lengths[4]; i4++)
                        {
                            if(i4 > 0 && strides[4] == 0)
                            {
                                continue;
                            }
                            std::vector<std::size_t> idx({i0, i1, i2, i3, i4});
                            std::cout << ck::type_convert<float>(tensor(idx)) << ", ";
                        }
                        if(lengths[4] > 1)
                        {
                            std::cout << "]";
                        }
                        if(lengths[4] > 3)
                        {
                            std::cout << std::endl;
                        }
                    }
                    if(lengths[3] > 1)
                    {
                        std::cout << "]" << std::endl;
                    }
                }
                if(lengths[2] > 1)
                {
                    std::cout << "]" << std::endl;
                }
            }
            if(lengths[1] > 1)
            {
                std::cout << "]" << std::endl;
            }
        }
        std::cout << "]" << std::endl;
    }
    else if(numDim == 3)
    {
        std::cout << str << "  [ " << std::endl;
        for(uint32_t i0 = 0; i0 < lengths[0]; i0++)
        {
            if(lengths[1] > 1)
            {
                std::cout << "  [";
            }
            if(i0 > 0 && strides[0] == 0)
            {
                continue;
            }
            for(uint32_t i1 = 0; i1 < lengths[1]; i1++)
            {
                if(lengths[2] > 1)
                {
                    std::cout << "  [";
                }
                if(i1 > 0 && strides[1] == 0)
                {
                    continue;
                }
                for(uint32_t i2 = 0; i2 < lengths[2]; i2++)
                {
                    if(lengths[3] > 1)
                    {
                        std::cout << "  [";
                    }
                    if(i2 > 0 && strides[2] == 0)
                    {
                        continue;
                    }
                    for(uint32_t i3 = 0; i3 < lengths[3]; i3++)
                    {
                        if(lengths[4] > 1)
                        {
                            std::cout << "  [";
                        }
                        if(i3 > 0 && strides[3] == 0)
                        {
                            continue;
                        }
                        for(uint32_t i4 = 0; i4 < lengths[4]; i4++)
                        {
                            if(lengths[5] > 1)
                            {
                                std::cout << "  [";
                            }
                            if(i4 > 0 && strides[4] == 0)
                            {
                                continue;
                            }
                            for(uint32_t i5 = 0; i5 < lengths[5]; i5++)
                            {
                                if(i5 > 0 && strides[5] == 0)
                                {
                                    continue;
                                }
                                std::vector<std::size_t> idx({i0, i1, i2, i3, i4, i5});
                                std::cout << ck::type_convert<float>(tensor(idx)) << ", ";
                            }
                            if(lengths[5] > 1)
                            {
                                std::cout << "]";
                            }
                            if(lengths[5] > 3)
                            {
                                std::cout << std::endl;
                            }
                        }
                        if(lengths[4] > 1)
                        {
                            std::cout << "]";
                        }
                    }
                    if(lengths[3] > 1)
                    {
                        std::cout << "]" << std::endl;
                    }
                }
                if(lengths[2] > 1)
                {
                    std::cout << "]" << std::endl;
                }
            }
            if(lengths[1] > 1)
            {
                std::cout << "]" << std::endl;
            }
        }
        std::cout << "]" << std::endl;
    }
}
