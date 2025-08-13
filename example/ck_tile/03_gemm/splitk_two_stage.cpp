// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include "gemm_utils.hpp"
#include "run_gemm_example.inc"
#include "run_gemm_example_common.hpp"
#include "splitk_two_stage_invoker.hpp"

int run_gemm_example(ck_tile::ArgParser& arg_parser)
{
    std::string data_type   = arg_parser.get_str("prec");
    std::string w_data_type = arg_parser.get_str("w_prec");
    std::string a_layout    = arg_parser.get_str("a_layout");
    std::string b_layout    = arg_parser.get_str("b_layout");

    using Invoker = SplitKTwoStageInvoker;

    if(data_type != "bf16")
    {
        std::cout << "WARNING: Ignoring prec option for splitk two stage, and using bf16."
                  << std::endl;
    }

    if(w_data_type == "fp16")
    {
        return run_gemm_example_prec_type<GemmConfigTwoStage<ck_tile::half_t>,
                                          Invoker,
                                          ck_tile::bf16_t>(a_layout, b_layout, arg_parser);
    }
    else if(w_data_type == "fp32")
    {
        return run_gemm_example_prec_type<GemmConfigTwoStage<float>, Invoker, ck_tile::bf16_t>(
            a_layout, b_layout, arg_parser);
    }
    else
    {
        throw std::runtime_error("Unsupported data type for this operation !!!");
    }
}

int main(int argc, char* argv[])
{
    auto arg_parser = create_args();
    arg_parser.insert("w_prec", "fp16", "workspace data type. fp16/fp32");
    auto result = arg_parser.parse(argc, argv);

    if(!result)
        return -1;

    try
    {
        return !run_gemm_example(arg_parser);
    }
    catch(const std::runtime_error& e)
    {
        std::cerr << "Runtime error: " << e.what() << '\n';
        return EXIT_FAILURE;
    }
}
