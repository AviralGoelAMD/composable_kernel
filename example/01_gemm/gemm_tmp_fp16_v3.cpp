// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#define CK_ENABLE_DYNAMIC_WARP_SIZE 1
#include "common.hpp"

#include "ck/tensor_operation/gpu/device/impl/device_gemm_xdl_cshuffle_v3.hpp"

using ADataType        = ck::half_t;
using BDataType        = ck::half_t;
using AccDataType      = float;
using CShuffleDataType = ck::half_t;
using CDataType        = ck::half_t;

using ALayout = Row;
using BLayout = Row;
using CLayout = Row;

using AElementOp = PassThrough;
using BElementOp = PassThrough;
using CElementOp = PassThrough;

static constexpr auto GemmDefault = ck::tensor_operation::device::GemmSpecialization::Default;

// clang-format off
#if 1
// valid on both wave32 and wave64
using DeviceGemmV2Instance = 
    ck::tensor_operation::device::DeviceGemm_Xdl_CShuffleV3<
        ALayout,   BLayout,  CLayout,   
        ADataType,   BDataType,  CDataType,  AccDataType,  CShuffleDataType, 
        PassThrough, PassThrough, PassThrough, GemmDefault, 
        #if 0
        64,
        16, 32, //32
        64, 8, 8,
        16,   16,
        1,   2, // 2,
        S<8, 8, 1>,  S<1, 0, 2>,  S<1, 0, 2>, 
        2, 8, 8, 0,
        S<8, 8, 1>,  S<1, 0, 2>,  S<1, 0, 2>, 
        2, 8, 8, 0,
        1, 1, S<1, 16, 1, 4>, 4,
        #endif

    256, 128, 128,
     64, 8, 8, 
     16, 16, 4, 4,
    S<4, 64, 1>, S<1, 0, 2>, S<1, 0, 2>,
    2, 8, 8, 1,
    S<4, 64, 1>, S<0, 2, 1>, S<0, 2, 1>,
    1, 1, 8, 1,
    1, 1, S<1, 32, 1, 8>, 8,
        ck::BlockGemmPipelineScheduler::Intrawave, ck::BlockGemmPipelineVersion::v3>;
#endif

#if 0
// valid on wave64 only
using DeviceGemmV2Instance = 
    ck::tensor_operation::device::DeviceGemm_Xdl_CShuffleV3<
        ALayout,   BLayout,  CLayout,   
        ADataType,   BDataType,  CDataType,  AccDataType,  CShuffleDataType, 
        PassThrough, PassThrough, PassThrough, GemmDefault, 
        64,
        16, 16,
        64, 8, 8,
        16,   16,
        1,   1,
        S<8, 8, 1>,  S<1, 0, 2>,  S<1, 0, 2>, 
        2, 8, 8, 0,
        S<8, 8, 1>,  S<1, 0, 2>,  S<1, 0, 2>, 
        2, 8, 8, 0,
        1, 1, S<1, 16, 1, 4>, 4,
        ck::BlockGemmPipelineScheduler::Interwave, ck::BlockGemmPipelineVersion::v1>;
#endif

#if 0
// valid on wave32 only
using DeviceGemmV2Instance = 
    ck::tensor_operation::device::DeviceGemm_Xdl_CShuffleV3<
        ALayout,   BLayout,  CLayout,   
        ADataType,   BDataType,  CDataType,  AccDataType,  CShuffleDataType, 
        PassThrough, PassThrough, PassThrough, GemmDefault, 
        32,
        16, 16,
        64, 8, 8,
        16,   16,
        1,   1,
        S<4, 8, 1>,  S<1, 0, 2>,  S<1, 0, 2>, 
        2, 4, 8, 0,
        S<4, 8, 1>,  S<1, 0, 2>,  S<1, 0, 2>, 
        2, 4, 8, 0,
        1, 1, S<1, 8, 1, 4>, 4,
        ck::BlockGemmPipelineScheduler::Interwave, ck::BlockGemmPipelineVersion::v1>;
#endif
// clang-format on

using ReferenceGemmInstance = ck::tensor_operation::host::
    ReferenceGemm<ADataType, BDataType, CDataType, AccDataType, AElementOp, BElementOp, CElementOp>;

#include "run_gemm_example_v2.inc"

int main(int argc, char* argv[]) { return !run_gemm_splitk_example(argc, argv); }
