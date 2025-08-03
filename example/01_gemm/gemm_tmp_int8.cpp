// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#define CK_ENABLE_DYNAMIC_WARP_SIZE 1
#include "common.hpp"

#include "ck/tensor_operation/gpu/device/impl/device_gemm_xdl_cshuffle_v3.hpp"

using ADataType        = int8_t;
using BDataType        = int8_t;
using CDataType        = int8_t;
using AccDataType      = int32_t;
using CShuffleDataType = int8_t;

using ALayout = Row;
using BLayout = Col;
using CLayout = Row;

using AElementOp = PassThrough;
using BElementOp = PassThrough;
using CElementOp = PassThrough;

static constexpr auto GemmDefault = ck::tensor_operation::device::GemmSpecialization::Default;

// clang-format off
using DeviceGemmV2Instance = ck::tensor_operation::device::DeviceGemm_Xdl_CShuffleV3<
    ALayout, BLayout, CLayout,
    ADataType, BDataType, CDataType,
    AccDataType, CShuffleDataType,
    AElementOp, BElementOp, CElementOp, GemmDefault,
    256, 128, 128, 64,
    8, 8, 16, 16, 4, 4,
    S<8, 32, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 0,
    S<8, 32, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 0,
    1, 2, S<1, 32, 1, 8>, 8>;
// clang-format on

using ReferenceGemmInstance = ck::tensor_operation::host::
    ReferenceGemm<ADataType, BDataType, CDataType, AccDataType, AElementOp, BElementOp, CElementOp>;

using ReferenceGemmInstanceGPU = ck::tensor_operation::device::ReferenceGemm<ALayout,
                                                                             BLayout,
                                                                             CLayout,
                                                                             ADataType,
                                                                             BDataType,
                                                                             CDataType,
                                                                             AccDataType,
                                                                             AElementOp,
                                                                             BElementOp,
                                                                             CElementOp>;

#include "run_gemm_example_v2.inc"

int main(int argc, char* argv[]) { return !run_gemm_splitk_example(argc, argv); }
