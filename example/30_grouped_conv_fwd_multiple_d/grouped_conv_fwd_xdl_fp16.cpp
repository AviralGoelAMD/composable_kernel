// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#define ENABLE_CONV_FACTORY 1

#include "device_grouped_conv_fwd_dl_v4.hpp"
#include "common.hpp"

// kernel data types
using InKernelDataType  = FP16;
using WeiKernelDataType = FP16;
using AccDataType       = FP32;
using CShuffleDataType  = FP16;
using OutKernelDataType = FP16;

// tensor data types
using InUserDataType  = InKernelDataType;
using WeiUserDataType = WeiKernelDataType;
using OutUserDataType = OutKernelDataType;

using InElementOp  = PassThrough;
using WeiElementOp = PassThrough;
using OutElementOp = PassThrough;

#if !defined(ENABLE_CONV_FACTORY)
template <ck::index_t NDimSpatial>
using DeviceConvFwdInstance =
    ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<
        NDimSpatial,
        InputLayout<NDimSpatial>,
        WeightLayout<NDimSpatial>,
        ck::Tuple<>,
        OutputLayout<NDimSpatial>,
        InKernelDataType,
        WeiKernelDataType,
        AccDataType,
        CShuffleDataType,
        ck::Tuple<>,
        OutKernelDataType,
        InElementOp,
        WeiElementOp,
        OutElementOp,
        ConvSpec,    // ConvForwardSpecialization
        GemmSpec,    // GemmSpecialization    
        #if 0 // pass 1
        1,   64,   32,    32,    32,   4,   4,   32,   32,    1,    1,   
       // ABlockTransferThreadClusterLengths_AK0_M_AK1   ABlockTransferThreadClusterArrangeOrder     ABlockTransferSrcAccessOrder    ABlockTransferSrcVectorDim     ABlockTransferSrcScalarPerVector     ABlockTransferDstScalarPerVector_AK1      ABlockLdsExtraM  
        S<8, 8, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              4,              4,         1,     
        S<8, 8, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              4,              4,         1,
        1,           1,
        S<1, 8, 1, 8>,               1>;
      #endif
        #if 1
        1,   256,   128,    64,    32,   4,   8,   32,   32,    2,    1,   
       // ABlockTransferThreadClusterLengths_AK0_M_AK1   ABlockTransferThreadClusterArrangeOrder     ABlockTransferSrcAccessOrder    ABlockTransferSrcVectorDim     ABlockTransferSrcScalarPerVector     ABlockTransferDstScalarPerVector_AK1      ABlockLdsExtraM  
        S<8, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              4,              4,         1,     
        S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0 , 2>,            2,              8,              8,         1,
        1,           1,
         S<1, 32, 1, 8>,               1>;
         #endif
#else
constexpr ck::index_t NDimSpatial = 2;
using ALayout = InputLayout<NDimSpatial>;
using BLayout = WeightLayout<NDimSpatial>;
using ELayout = OutputLayout<NDimSpatial>;
using DsLayout =  ck::Tuple<>;
using DsDataTypes =  ck::Tuple<>;
using F16 = FP16;
using F32 = FP32;
constexpr auto GemmMNKPadding = GemmSpec;
//using DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle = ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle;

using DeviceConvFwdFactory = std::tuple<
   // clang-format off
  //########################################|     NumDim|      A|      B|          Ds|      E| AData| BData| AccData| CShuffle|          Ds| EData|           A|           B|         CDE|    ConvForward|           GEMM| NumGemmK| Block|  MPer|  NPer|  KPer| AK1| BK1| MPer| NPer| MXdl| NXdl|  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|  BBlockTransfer| BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds|    CShuffle|    CShuffle| CBlockTransferClusterLengths|  CBlockTransfer|
  //########################################|    Spatial| Layout| Layout|      Layout| Layout|  Type|  Type|    Type| DataType|    DataType|  Type| Elementwise| Elementwise| Elementwise| Specialization| Specialization| Prefetch|  Size| Block| Block| Block|    |    |  XDL|  XDL|  Per|  Per|   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|   ThreadCluster|  ThreadCluster| SrcAccessOrder|  SrcVectorDim|      SrcScalar|      DstScalar| AddExtraN| MXdlPerWave| NXdlPerWave|         _MBlock_MWaveMPerXdl| ScalarPerVector|
  //########################################|           |       |       |            |       |      |      |        |         |            |      |   Operation|   Operation|   Operation|               |               |    Stage|      |      |      |      |    |    |     |     | Wave| Wave| Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          | Lengths_K0_N_K1|   ArrangeOrder|               |              |      PerVector|   PerVector_K1|          |  PerShuffle|  PerShuffle|         _NBlock_NWaveNPerXdl|   _NWaveNPerXdl|
  //########################################|           |       |       |            |       |      |      |        |         |            |      |            |            |            |               |               |         |      |      |      |      |    |    |     |     |     |     |                |               |               |               |               |               |          |                |               |               |              |               |               |          |            |            |                             |                |
    // generic instance                 
    ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NDimSpatial,ALayout,BLayout,    DsLayout,ELayout,   F16,   F16,     F32,      F16,    DsDataTypes,   F16, PassThrough, PassThrough, OutElementOp,       ConvSpec, GemmMNKPadding,        1,    64,    64,    64,    32,   4,   8,   32,   32,    2,    2,     S<4, 16, 1>,     S<0, 2, 1>,     S<0, 2, 1>, 1,              1,              4,         1,     S<4, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              1,              8,         1,           1,           1,               S<1, 16, 1, 4>,               1>
   
    // instances for small conv.K and conv.C
    ,ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NDimSpatial,ALayout,BLayout,    DsLayout,ELayout,   F16,   F16,     F32,      F16,    DsDataTypes,   F16, PassThrough, PassThrough, OutElementOp,       ConvSpec, GemmMNKPadding,        1,    64,    64,    32,    32,   4,   8,   32,   32,    2,    1,     S<8, 8, 1>,      S<0, 2, 1>,     S<0, 2, 1>, 1,              4,              4,         1,     S<4, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,         1,           1,           1,               S<1, 16, 1, 4>,               1>
    ,ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NDimSpatial,ALayout,BLayout,    DsLayout,ELayout,   F16,   F16,     F32,      F16,    DsDataTypes,   F16, PassThrough, PassThrough, OutElementOp,       ConvSpec, GemmMNKPadding,        1,   256,   128,   128,    32,   8,   8,   32,   32,    2,    2,     S<4, 64, 1>,     S<0, 2, 1>,     S<0, 2, 1>, 1,              1,              8,         1,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              1,              8,         1,           1,           1,               S<1, 32, 1, 8>,               1>
    ,ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NDimSpatial,ALayout,BLayout,    DsLayout,ELayout,   F16,   F16,     F32,      F16,    DsDataTypes,   F16, PassThrough, PassThrough, OutElementOp,       ConvSpec, GemmMNKPadding,        1,   256,   256,   128,    32,   4,   8,   32,   32,    4,    2,     S<8, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>, 1,              8,              4,         1,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,         1,           1,           1,               S<1, 32, 1, 8>,               1>
    
    ,ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NDimSpatial,ALayout,BLayout,    DsLayout,ELayout,   F16,   F16,     F32,      F16,    DsDataTypes,   F16, PassThrough, PassThrough, OutElementOp,       ConvSpec, GemmMNKPadding,        1,   256,   128,   256,    32,   4,   8,   32,   32,    2,    4,     S<8, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>, 1,              4,              4,         1,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,         1,           1,           1,               S<1, 32, 1, 8>,               1>
    ,ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NDimSpatial,ALayout,BLayout,    DsLayout,ELayout,   F16,   F16,     F32,      F16,    DsDataTypes,   F16, PassThrough, PassThrough, OutElementOp,       ConvSpec, GemmMNKPadding,        1,   128,   128,   128,    32,   4,   8,   32,   32,    4,    2,     S<8, 16, 1>,     S<0, 2, 1>,     S<0, 2, 1>, 1,              8,              4,         1,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,         1,           1,           1,               S<1, 16, 1, 8>,               1>
    ,ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NDimSpatial,ALayout,BLayout,    DsLayout,ELayout,   F16,   F16,     F32,      F16,    DsDataTypes,   F16, PassThrough, PassThrough, OutElementOp,       ConvSpec, GemmMNKPadding,        1,   256,   128,   128,    32,   4,   8,   32,   32,    2,    2,     S<8, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>, 1,              4,              4,         1,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,         1,           1,           1,               S<1, 32, 1, 8>,               1>
    ,ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NDimSpatial,ALayout,BLayout,    DsLayout,ELayout,   F16,   F16,     F32,      F16,    DsDataTypes,   F16, PassThrough, PassThrough, OutElementOp,       ConvSpec, GemmMNKPadding,        1,   128,   128,    64,    32,   4,   8,   32,   32,    2,    2,     S<8, 16, 1>,     S<0, 2, 1>,     S<0, 2, 1>, 1,              8,              4,         1,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,         1,           1,           1,               S<1, 32, 1, 4>,               1>
    ,ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NDimSpatial,ALayout,BLayout,    DsLayout,ELayout,   F16,   F16,     F32,      F16,    DsDataTypes,   F16, PassThrough, PassThrough, OutElementOp,       ConvSpec, GemmMNKPadding,        1,   128,    64,   128,    32,   4,   8,   32,   32,    2,    2,     S<8, 16, 1>,     S<0, 2, 1>,     S<0, 2, 1>, 1,              4,              4,         1,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,         1,           1,           1,               S<1, 16, 1, 8>,               1>
    ,ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NDimSpatial,ALayout,BLayout,    DsLayout,ELayout,   F16,   F16,     F32,      F16,    DsDataTypes,   F16, PassThrough, PassThrough, OutElementOp,       ConvSpec, GemmMNKPadding,        1,    64,    64,    64,    32,   4,   8,   32,   32,    2,    2,     S<8, 8, 1>,      S<0, 2, 1>,     S<0, 2, 1>, 1,              4,              4,         1,     S<4, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,         1,           1,           1,               S<1, 16, 1, 4>,               1>
    ,ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NDimSpatial,ALayout,BLayout,    DsLayout,ELayout,   F16,   F16,     F32,      F16,    DsDataTypes,   F16, PassThrough, PassThrough, OutElementOp,       ConvSpec, GemmMNKPadding,        1,   256,   128,    64,    32,   4,   8,   32,   32,    2,    1,     S<8, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>, 1,              4,              4,         1,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,         1,           1,           1,               S<1, 32, 1, 8>,               1>
    ,ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NDimSpatial,ALayout,BLayout,    DsLayout,ELayout,   F16,   F16,     F32,      F16,    DsDataTypes,   F16, PassThrough, PassThrough, OutElementOp,       ConvSpec, GemmMNKPadding,        1,   256,    64,   128,    32,   2,   8,   32,   32,    1,    2,     S<16,16, 1>,     S<0, 2, 1>,     S<0, 2, 1>, 1,              4,              2,         1,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,         1,           1,           1,               S<1, 32, 1, 8>,               1>
    ,ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NDimSpatial,ALayout,BLayout,    DsLayout,ELayout,   F16,   F16,     F32,      F16,    DsDataTypes,   F16, PassThrough, PassThrough, OutElementOp,       ConvSpec, GemmMNKPadding,        1,   128,   128,    32,    32,   4,   8,   32,   32,    2,    1,     S<8, 16, 1>,     S<0, 2, 1>,     S<0, 2, 1>, 1,              8,              4,         1,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,         1,           1,           1,               S<1, 32, 1, 4>,               1>
    ,ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NDimSpatial,ALayout,BLayout,    DsLayout,ELayout,   F16,   F16,     F32,      F16,    DsDataTypes,   F16, PassThrough, PassThrough, OutElementOp,       ConvSpec, GemmMNKPadding,        1,   128,    32,   128,    32,   4,   8,   32,   32,    1,    2,     S<8, 16, 1>,     S<0, 2, 1>,     S<0, 2, 1>, 1,              2,              4,         1,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,         1,           1,           1,               S<1, 16, 1, 8>,               1>
    ,ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NDimSpatial,ALayout,BLayout,    DsLayout,ELayout,   F16,   F16,     F32,      F16,    DsDataTypes,   F16, PassThrough, PassThrough, OutElementOp,       ConvSpec, GemmMNKPadding,        1,    64,    64,    32,    32,   4,   8,   32,   32,    2,    1,     S<8, 8,  1>,     S<0, 2, 1>,     S<0, 2, 1>, 1,              4,              4,         1,     S<4, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,         1,           1,           1,               S<1, 16, 1, 4>,               1>
    ,ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NDimSpatial,ALayout,BLayout,    DsLayout,ELayout,   F16,   F16,     F32,      F16,    DsDataTypes,   F16, PassThrough, PassThrough, OutElementOp,       ConvSpec, GemmMNKPadding,        1,    64,    32,    64,    32,   4,   8,   32,   32,    1,    2,     S<8, 8, 1>,     S<0, 2, 1>,     S<0, 2, 1>, 1,               4,              4,         1,     S<4, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,         1,           1,           1,               S<1, 16, 1, 4>,               1>

    ,ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NDimSpatial,ALayout,BLayout,    DsLayout,ELayout,   F16,   F16,     F32,      F16,    DsDataTypes,   F16, PassThrough, PassThrough, OutElementOp,       ConvSpec, GemmMNKPadding,        1,    256,    64,    64,    32,   4,   8,  16,   16,    2,    2,     S<8, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>, 1,              1,              4,         1,     S<4, 64, 1>,     S<1, 0, 2>,    S<1, 0, 2>,              2,              2,              8,          1,          1,           1,               S<1, 32, 1, 4>,               1>
    ,ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NDimSpatial,ALayout,BLayout,    DsLayout,ELayout,   F16,   F16,     F32,      F16,    DsDataTypes,   F16, PassThrough, PassThrough, OutElementOp,       ConvSpec, GemmMNKPadding,        1,    256,    64,    64,    32,   4,   8,  16,   16,    2,    2,     S<8, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>, 1,              2,              4,         1,     S<4, 64, 1>,     S<1, 0, 2>,    S<1, 0, 2>,              2,              1,              8,          1,          1,           1,               S<1, 32, 1, 4>,               1>
    //DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NDimSpatial,ALayout,BLayout,    DsLayout,ELayout,   F16,   F16,     F32,      F16,    DsDataTypes,   F16, PassThrough, PassThrough, OutElementOp,       ConvSpec, GemmMNKPadding,        1,    256,    64,    64,    32,   8,   8,  16,   16,    2,    2,     S<8, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>, 1,              4,              8,         1,     S<4, 64, 1>,     S<1, 0, 2>,    S<1, 0, 2>,              2,              4,              8,          1,          1,           1,               S<1, 32, 1, 4>,               4>,
    //DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NDimSpatial,ALayout,BLayout,    DsLayout,ELayout,   F16,   F16,     F32,      F16,    DsDataTypes,   F16, PassThrough, PassThrough, OutElementOp,       ConvSpec, GemmMNKPadding,        1,    256,    64,    64,    32,   8,   8,  16,   16,    2,    2,     S<8, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>, 1,              8,              8,         1,     S<4, 64, 1>,     S<1, 0, 2>,    S<1, 0, 2>,              2,              8,              8,          1,          1,           1,               S<1, 32, 1, 4>,               8>
  
>;
#endif
#include "run_grouped_conv_fwd_example.inc"

int main(int argc, char* argv[]) { return !run_grouped_conv_fwd_example(argc, argv); }
