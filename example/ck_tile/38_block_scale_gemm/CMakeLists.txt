include(GpuTargetsMatchAllowedList)

set(EXAMPLE_GEMM_COMPILE_OPTIONS)
if(CK_USE_OCP_FP8)
  list(APPEND EXAMPLE_GEMM_COMPILE_OPTIONS -DCK_TILE_USE_OCP_FP8)
endif()

list(APPEND EXAMPLE_GEMM_COMPILE_OPTIONS -mllvm -enable-noalias-to-md-conversion=0)

ContainsOnlySpecifiedGpuTargets("${SUPPORTED_GPU_TARGETS}" "gfx942" is_only_gfx942)
ContainsOnlySpecifiedGpuTargets("${SUPPORTED_GPU_TARGETS}" "gfx950" is_only_gfx950) 

if(is_only_gfx942 OR is_only_gfx950)
    add_executable(tile_example_gemm_aquant_basic EXCLUDE_FROM_ALL gemm_aquant_basic.cpp)
    target_compile_options(tile_example_gemm_aquant_basic PRIVATE ${EXAMPLE_GEMM_COMPILE_OPTIONS})
else()
    message(DEBUG "Unit tests for ck_tile quantization GEMM have been omitted due to compatibility "
        "constraints. The ck_tile quantization GEMM test suite exclusively supports gfx942 and gfx950 "
        "targets, while the current target configuration includes architectures that are not supported "
        "by this test.")
endif()
