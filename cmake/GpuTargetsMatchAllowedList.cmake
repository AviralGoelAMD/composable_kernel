# SPDX-License-Identifier: MIT
# Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

function(ContainsOnlyOneTarget GPU_TARGETS result)
    list(LENGTH GPU_TARGETS target_count)
    if(target_count EQUAL 1)
        set(${result} TRUE PARENT_SCOPE)
    else()
        set(${result} FALSE PARENT_SCOPE)
    endif()
endfunction()

function(ContainsOnlySpecifiedGpuTargets GPU_TARGETS AllowedList result)
    string(REPLACE " " ";" allowed_list "${AllowedList}")

    # Remove any empty elements
    list(REMOVE_ITEM allowed_list "")

    # Check if all elements in allowed_list are in GPU_TARGETS
    foreach(target ${GPU_TARGETS})
        if(NOT target IN_LIST allowed_list)
            set(${result} FALSE PARENT_SCOPE)
            return()
        endif()
    endforeach()

    set(${result} TRUE PARENT_SCOPE)
endfunction()

