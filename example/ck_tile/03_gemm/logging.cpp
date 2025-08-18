// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

#include <iostream>
#include <iomanip>
#include "logging.hpp"

extern "C" void ck_tile_log_kbatch(long kbatch)
{
    (void)kbatch;
    // std::cout << "k_batch: " << kbatch << std::endl;
}

extern "C" bool ck_tile_log_is_enabled()
{
    // Toggle globally here (env var, static flag, etc.). For now: always on.
    return true;
}

extern "C" void ck_tile_log_gemm_cli_args(const char* data_type,
                                          const char* a_layout,
                                          const char* b_layout)
{
    CK_LOG_STR("data_type", data_type);
    CK_LOG_STR("a_layout", a_layout);
    CK_LOG_STR("b_layout", b_layout);
}

extern "C" void ck_tile_log_i64(const char* tag,
                                 long value,
                                 const char* file,
                                 int line,
                                 const char* func)
{
    std::cout << "[i64] " << (tag ? tag : "")
              << " = " << value
              << " (" << (file ? file : "?") << ":" << line << ", " << (func ? func : "?")
              << ")" << std::endl;
}

extern "C" void ck_tile_log_f64(const char* tag,
                                 double value,
                                 const char* file,
                                 int line,
                                 const char* func)
{
    std::cout << std::fixed << std::setprecision(6)
              << "[f64] " << (tag ? tag : "")
              << " = " << value
              << " (" << (file ? file : "?") << ":" << line << ", " << (func ? func : "?")
              << ")" << std::endl;
}

extern "C" void ck_tile_log_str(const char* tag,
                                 const char* value,
                                 const char* file,
                                 int line,
                                 const char* func)
{
    std::cout << "[str] " << (tag ? tag : "")
              << " = " << (value ? value : "")
              << " (" << (file ? file : "?") << ":" << line << ", " << (func ? func : "?")
              << ")" << std::endl;
}


