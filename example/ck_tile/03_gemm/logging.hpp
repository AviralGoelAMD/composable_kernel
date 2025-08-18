// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

extern "C" void ck_tile_log_kbatch(long kbatch);

extern "C" void ck_tile_log_i64(const char* tag,
                                 long value,
                                 const char* file,
                                 int line,
                                 const char* func);

extern "C" void ck_tile_log_f64(const char* tag,
                                 double value,
                                 const char* file,
                                 int line,
                                 const char* func);

extern "C" void ck_tile_log_str(const char* tag,
                                 const char* value,
                                 const char* file,
                                 int line,
                                 const char* func);

extern "C" bool ck_tile_log_is_enabled();

#define CK_LOG_I64(TAG, VALUE) ck_tile_log_i64(TAG, static_cast<long>(VALUE), __FILE__, __LINE__, __func__)
#define CK_LOG_F64(TAG, VALUE) ck_tile_log_f64(TAG, static_cast<double>(VALUE), __FILE__, __LINE__, __func__)
#define CK_LOG_STR(TAG, VALUE) ck_tile_log_str(TAG, VALUE, __FILE__, __LINE__, __func__)

// X-macro helpers to keep call-sites clean: define a list and apply once
#define CK_LOG_APPLY_ONE(KIND, TAG, VALUE) CK_LOG_##KIND(TAG, VALUE)
#define CK_LOG_LIST(LIST_MACRO)                                                                    \
    do {                                                                                           \
        if(ck_tile_log_is_enabled()) { LIST_MACRO(CK_LOG_APPLY_ONE); }                             \
    } while(false)

// Clean call-site for common contexts
extern "C" void ck_tile_log_gemm_cli_args(const char* data_type,
                                          const char* a_layout,
                                          const char* b_layout);

#define CK_LOG_GEMM_CLI_ARGS(DATA_TYPE, A_LAYOUT, B_LAYOUT)                                        \
    do {                                                                                           \
        if(ck_tile_log_is_enabled())                                                               \
            ck_tile_log_gemm_cli_args((DATA_TYPE).c_str(), (A_LAYOUT).c_str(), (B_LAYOUT).c_str()); \
    } while(false)


