// Copyright (c) Advanced Micro Devices, Inc. or its affiliates. 
// SPDX-License-Identifier: MIT

#pragma once

namespace ck_tile {

// This class is used for codegen pattern matching
enum class BlockFmhaBwdPipelineEnum
{
    KRKTRVR_IGLP = 0,
    KRKTRVR,
};

} // namespace ck_tile
