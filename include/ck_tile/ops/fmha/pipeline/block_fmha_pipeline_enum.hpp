// Copyright (c) Advanced Micro Devices, Inc. or its affiliates. 
// SPDX-License-Identifier: MIT

#pragma once

namespace ck_tile {

// This class is used for codegen pattern matching
enum class BlockFmhaPipelineEnum
{
    QRKSVS = 0,
    QRKSVS_ASYNC,
    QSKSVS,
};

template <BlockFmhaPipelineEnum>
struct BlockFmhaPipelineEnumToStr;

template <>
struct BlockFmhaPipelineEnumToStr<BlockFmhaPipelineEnum::QRKSVS>
{
    static constexpr const char* name = "qr";
};
template <>
struct BlockFmhaPipelineEnumToStr<BlockFmhaPipelineEnum::QRKSVS_ASYNC>
{
    static constexpr const char* name = "qr_async";
};
template <>
struct BlockFmhaPipelineEnumToStr<BlockFmhaPipelineEnum::QSKSVS>
{
    static constexpr const char* name = "qs";
};

} // namespace ck_tile
