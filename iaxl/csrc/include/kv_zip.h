// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "kv_pool.h"

namespace kv_zip {

// Compress block block_indices[i] of the CPU pool into out_bufs[i] (malloc'ed).
void kv_zip_compress_batch(const kv_pool::KVPool &pool, const std::vector<int64_t> &block_indices,
                           std::vector<char *> &out_bufs, std::vector<size_t> &out_sizes,
                           std::vector<size_t> &orig_sizes, bool compress = true);

// Decompress data_ptrs[i] into block block_indices[i] of the CPU pool.
void kv_zip_decompress_batch(const std::vector<const char *> &data_ptrs,
                             const kv_pool::KVPool &pool,
                             const std::vector<int64_t> &block_indices);

} // namespace kv_zip
