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

// Caller-owned memory for the functions below, which throw std::invalid_argument on bad input
// instead of aborting.
struct ConstBytes {
    const char *data;
    size_t size;
};

struct Bytes {
    char *data;
    size_t size;
};

// Largest block a tensor of orig_size bytes compresses into.
size_t kv_zip_max_block_size(size_t orig_size);

// Compress srcs[i] into one block in dsts[i], which must hold kv_zip_max_block_size(srcs[i].size)
// bytes and not overlap the source, and return the block sizes. Unlike kv_zip_compress_batch, it
// leaves the sources unchanged. elem_size sets the lossy truncation mask; data shuffling applies to
// bf16 only.
std::vector<size_t> kv_zip_compress_into(const std::vector<ConstBytes> &srcs,
                                         const std::vector<Bytes> &dsts, int elem_size, bool bf16);

// Decompress block srcs[i] into dsts[i] and return the tensor sizes. The blocks are checked to lie
// within their bounds before any is decoded.
std::vector<size_t> kv_zip_decompress_into(const std::vector<ConstBytes> &srcs,
                                           const std::vector<Bytes> &dsts);

} // namespace kv_zip
