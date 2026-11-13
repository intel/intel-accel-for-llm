// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include <c10/core/ScalarType.h>

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

// Pool-free entry points for equally sized contiguous CPU buffers of one dtype.
// Compress in_bufs[i] (nbytes each) into out_bufs[i] (malloc'ed, out_sizes[i] bytes); the output
// vectors are resized. Like the pool path, a buffer split into several segments
// (IAXL_ZIP_SEGMENT_SIZE_KB) is used as scratch and holds garbage afterwards.
void kv_zip_compress_tensors(const std::vector<char *> &in_bufs, size_t nbytes,
                             at::ScalarType dtype, std::vector<char *> &out_bufs,
                             std::vector<size_t> &out_sizes);
// Decompress block in_bufs[i] into out_bufs[i] (nbytes each). A block carries its own size.
void kv_zip_decompress_tensors(const std::vector<const char *> &in_bufs,
                               const std::vector<char *> &out_bufs, size_t nbytes,
                               at::ScalarType dtype);
// The same for tensors, which must be contiguous CPU tensors of one dtype and byte size.
void kv_zip_compress_tensors(const std::vector<torch::Tensor> &tensors,
                             std::vector<char *> &out_bufs, std::vector<size_t> &out_sizes);
void kv_zip_decompress_tensors(const std::vector<const char *> &in_bufs,
                               const std::vector<torch::Tensor> &tensors);

} // namespace kv_zip
