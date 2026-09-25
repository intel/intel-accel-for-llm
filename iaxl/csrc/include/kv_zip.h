// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace kv_zip {

// One KV block inside an inference tensor: `outer_dims` runs of `inner_size` contiguous bytes,
// `outer_block_size` bytes apart. A whole scratch tensor is the special case outer_dims == 1.
struct ChunkView {
    char *base;
    int64_t outer_dims;
    int64_t inner_size;
    int64_t outer_block_size;
    int element_size;
    bool is_bf16;

    size_t nbytes() const { return static_cast<size_t>(outer_dims) * inner_size; }
};

struct CopySegment {
    char *dst;
    const char *src;
    size_t n;
};

// Copies every segment, through Intel DSA when IAXL_DSA_MEMCPY_ENABLE is set and the hardware
// accepts the batch, otherwise with memcpy (spread over the codec team when `parallel`).
void copy_segments(const std::vector<CopySegment> &segments, bool parallel);

// Compresses each view straight from the inference tensor into the codec staging buffer and
// packs the result into freshly malloc'd cache buffers (header + payload).
void kv_zip_compress_views(const std::vector<ChunkView> &views, std::vector<char *> &out_bufs,
                           std::vector<size_t> &out_sizes, std::vector<size_t> &orig_sizes,
                           bool compress = true);

// Decompresses cache payloads and scatters them straight into the inference tensor views.
void kv_zip_decompress_views(const std::vector<const char *> &data_ptrs,
                             const std::vector<ChunkView> &views);

// Scratch-tensor convenience wrappers around the view API (used by the GPU transfer path).
void kv_zip_compress_batch(const std::vector<torch::Tensor> &tensors, std::vector<char *> &out_bufs,
                           std::vector<size_t> &out_sizes, std::vector<size_t> &orig_sizes,
                           bool compress = true);

void kv_zip_decompress_batch(const std::vector<const char *> &data_ptrs,
                             const std::vector<torch::Tensor> &tensors);

} // namespace kv_zip
