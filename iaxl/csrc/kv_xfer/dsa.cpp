// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <vector>

#include "kv_xfer.h"
#include "env.h"
#include "iaxl_common.h"

extern "C" {
#include "dsa_gd.h"
}

extern "C" int dsa_memcpy_batch_async(void *const dest[], const void *const src[], const size_t n[],
                                      size_t count);
extern "C" int dsa_memcpy_batch_wait(void);

namespace kv_xfer {

static std::mutex dsa_context_mutex;

void dsa_context_reset() {
    std::lock_guard<std::mutex> lock(dsa_context_mutex);
    dsa_gd_default_reset();
}

bool dsa_copy_chunks_batch(char *gpu_base, int64_t chunk_stride, int64_t outer_dims,
                           int64_t inner_size, int64_t outer_block_size, bool is_h2d,
                           const std::vector<int64_t> &chunk_indices,
                           const std::vector<char *> &cpu_ptrs) {

    if (!envs.IAXL_DSA_GD_ENABLE)
        return false;

    std::lock_guard<std::mutex> lock(dsa_context_mutex);

    if ((inner_size % 8) != 0)
        return false;

    const size_t num_chunks = chunk_indices.size();
    if (num_chunks != cpu_ptrs.size())
        return false;
    const size_t count = num_chunks * static_cast<size_t>(outer_dims);
    if (count == 0)
        return true;

    size_t span = 0;
    for (size_t i = 0; i < num_chunks; ++i) {
        const size_t end = static_cast<size_t>(chunk_indices[i]) * chunk_stride +
                           static_cast<size_t>(outer_dims - 1) * outer_block_size +
                           static_cast<size_t>(inner_size);
        if (end > span)
            span = end;
    }
    if (dsa_gd_default_register(reinterpret_cast<uint64_t>(gpu_base), span) != 0)
        return false;

    void *gpu_bar_base = nullptr;
    if (dsa_gd_default_gpu_bar_addr(reinterpret_cast<uint64_t>(gpu_base),
                                    static_cast<size_t>(inner_size), &gpu_bar_base) != 0)
        return false;

    std::vector<void *> dest(count), src(count);
    std::vector<size_t> nbytes(count);

    size_t k = 0;
    for (size_t i = 0; i < num_chunks; ++i) {
        const int64_t gpu_chunk_off = chunk_indices[i] * chunk_stride;
        char *cpu_chunk = cpu_ptrs[i];
        for (int64_t o = 0; o < outer_dims; ++o) {
            char *gpu_bar =
                static_cast<char *>(gpu_bar_base) + gpu_chunk_off + o * outer_block_size;
            char *cpu_ptr = cpu_chunk + o * inner_size;
            if (is_h2d) {
                dest[k] = gpu_bar;
                src[k] = cpu_ptr;
            } else {
                dest[k] = cpu_ptr;
                src[k] = gpu_bar;
            }
            nbytes[k] = static_cast<size_t>(inner_size);
            ++k;
        }
    }

    return dsa_memcpy_batch_async(dest.data(), const_cast<const void *const *>(src.data()),
                                  nbytes.data(), count) == 0;
}

void dsa_copy_wait(context_t) {
    if (!envs.IAXL_DSA_GD_ENABLE)
        return;
    std::lock_guard<std::mutex> lock(dsa_context_mutex);
    IAXL_CHECK(dsa_memcpy_batch_wait() == 0, "dsa_copy_wait: DSA batch failed");
}

} // namespace kv_xfer
