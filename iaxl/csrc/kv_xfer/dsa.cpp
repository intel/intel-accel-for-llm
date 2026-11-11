// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "kv_pool.h"
#include "kv_xfer.h"
#include "env.h"
#include "iaxl_common.h"

extern "C" {
#include "dsa_gd.h"
}

extern "C" int64_t dsa_memcpy_batch_async(void *const dest[], const void *const src[],
                                          const size_t n[], size_t count);
extern "C" int dsa_memcpy_batch_wait(uint64_t ticket);

namespace kv_xfer {

using kv_pool::KVPool;
using kv_pool::PoolDevice;

// Serialises batch submission and mapping release; dsa_memcpy_batch_wait() needs no lock.
static std::mutex dsa_context_mutex;

// One GDRCopy mapping per KV pool base: re-pinning on every copy overflowed libgdrapi's handles.
struct DsaRegion {
    void *bar_base = nullptr;
    int refs = 0;
};
static std::mutex dsa_region_mutex;
static std::unordered_map<uintptr_t, DsaRegion> dsa_regions; // by KV pool base

// kv_pool hooks: called with the registry lock held, so never call back into kv_pool here.
static void dsa_on_register(const KVPool &p) {
    if (!envs.IAXL_DSA_GD_ENABLE || p.device != PoolDevice::GPU)
        return;
    std::lock_guard<std::mutex> lock(dsa_region_mutex);
    auto [it, inserted] = dsa_regions.try_emplace(p.base);
    it->second.refs++;
    if (!inserted)
        return;
    void *bar_base = nullptr;
    const bool mapped = dsa_gd_default_register(p.base, p.total_bytes) == 0 &&
                        dsa_gd_default_gpu_bar_addr(p.base, p.total_bytes, &bar_base) == 0;
    if (!mapped)
        fprintf(stderr,
                "dsa: GDRCopy cannot map KV pool %s (%zu bytes); the GPU BAR1 may be smaller "
                "than the KV cache. Set IAXL_DSA_GD_ENABLE=0 to copy with CUDA instead.\n",
                p.name.c_str(), p.total_bytes);
    IAXL_CHECK(mapped, "dsa: KV pool registration failed");
    it->second.bar_base = bar_base;
}

static void dsa_on_unregister(const KVPool &p) {
    if (!envs.IAXL_DSA_GD_ENABLE || p.device != PoolDevice::GPU)
        return;
    std::lock_guard<std::mutex> lock(dsa_region_mutex);
    auto it = dsa_regions.find(p.base);
    IAXL_CHECK(it != dsa_regions.end(), "dsa: GPU KV pool was not registered");
    if (--it->second.refs > 0)
        return;
    // Wait for in-flight DSA batches before the mapping goes away.
    std::lock_guard<std::mutex> submit_lock(dsa_context_mutex);
    dsa_memcpy_batch_wait(UINT64_MAX);
    IAXL_CHECK(dsa_gd_default_unregister(p.base, p.total_bytes) == 0,
               "dsa: KV pool mapping not found");
    dsa_regions.erase(it);
}

void dsa_install_pool_hooks() { kv_pool::add_pool_hooks({dsa_on_register, dsa_on_unregister}); }

void *dsa_gpu_bar_base(char *gpu_base) {
    std::lock_guard<std::mutex> lock(dsa_region_mutex);
    auto it = dsa_regions.find(reinterpret_cast<uintptr_t>(gpu_base));
    if (it == dsa_regions.end())
        return nullptr;
    return it->second.bar_base;
}

bool dsa_copy_chunks_batch(char *gpu_bar_base, int64_t chunk_stride, int64_t outer_dims,
                           int64_t inner_size, int64_t outer_block_size, bool is_h2d,
                           const std::vector<int64_t> &chunk_indices,
                           const std::vector<char *> &cpu_ptrs, uint64_t &ticket) {

    if (gpu_bar_base == nullptr) // IAXL_DSA_GD_ENABLE=0: the KV pool has no BAR mapping
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

    std::vector<void *> dest(count), src(count);
    std::vector<size_t> nbytes(count);

    size_t k = 0;
    for (size_t i = 0; i < num_chunks; ++i) {
        const int64_t gpu_chunk_off = chunk_indices[i] * chunk_stride;
        char *cpu_chunk = cpu_ptrs[i];
        for (int64_t o = 0; o < outer_dims; ++o) {
            char *gpu_bar = gpu_bar_base + gpu_chunk_off + o * outer_block_size;
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

    const int64_t t = dsa_memcpy_batch_async(
        dest.data(), const_cast<const void *const *>(src.data()), nbytes.data(), count);
    if (t < 0)
        return false;
    // Tickets are monotonic, so remembering the latest one is enough to cover every batch
    // this context has issued.
    ticket = static_cast<uint64_t>(t);
    return true;
}

void dsa_wait_ticket(uint64_t ticket) {
    if (!envs.IAXL_DSA_GD_ENABLE || ticket == 0)
        return;
    IAXL_CHECK(dsa_memcpy_batch_wait(ticket) == 0, "dsa_wait_ticket: DSA batch failed");
}

} // namespace kv_xfer
