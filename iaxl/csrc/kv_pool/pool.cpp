// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <torch/extension.h>

#include <mutex>
#include <unordered_map>

#include "iaxl_common.h"
#include "kv_pool.h"

namespace kv_pool {

// ---- KVPool ----

void KVPool::check_indices(const std::vector<int64_t> &idx) const {
    for (int64_t i : idx)
        IAXL_CHECK(i >= 0 && i < num_blocks, "block index out of range");
}

void KVPool::check_same_block_geometry(const KVPool &other) const {
    IAXL_CHECK(block_bytes == other.block_bytes && elem_size == other.elem_size &&
                   dtype == other.dtype,
               "pools have different block geometry");
}

void KVPool::check_staging_geometry(const KVPool &staging) const {
    IAXL_CHECK(staging.block_bytes >= block_bytes && elem_size == staging.elem_size &&
                   dtype == staging.dtype,
               "staging pool is smaller than the KV block");
}

std::vector<char *> KVPool::block_ptrs(const std::vector<int64_t> &idx) const {
    std::vector<char *> ptrs(idx.size());
    for (size_t i = 0; i < idx.size(); i++)
        ptrs[i] = block_ptr(idx[i]);
    return ptrs;
}

KVPool make_kv_pool(uintptr_t base, PoolDevice device, int dev_id,
                    const std::vector<int64_t> &shape, int64_t elem_size, int32_t dtype,
                    int chunk_dim, int64_t chunk_stride) {
    IAXL_CHECK(chunk_dim >= 0 && static_cast<size_t>(chunk_dim) < shape.size(),
               "chunk_dim out of range");
    KVPool p;
    p.base = base;
    p.device = device;
    p.dev_id = dev_id;
    p.shape = shape;
    p.chunk_dim = chunk_dim;
    p.elem_size = elem_size;
    p.dtype = dtype;
    p.num_blocks = shape[chunk_dim];
    p.outer_dims = 1;
    for (int d = 0; d < chunk_dim; d++)
        p.outer_dims *= shape[d];
    p.inner_size = elem_size;
    for (size_t d = chunk_dim + 1; d < shape.size(); d++)
        p.inner_size *= shape[d];
    p.chunk_stride = chunk_stride > 0 ? chunk_stride : p.inner_size;
    // For a packed (strided) 2-D cache the gap belongs to the next page; for
    // outer-dimension layouts the segment blocks stay contiguous.
    p.outer_block_size = p.num_blocks * (p.outer_dims == 1 ? p.chunk_stride : p.inner_size);
    p.block_bytes = p.outer_dims * p.inner_size;
    p.total_bytes = static_cast<size_t>(p.outer_dims) * p.outer_block_size;
    return p;
}

// ---- registry ----

namespace {

struct Registry {
    std::mutex mu;
    std::unordered_map<std::string, KVPool> pools;
    std::unordered_map<std::string, torch::Tensor> tensors; // keeps CPU/GPU pools alive
    PoolHooks hooks;
};

Registry &registry() {
    static Registry r;
    return r;
}

bool same_geometry(const KVPool &a, const KVPool &b) {
    return a.base == b.base && a.device == b.device && a.shape == b.shape &&
           a.chunk_dim == b.chunk_dim && a.elem_size == b.elem_size && a.dtype == b.dtype &&
           a.chunk_stride == b.chunk_stride && a.block_bytes == b.block_bytes;
}

const KVPool &register_with_tensor(KVPool pool, const torch::Tensor *t) {
    auto &r = registry();
    std::lock_guard<std::mutex> lock(r.mu);
    auto it = r.pools.find(pool.name);
    if (it != r.pools.end()) {
        if (!same_geometry(it->second, pool))
            fprintf(stderr, "kv_pool \"%s\" already registered with a different base/shape\n",
                    pool.name.c_str());
        IAXL_CHECK(same_geometry(it->second, pool), "pool re-registered with a different tensor");
        return it->second;
    }
    for (auto &[_, other] : r.pools)
        if (other.base == pool.base) {
            if (pool.device == PoolDevice::GPU) {
                // Packed overlay: MLA and indexer views can share a block base
                // while covering different page sizes.
                IAXL_CHECK(other.chunk_dim == pool.chunk_dim &&
                               other.elem_size == pool.elem_size,
                           "GPU views sharing one buffer must have the same element geometry");
            } else {
                IAXL_CHECK(other.shape == pool.shape && other.chunk_dim == pool.chunk_dim &&
                               other.elem_size == pool.elem_size,
                           "pools sharing one buffer must have the same geometry");
            }
        }
    if (t)
        r.tensors.emplace(pool.name, *t);
    const KVPool &p = r.pools.emplace(pool.name, std::move(pool)).first->second;
    if (r.hooks.on_register)
        r.hooks.on_register(p);
    return p;
}

} // namespace

const KVPool &register_kv_pool(KVPool pool) { return register_with_tensor(std::move(pool), nullptr); }

const KVPool &get_kv_pool(std::string_view name) {
    const KVPool *p = find_kv_pool(name);
    if (!p)
        fprintf(stderr, "KVPool '%.*s' is not registered\n", (int)name.size(), name.data());
    IAXL_CHECK(p, "KVPool not registered");
    return *p;
}

const KVPool *find_kv_pool(std::string_view name) {
    auto &r = registry();
    std::lock_guard<std::mutex> lock(r.mu);
    auto it = r.pools.find(std::string(name));
    return it == r.pools.end() ? nullptr : &it->second;
}

const KVPool *find_kv_pool_by_base(uintptr_t base) {
    auto &r = registry();
    std::lock_guard<std::mutex> lock(r.mu);
    for (auto &[_, p] : r.pools)
        if (p.base == base)
            return &p;
    return nullptr;
}

void unregister_kv_pool(std::string_view name) {
    auto &r = registry();
    std::lock_guard<std::mutex> lock(r.mu);
    auto it = r.pools.find(std::string(name));
    if (it == r.pools.end())
        return;
    if (r.hooks.on_unregister)
        r.hooks.on_unregister(it->second);
    r.pools.erase(it);
    r.tensors.erase(std::string(name));
}

void set_pool_hooks(const PoolHooks &hooks) {
    auto &r = registry();
    std::lock_guard<std::mutex> lock(r.mu);
    r.hooks = hooks;
}

// ---- torch side ----

int64_t tensor_bytes(const torch::Tensor &t) { return t.numel() * t.element_size(); }

KVPool kv_pool_from_tensor(const torch::Tensor &t, PoolDevice device, int chunk_dim,
                           int64_t chunk_stride) {
    if (chunk_stride > 0) {
        IAXL_CHECK(t.stride(chunk_dim) * t.element_size() == chunk_stride,
                   "KVPool chunk_stride does not match the tensor stride");
        int64_t expected = t.element_size();
        for (int d = t.dim() - 1; d > chunk_dim; d--) {
            IAXL_CHECK(t.stride(d) == expected, "KVPool tensor inner dims must be contiguous");
            expected *= t.size(d);
        }
    } else {
        IAXL_CHECK(t.is_contiguous(), "a KVPool tensor must be contiguous");
    }
    KVPool p = make_kv_pool(reinterpret_cast<uintptr_t>(t.data_ptr()), device, t.device().index(),
                            t.sizes().vec(), t.element_size(),
                            static_cast<int32_t>(t.scalar_type()), chunk_dim, chunk_stride);
    p.pinned = t.is_cpu() && t.is_pinned();
    return p;
}

const KVPool &register_kv_pool_cpu(std::string name, const torch::Tensor &t, int chunk_dim) {
    IAXL_CHECK(t.is_cpu(), "register_kv_pool_cpu: tensor is not on the CPU");
    KVPool p = kv_pool_from_tensor(t, PoolDevice::CPU, chunk_dim);
    p.name = std::move(name);
    return register_with_tensor(std::move(p), &t);
}

const KVPool &register_kv_pool_gpu(std::string name, const torch::Tensor &t, int chunk_dim,
                                   int64_t chunk_stride) {
    IAXL_CHECK(!t.is_cpu(), "register_kv_pool_gpu: tensor is on the CPU");
    KVPool p = kv_pool_from_tensor(t, PoolDevice::GPU, chunk_dim, chunk_stride);
    p.name = std::move(name);
    return register_with_tensor(std::move(p), &t);
}

const KVPool &register_kv_pool_remote(std::string name, const std::string &peer, uintptr_t base,
                                      const std::vector<int64_t> &shape, int64_t elem_size,
                                      int32_t dtype, int dev_id, int chunk_dim) {
    KVPool p = make_kv_pool(base, PoolDevice::Remote, dev_id, shape, elem_size, dtype, chunk_dim);
    p.name = std::move(name);
    p.peer = peer;
    return register_kv_pool(std::move(p));
}

torch::Tensor block_tensor(const torch::Tensor &t, const KVPool &pool, int64_t idx) {
    IAXL_CHECK(reinterpret_cast<uintptr_t>(t.data_ptr()) == pool.base,
               "tensor is not the one the pool was registered with");
    IAXL_CHECK(idx >= 0 && idx < pool.num_blocks, "block index out of range");
    return t.select(pool.chunk_dim, idx);
}

std::vector<torch::Tensor> block_tensors(const torch::Tensor &t, const KVPool &pool,
                                         const std::vector<int64_t> &idx) {
    std::vector<torch::Tensor> out;
    out.reserve(idx.size());
    for (int64_t i : idx)
        out.push_back(block_tensor(t, pool, i));
    return out;
}

torch::Tensor block_tensor(const KVPool &pool, int64_t idx) {
    auto &r = registry();
    torch::Tensor t;
    {
        std::lock_guard<std::mutex> lock(r.mu);
        auto it = r.tensors.find(pool.name);
        IAXL_CHECK(it != r.tensors.end(), "pool has no local tensor");
        t = it->second;
    }
    return block_tensor(t, pool, idx);
}

int64_t block_index_of(const KVPool &pool, const torch::Tensor &block) {
    IAXL_CHECK(pool.contains(block.data_ptr()), "tensor is not a block of this pool");
    return pool.block_index(block.data_ptr());
}

std::vector<int64_t> block_indices_of(const KVPool &pool, const std::vector<torch::Tensor> &blocks) {
    std::vector<int64_t> out(blocks.size());
    for (size_t i = 0; i < blocks.size(); i++)
        out[i] = block_index_of(pool, blocks[i]);
    return out;
}

} // namespace kv_pool
