// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

// The torch-side helpers below only name torch::Tensor; the implementation (pool.cpp)
// and the torch_ext sources include <torch/extension.h> themselves.
namespace at {
class Tensor;
}
namespace torch {
using at::Tensor;
}

namespace kv_pool {

// ---------------------------------------------------------------------------
// KVPool: a registered memory region cut into num_blocks blocks along chunk_dim.
// Block idx consists of outer_dims segments; segment o starts at
// base + o * outer_block_size + idx * chunk_stride and spans inner_size bytes.
// A CPU pool (chunk_dim == 0) has outer_dims == 1, so a block is block_bytes contiguous bytes.
// ---------------------------------------------------------------------------

enum class PoolDevice : uint8_t { CPU, GPU, Remote };

struct KVPool {
    std::string name;   // registry key: KV cache layer -> tensor_key, ScratchPool -> "scratch"
    uintptr_t base = 0;
    PoolDevice device = PoolDevice::CPU;
    int dev_id = -1;    // GPU device index (the client's for Remote)
    std::string peer;   // NIXL agent name (Remote only)
    std::vector<int64_t> shape;
    int chunk_dim = 0;
    int64_t elem_size = 0;
    int32_t dtype = -1; // c10::ScalarType as an integer; this header stays torch-free
    bool pinned = false;

    int64_t num_blocks = 0;       // shape[chunk_dim]
    int64_t outer_dims = 0;       // prod(shape[:chunk_dim])
    int64_t inner_size = 0;       // elem_size * prod(shape[chunk_dim+1:])
    int64_t chunk_stride = 0;     // bytes between neighbouring blocks (== inner_size, contiguous)
    int64_t outer_block_size = 0; // num_blocks * inner_size
    int64_t block_bytes = 0;      // outer_dims * inner_size
    size_t total_bytes = 0;

    // Hot path, no bounds checks.
    char *block_ptr(int64_t idx, int64_t o = 0) const {
        return reinterpret_cast<char *>(base + o * outer_block_size + idx * chunk_stride);
    }
    int64_t block_index(const void *ptr) const {
        return static_cast<int64_t>(((reinterpret_cast<uintptr_t>(ptr) - base) % outer_block_size) /
                                    chunk_stride);
    }
    bool contains(const void *ptr) const {
        auto p = reinterpret_cast<uintptr_t>(ptr);
        return p >= base && p < base + total_bytes;
    }
    bool is_contiguous_blocks() const { return outer_dims == 1 && chunk_stride == inner_size; }

    // Entry-point checks (caller thread), done once per call.
    void check_indices(const std::vector<int64_t> &idx) const;
    void check_same_block_geometry(const KVPool &other) const;

    // ptrs[i] = block_ptr(idx[i]); only at the Ops::copy_chunks_batch boundary.
    std::vector<char *> block_ptrs(const std::vector<int64_t> &idx) const;
};

// The only geometry computation; `shape` is laid out contiguously.
KVPool make_kv_pool(uintptr_t base, PoolDevice device, int dev_id, const std::vector<int64_t> &shape,
                    int64_t elem_size, int32_t dtype, int chunk_dim);

// Process-wide registry keyed by name. Element addresses are stable, so a Context may
// keep a `const KVPool *`. Re-registering a name requires the same base and geometry and
// returns the existing entry; different names may share one base (hybrid layers).
const KVPool &register_kv_pool(KVPool pool);
const KVPool &get_kv_pool(std::string_view name);   // aborts if not registered
const KVPool *find_kv_pool(std::string_view name);  // nullptr if not registered
const KVPool *find_kv_pool_by_base(uintptr_t base);
void unregister_kv_pool(std::string_view name);

// Transport backend callbacks, called for every pool with the registry lock held. Each backend
// adds its own pair: rdma_init() the NIXL one, the DSA backend (CUDA builds) the GDRCopy one.
struct PoolHooks {
    void (*on_register)(const KVPool &) = nullptr;
    void (*on_unregister)(const KVPool &) = nullptr;
};
void add_pool_hooks(const PoolHooks &hooks);

// ---- torch side (pool.cpp) ----
// cpu/gpu: validates contiguity and device, records pinned, keeps `t` alive until unregistered.
const KVPool &register_kv_pool_cpu(std::string name, const torch::Tensor &t, int chunk_dim = 0);
const KVPool &register_kv_pool_gpu(std::string name, const torch::Tensor &t, int chunk_dim);
const KVPool &register_kv_pool_remote(std::string name, const std::string &peer, uintptr_t base,
                                      const std::vector<int64_t> &shape, int64_t elem_size,
                                      int32_t dtype, int dev_id, int chunk_dim);

// Block idx as a view of `t` (the tensor the pool was registered with).
torch::Tensor block_tensor(const torch::Tensor &t, const KVPool &pool, int64_t idx);
std::vector<torch::Tensor> block_tensors(const torch::Tensor &t, const KVPool &pool,
                                         const std::vector<int64_t> &idx);
// Same, using the tensor kept at registration (Remote pools have none and abort).
torch::Tensor block_tensor(const KVPool &pool, int64_t idx);
int64_t block_index_of(const KVPool &pool, const torch::Tensor &block);
std::vector<int64_t> block_indices_of(const KVPool &pool, const std::vector<torch::Tensor> &blocks);
int64_t tensor_bytes(const torch::Tensor &t);
KVPool kv_pool_from_tensor(const torch::Tensor &t, PoolDevice device, int chunk_dim);

inline constexpr char LABEL_SEP = ':';

[[noreturn]] inline void invalid_label(std::string_view value) {
    std::fprintf(stderr, "Invalid KV cache label: %.*s\n", static_cast<int>(value.size()),
                 value.data());
    std::abort();
}

inline void validate_label_component(std::string_view component) {
    if (component.empty() || component == "." || component == ".." ||
        component.find_first_of(":/\\") != std::string_view::npos) {
        invalid_label(component);
    }
}

inline std::pair<std::string_view, std::string_view> parse_group_key(std::string_view group_key) {
    const auto separator = group_key.find(LABEL_SEP);
    if (separator == std::string_view::npos ||
        group_key.find(LABEL_SEP, separator + 1) != std::string_view::npos) {
        invalid_label(group_key);
    }

    const auto label = group_key.substr(0, separator);
    const auto chunk_id = group_key.substr(separator + 1);
    validate_label_component(label);
    validate_label_component(chunk_id);
    return {label, chunk_id};
}

inline std::pair<std::string_view, std::string_view> parse_full_label(std::string_view full_label) {
    const auto first_separator = full_label.find(LABEL_SEP);
    const auto last_separator = full_label.rfind(LABEL_SEP);
    if (first_separator == std::string_view::npos || first_separator == last_separator) {
        invalid_label(full_label);
    }

    const auto group_key = full_label.substr(0, last_separator);
    const auto tensor_key = full_label.substr(last_separator + 1);
    parse_group_key(group_key);
    validate_label_component(tensor_key);
    return {group_key, tensor_key};
}

inline std::string make_group_key(const std::string &label, const std::string &chunk_id) {
    validate_label_component(label);
    validate_label_component(chunk_id);
    return label + LABEL_SEP + chunk_id;
}

inline std::vector<std::string> make_group_keys(const std::string &label,
                                                const std::vector<std::string> &chunk_ids) {
    std::vector<std::string> result;
    result.reserve(chunk_ids.size());
    for (const auto &chunk_id : chunk_ids) {
        result.push_back(make_group_key(label, chunk_id));
    }
    return result;
}

inline std::string make_full_label(const std::string &group_key, const std::string &tensor_key) {
    parse_group_key(group_key);
    validate_label_component(tensor_key);
    return group_key + LABEL_SEP + tensor_key;
}

inline std::string make_chunk_label(const std::string &label, const std::string &tensor_key,
                                    const std::string &chunk_id) {
    return make_full_label(make_group_key(label, chunk_id), tensor_key);
}

inline std::vector<std::string> make_chunk_labels(const std::string &label,
                                                  const std::string &tensor_key,
                                                  const std::vector<std::string> &chunk_ids) {
    std::vector<std::string> result;
    result.reserve(chunk_ids.size());
    for (const auto &chunk_id : chunk_ids) {
        result.push_back(make_chunk_label(label, tensor_key, chunk_id));
    }
    return result;
}

inline std::string get_path_from_label(const std::string &full_label) {
    const auto [group_key, tensor_key] = parse_full_label(full_label);
    const auto [label, chunk_id] = parse_group_key(group_key);

    std::filesystem::path path{label};
    for (size_t offset = 0; offset < chunk_id.size(); offset += 2) {
        path /= chunk_id.substr(offset, 2);
    }
    path /= tensor_key;
    return path.string();
}

class Record {
  public:
    explicit Record(const std::string &sqlite_path, bool cleanup_unpersisted = false);

    ~Record();

    Record(const Record &) = delete;
    Record &operator=(const Record &) = delete;

    void submit(const std::string &label, const std::vector<std::string> &chunk_labels);

    void touch(const std::string &label, const std::vector<std::string> &chunk_labels);

    std::pair<std::vector<std::string>, double> evict(double after, double before,
                                                      size_t max_count, size_t min_count = 0);

    void mark_persisted(const std::vector<std::string> &chunk_labels);

    void remove(const std::vector<std::string> &chunk_labels);

    std::vector<bool> is_persisted(const std::vector<std::string> &chunk_labels);

    void sync();

    std::vector<bool> has(const std::string &label, const std::vector<std::string> &chunk_labels);

    void shutdown();

    size_t pending_count() const;

  private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

class Storage {
  public:
    explicit Storage(const std::string &persist_dir);
    ~Storage();

    Storage(const Storage &) = delete;
    Storage &operator=(const Storage &) = delete;

    bool save(const std::string &full_label, const char *data, size_t size);

    std::pair<char *, size_t> load(const std::string &full_label);

  private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

class Mem {
  public:
    explicit Mem(size_t capacity_bytes = 0, Storage *storage = nullptr, Record *record = nullptr);
    ~Mem();

    Mem(const Mem &) = delete;
    Mem &operator=(const Mem &) = delete;

    void put(const std::vector<std::string> &keys, std::vector<char *> &&data_ptrs,
             const std::vector<size_t> &sizes, const std::vector<size_t> &unzip_sizes = {});

    std::vector<std::pair<const char *, size_t>> get(const std::vector<std::string> &keys,
                                                     bool allow_using_omp = false);

    // Looks up label:chunk_id:tensor_key for every chunk_id without building full labels.
    std::vector<std::pair<const char *, size_t>> get(const std::string &label,
                                                     const std::string &tensor_key,
                                                     const std::vector<std::string> &chunk_ids,
                                                     bool allow_using_omp = false);

    std::vector<std::pair<const char *, size_t>> get_unlocked(const std::vector<std::string> &keys,
                                                              bool allow_using_omp = false);

    std::vector<bool> has(const std::vector<std::string> &keys);

    size_t remove(const std::string &key);

    std::vector<std::tuple<std::string, size_t>> persist_groups(size_t max_groups);

    std::vector<std::tuple<std::string, size_t>> evict_groups(size_t max_groups);

    void evict_labels(const std::vector<std::string> &group_keys);

    size_t evict_to_size(size_t target_bytes);
    size_t force_evict_to_size(size_t target_bytes);

    void set_evict_enabled(bool enabled);
    bool is_evict_enabled() const;

    void lock();
    void unlock();

    void acquire_deletion_guard();
    void release_deletion_guard();
    bool is_deletion_guarded() const;

    std::vector<std::tuple<std::string, const char *, size_t>>
    get_unpersisted_unlocked(size_t max_count);

    std::vector<std::tuple<std::string, const char *, size_t>>
    get_lru_oldest_unlocked(size_t max_count);

    void mark_persisted(const std::vector<std::string> &keys);

    size_t size() const;
    size_t group_count() const;
    size_t current_bytes() const;
    size_t capacity_bytes() const;
    void set_capacity_bytes(size_t capacity);

    uint64_t hits() const;
    uint64_t hits_in_storage() const;
    uint64_t misses() const;
    uint64_t puts() const;
    uint64_t evictions() const;
    uint64_t total_unzip_bytes() const;
    uint64_t total_zip_bytes() const;
    double compression_ratio() const;
    size_t unpersisted_count() const;

    void reset_stats();
    void clear();

  private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace kv_pool
