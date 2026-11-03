// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "iaxl_common.h"
#include "env.h"
#include "kv_pool.h"
#include "parallel.h"
#include "task_queue.h"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <unistd.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace kv_pool {

namespace {

double now_timestamp() {
    const auto now = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(now.time_since_epoch()).count();
}

// Lets unordered_map::find take a string_view without building a std::string.
struct StringHash {
    using is_transparent = void;
    size_t operator()(std::string_view s) const noexcept {
        return std::hash<std::string_view>{}(s);
    }
};

template <typename V>
using StringMap = std::unordered_map<std::string, V, StringHash, std::equal_to<>>;

struct SubEntry {
    char *data = nullptr;
    size_t size = 0;

    ~SubEntry() {
        if (data) {
            free(data);
            data = nullptr;
        }
    }

    SubEntry(const SubEntry &) = delete;
    SubEntry &operator=(const SubEntry &) = delete;

    SubEntry() = default;

    SubEntry(SubEntry &&other) noexcept : data(other.data), size(other.size) {
        other.data = nullptr;
        other.size = 0;
    }

    SubEntry &operator=(SubEntry &&other) noexcept {
        if (this != &other) {
            if (data)
                free(data);
            data = other.data;
            size = other.size;
            other.data = nullptr;
            other.size = 0;
        }
        return *this;
    }
};

struct EntryGroup {
    EntryGroup *lru_prev = nullptr;
    EntryGroup *lru_next = nullptr;

    EntryGroup *unpersisted_prev = nullptr;
    EntryGroup *unpersisted_next = nullptr;
    bool is_persisted = false;

    const std::string *group_key_ptr = nullptr;

    double created_at = 0.0;
    double last_access = 0.0;

    StringMap<SubEntry> entries;

    size_t total_bytes() const {
        size_t total = 0;
        for (const auto &[_, sub] : entries)
            total += sub.size;
        return total;
    }

    size_t entry_count() const { return entries.size(); }

    SubEntry *find(std::string_view tensor_key) {
        auto it = entries.find(tensor_key);
        return it != entries.end() ? &it->second : nullptr;
    }

    const SubEntry *find(std::string_view tensor_key) const {
        auto it = entries.find(tensor_key);
        return it != entries.end() ? &it->second : nullptr;
    }

    SubEntry &get_or_create(const std::string &tensor_key) { return entries[tensor_key]; }

    ~EntryGroup() = default;

    EntryGroup() { entries.reserve((size_t)envs.IAXL_CACHE_CACHEGROUP_SIZE); }

    EntryGroup(const EntryGroup &) = delete;
    EntryGroup &operator=(const EntryGroup &) = delete;

    EntryGroup(EntryGroup &&other) noexcept
        : lru_prev(other.lru_prev), lru_next(other.lru_next),
          unpersisted_prev(other.unpersisted_prev), unpersisted_next(other.unpersisted_next),
          is_persisted(other.is_persisted), group_key_ptr(other.group_key_ptr),
          created_at(other.created_at), last_access(other.last_access),
          entries(std::move(other.entries)) {
        other.lru_prev = nullptr;
        other.lru_next = nullptr;
        other.unpersisted_prev = nullptr;
        other.unpersisted_next = nullptr;
        other.group_key_ptr = nullptr;
    }

    EntryGroup &operator=(EntryGroup &&other) noexcept {
        if (this != &other) {
            lru_prev = other.lru_prev;
            lru_next = other.lru_next;
            unpersisted_prev = other.unpersisted_prev;
            unpersisted_next = other.unpersisted_next;
            is_persisted = other.is_persisted;
            group_key_ptr = other.group_key_ptr;
            created_at = other.created_at;
            last_access = other.last_access;
            entries = std::move(other.entries);
            other.lru_prev = nullptr;
            other.lru_next = nullptr;
            other.unpersisted_prev = nullptr;
            other.unpersisted_next = nullptr;
            other.group_key_ptr = nullptr;
        }
        return *this;
    }
};

} // namespace

class Mem::Impl {
  public:
    explicit Impl(size_t capacity_bytes, Storage *storage, Record *record)
        : capacity_bytes_(capacity_bytes), storage_(storage), record_(record), current_bytes_(0),
          total_entries_(0), evict_enabled_(true) {
        cache_.reserve((size_t)envs.IAXL_CACHE_CACHEGROUP_NUM);

        lru_head_.lru_next = &lru_head_;
        lru_head_.lru_prev = &lru_head_;

        unpersisted_head_.unpersisted_next = &unpersisted_head_;
        unpersisted_head_.unpersisted_prev = &unpersisted_head_;
        constexpr size_t chunk = 2 << 20;
        const size_t page = sysconf(_SC_PAGESIZE);
        std::vector<void *> blocks(capacity_bytes / chunk + (capacity_bytes % chunk != 0));
        parallel::parallel_for(blocks.size(), [&](size_t i) {
            const size_t size = std::min(chunk, capacity_bytes - i * chunk);
            blocks[i] = malloc(size);
            IAXL_CHECK(blocks[i], "KV heap prefault allocation failed");
            for (size_t offset = 0; offset < size; offset += page)
                static_cast<volatile char *>(blocks[i])[offset] = 1;
        });
        for (void *p : blocks)
            free(p);
    }

    ~Impl() {
        free_queue_.shutdown(); // drains pending frees
        cache_.clear();
    }

    void lock() { mutex_.lock(); }
    void unlock() { mutex_.unlock(); }

    void acquire_deletion_guard() { deletion_guard_count_.fetch_add(1, std::memory_order_acq_rel); }

    void release_deletion_guard() {
        if (deletion_guard_count_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            std::lock_guard<std::mutex> lk(deletion_cv_mutex_);
            deletion_cv_.notify_all();
        }
    }

    bool is_deletion_guarded() const {
        return deletion_guard_count_.load(std::memory_order_acquire) > 0;
    }

    void wait_deletion_guard() {
        if (deletion_guard_count_.load(std::memory_order_acquire) == 0)
            return;
        std::unique_lock<std::mutex> lk(deletion_cv_mutex_);
        deletion_cv_.wait(
            lk, [this] { return deletion_guard_count_.load(std::memory_order_acquire) == 0; });
    }

    void put(const std::vector<std::string> &keys, std::vector<char *> &&data_ptrs,
             const std::vector<size_t> &sizes, const std::vector<size_t> &unzip_sizes) {
        IAXL_CHECK(keys.size() == data_ptrs.size(),
                   "Mem::put: keys and data pointers must have the same length");
        IAXL_CHECK(keys.size() == sizes.size(),
                   "Mem::put: keys and sizes must have the same length");
        IAXL_CHECK(unzip_sizes.empty() || keys.size() == unzip_sizes.size(),
                   "Mem::put: unzip sizes must be empty or match keys length");
        std::lock_guard<std::mutex> lock(mutex_);
        put_unlocked(keys, std::move(data_ptrs), sizes, unzip_sizes);
    }

    std::vector<std::pair<const char *, size_t>> get(const std::vector<std::string> &keys,
                                                     bool allow_using_omp) {
        std::lock_guard<std::mutex> lock(mutex_);
        return get_unlocked(keys, allow_using_omp);
    }

    std::vector<std::pair<const char *, size_t>> get(const std::string &label,
                                                     const std::string &tensor_key,
                                                     const std::vector<std::string> &chunk_ids,
                                                     bool allow_using_omp) {
        std::lock_guard<std::mutex> lock(mutex_);
        return get_unlocked(label, tensor_key, chunk_ids, allow_using_omp);
    }

    std::vector<bool> has(const std::vector<std::string> &keys) {
        std::lock_guard<std::mutex> lock(mutex_);
        return has_unlocked(keys);
    }

    std::vector<std::tuple<std::string, size_t>> persist_groups(size_t max_groups) {
        std::lock_guard<std::mutex> lock(mutex_);
        return persist_groups_unlocked(max_groups);
    }

    std::vector<std::tuple<std::string, size_t>> evict_groups(size_t max_groups) {
        wait_deletion_guard();
        std::lock_guard<std::mutex> lock(mutex_);
        return evict_groups_unlocked(max_groups);
    }

    // Unlinks the groups now; their buffers are freed on free_queue_'s thread.
    void evict_labels(const std::vector<std::string> &group_keys) {
        std::vector<StringMap<SubEntry>> detached;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            for (const auto &group_key : group_keys) {
                auto it = cache_.find(group_key);
                if (it == cache_.end())
                    continue;
                EntryGroup &group = it->second;
                current_bytes_ -= group.total_bytes();
                total_entries_ -= group.entry_count();
                lru_remove(&group);
                unpersisted_remove(&group);
                detached.push_back(std::move(group.entries));
                cache_.erase(it);
                evictions_.fetch_add(1, std::memory_order_relaxed);
            }
        }
        if (!detached.empty()) {
            free_queue_.init(); // idempotent: no thread until the first eviction
            free_queue_.submit([this, d = std::move(detached)]() mutable {
                // Safe without blocking new acquirers: a get() that acquired the guard after
                // the unlink above cannot find these groups, so only a get() that took its
                // pointers before the unlink can reference them, and that one holds the
                // guard until it has finished using them.
                wait_deletion_guard();
                d.clear();
            });
        }
    }

    void put_unlocked(const std::vector<std::string> &keys, std::vector<char *> &&data_ptrs,
                      const std::vector<size_t> &sizes, const std::vector<size_t> &unzip_sizes) {
        double now = now_timestamp();
        for (size_t i = 0; i < keys.size(); i++) {

            size_t unzip_size = (i < unzip_sizes.size()) ? unzip_sizes[i] : 0;
            put_one_unlocked(keys[i], data_ptrs[i], sizes[i], unzip_size, now);
            data_ptrs[i] = nullptr;
        }
    }

    std::vector<std::pair<const char *, size_t>> get_unlocked(const std::vector<std::string> &keys,
                                                              bool allow_using_omp) {
        std::vector<std::pair<const char *, size_t>> results(keys.size());
        double now = now_timestamp();

        std::vector<size_t> miss_indices;
        for (size_t i = 0; i < keys.size(); ++i) {
            results[i] = get_one_unlocked(keys[i], now);
            if (results[i].first == nullptr) {
                miss_indices.push_back(i);
            }
        }

        if (!miss_indices.empty())
            load_misses_unlocked(keys, miss_indices, results, now, allow_using_omp);
        return results;
    }

    // Same lookup as get_unlocked(keys) for label:chunk_id:tensor_key without building or
    // re-parsing one full label per chunk; the group key is assembled in a reused buffer.
    std::vector<std::pair<const char *, size_t>>
    get_unlocked(const std::string &label, const std::string &tensor_key,
                 const std::vector<std::string> &chunk_ids, bool allow_using_omp) {
        validate_label_component(label);
        validate_label_component(tensor_key);

        std::vector<std::pair<const char *, size_t>> results(chunk_ids.size());
        double now = now_timestamp();

        std::string group_key = label + LABEL_SEP;
        const size_t prefix = group_key.size();

        std::vector<size_t> miss_indices;
        for (size_t i = 0; i < chunk_ids.size(); ++i) {
            validate_label_component(chunk_ids[i]);
            group_key.resize(prefix);
            group_key.append(chunk_ids[i]);
            results[i] = get_one_unlocked(group_key, tensor_key, now);
            if (results[i].first == nullptr) {
                miss_indices.push_back(i);
            }
        }

        if (!miss_indices.empty()) {
            // The storage path works on full labels; only the missed ones are needed.
            std::vector<std::string> keys(chunk_ids.size());
            for (size_t idx : miss_indices)
                keys[idx] = make_chunk_label(label, tensor_key, chunk_ids[idx]);
            load_misses_unlocked(keys, miss_indices, results, now, allow_using_omp);
        }
        return results;
    }

    // Fills results[miss_indices] from storage when the groups are persisted; keys[i] must be
    // the full label of results[i] for every missed index.
    void load_misses_unlocked(const std::vector<std::string> &keys,
                              const std::vector<size_t> &miss_indices,
                              std::vector<std::pair<const char *, size_t>> &results, double now,
                              bool allow_using_omp) {
        if (!storage_) {
            misses_.fetch_add(miss_indices.size(), std::memory_order_relaxed);
            return;
        }

        std::vector<size_t> storage_indices;
        storage_indices.reserve(miss_indices.size());

        if (record_) {
            std::vector<std::string> check_keys;
            check_keys.reserve(miss_indices.size());
            for (size_t idx : miss_indices) {
                auto [group_key_sv, _] = parse_full_label(keys[idx]);
                check_keys.emplace_back(group_key_sv);
            }

            std::vector<bool> is_persisted = record_->is_persisted(check_keys);

            for (size_t i = 0; i < miss_indices.size(); ++i) {
                if (is_persisted[i]) {
                    storage_indices.push_back(miss_indices[i]);
                } else {
                }
            }
            misses_.fetch_add(miss_indices.size() - storage_indices.size(),
                              std::memory_order_relaxed);
        } else {

            misses_.fetch_add(miss_indices.size(), std::memory_order_relaxed);
            return;
        }

        if (storage_indices.empty()) {
            return;
        }

        struct LoadedData {
            size_t index;
            char *buffer;
            size_t size;
        };
        std::vector<LoadedData> loaded_data(storage_indices.size());

        if (allow_using_omp) {
#pragma omp parallel for schedule(dynamic)
            for (size_t i = 0; i < storage_indices.size(); ++i) {
                size_t key_index = storage_indices[i];
                auto [buffer, size] = storage_->load(keys[key_index]);
                loaded_data[i] = {key_index, buffer, size};
            }
        } else {
            for (size_t i = 0; i < storage_indices.size(); ++i) {
                size_t key_index = storage_indices[i];
                auto [buffer, size] = storage_->load(keys[key_index]);
                loaded_data[i] = {key_index, buffer, size};
            }
        }

        for (const auto &item : loaded_data) {
            if (item.buffer) {

                put_one_unlocked(keys[item.index], item.buffer, item.size, 0, now);

                mark_persisted_unlocked({keys[item.index]});

                hits_in_storage_.fetch_add(1, std::memory_order_relaxed);

                auto [group_key_sv, tensor_key_sv] = parse_full_label(keys[item.index]);
                auto it = cache_.find(group_key_sv);
                SubEntry *sub = it->second.find(tensor_key_sv);
                results[item.index] = {sub->data, sub->size};
            } else {

                misses_.fetch_add(1, std::memory_order_relaxed);
            }
        }
    }

    std::vector<bool> has_unlocked(const std::vector<std::string> &keys) const {
        std::vector<bool> results;
        results.reserve(keys.size());
        for (const auto &key : keys) {
            auto [group_key_sv, tensor_key_sv] = parse_full_label(key);
            auto it = cache_.find(group_key_sv);
            if (it != cache_.end()) {
                if (tensor_key_sv.empty()) {
                    results.push_back(true);
                } else {
                    const SubEntry *sub = it->second.find(tensor_key_sv);
                    results.push_back(sub != nullptr && sub->data != nullptr);
                }
            } else {
                results.push_back(false);
            }
        }
        return results;
    }

    std::vector<std::tuple<std::string, const char *, size_t>>
    get_unpersisted_unlocked(size_t max_count) {
        std::vector<std::tuple<std::string, const char *, size_t>> results;
        EntryGroup *group = unpersisted_head_.unpersisted_next;
        size_t groups_returned = 0;
        while (group != &unpersisted_head_ && groups_returned < max_count) {
            const std::string &group_key = *group->group_key_ptr;
            for (const auto &[tensor_key, sub] : group->entries) {
                std::string full_label = make_full_label(group_key, tensor_key);
                results.emplace_back(std::move(full_label), sub.data, sub.size);
            }
            groups_returned++;
            group = group->unpersisted_next;
        }
        return results;
    }

    std::vector<std::tuple<std::string, const char *, size_t>>
    get_lru_oldest_unlocked(size_t max_count) {
        std::vector<std::tuple<std::string, const char *, size_t>> results;
        EntryGroup *group = lru_head_.lru_next;
        size_t groups_returned = 0;
        while (group != &lru_head_ && groups_returned < max_count) {
            const std::string &group_key = *group->group_key_ptr;
            for (const auto &[tensor_key, sub] : group->entries) {
                std::string full_label = make_full_label(group_key, tensor_key);
                results.emplace_back(std::move(full_label), sub.data, sub.size);
            }
            groups_returned++;
            group = group->lru_next;
        }
        return results;
    }

    void mark_persisted(const std::vector<std::string> &keys) {
        std::lock_guard<std::mutex> lock(mutex_);
        mark_persisted_unlocked(keys);
    }

    void mark_persisted_unlocked(const std::vector<std::string> &keys) {
        for (const auto &key : keys) {
            auto [group_key_sv, _] = parse_full_label(key);
            auto it = cache_.find(group_key_sv);
            if (it != cache_.end()) {
                EntryGroup &group = it->second;
                if (!group.is_persisted) {
                    group.is_persisted = true;
                    unpersisted_remove(&group);
                }
            }
        }
    }

    size_t unpersisted_count() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return unpersisted_count_unlocked();
    }

    size_t unpersisted_count_unlocked() const {
        size_t count = 0;
        const EntryGroup *group = unpersisted_head_.unpersisted_next;
        while (group != &unpersisted_head_) {
            count++;
            group = group->unpersisted_next;
        }
        return count;
    }

    void set_evict_enabled(bool enabled) {
        evict_enabled_.store(enabled, std::memory_order_release);
    }

    bool is_evict_enabled() const { return evict_enabled_.load(std::memory_order_acquire); }

    size_t evict_to_size(size_t target_bytes) {
        if (!evict_enabled_.load(std::memory_order_acquire)) {
            return 0;
        }
        wait_deletion_guard();
        std::lock_guard<std::mutex> lock(mutex_);
        return evict_unlocked(target_bytes);
    }

    size_t force_evict_to_size(size_t target_bytes) {
        wait_deletion_guard();
        std::lock_guard<std::mutex> lock(mutex_);
        return evict_unlocked(target_bytes);
    }

    size_t remove(const std::string &key) {
        wait_deletion_guard();
        std::lock_guard<std::mutex> lock(mutex_);
        return remove_unlocked(key);
    }

    size_t remove_unlocked(const std::string &key) {
        auto [group_key_sv, _] = parse_full_label(key);
        auto it = cache_.find(group_key_sv);
        if (it != cache_.end()) {
            EntryGroup &group = it->second;
            size_t bytes = group.total_bytes();
            current_bytes_ -= bytes;
            total_entries_ -= group.entry_count();
            lru_remove(&group);
            unpersisted_remove(&group);
            cache_.erase(it);
            return bytes;
        }
        return 0;
    }

    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return total_entries_;
    }

    size_t group_count() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return cache_.size();
    }

    size_t current_bytes() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return current_bytes_;
    }

    size_t capacity_bytes() const { return capacity_bytes_; }
    void set_capacity_bytes(size_t capacity) { capacity_bytes_ = capacity; }

    uint64_t hits() const { return hits_.load(std::memory_order_relaxed); }
    uint64_t hits_in_storage() const { return hits_in_storage_.load(std::memory_order_relaxed); }
    uint64_t misses() const { return misses_.load(std::memory_order_relaxed); }
    uint64_t puts() const { return puts_.load(std::memory_order_relaxed); }
    uint64_t evictions() const { return evictions_.load(std::memory_order_relaxed); }

    uint64_t total_unzip_bytes() const {
        return total_unzip_bytes_.load(std::memory_order_relaxed);
    }
    uint64_t total_zip_bytes() const { return total_zip_bytes_.load(std::memory_order_relaxed); }

    double compression_ratio() const {
        uint64_t unzip = total_unzip_bytes_.load(std::memory_order_relaxed);
        uint64_t zip = total_zip_bytes_.load(std::memory_order_relaxed);
        return (zip > 0) ? static_cast<double>(unzip) / static_cast<double>(zip) : 0.0;
    }

    void reset_stats() {
        hits_.store(0, std::memory_order_relaxed);
        hits_in_storage_.store(0, std::memory_order_relaxed);
        misses_.store(0, std::memory_order_relaxed);
        puts_.store(0, std::memory_order_relaxed);
        evictions_.store(0, std::memory_order_relaxed);
        total_unzip_bytes_.store(0, std::memory_order_relaxed);
        total_zip_bytes_.store(0, std::memory_order_relaxed);
    }

    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        cache_.clear();
        current_bytes_ = 0;
        total_entries_ = 0;
        lru_head_.lru_next = &lru_head_;
        lru_head_.lru_prev = &lru_head_;
        unpersisted_head_.unpersisted_next = &unpersisted_head_;
        unpersisted_head_.unpersisted_prev = &unpersisted_head_;
    }

  private:
    void put_one_unlocked(const std::string &full_label, char *data, size_t size, size_t unzip_size,
                          double now) {
        if (!data) {
            throw std::runtime_error(
                "[Mem] memory allocation failed for '" + full_label + "' (" + std::to_string(size) +
                " bytes). DDR pool usage: " + std::to_string(current_bytes_ / (1024 * 1024)) +
                " MB / " + std::to_string(capacity_bytes_ / (1024 * 1024)) +
                " MB capacity. Increase IAXL_DDR_POOL_SIZE_GB or enable cache eviction.");
        }

        if (unzip_size > 0) {
            total_unzip_bytes_.fetch_add(unzip_size, std::memory_order_relaxed);
            total_zip_bytes_.fetch_add(size, std::memory_order_relaxed);
        }
        auto [group_key_sv, tensor_key_sv] = parse_full_label(full_label);
        std::string group_key(group_key_sv);
        std::string tensor_key(tensor_key_sv);

        auto it = cache_.find(group_key);
        if (it != cache_.end()) {

            EntryGroup &group = it->second;
            SubEntry *sub = group.find(tensor_key_sv);
            if (sub) {
                current_bytes_ -= sub->size;
                if (sub->data)
                    free(sub->data);
                sub->data = data;
                sub->size = size;
            } else {
                SubEntry &new_sub = group.get_or_create(tensor_key);
                new_sub.data = data;
                new_sub.size = size;
                total_entries_++;

                if (group.entries.size() > (size_t)envs.IAXL_CACHE_CACHEGROUP_SIZE) {
                    fprintf(stderr,
                            "[Mem] WARN: EntryGroup '%s' entries (%zu) "
                            "exceeded reserve (%zu), rehash triggered\n",
                            group_key.c_str(), group.entries.size(),
                            (size_t)envs.IAXL_CACHE_CACHEGROUP_SIZE);
                }
            }
            current_bytes_ += size;
            group.last_access = now;

            lru_remove(&group);
            lru_push_back(&group);

            if (group.is_persisted) {
                group.is_persisted = false;
                unpersisted_push_back(&group);
            }
        } else {

            decltype(cache_.begin()) new_it;
            try {
                auto [it2, inserted] =
                    cache_.emplace(std::piecewise_construct, std::forward_as_tuple(group_key),
                                   std::forward_as_tuple());
                new_it = it2;
            } catch (const std::bad_alloc &) {
                free(data);
                throw std::runtime_error(
                    "[Mem] failed to allocate new cache group '" + group_key +
                    "'. DDR pool usage: " + std::to_string(current_bytes_ / (1024 * 1024)) +
                    " MB / " + std::to_string(capacity_bytes_ / (1024 * 1024)) + " MB capacity, " +
                    std::to_string(cache_.size()) +
                    " groups. System memory exhausted. "
                    "Increase IAXL_DDR_POOL_SIZE_GB or enable cache eviction.");
            }
            EntryGroup &group = new_it->second;
            group.group_key_ptr = &new_it->first;

            SubEntry &sub = group.get_or_create(tensor_key);
            sub.data = data;
            sub.size = size;
            group.created_at = now;
            group.last_access = now;
            group.is_persisted = false;

            current_bytes_ += size;
            total_entries_++;

            lru_push_back(&group);
            unpersisted_push_back(&group);

            if (cache_.size() > (size_t)envs.IAXL_CACHE_CACHEGROUP_NUM) {
                fprintf(stderr,
                        "[Mem] WARN: cache groups (%zu) "
                        "exceeded reserve (%zu), rehash triggered\n",
                        cache_.size(), (size_t)envs.IAXL_CACHE_CACHEGROUP_NUM);
            }
        }

        puts_.fetch_add(1, std::memory_order_relaxed);

        if (capacity_bytes_ > 0 && current_bytes_ > capacity_bytes_) {
            static std::atomic<uint64_t> warn_count{0};
            if ((warn_count.fetch_add(1) % 10000) == 0) {
                fprintf(stderr, "[Mem] WARNING: Over budget! %zu MB > %zu MB capacity\n",
                        current_bytes_ / (1024 * 1024), capacity_bytes_ / (1024 * 1024));
            }
        }
    }

    std::pair<const char *, size_t> get_one_unlocked(const std::string &full_label, double now) {
        auto [group_key_sv, tensor_key_sv] = parse_full_label(full_label);
        return get_one_unlocked(group_key_sv, tensor_key_sv, now);
    }

    std::pair<const char *, size_t> get_one_unlocked(std::string_view group_key_sv,
                                                     std::string_view tensor_key_sv, double now) {
        auto it = cache_.find(group_key_sv);
        if (it != cache_.end()) {
            EntryGroup &group = it->second;
            SubEntry *sub = group.find(tensor_key_sv);
            if (sub && sub->data) {
                group.last_access = now;
                lru_remove(&group);
                lru_push_back(&group);
                hits_.fetch_add(1, std::memory_order_relaxed);
                return {sub->data, sub->size};
            }
        }
        return {nullptr, 0};
    }

    std::vector<std::tuple<std::string, size_t>> persist_groups_unlocked(size_t max_groups) {
        if (!storage_) {
            std::cerr << "\033[1;31m*** FATAL: No storage backend configured, cannot persist "
                         "groups.\033[0m"
                      << std::endl;
            std::abort();
        }
        auto entries_to_persist = get_unpersisted_unlocked(max_groups);
        if (entries_to_persist.empty()) {
            return {};
        }

        std::vector<std::tuple<std::string, size_t>> persisted_groups;
        std::vector<std::string> persisted_full_labels;
        std::unordered_map<std::string, size_t> group_bytes;

        for (const auto &entry_tuple : entries_to_persist) {
            const auto &full_label = std::get<0>(entry_tuple);
            const char *data = std::get<1>(entry_tuple);
            size_t size = std::get<2>(entry_tuple);

            if (storage_->save(full_label, data, size)) {
                persisted_full_labels.push_back(full_label);
                auto [group_key_sv, _] = parse_full_label(full_label);
                group_bytes[std::string(group_key_sv)] += size;
            }
        }

        mark_persisted_unlocked(persisted_full_labels);

        if (record_) {
            std::vector<std::string> group_keys;
            group_keys.reserve(persisted_full_labels.size());
            std::unordered_set<std::string> seen;
            seen.reserve(persisted_full_labels.size());
            for (const auto &fl : persisted_full_labels) {
                auto [gk_sv, _] = parse_full_label(fl);
                std::string gk(gk_sv);
                if (seen.insert(gk).second) {
                    group_keys.push_back(std::move(gk));
                }
            }
            record_->mark_persisted(group_keys);
        }

        for (const auto &pair : group_bytes) {
            persisted_groups.emplace_back(pair.first, pair.second);
        }

        return persisted_groups;
    }

    std::vector<std::tuple<std::string, size_t>> evict_groups_unlocked(size_t max_groups) {
        if (!evict_enabled_) {
            return {};
        }

        auto entries_to_evict = get_lru_oldest_unlocked(max_groups);
        if (entries_to_evict.empty()) {
            return {};
        }

        std::vector<std::string> unique_full_labels;
        std::vector<std::string> unique_group_keys;
        std::unordered_map<std::string, size_t> group_bytes;
        std::unordered_set<std::string> seen_groups;

        for (const auto &entry_tuple : entries_to_evict) {
            const auto &full_label = std::get<0>(entry_tuple);
            size_t size = std::get<2>(entry_tuple);
            auto [group_key_sv, _] = parse_full_label(full_label);
            std::string group_key(group_key_sv);

            group_bytes[group_key] += size;
            if (seen_groups.find(group_key) == seen_groups.end()) {
                seen_groups.insert(group_key);
                unique_group_keys.push_back(group_key);
                unique_full_labels.push_back(full_label);
            }
        }

        std::vector<std::string> unpersisted_groups_to_remove;
        if (record_) {
            std::vector<bool> is_persisted_results = record_->is_persisted(unique_group_keys);
            for (size_t i = 0; i < unique_group_keys.size(); ++i) {
                if (!is_persisted_results[i]) {
                    unpersisted_groups_to_remove.push_back(unique_group_keys[i]);
                }
            }
        }

        if (record_ && !unpersisted_groups_to_remove.empty()) {
            record_->remove(unpersisted_groups_to_remove);
        }

        for (const auto &full_label : unique_full_labels) {
            remove_unlocked(full_label);
        }

        std::vector<std::tuple<std::string, size_t>> evicted_groups;
        for (const auto &group_key : unique_group_keys) {
            evicted_groups.emplace_back(group_key, group_bytes[group_key]);
        }

        return evicted_groups;
    }

    size_t evict_unlocked(size_t target_bytes) {
        size_t evicted = 0;
        while (current_bytes_ > target_bytes && lru_head_.lru_next != &lru_head_) {
            EntryGroup *oldest = lru_head_.lru_next;
            size_t bytes = oldest->total_bytes();

            evicted += bytes;
            current_bytes_ -= bytes;
            total_entries_ -= oldest->entry_count();

            lru_remove(oldest);
            unpersisted_remove(oldest);

            cache_.erase(*oldest->group_key_ptr);
            evictions_.fetch_add(1, std::memory_order_relaxed);
        }
        return evicted;
    }

    void lru_remove(EntryGroup *group) {
        if (group->lru_prev) {
            group->lru_prev->lru_next = group->lru_next;
        }
        if (group->lru_next) {
            group->lru_next->lru_prev = group->lru_prev;
        }
        group->lru_prev = nullptr;
        group->lru_next = nullptr;
    }

    void lru_push_back(EntryGroup *group) {
        group->lru_prev = lru_head_.lru_prev;
        group->lru_next = &lru_head_;
        lru_head_.lru_prev->lru_next = group;
        lru_head_.lru_prev = group;
    }

    void unpersisted_remove(EntryGroup *group) {
        if (group->unpersisted_prev) {
            group->unpersisted_prev->unpersisted_next = group->unpersisted_next;
        }
        if (group->unpersisted_next) {
            group->unpersisted_next->unpersisted_prev = group->unpersisted_prev;
        }
        group->unpersisted_prev = nullptr;
        group->unpersisted_next = nullptr;
    }

    void unpersisted_push_back(EntryGroup *group) {
        group->unpersisted_prev = unpersisted_head_.unpersisted_prev;
        group->unpersisted_next = &unpersisted_head_;
        unpersisted_head_.unpersisted_prev->unpersisted_next = group;
        unpersisted_head_.unpersisted_prev = group;
    }

    StringMap<EntryGroup> cache_;

    EntryGroup lru_head_;
    EntryGroup unpersisted_head_;

    size_t capacity_bytes_;
    size_t current_bytes_;
    size_t total_entries_;

    std::atomic<bool> evict_enabled_;

    mutable std::mutex mutex_;

    std::atomic<int> deletion_guard_count_{0};
    std::mutex deletion_cv_mutex_;
    std::condition_variable deletion_cv_;

    std::atomic<uint64_t> hits_{0};
    std::atomic<uint64_t> misses_{0};
    std::atomic<uint64_t> hits_in_storage_{0};
    std::atomic<uint64_t> puts_{0};
    std::atomic<uint64_t> evictions_{0};

    Storage *storage_ = nullptr;
    Record *record_ = nullptr;

    std::atomic<uint64_t> total_unzip_bytes_{0};
    std::atomic<uint64_t> total_zip_bytes_{0};

    TaskQueue free_queue_{"iaxl-mem-free"};
};

Mem::Mem(size_t capacity_bytes, Storage *storage, Record *record)
    : impl_(std::make_unique<Impl>(capacity_bytes, storage, record)) {}
Mem::~Mem() = default;

void Mem::put(const std::vector<std::string> &keys, std::vector<char *> &&data_ptrs,
              const std::vector<size_t> &sizes, const std::vector<size_t> &unzip_sizes) {
    impl_->put(keys, std::move(data_ptrs), sizes, unzip_sizes);
}

std::vector<std::pair<const char *, size_t>> Mem::get(const std::vector<std::string> &keys,
                                                      bool allow_using_omp) {
    return impl_->get(keys, allow_using_omp);
}

std::vector<std::pair<const char *, size_t>> Mem::get(const std::string &label,
                                                      const std::string &tensor_key,
                                                      const std::vector<std::string> &chunk_ids,
                                                      bool allow_using_omp) {
    return impl_->get(label, tensor_key, chunk_ids, allow_using_omp);
}

std::vector<std::pair<const char *, size_t>> Mem::get_unlocked(const std::vector<std::string> &keys,
                                                               bool allow_using_omp) {
    return impl_->get_unlocked(keys, allow_using_omp);
}

std::vector<bool> Mem::has(const std::vector<std::string> &keys) { return impl_->has(keys); }

size_t Mem::remove(const std::string &key) { return impl_->remove(key); }

std::vector<std::tuple<std::string, size_t>> Mem::persist_groups(size_t max_groups) {
    return impl_->persist_groups(max_groups);
}

std::vector<std::tuple<std::string, size_t>> Mem::evict_groups(size_t max_groups) {
    return impl_->evict_groups(max_groups);
}

void Mem::evict_labels(const std::vector<std::string> &group_keys) {
    impl_->evict_labels(group_keys);
}

size_t Mem::evict_to_size(size_t target_bytes) { return impl_->evict_to_size(target_bytes); }
size_t Mem::force_evict_to_size(size_t target_bytes) {
    return impl_->force_evict_to_size(target_bytes);
}

void Mem::set_evict_enabled(bool enabled) { impl_->set_evict_enabled(enabled); }
bool Mem::is_evict_enabled() const { return impl_->is_evict_enabled(); }

void Mem::lock() { impl_->lock(); }
void Mem::unlock() { impl_->unlock(); }

void Mem::acquire_deletion_guard() { impl_->acquire_deletion_guard(); }
void Mem::release_deletion_guard() { impl_->release_deletion_guard(); }
bool Mem::is_deletion_guarded() const { return impl_->is_deletion_guarded(); }

std::vector<std::tuple<std::string, const char *, size_t>>
Mem::get_unpersisted_unlocked(size_t max_count) {
    return impl_->get_unpersisted_unlocked(max_count);
}

std::vector<std::tuple<std::string, const char *, size_t>>
Mem::get_lru_oldest_unlocked(size_t max_count) {
    return impl_->get_lru_oldest_unlocked(max_count);
}

void Mem::mark_persisted(const std::vector<std::string> &keys) { impl_->mark_persisted(keys); }

size_t Mem::size() const { return impl_->size(); }
size_t Mem::group_count() const { return impl_->group_count(); }
size_t Mem::current_bytes() const { return impl_->current_bytes(); }
size_t Mem::capacity_bytes() const { return impl_->capacity_bytes(); }
void Mem::set_capacity_bytes(size_t capacity) { impl_->set_capacity_bytes(capacity); }

uint64_t Mem::hits() const { return impl_->hits(); }
uint64_t Mem::hits_in_storage() const { return impl_->hits_in_storage(); }
uint64_t Mem::misses() const { return impl_->misses(); }
uint64_t Mem::puts() const { return impl_->puts(); }
uint64_t Mem::evictions() const { return impl_->evictions(); }
uint64_t Mem::total_unzip_bytes() const { return impl_->total_unzip_bytes(); }
uint64_t Mem::total_zip_bytes() const { return impl_->total_zip_bytes(); }
double Mem::compression_ratio() const { return impl_->compression_ratio(); }
size_t Mem::unpersisted_count() const { return impl_->unpersisted_count(); }

void Mem::reset_stats() { impl_->reset_stats(); }
void Mem::clear() { impl_->clear(); }

} // namespace kv_pool
