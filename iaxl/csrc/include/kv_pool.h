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

namespace kv_pool {

inline constexpr char LABEL_SEP = ':';

// Layout of the persisted chunk files (header flags, payload encoding). Bump whenever a block
// written by the previous version would decode differently; Record refuses to open a mismatch.
//   1: header word 0 carries the IAA flag (bit 31) and the byte-plane shuffle flag (bit 30).
inline constexpr int CACHE_FORMAT_VERSION = 1;

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

    std::vector<std::pair<const char *, size_t>> get_unlocked(const std::vector<std::string> &keys,
                                                              bool allow_using_omp = false);

    std::vector<bool> has(const std::vector<std::string> &keys);

    size_t remove(const std::string &key);

    std::vector<std::tuple<std::string, size_t>> persist_groups(size_t max_groups);

    std::vector<std::tuple<std::string, size_t>> evict_groups(size_t max_groups);

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
