// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <torch/extension.h>

#include <algorithm>
#include <iterator>
#include <sstream>

#include "iaxl_common.h"
#include "kv_pool.h"

namespace kv_pool {

ScratchPool::ScratchPool(size_t bytes, bool pin_memory) {
    const int64_t num_slots = static_cast<int64_t>(bytes / SCRATCH_SLOT_BYTES);
    IAXL_CHECK(num_slots > 0, "ScratchPool: size is below one slot");
    auto t = torch::empty({num_slots, SCRATCH_SLOT_BYTES},
                          torch::TensorOptions().dtype(torch::kUInt8).pinned_memory(pin_memory));
    IAXL_CHECK(!pin_memory || t.is_pinned(), "ScratchPool: pin_memory=true did not take effect");
    // The registry keeps the tensor alive until close().
    register_kv_pool_cpu_slots(NAME, t);
    num_slots_ = num_slots;
    free_.emplace(0, num_slots);
}

ScratchPool::~ScratchPool() { close(); }

void ScratchPool::close() {
    if (num_slots_ == 0)
        return;
    unregister_kv_pool(NAME);
    num_slots_ = 0;
    free_.clear();
}

std::optional<ScratchAlloc> ScratchPool::allocate(int64_t pages, int64_t page_bytes) {
    IAXL_CHECK(pages >= 0 && page_bytes > 0, "ScratchPool::allocate: bad request");
    const int64_t per_page = (page_bytes + SCRATCH_SLOT_BYTES - 1) / SCRATCH_SLOT_BYTES;

    // All or nothing: count fitting pages, lowest address first, and stop once enough.
    int64_t fit = 0;
    for (auto it = free_.begin(); it != free_.end() && fit < pages; ++it)
        fit += it->second / per_page;
    if (fit < pages)
        return std::nullopt;

    // Cut the pages from the same ranges in place; ranges past the last one used stay untouched.
    ScratchAlloc alloc{{}, per_page};
    alloc.indices.reserve(pages);
    for (auto it = free_.begin(); static_cast<int64_t>(alloc.indices.size()) < pages;) {
        const auto [start, length] = *it;
        const int64_t take = std::min(length / per_page, pages - static_cast<int64_t>(alloc.indices.size()));
        for (int64_t j = 0; j < take; j++)
            alloc.indices.push_back(start + j * per_page);
        if (take == 0) {
            ++it;
            continue;
        }
        it = free_.erase(it);
        if (length > take * per_page)
            it = std::next(free_.emplace_hint(it, start + take * per_page, length - take * per_page));
    }
    used_ += pages * per_page;
    allocate_count_++;
    return alloc;
}

void ScratchPool::release(const ScratchAlloc &alloc) {
    for (int64_t start : alloc.indices) {
        int64_t end = start + alloc.per_page;
        auto next = free_.lower_bound(start);
        if (next != free_.end() && next->first == end) { // merge the right neighbour
            end += next->second;
            next = free_.erase(next);
        }
        if (next != free_.begin()) { // merge the left neighbour
            auto prev = std::prev(next);
            if (prev->first + prev->second == start) {
                prev->second = end - prev->first;
                continue;
            }
        }
        free_.emplace_hint(next, start, end - start);
    }
    used_ -= static_cast<int64_t>(alloc.indices.size()) * alloc.per_page;
    release_count_++;
}

std::vector<std::pair<int64_t, int64_t>> ScratchPool::free_ranges() const {
    return {free_.begin(), free_.end()};
}

std::string ScratchPool::status() const {
    int64_t largest = 0;
    for (const auto &[_, length] : free_)
        largest = std::max(largest, length);
    std::ostringstream s;
    s << "ScratchPool: " << num_slots_ << " slots x " << SCRATCH_SLOT_BYTES << " B, " << available()
      << " available in " << free_.size() << " ranges (largest " << largest
      << "), alloc/release=" << allocate_count_ << "/" << release_count_ << ", "
      << total_bytes() / (1 << 20) << " MB";
    return s.str();
}

} // namespace kv_pool
