// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

#include "context.h"
#include "kv_pool.h"
#include "kv_zip.h"

using namespace profiler;
using namespace kv_zip;

namespace {

static inline uint64_t metrics_sum_sizes(const std::vector<size_t> &sizes) {
    uint64_t total = 0;
    for (size_t s : sizes)
        total += s;
    return total;
}

static inline uint64_t metrics_sum_tensor_bytes(const std::vector<torch::Tensor> &tensors) {
    uint64_t total = 0;
    for (const auto &t : tensors) {
        total += static_cast<uint64_t>(t.numel()) * t.element_size();
    }
    return total;
}

// Every layer of a load hits the same missing blocks; log the first few, then every 1000th.
static inline bool should_log(std::atomic<uint64_t> &count) {
    uint64_t n = count.fetch_add(1, std::memory_order_relaxed);
    return n < 10 || n % 1000 == 0;
}

struct UnzipFromMemWork : std::enable_shared_from_this<UnzipFromMemWork> {
    Context *ctx;
    kv_pool::Mem *mem;
    std::vector<std::string> chunk_labels;
    std::vector<int64_t> chunk_indices;
    std::vector<torch::Tensor> cpu_tensors;
    std::shared_ptr<std::promise<void>> promise;

    void execute(bool is_retry) {
        PROFILE_SCOPE_FMT("unzip_from_mem_work(%s,retry=%d,l=<%zu,l0=%s>,i=<%zu,i0=%ld>)",
                          ctx->name().c_str(), is_retry, chunk_labels.size(),
                          chunk_labels[0].c_str(), chunk_indices.size(), chunk_indices[0]);

        const size_t n = chunk_labels.size();

        // Pinned groups cannot be evicted while their buffers are being decompressed.
        std::vector<std::pair<const char *, size_t>> results;
        std::vector<kv_pool::Mem::PinHandle> pinned;
        pinned.reserve(n);
        {
            PROFILE_SCOPE("cache_get");
            results = mem->get_pinned(chunk_labels, pinned, true);
        }

        std::vector<size_t> hits;
        std::vector<int64_t> failed;
        size_t first_miss = n;
        hits.reserve(n);
        for (size_t i = 0; i < n; i++) {
            if (results[i].first) {
                hits.push_back(i);
            } else {
                if (first_miss == n)
                    first_miss = i;
                failed.push_back(chunk_indices[i]);
            }
        }

        if (!failed.empty() && !is_retry) {
            // An in-flight zip_to_mem for the same blocks may still be queued; retry behind it.
            mem->unpin(pinned);
            static std::atomic<uint64_t> retry_logs{0};
            if (should_log(retry_logs)) {
                std::cout << "[IAXL] unzip_from_mem(" << ctx->name() << "): " << failed.size()
                          << "/" << n << " blocks not in the pool (first "
                          << chunk_labels[first_miss] << "), retrying\n";
            }
            auto self = shared_from_this();
            omp_queue().submit(
                [self]() {
                    try {
                        self->execute(true);
                    } catch (...) {
                        IAXL_CHECK(false,
                                   "unzip_from_mem retry: unexpected asynchronous exception");
                    }
                },
                TaskQueue::PRIORITY_LOW);
            return;
        }

        static std::atomic<uint64_t> fail_logs{0};
        if (!failed.empty() && should_log(fail_logs)) {
            std::cerr << "[IAXL] unzip_from_mem(" << ctx->name() << "): " << failed.size() << "/"
                      << n << " blocks missing from the pool after retry; reporting them as "
                      << "failed loads\n";
        }

        if (!hits.empty()) {
            const bool all_hit = hits.size() == n;
            std::vector<const char *> data_ptrs;
            std::vector<torch::Tensor> hit_tensors;
            std::vector<int64_t> hit_indices;
            data_ptrs.reserve(hits.size());
            for (size_t i : hits) {
                data_ptrs.push_back(results[i].first);
            }
            if (!all_hit) {
                hit_tensors.reserve(hits.size());
                hit_indices.reserve(hits.size());
                for (size_t i : hits) {
                    hit_tensors.push_back(cpu_tensors[i]);
                    hit_indices.push_back(chunk_indices[i]);
                }
            }
            const auto &tensors = all_hit ? cpu_tensors : hit_tensors;
            const auto &indices = all_hit ? chunk_indices : hit_indices;

            {
                PROFILE_SCOPE("decompress");
                METRICS_TIMER_START(metrics_decompress);
                kv_zip_decompress_batch(data_ptrs, tensors);
                METRICS_ADD_DECOMPRESS(metrics_decompress, metrics_sum_tensor_bytes(tensors));
            }

            mem->unpin(pinned);

            ctx->xfer_chunks_batch(indices, tensors);
        }

        ctx->xfer_finish();
        ctx->set_unzip_failed(std::move(failed));

        promise->set_value();
    }
};

} // namespace

void Context::zip_to_mem(kv_pool::Mem &mem, const std::string &label, const std::string &tensor_key,
                         const std::vector<std::string> &chunk_labels,
                         const std::vector<torch::Tensor> &cpu_tensors, bool compress) {
    PROFILE_SCOPE_FMT("zip_to_mem(%s,l=<%zu,l0=%s>)", name().c_str(), chunk_labels.size(),
                      chunk_labels[0].c_str());

    const size_t n = cpu_tensors.size();

    auto full_chunk_labels = kv_pool::make_chunk_labels(label, tensor_key, chunk_labels);

    reset_async_state();

    auto *cache_ptr = &mem;
    auto chunk_labels_copy = std::move(full_chunk_labels);
    auto cpu_tensors_copy = cpu_tensors;
    auto *self = this;

    zip_future_ = omp_queue().submit(
        [=]() {
            PROFILE_SCOPE_FMT("zip_to_mem_work(%s,l=<%zu,l0=%s>)", self->name().c_str(),
                              chunk_labels_copy.size(), chunk_labels_copy[0].c_str());

            self->xfer_wait();

            try {
                std::vector<char *> compressed_bufs(n);
                std::vector<size_t> compressed_sizes(n);
                std::vector<size_t> unzip_sizes(n);

                {
                    PROFILE_SCOPE("compress");
                    METRICS_TIMER_START(metrics_compress);
                    kv_zip_compress_batch(cpu_tensors_copy, compressed_bufs, compressed_sizes,
                                          unzip_sizes, compress);
                    METRICS_ADD_COMPRESS(metrics_compress, metrics_sum_sizes(unzip_sizes));
                }

                {
                    cache_ptr->put(chunk_labels_copy, std::move(compressed_bufs), compressed_sizes,
                                   unzip_sizes);
                }
            } catch (...) {
                IAXL_CHECK(false, "zip_to_mem: unexpected asynchronous exception");
            }
        },
        TaskQueue::PRIORITY_LOW);
}

void Context::unzip_from_mem(kv_pool::Mem &mem, const std::string &label,
                             const std::string &tensor_key,
                             const std::vector<std::string> &chunk_labels,
                             const std::vector<int64_t> &chunk_indices,
                             const std::vector<torch::Tensor> &cpu_tensors) {
    PROFILE_SCOPE_FMT("unzip_from_mem %s/%s", label.c_str(), tensor_key.c_str());

    auto full_chunk_labels = kv_pool::make_chunk_labels(label, tensor_key, chunk_labels);
    const size_t n = full_chunk_labels.size();

    reset_async_state();
    unzip_failed_.clear();

    auto promise = std::make_shared<std::promise<void>>();
    unzip_future_ = promise->get_future();

    auto work = std::make_shared<UnzipFromMemWork>();
    work->ctx = this;
    work->mem = &mem;
    work->chunk_labels = std::move(full_chunk_labels);
    work->chunk_indices = chunk_indices;
    work->cpu_tensors = cpu_tensors;
    work->promise = promise;

    omp_queue().submit(
        [work]() {
            try {
                work->execute(false);
            } catch (...) {
                IAXL_CHECK(false, "unzip_from_mem: unexpected asynchronous exception");
            }
        },
        TaskQueue::PRIORITY_HIGH);
}

void Context::zip_wait() {
    IAXL_CHECK(zip_future_.valid(), "zip_wait: zip_to_mem must be called first");
    PROFILE_SCOPE_FMT("zip_wait(%s)", name().c_str());
    zip_future_.get();
    zip_future_ = std::future<void>();
}

bool Context::zip_is_complete() {
    if (zip_future_.valid() &&
        zip_future_.wait_for(std::chrono::seconds(0)) != std::future_status::ready)
        return false;
    return true;
}

void Context::unzip_wait() {
    IAXL_CHECK(unzip_future_.valid(), "unzip_wait: unzip_from_mem must be called first");
    PROFILE_SCOPE_FMT("unzip_wait(%s)", name().c_str());
    unzip_future_.get();
    unzip_future_ = std::future<void>();
}

bool Context::unzip_is_complete() {
    if (unzip_future_.valid() &&
        unzip_future_.wait_for(std::chrono::seconds(0)) != std::future_status::ready) {
        if (envs.IAXL_DEBUG_LOG)
            fprintf(stderr, "[unzip_is_complete] %s: future not ready\n", name().c_str());
        return false;
    }
    return true;
}
