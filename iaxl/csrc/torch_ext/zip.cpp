// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

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

struct UnzipFromMemWork : std::enable_shared_from_this<UnzipFromMemWork> {
    Context *ctx;
    kv_pool::Mem *mem;
    std::vector<std::string> chunk_labels;
    std::vector<int64_t> chunk_indices;
    std::vector<torch::Tensor> cpu_tensors;
    bool direct = false;
    std::shared_ptr<std::promise<void>> promise;

    void execute(bool is_retry) {
        PROFILE_SCOPE_FMT("unzip_from_mem_work(%s,retry=%d,direct=%d,l=<%zu,l0=%s>,i=<%zu,i0=%ld>)",
                          ctx->name().c_str(), is_retry, direct, chunk_labels.size(),
                          chunk_labels[0].c_str(), chunk_indices.size(), chunk_indices[0]);

        const size_t n = chunk_labels.size();

        mem->acquire_deletion_guard();

        std::vector<const char *> data_ptrs(n);
        {
            PROFILE_SCOPE("cache_get");
            auto results = mem->get(chunk_labels, true);

            for (size_t i = 0; i < n; i++) {
                const auto &[ptr, size] = results[i];
                (void)size;
                if (!ptr) {
                    mem->release_deletion_guard();

                    if (!is_retry) {
                        std::cout << "[KVClip] cache miss on unzip (will retry): "
                                  << chunk_labels[i] << "\n";
                        auto self = shared_from_this();
                        omp_queue().submit(
                            [self]() {
                                try {
                                    self->execute(true);
                                } catch (...) {
                                    IAXL_CHECK(
                                        false,
                                        "unzip_from_mem retry: unexpected asynchronous exception");
                                }
                            },
                            TaskQueue::PRIORITY_LOW);
                        return;
                    }

                    IAXL_CHECK(false, "unzip_from_mem: cache key missing after retry");
                }
                data_ptrs[i] = ptr;
            }
        }

        {
            PROFILE_SCOPE("decompress");
            METRICS_TIMER_START(metrics_decompress);
            if (direct) {
                std::vector<kv_zip::ChunkView> views;
                views.reserve(n);
                uint64_t total = 0;
                for (int64_t idx : chunk_indices) {
                    views.push_back(ctx->chunk_view(idx));
                    total += views.back().nbytes();
                }
                kv_zip_decompress_views(data_ptrs, views);
                METRICS_ADD_DECOMPRESS(metrics_decompress, total);
            } else {
                kv_zip_decompress_batch(data_ptrs, cpu_tensors);
                METRICS_ADD_DECOMPRESS(metrics_decompress, metrics_sum_tensor_bytes(cpu_tensors));
            }
        }

        mem->release_deletion_guard();

        {
            // Direct mode already wrote the tensor; the finish marker keeps xfer_wait() valid.
            if (!direct)
                ctx->xfer_chunks_batch(chunk_indices, cpu_tensors);

            ctx->xfer_finish();
        }

        promise->set_value();
    }
};

static void check_direct_supported() {
#ifndef CPU_SUPPORT
    TORCH_CHECK(false, "direct codec paths read/write the inference tensor from the host and "
                       "require a DEVICE=cpu build");
#endif
}

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

void Context::zip_to_mem_direct(kv_pool::Mem &mem, const std::string &label,
                                const std::string &tensor_key,
                                const std::vector<std::string> &chunk_labels,
                                const std::vector<int64_t> &chunk_indices, bool compress) {
    check_direct_supported();
    PROFILE_SCOPE_FMT("zip_to_mem_direct(%s,l=<%zu,l0=%s>)", name().c_str(), chunk_labels.size(),
                      chunk_labels[0].c_str());
    TORCH_CHECK(chunk_indices.size() == chunk_labels.size(),
                "zip_to_mem_direct: chunk_indices and chunk_labels must match");
    const int64_t limit = num_chunks();
    for (int64_t idx : chunk_indices)
        TORCH_CHECK(idx >= 0 && idx < limit, "zip_to_mem_direct: chunk index out of range");

    const size_t n = chunk_indices.size();
    std::vector<kv_zip::ChunkView> views;
    views.reserve(n);
    for (int64_t idx : chunk_indices)
        views.push_back(chunk_view(idx));

    reset_async_state();

    auto *cache_ptr = &mem;
    auto chunk_labels_copy = kv_pool::make_chunk_labels(label, tensor_key, chunk_labels);
    auto *self = this;

    zip_future_ = omp_queue().submit(
        [=, views = std::move(views)]() {
            PROFILE_SCOPE_FMT("zip_to_mem_direct_work(%s,l=<%zu,l0=%s>)", self->name().c_str(),
                              chunk_labels_copy.size(), chunk_labels_copy[0].c_str());
            try {
                std::vector<char *> compressed_bufs(n);
                std::vector<size_t> compressed_sizes(n);
                std::vector<size_t> unzip_sizes(n);
                {
                    PROFILE_SCOPE("compress");
                    METRICS_TIMER_START(metrics_compress);
                    kv_zip_compress_views(views, compressed_bufs, compressed_sizes, unzip_sizes,
                                          compress);
                    METRICS_ADD_COMPRESS(metrics_compress, metrics_sum_sizes(unzip_sizes));
                }
                cache_ptr->put(chunk_labels_copy, std::move(compressed_bufs), compressed_sizes,
                               unzip_sizes);
            } catch (...) {
                IAXL_CHECK(false, "zip_to_mem_direct: unexpected asynchronous exception");
            }
        },
        TaskQueue::PRIORITY_LOW);
}

void Context::unzip_from_mem_direct(kv_pool::Mem &mem, const std::string &label,
                                    const std::string &tensor_key,
                                    const std::vector<std::string> &chunk_labels,
                                    const std::vector<int64_t> &chunk_indices) {
    check_direct_supported();
    PROFILE_SCOPE_FMT("unzip_from_mem_direct %s/%s", label.c_str(), tensor_key.c_str());
    TORCH_CHECK(chunk_indices.size() == chunk_labels.size(),
                "unzip_from_mem_direct: chunk_indices and chunk_labels must match");
    const int64_t limit = num_chunks();
    for (int64_t idx : chunk_indices)
        TORCH_CHECK(idx >= 0 && idx < limit, "unzip_from_mem_direct: chunk index out of range");

    reset_async_state();

    auto promise = std::make_shared<std::promise<void>>();
    unzip_future_ = promise->get_future();

    auto work = std::make_shared<UnzipFromMemWork>();
    work->ctx = this;
    work->mem = &mem;
    work->chunk_labels = kv_pool::make_chunk_labels(label, tensor_key, chunk_labels);
    work->chunk_indices = chunk_indices;
    work->direct = true;
    work->promise = promise;

    omp_queue().submit(
        [work]() {
            try {
                work->execute(false);
            } catch (...) {
                IAXL_CHECK(false, "unzip_from_mem_direct: unexpected asynchronous exception");
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
