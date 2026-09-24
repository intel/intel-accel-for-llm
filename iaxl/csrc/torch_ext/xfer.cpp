// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "context.h"

#include <chrono>
#include <vector>

void Context::xfer_chunk(const torch::Tensor &cpu_tensor, int64_t chunk_idx) {
    PROFILE_SCOPE_FMT("xfer_chunk(%s,stream=%llu,idx=%ld)", name_.c_str(), stream_id_, chunk_idx);
    char *cpu_base = (char *)cpu_tensor.data_ptr();
    bool h2d = (direction_ == GpuTransferDirection::H2D);
    auto x = xctx_;
    auto ops = ops_;

    queue_->submit([=]() {
        PROFILE_SCOPE_FMT("xfer_chunk(%s,stream=%llu,idx=%ld)", name_.c_str(), stream_id_,
                          chunk_idx);
        ops->copy_chunk(x, cpu_base, chunk_idx, h2d);
    });
}

void Context::xfer_chunks_batch(const std::vector<int64_t> &chunk_indices,
                                const std::vector<torch::Tensor> &cpu_tensors) {
    PROFILE_SCOPE_FMT("xfer_chunks_batch(%s,stream=%llu,n=%zu,i0=%ld)", name_.c_str(), stream_id_,
                      chunk_indices.size(), chunk_indices[0]);

    auto chunk_indices_copy = chunk_indices;
    bool h2d = (direction_ == GpuTransferDirection::H2D);
    auto x = xctx_;
    auto ops = ops_;

    std::vector<char *> cpu_ptrs;
    cpu_ptrs.reserve(cpu_tensors.size());
    for (const auto &tensor : cpu_tensors) {
        cpu_ptrs.push_back((char *)tensor.data_ptr());
    }

    queue_->submit([=, chunk_indices = std::move(chunk_indices_copy)]() {
        PROFILE_SCOPE_FMT("xfer_chunks_batch(%s,stream=%llu,n=%zu,i0=%ld)", name_.c_str(),
                          stream_id_, chunk_indices.size(), chunk_indices[0]);
        ops->copy_chunks_batch(x, chunk_indices, cpu_ptrs, h2d);
    });
}

void Context::xfer_chunks_batch_fast(const torch::Tensor &chunk_indices,
                                     const torch::Tensor &cpu_ptrs) {
    const int64_t n = chunk_indices.numel();
    IAXL_CHECK(n == cpu_ptrs.numel(),
               "xfer_chunks_batch_fast: chunk_indices and cpu_ptrs must have the same length");
    IAXL_CHECK(chunk_indices.scalar_type() == torch::kInt64 && chunk_indices.is_cpu() &&
                   chunk_indices.is_contiguous() && cpu_ptrs.scalar_type() == torch::kInt64 &&
                   cpu_ptrs.is_cpu() && cpu_ptrs.is_contiguous(),
               "xfer_chunks_batch_fast: expected contiguous int64 CPU tensors");

    const int64_t *index_data = chunk_indices.data_ptr<int64_t>();
    char *const *ptr_data = (char *const *)cpu_ptrs.data_ptr<int64_t>();

    PROFILE_SCOPE_FMT("xfer_chunks_batch_fast(%s,stream=%llu,n=%ld,i0=%ld)", name_.c_str(),
                      stream_id_, n, index_data[0]);

    // The caller already resolved the addresses, so this is two bulk copies instead of a scan.
    std::vector<int64_t> indices(index_data, index_data + n);
    std::vector<char *> ptrs(ptr_data, ptr_data + n);

    bool h2d = (direction_ == GpuTransferDirection::H2D);
    auto x = xctx_;
    auto ops = ops_;

    queue_->submit([=, indices = std::move(indices), ptrs = std::move(ptrs)]() {
        PROFILE_SCOPE_FMT("xfer_chunks_batch_fast(%s,stream=%llu,n=%zu,i0=%ld)", name_.c_str(),
                          stream_id_, indices.size(), indices[0]);
        ops->copy_chunks_batch(x, indices, ptrs, h2d);
    });
}

void Context::xfer_finish() {
    PROFILE_SCOPE_FMT("xfer_finish(%s,stream=%llu)", name_.c_str(), stream_id_);
    auto event = event_;
    auto x = xctx_;
    auto ops = ops_;
    auto *ev_flag = &event_recorded_;

    xfer_last_future_ = queue_->submit([=]() {
        PROFILE_SCOPE_FMT("xfer_finish(%s,stream=%llu)", name_.c_str(), stream_id_);
        ops->context_record_event(x, event);
        ev_flag->store(true, std::memory_order_release);
    });
}

void Context::xfer_wait() {
    IAXL_CHECK(xfer_last_future_.valid(), "xfer_wait: xfer_finish must be called first");
    {
        PROFILE_SCOPE_FMT("xfer_wait(%s,stream=%llu)", name_.c_str(), stream_id_);
        xfer_last_future_.get();
        xfer_last_future_ = std::future<void>();
    }
    if (ops_->copy_wait) {
        PROFILE_SCOPE_FMT("copy_wait(%s,stream=%llu)", name_.c_str(), stream_id_);
        ops_->copy_wait(xctx_);
    }
    {
        PROFILE_SCOPE_FMT("event_wait(%s,stream=%llu)", name_.c_str(), stream_id_);
        ops_->event_synchronize(event_);

        if (direction_ == GpuTransferDirection::H2D && !ops_->context_same_stream(xctx_)) {
            ops_->context_cur_wait_event(xctx_, event_);
        }
    }
}

bool Context::xfer_is_complete() {
    if (xfer_last_future_.valid() &&
        xfer_last_future_.wait_for(std::chrono::seconds(0)) != std::future_status::ready) {
        if (envs.IAXL_DEBUG_LOG)
            fprintf(stderr, "[xfer_is_complete] %s: future not ready\n", name_.c_str());
        return false;
    }
    bool recorded = event_recorded_.load(std::memory_order_acquire);
    if (!recorded && envs.IAXL_DEBUG_LOG)
        fprintf(stderr, "[xfer_is_complete] %s: future=%s but event_recorded=false\n",
                name_.c_str(), xfer_last_future_.valid() ? "valid+ready" : "invalid");
    return recorded;
}

void Context::xfer_wait_cur_stream(bool sync_cur_stream) {
    if (ops_->context_same_stream(xctx_) && !sync_cur_stream)
        return;
    if (sync_cur_stream) {
        PROFILE_SCOPE("sync_cur_stream");

        ops_->context_sync_cur(xctx_);
    }
    auto x = xctx_;
    auto ops = ops_;
    queue_->submit([x, ops]() {
        PROFILE_SCOPE("wait_cur_stream");
        ops->context_work_wait_cur(x);
    });
}

void Context::xfer_wait_stream(kv_xfer::event_t wait_event) {
    auto x = xctx_;
    auto ops = ops_;
    queue_->submit([x, ops, wait_event]() {
        PROFILE_SCOPE("wait_stream");
        ops->context_work_wait_event(x, wait_event);
        kv_xfer::event_destroy(wait_event); // wait_event always comes from the GPU backend
    });
}

bool Context::xfer_wait_stream(pybind11::object cur_stream) {
    PROFILE_SCOPE_FMT("wait_stream(%s)", name_.c_str());
    kv_xfer::event_t wait_event = kv_xfer::wait_stream_from_py(cur_stream);
    if (wait_event) {
        xfer_wait_stream(wait_event);
        return true;
    }
    return false;
}

void h2d_xfer_chunks_batch(const torch::Tensor &gpu_tensor, int chunk_dim,
                           const std::vector<int64_t> &chunk_indices,
                           const std::vector<torch::Tensor> &cpu_tensors) {
    auto ctx = Context::create(gpu_tensor, chunk_dim, GpuTransferDirection::H2D);
    ctx.xfer_chunks_batch(chunk_indices, cpu_tensors);
}

void d2h_xfer_chunks_batch(const torch::Tensor &gpu_tensor, int chunk_dim,
                           const std::vector<int64_t> &chunk_indices,
                           const std::vector<torch::Tensor> &cpu_tensors) {
    auto ctx = Context::create(gpu_tensor, chunk_dim, GpuTransferDirection::D2H);
    ctx.xfer_chunks_batch(chunk_indices, cpu_tensors);
}
