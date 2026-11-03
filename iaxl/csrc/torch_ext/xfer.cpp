// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "context.h"

#include <chrono>
#include <vector>

void Context::xfer_chunks_batch(const std::vector<int64_t> &chunk_indices,
                                const std::vector<int64_t> &cpu_block_indices) {
    PROFILE_SCOPE_FMT("xfer_chunks_batch(%s,stream=%llu,n=%zu,i0=%ld)", name_.c_str(), stream_id_,
                      chunk_indices.size(), chunk_indices[0]);
    IAXL_CHECK(chunk_indices.size() == cpu_block_indices.size(),
               "xfer_chunks_batch: chunk_indices and cpu_block_indices must have the same length");
    pool_->check_indices(chunk_indices);
    cpu_pool_->check_indices(cpu_block_indices);

    const bool h2d = (direction_ == GpuTransferDirection::H2D);
    auto x = xctx_;
    auto ops = ops_;

    queue_->submit([=, this, chunk_indices = chunk_indices, cpu_block_indices = cpu_block_indices]() {
        PROFILE_SCOPE_FMT("xfer_chunks_batch(%s,stream=%llu,n=%zu,i0=%ld)", name_.c_str(),
                          stream_id_, chunk_indices.size(), chunk_indices[0]);
        xfer_metrics_begin(chunk_indices.size() * pool_->block_bytes);
        ops->copy_chunks_batch(x, chunk_indices, cpu_pool_->block_ptrs(cpu_block_indices), h2d);
    });
}

void Context::xfer_finish() {
    PROFILE_SCOPE_FMT("xfer_finish(%s,stream=%llu)", name_.c_str(), stream_id_);
    auto event = event_;
    auto x = xctx_;
    auto ops = ops_;
    auto *ev_flag = &event_recorded_;

    xfer_last_future_ = queue_->submit([=, this]() {
        PROFILE_SCOPE_FMT("xfer_finish(%s,stream=%llu)", name_.c_str(), stream_id_);
        ops->context_record_event(x, event);
        ev_flag->store(true, std::memory_order_release);
    });
}

void Context::xfer_wait() {
    IAXL_CHECK(xfer_last_future_.valid(), "xfer_wait: xfer_finish must be called first");
    const bool h2d = (direction_ == GpuTransferDirection::H2D);
    METRICS_TIMER_START(metrics_wait);
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
        // The data is in place here; the real completion may have been earlier.
        xfer_metrics_end();
        METRICS_ADD_NS(metrics_wait, g_metrics.xfer(h2d).wait_ns);

        if (h2d && !ops_->context_same_stream(xctx_)) {
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

