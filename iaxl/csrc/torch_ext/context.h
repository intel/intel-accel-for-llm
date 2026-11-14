// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <torch/extension.h>
#include <vector>
#include <cstdint>
#include <future>
#include <atomic>
#include <queue>
#include <mutex>
#include <omp.h>

#include "task_queue.h"
#include "profiler.h"
#include "env.h"
#include "iaxl_common.h"

using namespace profiler;

#include "kv_pool.h"
#include "kv_xfer.h"
#include "kv_xfer_rdma.h"

enum class GpuTransferDirection { H2D, D2H };

inline TaskQueue &h2d_queue();
inline TaskQueue &d2h_queue();
inline TaskQueue &omp_queue();

class Context {
  public:
    // pool_name: the registered GPU / Remote / CPU-in-place KV pool (= tensor_key);
    // cpu_pool: the CPU pool its blocks are staged in ("scratch", or pool_name when in place).
    static Context create(const std::string &pool_name, const std::string &cpu_pool,
                          GpuTransferDirection direction = GpuTransferDirection::H2D,
                          const std::string &name = "", kv_xfer::stream_t work_stream = nullptr) {
        using kv_pool::PoolDevice;
        Context ctx;
        ctx.name_ = name;
        ctx.direction_ = direction;
        ctx.pool_ = &kv_pool::get_kv_pool(pool_name);
        ctx.cpu_pool_ = &kv_pool::get_kv_pool(cpu_pool);
        IAXL_CHECK(ctx.cpu_pool_->device == PoolDevice::CPU, "cpu_pool is not in CPU memory");

        const kv_pool::KVPool &p = *ctx.pool_;
        // Mixed page sizes share one max-page staging pool, remote pools included:
        // the staging block only has to cover the block being transferred.
        ctx.pool_->check_staging_geometry(*ctx.cpu_pool_);
        if (p.device == PoolDevice::Remote) {
#ifdef CPU_SUPPORT
            TORCH_CHECK(false, "RDMA (remote_pool) needs NIXL, which DEVICE=cpu builds do not link");
#else
            ctx.ops_ = &kv_xfer::rdma_ops();
            ctx.xctx_ = kv_xfer::rdma_context_create(p, *ctx.cpu_pool_);
#endif
        } else {
            ctx.ops_ = &kv_xfer::gpu_ops();
            ctx.xctx_ = kv_xfer::context_create((char *)p.base, p.dev_id, p.chunk_stride,
                                                p.outer_dims, p.inner_size, p.outer_block_size,
                                                work_stream);
        }
        ctx.event_ = ctx.ops_->event_acquire();
        ctx.queue_ = &((direction == GpuTransferDirection::H2D) ? h2d_queue() : d2h_queue());
        ctx.stream_id_ = ctx.ops_->context_stream_id(ctx.xctx_);

        PROFILE_SCOPE_FMT("ctx_create(%s,stream=%llu)", name.c_str(), ctx.stream_id_);
        return ctx;
    }

    // pool_[chunk_indices[i]] <-> cpu_pool_[cpu_block_indices[i]] (async).
    void xfer_chunks_batch(const std::vector<int64_t> &chunk_indices,
                           const std::vector<int64_t> &cpu_block_indices);
    void xfer_finish();
    void xfer_wait();
    bool xfer_is_complete();

    void xfer_wait_cur_stream(bool sync_cur_stream = false);
    void xfer_wait_stream(kv_xfer::event_t wait_event);
    bool xfer_wait_stream(pybind11::object cur_stream);

    void zip_to_mem(kv_pool::Mem &mem, const std::string &label, const std::string &tensor_key,
                    const std::vector<std::string> &chunk_labels,
                    const std::vector<int64_t> &cpu_block_indices, bool compress = true);

    // priority / bundle_id / sort_key go to TaskQueue::submit() of the decompression task.
    void unzip_from_mem(kv_pool::Mem &mem, const std::string &label, const std::string &tensor_key,
                        const std::vector<std::string> &chunk_labels,
                        const std::vector<int64_t> &chunk_indices,
                        const std::vector<int64_t> &cpu_block_indices,
                        int priority = TaskQueue::PRIORITY_HIGH, uint64_t bundle_id = 0,
                        int64_t sort_key = -1);

    void zip_wait();
    bool zip_is_complete();
    void unzip_wait();
    bool unzip_is_complete();

    void reset_async_state() { event_recorded_.store(false, std::memory_order_release); }

    unsigned long long stream_id() const { return stream_id_; }
    kv_xfer::event_t event() const { return event_; }
    TaskQueue &queue() const { return *queue_; }
    const std::string &name() const { return name_; }
    const kv_pool::KVPool &pool() const { return *pool_; }
    const kv_pool::KVPool &cpu_pool() const { return *cpu_pool_; }

    Context() = default;
    ~Context() {
        check_no_pending_work();
        xfer_metrics_end();
        ops_->event_release(event_);
        ops_->context_destroy(xctx_);
    }
    Context(Context &&other) noexcept { *this = std::move(other); }
    Context &operator=(Context &&other) noexcept {
        if (this != &other) {
            check_no_pending_work();
            xfer_metrics_end();
            ops_->event_release(event_);
            ops_->context_destroy(xctx_);

            ops_ = other.ops_;
            xctx_ = other.xctx_;
            stream_id_ = other.stream_id_;
            pool_ = other.pool_;
            cpu_pool_ = other.cpu_pool_;
            direction_ = other.direction_;
            queue_ = other.queue_;
            event_ = other.event_;
            xfer_last_future_ = std::move(other.xfer_last_future_);
            event_recorded_.store(other.event_recorded_.load(std::memory_order_relaxed),
                                  std::memory_order_relaxed);
            xfer_metrics_active_.store(other.xfer_metrics_active_.exchange(false),
                                       std::memory_order_relaxed);
            name_ = std::move(other.name_);
            zip_future_ = std::move(other.zip_future_);
            unzip_future_ = std::move(other.unzip_future_);

            other.xctx_ = nullptr;
            other.event_ = nullptr;
            other.event_recorded_.store(false, std::memory_order_relaxed);
        }
        return *this;
    }
    Context(const Context &) = delete;
    Context &operator=(const Context &) = delete;

  private:
    void check_no_pending_work() const {
        IAXL_CHECK(!xfer_last_future_.valid(), "Context destroyed before xfer_wait completed");
        IAXL_CHECK(!zip_future_.valid(), "Context destroyed before zip_wait completed");
        IAXL_CHECK(!unzip_future_.valid(), "Context destroyed before unzip_wait completed");
    }

    // Called on the H2D/D2H thread as each copy op starts; the first one opens this
    // context's transfer window in g_metrics.
    void xfer_metrics_begin(uint64_t bytes) {
        if (KVCLIP_LIKELY(!g_metrics_enabled))
            return;
        auto &m = g_metrics.xfer(direction_ == GpuTransferDirection::H2D);
        m.bytes.fetch_add(bytes, std::memory_order_relaxed);
        if (!xfer_metrics_active_.exchange(true))
            m.begin();
    }
    void xfer_metrics_end() {
        if (xfer_metrics_active_.exchange(false))
            g_metrics.xfer(direction_ == GpuTransferDirection::H2D).end();
    }

    const kv_xfer::Ops *ops_ = &kv_xfer::gpu_ops();
    kv_xfer::context_t xctx_ = nullptr;
    unsigned long long stream_id_ = 0;
    const kv_pool::KVPool *pool_ = nullptr;
    const kv_pool::KVPool *cpu_pool_ = nullptr;
    GpuTransferDirection direction_ = GpuTransferDirection::H2D;
    TaskQueue *queue_ = nullptr;
    kv_xfer::event_t event_ = nullptr;
    std::future<void> xfer_last_future_;
    std::atomic<bool> event_recorded_{false};
    std::atomic<bool> xfer_metrics_active_{false};
    std::string name_;
    std::future<void> zip_future_;
    std::future<void> unzip_future_;
};

inline TaskQueue &h2d_queue() {
    static TaskQueue queue("H2D");
    static const bool initialized = []() {
        queue.init();
        return true;
    }();
    (void)initialized;
    return queue;
}

inline TaskQueue &d2h_queue() {
    static TaskQueue queue("D2H");
    static const bool initialized = []() {
        queue.init();
        return true;
    }();
    (void)initialized;
    return queue;
}

inline TaskQueue &omp_queue() {
    static TaskQueue queue("OMP-Main");
    static const bool initialized = []() {
        if (envs.IAXL_USE_OMP) {
            int omp_threads = envs.IAXL_OMP_THREAD_NUM;

#pragma omp parallel num_threads(omp_threads)
            {
                IAXL_CHECK(omp_get_num_threads() == omp_threads,
                           "omp_queue: OpenMP did not create the configured worker team");
                int tid = omp_get_thread_num();
                std::string name = "OMP-" + std::to_string(tid);
                profiler::set_thread_name(name.c_str());
            }
        }

        queue.init();
        return true;
    }();
    (void)initialized;

    return queue;
}
