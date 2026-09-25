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

#include "kv_xfer.h"
#include "kv_xfer_rdma.h"
#include "kv_zip.h"

namespace kv_pool {
class Mem;
}

enum class GpuTransferDirection { H2D, D2H };

inline TaskQueue &h2d_queue();
inline TaskQueue &d2h_queue();
inline TaskQueue &omp_queue();

class Context {
  public:
    static Context create(const torch::Tensor &tensor, int chunk_dim,
                          GpuTransferDirection direction = GpuTransferDirection::H2D,
                          const std::string &name = "", kv_xfer::stream_t work_stream = nullptr) {
        TORCH_CHECK(tensor.device().type() == c10::Device(IAXL_DEVICE).type(),
                    "IAXL was built for " IAXL_DEVICE " tensors, got ", tensor.device());
        TORCH_CHECK(tensor.is_contiguous(), "IAXL transfer tensors must be contiguous");
        TORCH_CHECK(chunk_dim >= 0 && chunk_dim < tensor.dim(), "Chunk dimension out of range");
        Context ctx;
        ctx.name_ = name;
        ctx.gpu_tensor_ = tensor;
        ctx.direction_ = direction;
        ctx.event_ = ctx.ops_->event_acquire();
        ctx.queue_ = &((direction == GpuTransferDirection::H2D) ? h2d_queue() : d2h_queue());

        int64_t outer_dims = 1;
        for (int d = 0; d < chunk_dim; d++)
            outer_dims *= tensor.size(d);
        int64_t inner_size = tensor.element_size();
        for (int d = chunk_dim + 1; d < tensor.dim(); d++)
            inner_size *= tensor.size(d);
        int64_t chunk_stride = tensor.stride(chunk_dim) * tensor.element_size();
        int64_t outer_block_size = tensor.size(chunk_dim) * inner_size;

        ctx.tensor_base_ = static_cast<char *>(tensor.data_ptr());
        ctx.chunk_dim_ = chunk_dim;
        ctx.chunk_stride_ = chunk_stride;
        ctx.outer_dims_ = outer_dims;
        ctx.inner_size_ = inner_size;
        ctx.outer_block_size_ = outer_block_size;
        ctx.element_size_ = static_cast<int>(tensor.element_size());
        ctx.is_bf16_ = tensor.dtype() == torch::kBFloat16;

        ctx.xctx_ = kv_xfer::context_create((char *)tensor.data_ptr(), tensor.device().index(),
                                            chunk_stride, outer_dims, inner_size, outer_block_size,
                                            work_stream);
        ctx.stream_id_ = ctx.ops_->context_stream_id(ctx.xctx_);

        PROFILE_SCOPE_FMT("ctx_create(%s,stream=%llu)", name.c_str(), ctx.stream_id_);
        return ctx;
    }

    // Remote (RDMA) tensor: only its address and contiguous geometry are known here.
    static Context create_remote(uintptr_t base, int dev_id, const std::vector<int64_t> &shape,
                                 int64_t elem_size, int chunk_dim,
                                 GpuTransferDirection direction = GpuTransferDirection::H2D,
                                 const std::string &name = "") {
        Context ctx;
        ctx.ops_ = &kv_xfer::rdma_ops();
        ctx.name_ = name;
        ctx.direction_ = direction;
        ctx.event_ = ctx.ops_->event_acquire();
        ctx.queue_ = &((direction == GpuTransferDirection::H2D) ? h2d_queue() : d2h_queue());

        int64_t outer_dims = 1;
        for (int d = 0; d < chunk_dim; d++)
            outer_dims *= shape[d];
        int64_t inner_size = elem_size;
        for (size_t d = chunk_dim + 1; d < shape.size(); d++)
            inner_size *= shape[d];
        int64_t outer_block_size = shape[chunk_dim] * inner_size;

        ctx.xctx_ = kv_xfer::rdma_context_create((char *)base, inner_size, outer_dims, inner_size,
                                                 outer_block_size);
        PROFILE_SCOPE_FMT("ctx_create_remote(%s)", name.c_str());
        return ctx;
    }

    void xfer_chunk(const torch::Tensor &cpu_tensor, int64_t chunk_idx);
    void xfer_chunks_batch(const std::vector<int64_t> &chunk_indices,
                           const std::vector<torch::Tensor> &cpu_tensors);
    // Same transfer, but takes the chunk indices and CPU addresses as prebuilt int64 tensors.
    void xfer_chunks_batch_fast(const torch::Tensor &chunk_indices, const torch::Tensor &cpu_ptrs);
    void xfer_finish();
    void xfer_wait();
    bool xfer_is_complete();

    void xfer_wait_cur_stream(bool sync_cur_stream = false);
    void xfer_wait_stream(kv_xfer::event_t wait_event);
    bool xfer_wait_stream(pybind11::object cur_stream);

    void zip_to_mem(kv_pool::Mem &mem, const std::string &label, const std::string &tensor_key,
                    const std::vector<std::string> &chunk_labels,
                    const std::vector<torch::Tensor> &cpu_tensors, bool compress = true);

    void unzip_from_mem(kv_pool::Mem &mem, const std::string &label, const std::string &tensor_key,
                        const std::vector<std::string> &chunk_labels,
                        const std::vector<int64_t> &chunk_indices,
                        const std::vector<torch::Tensor> &cpu_tensors);

    // CPU-build paths that codec straight from / into the inference tensor, no scratch buffers.
    void zip_to_mem_direct(kv_pool::Mem &mem, const std::string &label,
                           const std::string &tensor_key,
                           const std::vector<std::string> &chunk_labels,
                           const std::vector<int64_t> &chunk_indices, bool compress = true);

    void unzip_from_mem_direct(kv_pool::Mem &mem, const std::string &label,
                               const std::string &tensor_key,
                               const std::vector<std::string> &chunk_labels,
                               const std::vector<int64_t> &chunk_indices);

    kv_zip::ChunkView chunk_view(int64_t chunk_idx) const {
        return kv_zip::ChunkView{tensor_base_ + chunk_idx * chunk_stride_, outer_dims_,
                                 inner_size_, outer_block_size_, element_size_, is_bf16_};
    }
    int64_t num_chunks() const { return gpu_tensor_.defined() ? gpu_tensor_.size(chunk_dim_) : 0; }

    void zip_wait();
    bool zip_is_complete();
    void unzip_wait();
    bool unzip_is_complete();

    void reset_async_state() { event_recorded_.store(false, std::memory_order_release); }

    unsigned long long stream_id() const { return stream_id_; }
    kv_xfer::event_t event() const { return event_; }
    TaskQueue &queue() const { return *queue_; }
    const std::string &name() const { return name_; }

    Context() = default;
    ~Context() {
        check_no_pending_work();
        ops_->event_release(event_);
        ops_->context_destroy(xctx_);
    }
    Context(Context &&other) noexcept { *this = std::move(other); }
    Context &operator=(Context &&other) noexcept {
        if (this != &other) {
            check_no_pending_work();
            ops_->event_release(event_);
            ops_->context_destroy(xctx_);

            ops_ = other.ops_;
            xctx_ = other.xctx_;
            stream_id_ = other.stream_id_;
            gpu_tensor_ = std::move(other.gpu_tensor_);
            direction_ = other.direction_;
            tensor_base_ = other.tensor_base_;
            chunk_dim_ = other.chunk_dim_;
            chunk_stride_ = other.chunk_stride_;
            outer_dims_ = other.outer_dims_;
            inner_size_ = other.inner_size_;
            outer_block_size_ = other.outer_block_size_;
            element_size_ = other.element_size_;
            is_bf16_ = other.is_bf16_;
            queue_ = other.queue_;
            event_ = other.event_;
            xfer_last_future_ = std::move(other.xfer_last_future_);
            event_recorded_.store(other.event_recorded_.load(std::memory_order_relaxed),
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

    const kv_xfer::Ops *ops_ = &kv_xfer::gpu_ops();
    kv_xfer::context_t xctx_ = nullptr;
    unsigned long long stream_id_ = 0;
    torch::Tensor gpu_tensor_;
    GpuTransferDirection direction_ = GpuTransferDirection::H2D;
    char *tensor_base_ = nullptr;
    int chunk_dim_ = 0;
    int64_t chunk_stride_ = 0;
    int64_t outer_dims_ = 0;
    int64_t inner_size_ = 0;
    int64_t outer_block_size_ = 0;
    int element_size_ = 0;
    bool is_bf16_ = false;
    TaskQueue *queue_ = nullptr;
    kv_xfer::event_t event_ = nullptr;
    std::future<void> xfer_last_future_;
    std::atomic<bool> event_recorded_{false};
    std::string name_;
    std::future<void> zip_future_;
    std::future<void> unzip_future_;
};

inline void gpu_transfer_batch_pytorch(torch::Tensor &gpu_tensor, int chunk_dim,
                                       const std::vector<int64_t> &chunk_indices,
                                       const std::vector<torch::Tensor> &cpu_tensors,
                                       GpuTransferDirection direction) {
    for (size_t i = 0; i < chunk_indices.size(); i++) {
        auto gpu_slice = gpu_tensor.select(chunk_dim, chunk_indices[i]);
        if (direction == GpuTransferDirection::H2D) {
            gpu_slice.copy_(cpu_tensors[i], true);
        } else {
            cpu_tensors[i].copy_(gpu_slice, true);
        }
    }
}

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
    int omp_threads = envs.IAXL_OMP_THREAD_NUM;

#pragma omp parallel num_threads(omp_threads)
        {
        IAXL_CHECK(omp_get_num_threads() == omp_threads,
               "omp_queue: OpenMP did not create the configured worker team");
            int tid = omp_get_thread_num();
            std::string name = "OMP-" + std::to_string(tid);
            profiler::set_thread_name(name.c_str());
        }

        queue.init();
        return true;
    }();
    (void)initialized;

    return queue;
}
