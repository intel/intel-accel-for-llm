// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cassert>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <queue>
#include <vector>

#include <pybind11/pybind11.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>

#include "env.h"
#include "kv_xfer.h"

namespace py = pybind11;

#define GPU_CHECK(call)                                                                            \
    do {                                                                                           \
        cudaError_t err = (call);                                                                  \
        if (err != cudaSuccess) {                                                                  \
            fprintf(stderr, "CUDA error %s:%d: %s returned %s\n", __FILE__, __LINE__, #call,       \
                    cudaGetErrorString(err));                                                      \
            abort();                                                                               \
        }                                                                                          \
    } while (0)

#define CUDA_CHECK(call) GPU_CHECK(call)

using GpuStream_t = cudaStream_t;
using GpuEvent_t = cudaEvent_t;

class GpuEventPool {
  public:
    static GpuEventPool &instance() {
        static GpuEventPool pool;
        return pool;
    }

    GpuEvent_t acquire() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!pool_.empty()) {
            GpuEvent_t event = pool_.front();
            pool_.pop();
            return event;
        }
        GpuEvent_t event;
        GPU_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
        return event;
    }

    void release(GpuEvent_t event) {
        if (event) {
            std::lock_guard<std::mutex> lock(mutex_);
            pool_.push(event);
        }
    }

    ~GpuEventPool() {
        while (!pool_.empty()) {
            cudaEventDestroy(pool_.front());
            pool_.pop();
        }
    }

  private:
    GpuEventPool() = default;
    std::mutex mutex_;
    std::queue<GpuEvent_t> pool_;
};

static inline void gpu_set_device(int device_index) { GPU_CHECK(cudaSetDevice(device_index)); }

static inline GpuStream_t gpu_get_current_stream(int device_index) {
    return at::cuda::getCurrentCUDAStream(device_index).stream();
}

static inline unsigned long long gpu_get_stream_id(GpuStream_t stream) {
    unsigned long long id = 0;
    cudaStreamGetId(stream, &id);
    return id;
}

static inline void gpu_event_record(GpuEvent_t event, GpuStream_t stream) {
    GPU_CHECK(cudaEventRecord(event, stream));
}

static inline void gpu_event_synchronize(GpuEvent_t event) {
    GPU_CHECK(cudaEventSynchronize(event));
}

static inline void gpu_stream_wait_event(GpuStream_t stream, GpuEvent_t event) {
    GPU_CHECK(cudaStreamWaitEvent(stream, event, 0));
}

static inline void gpu_stream_synchronize(GpuStream_t stream) {
    GPU_CHECK(cudaStreamSynchronize(stream));
}

static inline GpuEvent_t gpu_event_create() {
    GpuEvent_t event;
    GPU_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
    return event;
}

static inline void gpu_event_destroy(GpuEvent_t event) { cudaEventDestroy(event); }

static inline GpuStream_t gpu_extract_stream(py::object work_stream) {
    uintptr_t stream_ptr = work_stream.attr("cuda_stream").cast<uintptr_t>();
    return reinterpret_cast<GpuStream_t>(stream_ptr);
}

static inline GpuEvent_t gpu_wait_stream_from_py(py::object cur_stream) {
    GpuStream_t cuda_stream = gpu_extract_stream(cur_stream);
    GpuEvent_t wait_event = gpu_event_create();
    gpu_event_record(wait_event, cuda_stream);
    return wait_event;
}

#define USE_CUDA_MEMCPY_3D_BATCH 1

#define USE_CUDA_MEMCPY_2D 0

namespace kv_xfer {

struct XferContext {
    int device_index;
    GpuStream_t cur_stream;
    GpuStream_t work_stream;
    unsigned long long stream_id;
    char *gpu_base_ptr;
    int64_t chunk_stride;
    int64_t outer_dims;
    int64_t inner_size;
    int64_t outer_block_size;
};

static inline XferContext *as_ctx(context_t c) { return static_cast<XferContext *>(c); }

event_t event_acquire() { return GpuEventPool::instance().acquire(); }

void event_release(event_t event) {
    GpuEventPool::instance().release(static_cast<GpuEvent_t>(event));
}

event_t event_create() { return gpu_event_create(); }

void event_destroy(event_t event) { gpu_event_destroy(static_cast<GpuEvent_t>(event)); }

void event_synchronize(event_t event) { gpu_event_synchronize(static_cast<GpuEvent_t>(event)); }

stream_t extract_stream(pybind11::object stream) { return gpu_extract_stream(stream); }

event_t wait_stream_from_py(pybind11::object stream) { return gpu_wait_stream_from_py(stream); }

context_t context_create(char *gpu_base_ptr, int device_index, int64_t chunk_stride,
                         int64_t outer_dims, int64_t inner_size, int64_t outer_block_size,
                         stream_t work_stream) {
    XferContext *x = new XferContext();
    x->device_index = device_index;
    x->cur_stream = gpu_get_current_stream(device_index);
    x->work_stream = work_stream ? static_cast<GpuStream_t>(work_stream) : x->cur_stream;
    x->stream_id = gpu_get_stream_id(x->work_stream);
    x->gpu_base_ptr = gpu_base_ptr;
    x->chunk_stride = chunk_stride;
    x->outer_dims = outer_dims;
    x->inner_size = inner_size;
    x->outer_block_size = outer_block_size;
    return x;
}

void context_destroy(context_t ctx) {
    if (!ctx)
        return;
#if defined(DSA_SUPPORT)
    if (envs.IAXL_DSA_GD_RESET_ON_DESTROY)
        dsa_context_reset();
#endif
    delete as_ctx(ctx);
}

unsigned long long context_stream_id(context_t ctx) { return as_ctx(ctx)->stream_id; }

bool context_same_stream(context_t ctx) {
    XferContext *x = as_ctx(ctx);
    return x->cur_stream == x->work_stream;
}

void context_record_event(context_t ctx, event_t event) {
    XferContext *x = as_ctx(ctx);
    gpu_set_device(x->device_index);
    gpu_event_record(static_cast<GpuEvent_t>(event), x->work_stream);
}

void context_work_wait_event(context_t ctx, event_t event) {
    gpu_stream_wait_event(as_ctx(ctx)->work_stream, static_cast<GpuEvent_t>(event));
}

void context_cur_wait_event(context_t ctx, event_t event) {
    gpu_stream_wait_event(as_ctx(ctx)->cur_stream, static_cast<GpuEvent_t>(event));
}

void context_work_wait_cur(context_t ctx) {
    XferContext *x = as_ctx(ctx);
    if (x->cur_stream == x->work_stream)
        return;
    gpu_set_device(x->device_index);
    GpuEvent_t ev = gpu_event_create();
    gpu_event_record(ev, x->cur_stream);
    gpu_stream_wait_event(x->work_stream, ev);
    gpu_event_destroy(ev);
}

void context_sync_cur(context_t ctx) { gpu_stream_synchronize(as_ctx(ctx)->cur_stream); }

// Size of the largest PCI memory aperture of the GPU owning `ptr` (what nvidia-smi reports
// as BAR1 and GDRCopy maps into), read from sysfs; 0 if unavailable.
size_t gpu_bar_total(uintptr_t ptr) {
    cudaPointerAttributes attr{};
    char bus[32], path[96];
    if (cudaPointerGetAttributes(&attr, reinterpret_cast<void *>(ptr)) != cudaSuccess ||
        cudaDeviceGetPCIBusId(bus, sizeof bus, attr.device) != cudaSuccess)
        return 0;
    for (char *c = bus; *c; c++) // CUDA gives "0000:3A:00.0", sysfs names are lowercase
        *c = tolower(*c);
    snprintf(path, sizeof path, "/sys/bus/pci/devices/%s/resource", bus);
    FILE *f = fopen(path, "r");
    if (!f)
        return 0;
    // Only the six standard BARs: later lines are the ROM and SR-IOV VF windows.
    unsigned long long start, end, flags, best = 0;
    for (int i = 0; i < 6 && fscanf(f, "%llx %llx %llx", &start, &end, &flags) == 3; i++)
        if (end > start)
            best = std::max(best, end - start + 1);
    fclose(f);
    return best;
}

void copy_chunk(context_t ctx, char *cpu_base, int64_t chunk_index, bool h2d) {
    XferContext *x = as_ctx(ctx);
    gpu_set_device(x->device_index);
    char *gpu_base = x->gpu_base_ptr + chunk_index * x->chunk_stride;
    GpuStream_t stream = x->work_stream;
    for (int64_t outer = 0; outer < x->outer_dims; outer++) {
        char *gpu_ptr = gpu_base + outer * x->outer_block_size;
        char *cpu_ptr = cpu_base + outer * x->inner_size;
        if (h2d) {
            CUDA_CHECK(
                cudaMemcpyAsync(gpu_ptr, cpu_ptr, x->inner_size, cudaMemcpyHostToDevice, stream));
        } else {
            CUDA_CHECK(
                cudaMemcpyAsync(cpu_ptr, gpu_ptr, x->inner_size, cudaMemcpyDeviceToHost, stream));
        }
    }
}

void copy_chunks_batch(context_t ctx, const std::vector<int64_t> &chunk_indices,
                       const std::vector<char *> &cpu_ptrs, bool h2d) {
    XferContext *x = as_ctx(ctx);
    gpu_set_device(x->device_index);

    char *gpu_base_ptr = x->gpu_base_ptr;
    int64_t chunk_stride = x->chunk_stride;
    int64_t outer_dims = x->outer_dims;
    int64_t inner_size = x->inner_size;
    int64_t outer_block_size = x->outer_block_size;

    static std::once_flag layout_noted;
    std::call_once(layout_noted, [&] {
        fprintf(stderr,
                "[kv_xfer] copy_chunks_batch: contiguous block size = %.2f KB, blocks = %zu\n",
                inner_size / 1024.0, chunk_indices.size());
    });
#if defined(DSA_SUPPORT)

    if (dsa_copy_chunks_batch(gpu_base_ptr, chunk_stride, outer_dims, inner_size, outer_block_size,
                              h2d, chunk_indices, cpu_ptrs)) {
        static std::once_flag dsa_noted;
        std::call_once(dsa_noted, [] {
            fprintf(stderr, "[kv_xfer] copy_chunks_batch: using Intel DSA backend\n");
        });
        return;
    }
#endif

#if USE_CUDA_MEMCPY_3D_BATCH
    static const char *kCudaApi = "cudaMemcpy3DBatchAsync";
#elif USE_CUDA_MEMCPY_2D
    static const char *kCudaApi = "cudaMemcpy2DAsync";
#else
    static const char *kCudaApi = "cudaMemcpyAsync";
#endif
    static std::once_flag cuda_noted;
    std::call_once(cuda_noted, [] {
        fprintf(stderr, "[kv_xfer] copy_chunks_batch: using CUDA backend (%s)\n", kCudaApi);
    });

    GpuStream_t stream = x->work_stream;
#if USE_CUDA_MEMCPY_3D_BATCH

    size_t num_chunks = chunk_indices.size();
    std::vector<cudaMemcpy3DBatchOp> ops(num_chunks);

    for (size_t i = 0; i < num_chunks; i++) {
        memset(&ops[i], 0, sizeof(cudaMemcpy3DBatchOp));

        char *gpu_base = gpu_base_ptr + chunk_indices[i] * chunk_stride;
        char *cpu_base = cpu_ptrs[i];

        ops[i].src.type = cudaMemcpyOperandTypePointer;

        ops[i].dst.type = cudaMemcpyOperandTypePointer;

        if (h2d) {

            ops[i].src.op.ptr.ptr = cpu_base;
            ops[i].src.op.ptr.rowLength = 0;
            ops[i].src.op.ptr.layerHeight = 0;

            ops[i].dst.op.ptr.ptr = gpu_base;
            ops[i].dst.op.ptr.rowLength = outer_block_size;
            ops[i].dst.op.ptr.layerHeight = 0;
        } else {

            ops[i].src.op.ptr.ptr = gpu_base;
            ops[i].src.op.ptr.rowLength = outer_block_size;
            ops[i].src.op.ptr.layerHeight = 0;

            ops[i].dst.op.ptr.ptr = cpu_base;
            ops[i].dst.op.ptr.rowLength = 0;
            ops[i].dst.op.ptr.layerHeight = 0;
        }

        ops[i].extent = make_cudaExtent(inner_size, outer_dims, 1);

        ops[i].srcAccessOrder = cudaMemcpySrcAccessOrderStream;
        ops[i].flags = 0;
    }

    size_t failIdx = SIZE_MAX;
#if CUDART_VERSION >= 13000

    (void)failIdx;
    CUDA_CHECK(cudaMemcpy3DBatchAsync(num_chunks, ops.data(), 0, stream));
#else

    CUDA_CHECK(cudaMemcpy3DBatchAsync(num_chunks, ops.data(), &failIdx, 0, stream));
    assert(failIdx == SIZE_MAX && "cudaMemcpy3DBatchAsync: operation failed");
#endif

#elif USE_CUDA_MEMCPY_2D

    for (size_t i = 0; i < chunk_indices.size(); i++) {
        char *gpu_base = gpu_base_ptr + chunk_indices[i] * chunk_stride;
        char *cpu_base = cpu_ptrs[i];

        if (h2d) {

            CUDA_CHECK(cudaMemcpy2DAsync(gpu_base, outer_block_size, cpu_base, inner_size,
                                         inner_size, outer_dims, cudaMemcpyHostToDevice, stream));
        } else {

            CUDA_CHECK(cudaMemcpy2DAsync(cpu_base, inner_size, gpu_base, outer_block_size,
                                         inner_size, outer_dims, cudaMemcpyDeviceToHost, stream));
        }
    }

#else

    for (size_t i = 0; i < chunk_indices.size(); i++) {
        char *gpu_base = gpu_base_ptr + chunk_indices[i] * chunk_stride;
        char *cpu_base = cpu_ptrs[i];

        for (int64_t outer = 0; outer < outer_dims; outer++) {
            char *gpu_ptr = gpu_base + outer * outer_block_size;
            char *cpu_ptr = cpu_base + outer * inner_size;

            if (h2d) {
                CUDA_CHECK(
                    cudaMemcpyAsync(gpu_ptr, cpu_ptr, inner_size, cudaMemcpyHostToDevice, stream));
            } else {
                CUDA_CHECK(
                    cudaMemcpyAsync(cpu_ptr, gpu_ptr, inner_size, cudaMemcpyDeviceToHost, stream));
            }
        }
    }
#endif
}

} // namespace kv_xfer
