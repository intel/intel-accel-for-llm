// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

// DSA "v1" flavour of kv_xfer::Ops (CUDA + DSA builds; IAXL_DSA_V1_ENABLE selects
// Context::create_dsa_v1). Like rdma.cpp the GPU memory is registered once, up front:
// dsa_v1_register_mem GDR-maps a whole kvcache region and every later context on a
// tensor inside it is plain address arithmetic on the cached BAR alias, instead of
// dsa.cpp's per-call mapping lookups. Copies are queued with dsa_memcpy_batch_async and
// drained by copy_wait; the per-context event is unused. The wrapped CUDA context only
// serves stream synchronisation.

#include <cstdint>
#include <cstdio>
#include <map>
#include <mutex>
#include <vector>

#include "env.h"
#include "iaxl_common.h"
#include "kv_xfer.h"

extern "C" {
#include "dsa_gd.h"
}

extern "C" int dsa_memcpy_batch_async(void *const dest[], const void *const src[], const size_t n[],
                                      size_t count);
extern "C" int dsa_memcpy_batch_wait(void);

namespace kv_xfer {

namespace {

struct XferContext {
    context_t cuda; // CUDA backend context, used only for stream synchronisation
    char *bar;      // BAR-mapped CPU alias of the tensor base
    int64_t chunk_stride, outer_dims, inner_size, outer_block_size;
};

struct Reg {
    char *bar;
    size_t bytes;
};

// Guards the registry and the single-threaded dsa_memcpy_batch_* queue state.
std::mutex mu;
std::map<uintptr_t, Reg> regs; // registered GPU regions keyed by base address

inline XferContext *as_ctx(context_t c) { return static_cast<XferContext *>(c); }

// BAR alias of base if [base, base + bytes) lies inside one registered region, else null.
char *lookup_locked(uintptr_t base, size_t bytes) {
    auto it = regs.upper_bound(base);
    if (it == regs.begin())
        return nullptr;
    --it;
    if (base + bytes > it->first + it->second.bytes)
        return nullptr;
    return it->second.bar + (base - it->first);
}

char *register_locked(uintptr_t base, size_t bytes) {
    IAXL_CHECK(envs.IAXL_DSA_GD_ENABLE && !envs.IAXL_DSA_GD_RESET_ON_DESTROY,
               "dsa_v1 needs IAXL_DSA_GD_ENABLE=1 and IAXL_DSA_GD_RESET_ON_DESTROY=0");
    void *bar = nullptr;
    IAXL_CHECK(dsa_gd_default_gpu_bar_addr(base, bytes, &bar) == 0,
               "dsa_v1: GDRCopy mapping of the GPU region failed");
    regs[base] = Reg{static_cast<char *>(bar), bytes};
    fprintf(stderr, "[kv_xfer/dsa_v1] registered 0x%lx (%.1f MB) -> %p\n", (unsigned long)base,
            bytes / 1048576.0, bar);
    return static_cast<char *>(bar);
}

} // namespace

namespace dsa_v1 {

// Completion is already guaranteed by xfer_wait (future.get + copy_wait) and tracked by
// Context::event_recorded_, so the per-context event carries no state here.
event_t event_acquire() { return nullptr; }
void event_release(event_t) {}
void event_synchronize(event_t) {}
void context_record_event(context_t, event_t) {}
void context_cur_wait_event(context_t, event_t) {}

void context_destroy(context_t ctx) {
    XferContext *x = as_ctx(ctx);
    if (!x)
        return;
    kv_xfer::context_destroy(x->cuda);
    delete x;
}

unsigned long long context_stream_id(context_t ctx) {
    return kv_xfer::context_stream_id(as_ctx(ctx)->cuda);
}
bool context_same_stream(context_t ctx) {
    return kv_xfer::context_same_stream(as_ctx(ctx)->cuda);
}

// `event` is a CUDA event from wait_stream_from_py; CPU-driven DMA cannot be ordered on a
// stream, so block this worker until the GPU reaches it.
void context_work_wait_event(context_t, event_t event) { kv_xfer::event_synchronize(event); }
void context_work_wait_cur(context_t ctx) { kv_xfer::context_sync_cur(as_ctx(ctx)->cuda); }
void context_sync_cur(context_t ctx) { kv_xfer::context_sync_cur(as_ctx(ctx)->cuda); }

void copy_chunks_batch(context_t ctx, const std::vector<int64_t> &chunk_indices,
                       const std::vector<char *> &cpu_ptrs, bool h2d) {
    XferContext *x = as_ctx(ctx);
    const size_t n = chunk_indices.size();
    IAXL_CHECK(n == cpu_ptrs.size(), "dsa_v1: chunk_indices and cpu_ptrs length mismatch");
    const size_t count = n * static_cast<size_t>(x->outer_dims);
    if (count == 0)
        return;

    std::vector<void *> dest(count);
    std::vector<const void *> src(count);
    std::vector<size_t> nbytes(count, static_cast<size_t>(x->inner_size));
    for (size_t i = 0, k = 0; i < n; i++) {
        char *gpu_chunk = x->bar + chunk_indices[i] * x->chunk_stride;
        for (int64_t o = 0; o < x->outer_dims; o++, k++) {
            char *gpu = gpu_chunk + o * x->outer_block_size;
            char *cpu = cpu_ptrs[i] + o * x->inner_size;
            dest[k] = h2d ? gpu : cpu;
            src[k] = h2d ? cpu : gpu;
        }
    }

    static std::once_flag noted;
    std::call_once(noted, [&] {
        fprintf(stderr, "[kv_xfer] copy_chunks_batch: using DSA v1 backend (%ld B x %ld per block)\n",
                x->inner_size, x->outer_dims);
    });

    std::lock_guard<std::mutex> lock(mu);
    IAXL_CHECK(dsa_memcpy_batch_async(dest.data(), src.data(), nbytes.data(), count) == 0,
               "dsa_v1: DSA batch submit failed");
}

void copy_chunk(context_t ctx, char *cpu_base, int64_t chunk_index, bool h2d) {
    copy_chunks_batch(ctx, {chunk_index}, {cpu_base}, h2d);
}

void copy_wait(context_t) {
    std::lock_guard<std::mutex> lock(mu);
    IAXL_CHECK(dsa_memcpy_batch_wait() == 0, "dsa_v1: DSA batch failed");
}

} // namespace dsa_v1

const Ops &dsa_v1_ops() {
    static const Ops ops{dsa_v1::event_acquire,         dsa_v1::event_release,
                         dsa_v1::event_synchronize,     dsa_v1::context_destroy,
                         dsa_v1::context_stream_id,     dsa_v1::context_same_stream,
                         dsa_v1::copy_chunk,            dsa_v1::copy_chunks_batch,
                         dsa_v1::context_record_event,  dsa_v1::context_work_wait_event,
                         dsa_v1::context_cur_wait_event, dsa_v1::context_work_wait_cur,
                         dsa_v1::context_sync_cur,      dsa_v1::copy_wait};
    return ops;
}

void dsa_v1_register_mem(uintptr_t base, size_t bytes) {
    std::lock_guard<std::mutex> lock(mu);
    if (!lookup_locked(base, bytes))
        register_locked(base, bytes);
}

context_t dsa_v1_context_create(char *gpu_base_ptr, int device_index, int64_t chunk_stride,
                                int64_t outer_dims, int64_t inner_size, int64_t outer_block_size,
                                stream_t work_stream) {
    IAXL_CHECK(chunk_stride == inner_size && inner_size % 8 == 0,
               "create_dsa_v1: tensor must be contiguous with 8-byte aligned blocks");
    XferContext *x = new XferContext();
    x->cuda = context_create(gpu_base_ptr, device_index, chunk_stride, outer_dims, inner_size,
                             outer_block_size, work_stream);
    {
        const uintptr_t base = reinterpret_cast<uintptr_t>(gpu_base_ptr);
        const size_t bytes = static_cast<size_t>(outer_dims) * outer_block_size;
        std::lock_guard<std::mutex> lock(mu);
        x->bar = lookup_locked(base, bytes);
        if (!x->bar) {
            static std::once_flag warned;
            std::call_once(warned, [&] {
                fprintf(stderr,
                        "[kv_xfer/dsa_v1] warning: 0x%lx (%.1f MB) not covered by "
                        "dsa_v1_register_mem, mapping it now (reported once)\n",
                        (unsigned long)base, bytes / 1048576.0);
            });
            x->bar = register_locked(base, bytes);
        }
    }
    x->chunk_stride = chunk_stride;
    x->outer_dims = outer_dims;
    x->inner_size = inner_size;
    x->outer_block_size = outer_block_size;
    return x;
}

} // namespace kv_xfer
