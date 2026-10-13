// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <vector>

#include <pybind11/pybind11.h>

namespace kv_xfer {

using stream_t = void *;
using event_t = void *;
using context_t = void *;

event_t event_acquire();
void event_release(event_t event);

event_t event_create();
void event_destroy(event_t event);

void event_synchronize(event_t event);

stream_t extract_stream(pybind11::object stream);
event_t wait_stream_from_py(pybind11::object stream);

context_t context_create(char *gpu_base_ptr, int device_index, int64_t chunk_stride,
                         int64_t outer_dims, int64_t inner_size, int64_t outer_block_size,
                         stream_t work_stream);
void context_destroy(context_t ctx);

unsigned long long context_stream_id(context_t ctx);
bool context_same_stream(context_t ctx);

void copy_chunk(context_t ctx, char *cpu_base, int64_t chunk_index, bool h2d);
void copy_chunks_batch(context_t ctx, const std::vector<int64_t> &chunk_indices,
                       const std::vector<char *> &cpu_ptrs, bool h2d);

void context_record_event(context_t ctx, event_t event);
void context_work_wait_event(context_t ctx, event_t event);
void context_cur_wait_event(context_t ctx, event_t event);
void context_work_wait_cur(context_t ctx);
void context_sync_cur(context_t ctx);

#if defined(CUDA_SUPPORT) && defined(DSA_SUPPORT)

// Largest PCI BAR of the GPU owning `ptr` (nvidia-smi's "BAR1 Total"); 0 if unknown. cuda.cpp.
size_t gpu_bar_total(uintptr_t ptr);

void dsa_context_reset();

bool dsa_copy_chunks_batch(char *gpu_base, int64_t chunk_stride, int64_t outer_dims,
                           int64_t inner_size, int64_t outer_block_size, bool is_h2d,
                           const std::vector<int64_t> &chunk_indices,
                           const std::vector<char *> &cpu_ptrs);
// Blocks until every batch submitted by dsa_copy_chunks_batch has completed.
void dsa_copy_wait(context_t ctx);

// DSA v1 backend (dsa_v1.cpp), selected by Context::create_dsa_v1. Register the whole
// kvcache once with dsa_v1_register_mem; contexts on tensors inside it only compute offsets.
struct Ops;
const Ops &dsa_v1_ops();
void dsa_v1_register_mem(uintptr_t base, size_t bytes);
// Fraction [0, 1] of each batch's chunks copied by the GPU copy engine (cuda.cpp) instead of
// DSA, so both engines run in parallel; default 0 = DSA only.
void dsa_v1_set_cuda_ratio(double ratio);
context_t dsa_v1_context_create(char *gpu_base_ptr, int device_index, int64_t chunk_stride,
                                int64_t outer_dims, int64_t inner_size, int64_t outer_block_size,
                                stream_t work_stream);
#endif

// Per-context backend: the GPU (cuda/xpu) free functions above, or the RDMA
// backend in rdma.cpp (kv_xfer_rdma.h). Selected when a Context is created.
struct Ops {
    event_t (*event_acquire)();
    void (*event_release)(event_t);
    void (*event_synchronize)(event_t);
    void (*context_destroy)(context_t);
    unsigned long long (*context_stream_id)(context_t);
    bool (*context_same_stream)(context_t);
    void (*copy_chunk)(context_t, char *, int64_t, bool);
    void (*copy_chunks_batch)(context_t, const std::vector<int64_t> &, const std::vector<char *> &,
                              bool);
    void (*context_record_event)(context_t, event_t);
    void (*context_work_wait_event)(context_t, event_t);
    void (*context_cur_wait_event)(context_t, event_t);
    void (*context_work_wait_cur)(context_t);
    void (*context_sync_cur)(context_t);
    // Waits for copies that complete outside the stream/event (DSA); nullptr if not needed.
    void (*copy_wait)(context_t);
};

inline const Ops &gpu_ops() {
    static const Ops ops{event_acquire,         event_release,          event_synchronize,
                         context_destroy,       context_stream_id,      context_same_stream,
                         copy_chunk,            copy_chunks_batch,      context_record_event,
                         context_work_wait_event, context_cur_wait_event, context_work_wait_cur,
                         context_sync_cur,
#if defined(CUDA_SUPPORT) && defined(DSA_SUPPORT)
                         dsa_copy_wait
#else
                         nullptr
#endif
    };
    return ops;
}

} // namespace kv_xfer
