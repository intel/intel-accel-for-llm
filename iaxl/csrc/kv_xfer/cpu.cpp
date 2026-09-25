#include "kv_xfer.h"

#include <torch/extension.h>

#include <cstring>
#include <stdexcept>

#include "kv_zip.h"

namespace kv_xfer {

struct XferContext {
    char *tensor_base;
    int64_t chunk_stride;
    int64_t outer_dims;
    int64_t inner_size;
    int64_t outer_block_size;
};

event_t event_acquire() { return nullptr; }

void event_release(event_t) {}

event_t event_create() { return nullptr; }

void event_destroy(event_t) {}

void event_synchronize(event_t) {}

stream_t extract_stream(pybind11::object stream) {
    if (!stream.is_none()) {
        throw std::invalid_argument("CPU transfers do not accept accelerator streams");
    }
    return nullptr;
}

event_t wait_stream_from_py(pybind11::object stream) {
    return extract_stream(stream);
}

context_t context_create(char *tensor_base, int, int64_t chunk_stride,
                         int64_t outer_dims, int64_t inner_size, int64_t outer_block_size,
                         stream_t work_stream) {
    if (work_stream) {
        throw std::invalid_argument("CPU transfers do not accept accelerator streams");
    }
    return new XferContext{tensor_base, chunk_stride, outer_dims, inner_size, outer_block_size};
}

void context_destroy(context_t ctx) { delete static_cast<XferContext *>(ctx); }

unsigned long long context_stream_id(context_t) { return 0; }

bool context_same_stream(context_t) { return true; }

void context_record_event(context_t, event_t) {}

void context_work_wait_event(context_t, event_t) {}

void context_cur_wait_event(context_t, event_t) {}

void context_work_wait_cur(context_t) {}

void context_sync_cur(context_t) {}

void copy_chunk(context_t ctx, char *cpu_base, int64_t chunk_index, bool h2d) {
    const auto *context = static_cast<const XferContext *>(ctx);
    char *tensor_base = context->tensor_base + chunk_index * context->chunk_stride;
    for (int64_t outer = 0; outer < context->outer_dims; outer++) {
        char *tensor_ptr = tensor_base + outer * context->outer_block_size;
        char *scratch_ptr = cpu_base + outer * context->inner_size;
        if (h2d) {
            std::memcpy(tensor_ptr, scratch_ptr, context->inner_size);
        } else {
            std::memcpy(scratch_ptr, tensor_ptr, context->inner_size);
        }
    }
}

void copy_chunks_batch(context_t ctx, const std::vector<int64_t> &chunk_indices,
                       const std::vector<char *> &cpu_ptrs, bool h2d) {
    if (chunk_indices.size() != cpu_ptrs.size()) {
        throw std::invalid_argument("Chunk index and scratch buffer counts must match");
    }
    const auto *context = static_cast<const XferContext *>(ctx);
    std::vector<kv_zip::CopySegment> segments;
    segments.reserve(chunk_indices.size() * context->outer_dims);
    for (size_t chunk = 0; chunk < chunk_indices.size(); chunk++) {
        char *tensor_base = context->tensor_base + chunk_indices[chunk] * context->chunk_stride;
        for (int64_t outer = 0; outer < context->outer_dims; outer++) {
            char *tensor_ptr = tensor_base + outer * context->outer_block_size;
            char *scratch_ptr = cpu_ptrs[chunk] + outer * context->inner_size;
            const size_t n = static_cast<size_t>(context->inner_size);
            segments.push_back(h2d ? kv_zip::CopySegment{tensor_ptr, scratch_ptr, n}
                                   : kv_zip::CopySegment{scratch_ptr, tensor_ptr, n});
        }
    }
    // Runs on the single transfer thread, so DSA (when enabled) is the only way to parallelise.
    kv_zip::copy_segments(segments, false);
}

}