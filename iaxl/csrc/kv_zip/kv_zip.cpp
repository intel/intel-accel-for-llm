// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

// QAT, IAA and CPU workers share one task pool. Each worker claims another item when a request
// completes, so the faster backend naturally processes more of the batch. QAT and IAA workers keep
// multiple asynchronous requests in flight, while CPU workers run one synchronous raw-DEFLATE
// request each.
//
// QAT and CPU streams are mutually compatible, IAA streams are compatible with neither, so a
// compressed block records in its header whether IAA produced it and decompression only hands it
// to a backend that can decode it.

#include <torch/extension.h>

#include <omp.h>
#include <atomic>
#include <climits>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <vector>

#include "env.h"
#include "iaxl_common.h"
#include "cpu_zip.h"
#include "iaa_zip.h"
#include "qat_zip.h"
#include "data_shuffle.h"
#include "lossy.h"
#include "kv_zip.h"

#define OMP_SCHEDULE dynamic

// Header word 0 is the payload length with two flag bits: IAA produced the stream (it cannot be
// decoded by QAT/CPU), and the block was byte-plane shuffled before compression. GET must obey
// these rather than the current environment so a persisted cache survives a config change.
#define KV_ZIP_IAA_FLAG (1u << 31)
#define KV_ZIP_SHUFFLE_FLAG (1u << 30)
#define KV_ZIP_LEN_MASK (~(KV_ZIP_IAA_FLAG | KV_ZIP_SHUFFLE_FLAG))

namespace kv_zip {

enum class ZipBackend { QAT, IAA, CPU };

// Backend entry points, indexed by ZipBackend.
struct ZipOps {
    int (*compress)(int slot, void *src, int len);
    int (*decompress)(int slot, void *src, int len);
    int (*wait)(int slot, void **dest, int *len);
    int (*src_cap)(void);
    int (*queue_depth)(void);
};

static const ZipOps kZipOps[] = {
    {qat_zip_compress, qat_zip_decompress, qat_zip_wait, qat_zip_src_cap, qat_zip_queue_depth},
    {iaa_zip_compress, iaa_zip_decompress, iaa_zip_wait, iaa_zip_src_cap, iaa_zip_queue_depth},
    {cpu_zip_compress, cpu_zip_decompress, cpu_zip_wait, cpu_zip_src_cap, cpu_zip_queue_depth},
};

static const ZipOps &ops(ZipBackend backend) { return kZipOps[static_cast<int>(backend)]; }

// Returned by get_next once a backend has nothing left to claim.
static constexpr size_t kNoTask = static_cast<size_t>(-1);

static void pin_codec_thread() {
    static thread_local bool pinned = false;
    if (!pinned) {
        pinned = true;
        iaxl_apply_thread_affinity();
    }
}

static void ensure_zip_init() {
    static std::once_flag flag;
    std::call_once(flag, [] {
        if (envs.IAXL_QAT_ZIP_ENABLE)
            IAXL_CHECK(qat_zip_init() == 0, "kv_zip: qat_zip_init failed");
        if (envs.IAXL_IAA_ZIP_ENABLE)
            IAXL_CHECK(iaa_zip_init() == 0, "kv_zip: iaa_zip_init failed");
        if (envs.IAXL_CPU_ZIP_ENABLE)
            IAXL_CHECK(cpu_zip_init() == 0, "kv_zip: cpu_zip_init failed");
    });
}

template <class Next, class Submit, class Complete>
static void zip_pipeline(Next &&get_next, Submit &&submit, Complete &&complete) {
    ensure_zip_init();
    const int qat_workers = envs.IAXL_QAT_ZIP_ENABLE ? envs.IAXL_QAT_INSTANCE_NUM : 0;
    const int iaa_workers = envs.IAXL_IAA_ZIP_ENABLE ? envs.IAXL_IAA_INSTANCE_NUM : 0;
    const int cpu_workers = envs.IAXL_CPU_ZIP_ENABLE ? cpu_zip_num_slots() : 0;
    const int worker_count = qat_workers + iaa_workers + cpu_workers;
    IAXL_CHECK(qat_workers == 0 || qat_workers <= qat_zip_num_slots() / qat_zip_queue_depth(),
               "kv_zip: IAXL_QAT_INSTANCE_NUM exceeds available QAT instances");
    IAXL_CHECK(iaa_workers == 0 || iaa_workers <= iaa_zip_num_slots() / iaa_zip_queue_depth(),
               "kv_zip: IAXL_IAA_INSTANCE_NUM exceeds available IAA instances");
    IAXL_CHECK(worker_count == envs.IAXL_OMP_THREAD_NUM,
               "kv_zip: compression workers do not match the configured OpenMP team");

#pragma omp parallel num_threads(worker_count)
    {
        pin_codec_thread();
        const int t = omp_get_thread_num();
        ZipBackend backend = ZipBackend::CPU;
        int first = qat_workers + iaa_workers;
        if (t < qat_workers) {
            backend = ZipBackend::QAT;
            first = 0;
        } else if (t < qat_workers + iaa_workers) {
            backend = ZipBackend::IAA;
            first = qat_workers;
        }
        const int depth = ops(backend).queue_depth();
        const int base = (t - first) * depth;
        IAXL_CHECK(omp_get_num_threads() == worker_count,
                   "kv_zip: OpenMP did not create the configured worker team");

        int active_depth = 0;
        std::vector<size_t> slot_item(static_cast<size_t>(depth));
        for (int k = 0; k < depth; k++) {
            const size_t i = get_next(backend);
            if (i == kNoTask)
                break;
            submit(backend, base + k, i);
            slot_item[k] = i;
            active_depth++;
        }

        int in_flight = active_depth;
        bool draining = false;
        for (int s = 0; in_flight > 0; s = (s + 1) % active_depth) {
            void *out;
            int out_len;
            const int status = ops(backend).wait(base + s, &out, &out_len);
            IAXL_CHECK(status == 0, "kv_zip: zip wait failed");
            complete(backend, slot_item[s], out, out_len);

            const size_t i = draining ? kNoTask : get_next(backend);
            if (i != kNoTask) {
                submit(backend, base + s, i);
                slot_item[s] = i;
            } else {
                draining = true;
                in_flight--;
            }
        }
    }
}

static void pack_header(char *buf, uint32_t payload_len, ZipBackend backend, size_t orig_size,
                        bool shuffled) {
    IAXL_CHECK((payload_len & ~KV_ZIP_LEN_MASK) == 0, "kv_zip: payload length overflows header");
    reinterpret_cast<uint32_t *>(buf)[0] = payload_len |
                                           (backend == ZipBackend::IAA ? KV_ZIP_IAA_FLAG : 0u) |
                                           (shuffled ? KV_ZIP_SHUFFLE_FLAG : 0u);
    reinterpret_cast<int *>(buf)[1] = static_cast<int>(orig_size);
}

void kv_zip_compress_batch(const std::vector<torch::Tensor> &tensors, std::vector<char *> &out_bufs,
                           std::vector<size_t> &out_sizes, std::vector<size_t> &orig_sizes,
                           bool compress) {
    const size_t n = tensors.size();

    if (!compress || !envs.IAXL_KV_COMPRESSION) {
#pragma omp parallel for schedule(OMP_SCHEDULE) num_threads(envs.IAXL_OMP_THREAD_NUM)
        for (size_t i = 0; i < n; i++) {
            const auto &tensor = tensors[i];
            IAXL_CHECK(tensor.is_contiguous() && tensor.device().type() == c10::DeviceType::CPU,
                       "kv_zip: tensor must be a contiguous CPU tensor");
            const size_t nbytes = tensor.numel() * tensor.element_size();
            char *buffer = static_cast<char *>(malloc(sizeof(int) * 2 + nbytes));
            IAXL_CHECK(buffer != nullptr, "kv_zip: raw cache buffer allocation failed");
            reinterpret_cast<int *>(buffer)[0] = 0;
            reinterpret_cast<int *>(buffer)[1] = 0;
            memcpy(buffer + sizeof(int) * 2, tensor.data_ptr(), nbytes);
            out_bufs[i] = buffer;
            out_sizes[i] = sizeof(int) * 2 + nbytes;
            orig_sizes[i] = nbytes;
        }
        return;
    }

    auto prep = [&](size_t i, char **data, size_t *nbytes) {
        const auto &t = tensors[i];
        IAXL_CHECK(t.is_contiguous() && t.device().type() == c10::DeviceType::CPU,
                   "kv_zip: tensor must be a contiguous CPU tensor");
        size_t nb = t.numel() * t.element_size();
        char *p = static_cast<char *>(t.data_ptr());
        lossy_trunc(p, nb, t.element_size());
        data_shuffle(p, nb, t.dtype() == torch::kBFloat16, data_shuffle_enabled());
        orig_sizes[i] = nb;
        *data = p;
        *nbytes = nb;
    };

    auto pack = [&](size_t i, ZipBackend backend, const void *payload, int payload_len) {
        char *buf = static_cast<char *>(malloc(sizeof(int) * 2 + payload_len));
        IAXL_CHECK(buf != nullptr, "kv_zip: cache buffer allocation failed");
        // data_shuffle() only transforms bf16 blocks, so this is what prep actually applied.
        const bool shuffled =
            data_shuffle_enabled() && tensors[i].dtype() == torch::kBFloat16;
        pack_header(buf, static_cast<uint32_t>(payload_len), backend, orig_sizes[i], shuffled);
        memcpy(buf + sizeof(int) * 2, payload, payload_len);
        out_bufs[i] = buf;
        out_sizes[i] = sizeof(int) * 2 + payload_len;
    };

    std::atomic<size_t> next{0};
    zip_pipeline(
        [&](ZipBackend) {
            const size_t i = next.fetch_add(1, std::memory_order_relaxed);
            return i < n ? i : kNoTask;
        },
        [&](ZipBackend backend, int slot, size_t i) {
            char *data;
            size_t nb;
            prep(i, &data, &nb);
            IAXL_CHECK(nb <= static_cast<size_t>(INT_MAX),
                       "kv_zip: tensor byte size exceeds zip integer length range");
            IAXL_CHECK(nb <= static_cast<size_t>(ops(backend).src_cap()),
                       "kv_zip: tensor byte size exceeds zip source capacity");
            const int status = ops(backend).compress(slot, data, static_cast<int>(nb));
            IAXL_CHECK(status == 0, "kv_zip: zip compress failed");
        },
        [&](ZipBackend backend, size_t i, void *out, int out_len) {
            pack(i, backend, out, out_len);
        });
}

void kv_zip_decompress_batch(const std::vector<const char *> &data_ptrs,
                             const std::vector<torch::Tensor> &tensors) {
    const size_t n = tensors.size();
    IAXL_CHECK(data_ptrs.size() == n, "kv_zip: decompression inputs must have matching lengths");

    auto copy_raw = [&](size_t i) {
        const auto &t = tensors[i];
        const size_t nb = t.numel() * t.element_size();
        memcpy(t.data_ptr(), data_ptrs[i] + sizeof(int) * 2, nb);
    };

    // IAA-produced blocks must go back to IAA, everything else to QAT/CPU.
    std::vector<size_t> iaa_items, other_items;
    for (size_t i = 0; i < n; i++) {
        const int *header = reinterpret_cast<const int *>(data_ptrs[i]);
        if (header[1] == 0)
            continue;
        const uint32_t encoded = static_cast<uint32_t>(header[0]);
        IAXL_CHECK((encoded & KV_ZIP_LEN_MASK) != 0,
                   "kv_zip: invalid compressed payload length");
        ((encoded & KV_ZIP_IAA_FLAG) ? iaa_items : other_items).push_back(i);
    }

#pragma omp parallel for schedule(OMP_SCHEDULE) num_threads(envs.IAXL_OMP_THREAD_NUM)
    for (size_t i = 0; i < n; i++) {
        const int *header = reinterpret_cast<const int *>(data_ptrs[i]);
        if (header[1] == 0)
            copy_raw(i);
    }

    if (iaa_items.empty() && other_items.empty())
        return;
    IAXL_CHECK(iaa_items.empty() || envs.IAXL_IAA_ZIP_ENABLE,
               "kv_zip: batch holds IAA-compressed blocks but the IAA backend is disabled");
    IAXL_CHECK(other_items.empty() || envs.IAXL_QAT_ZIP_ENABLE || envs.IAXL_CPU_ZIP_ENABLE,
               "kv_zip: batch holds QAT/CPU-compressed blocks but both backends are disabled");

    auto finish = [&](size_t i, const void *out, int out_len) {
        const auto &t = tensors[i];
        const size_t nb = t.numel() * t.element_size();
        IAXL_CHECK(out_len >= 0 && static_cast<size_t>(out_len) == nb,
                   "kv_zip: decompressed size does not match tensor byte size");
        char *dst = static_cast<char *>(t.data_ptr());
        memcpy(dst, out, nb);
        // The header, not the environment, says whether the payload was shuffled.
        const bool shuffled =
            reinterpret_cast<const uint32_t *>(data_ptrs[i])[0] & KV_ZIP_SHUFFLE_FLAG;
        data_shuffle(dst, nb, t.dtype() == torch::kBFloat16, shuffled);
    };

    std::atomic<size_t> iaa_next{0}, other_next{0};
    zip_pipeline(
        [&](ZipBackend backend) {
            const bool iaa = backend == ZipBackend::IAA;
            const std::vector<size_t> &items = iaa ? iaa_items : other_items;
            std::atomic<size_t> &cursor = iaa ? iaa_next : other_next;
            const size_t k = cursor.fetch_add(1, std::memory_order_relaxed);
            return k < items.size() ? items[k] : kNoTask;
        },
        [&](ZipBackend backend, int slot, size_t i) {
            const uint32_t encoded = reinterpret_cast<const uint32_t *>(data_ptrs[i])[0];
            const char *payload = data_ptrs[i] + sizeof(int) * 2;
            const int payload_len = static_cast<int>(encoded & KV_ZIP_LEN_MASK);
            const int status =
                ops(backend).decompress(slot, const_cast<char *>(payload), payload_len);
            IAXL_CHECK(status == 0, "kv_zip: zip decompress failed");
        },
        [&](ZipBackend, size_t i, void *out, int out_len) { finish(i, out, out_len); });
}

} // namespace kv_zip
