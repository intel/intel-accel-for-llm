// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

// QAT, IAA and CPU workers share one task pool. Each worker claims another item when a request
// completes, so the faster backend naturally processes more of the batch. QAT and IAA workers keep
// multiple asynchronous requests in flight, while CPU workers run one synchronous raw-DEFLATE
// request each. Up to IAXL_OMP_THREAD_NUM threads (see parallel.h) drive the workers, each polling
// its slots without blocking, so a single thread can drive them all. With more threads than
// workers, the slots of one worker are split between threads.
//
// An item is one segment of a tensor: the whole tensor, or IAXL_ZIP_SEGMENT_SIZE_KB of it so that a
// large tensor keeps several requests in flight. Each tensor still becomes exactly one block: a
// header, a segment table and the segment payloads.
//
// QAT and CPU streams are mutually compatible, IAA streams are compatible with neither, so each
// segment records whether IAA produced it and decompression only hands it to a backend that can
// decode it.

#include <torch/extension.h>

#include <algorithm>
#include <atomic>
#include <climits>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include "env.h"
#include "iaxl_common.h"
#include "parallel.h"
#include "cpu_zip.h"
#include "iaa_zip.h"
#include "qat_zip.h"
#include "data_shuffle.h"
#include "lossy.h"
#include "kv_zip.h"

namespace kv_zip {

// A block holds one tensor: a BlockHeader, `segments` SegmentEntry records, then the payloads.
struct BlockHeader {
    uint32_t magic;        // kBlockMagic
    uint16_t version;      // kBlockVersion
    uint16_t flags;        // kBlockShuffled, kBlockExponentOnly
    uint32_t orig_size;    // tensor bytes
    uint32_t segment_size; // tensor bytes per segment, the last one also takes the remainder
    uint32_t segments;     // SegmentEntry records that follow
};

struct SegmentEntry {
    uint32_t offset; // payload start, from the start of the block
    uint32_t length; // payload bytes
    uint32_t flags;  // kSegmentStored, kSegmentIaa
};

static_assert(sizeof(BlockHeader) == 20 && sizeof(SegmentEntry) == 12,
              "the block layout is persisted and must not change");

static constexpr uint32_t kBlockMagic = 0x5A584149; // "IAXZ" in memory
static constexpr uint16_t kBlockVersion = 1;
// Every segment was byte-shuffled on its own before compression.
static constexpr uint16_t kBlockShuffled = 1u << 0;
// Only the first half of each shuffled segment went through the backend; the second half follows
// the stream raw. Stored segments are whole.
static constexpr uint16_t kBlockExponentOnly = 1u << 1;
// The payload is the segment itself: compression was off or did not make it smaller.
static constexpr uint32_t kSegmentStored = 1u << 0;
// IAA produced the stream, so only IAA can decode it.
static constexpr uint32_t kSegmentIaa = 1u << 1;

// One segment of one block in a batch.
struct SegmentRef {
    uint32_t block;
    uint32_t index;
};

static uint32_t segment_count(uint32_t orig_size, uint32_t segment_size) {
    return std::max<uint32_t>(1, orig_size / segment_size);
}

// IAXL_ZIP_SEGMENT_SIZE_KB in bytes; 0, or a size not below the tensor's, keeps one segment.
static uint32_t segment_size_for(uint32_t orig_size) {
    const uint64_t size = uint64_t(envs.IAXL_ZIP_SEGMENT_SIZE_KB) * 1024;
    if (size == 0 || size >= orig_size)
        return std::max<uint32_t>(orig_size, 1);
    return static_cast<uint32_t>(size);
}

static BlockHeader make_header(size_t orig_size, uint16_t flags) {
    const auto size = static_cast<uint32_t>(orig_size);
    const uint32_t segment_size = segment_size_for(size);
    const uint32_t segments = segment_count(size, segment_size);
    return {kBlockMagic, kBlockVersion, flags, size, segment_size, segments};
}

// Tensor bytes of segment k, which starts at k * segment_size.
static uint32_t segment_length(const BlockHeader &h, uint32_t k) {
    return k + 1 < h.segments ? h.segment_size : h.orig_size - k * h.segment_size;
}

// Bytes of segment k that go through the backend.
static uint32_t zip_length(const BlockHeader &h, uint32_t k) {
    const uint32_t len = segment_length(h, k);
    return h.flags & kBlockExponentOnly ? len / 2 : len;
}

static size_t payload_start(uint32_t segments) {
    return sizeof(BlockHeader) + segments * sizeof(SegmentEntry);
}

static const BlockHeader &block_header(const char *block) {
    return *reinterpret_cast<const BlockHeader *>(block);
}

static const SegmentEntry *segment_table(const char *block) {
    return reinterpret_cast<const SegmentEntry *>(block + sizeof(BlockHeader));
}

enum class ZipBackend { QAT, IAA, CPU };

// Backend entry points, indexed by ZipBackend.
struct ZipOps {
    int (*compress)(int slot, void *src, int len);
    int (*decompress)(int slot, void *src, int len);
    int (*wait)(int slot, void **dest, int *len, int non_block);
    void *(*src_buf)(int slot);
    int (*queue_depth)(void);
};

static const ZipOps kZipOps[] = {
    {qat_zip_compress, qat_zip_decompress, qat_zip_wait, qat_zip_src_buf, qat_zip_queue_depth},
    {iaa_zip_compress, iaa_zip_decompress, iaa_zip_wait, iaa_zip_src_buf, iaa_zip_queue_depth},
    {cpu_zip_compress, cpu_zip_decompress, cpu_zip_wait, cpu_zip_src_buf, cpu_zip_queue_depth},
};

static const ZipOps &ops(ZipBackend backend) { return kZipOps[static_cast<int>(backend)]; }

// Returned by get_next once a backend has nothing left to claim.
static constexpr size_t kNoTask = static_cast<size_t>(-1);

static void ensure_zip_init() {
    static std::once_flag flag;
    std::call_once(flag, [] {
        if (envs.IAXL_QAT_ZIP_ENABLE) {
            IAXL_CHECK(qat_zip_init() == 0, "kv_zip: qat_zip_init failed");
            IAXL_CHECK(envs.IAXL_QAT_INSTANCE_NUM <= qat_zip_num_slots() / qat_zip_queue_depth(),
                       "kv_zip: IAXL_QAT_INSTANCE_NUM exceeds available QAT instances");
        }
        if (envs.IAXL_IAA_ZIP_ENABLE) {
            IAXL_CHECK(iaa_zip_init() == 0, "kv_zip: iaa_zip_init failed");
            IAXL_CHECK(envs.IAXL_IAA_INSTANCE_NUM <= iaa_zip_num_slots() / iaa_zip_queue_depth(),
                       "kv_zip: IAXL_IAA_INSTANCE_NUM exceeds available IAA instances");
        }
        if (envs.IAXL_CPU_ZIP_ENABLE)
            IAXL_CHECK(cpu_zip_init() == 0, "kv_zip: cpu_zip_init failed");
    });
}

// A backend slot, the item in flight on it and the thread that drives it.
struct Lane {
    ZipBackend backend;
    int slot;
    size_t item;
    int owner; // driving thread
};

struct LanePlan {
    std::vector<Lane> lanes;
    int threads;
};

// Which thread drives which slot depends only on the configuration, so it is planned once.
static const LanePlan &lane_plan() {
    static const LanePlan plan = [] {
        ensure_zip_init();
        const int qat_workers = envs.IAXL_QAT_ZIP_ENABLE ? envs.IAXL_QAT_INSTANCE_NUM : 0;
        const int iaa_workers = envs.IAXL_IAA_ZIP_ENABLE ? envs.IAXL_IAA_INSTANCE_NUM : 0;
        const int cpu_workers = envs.IAXL_CPU_ZIP_ENABLE ? cpu_zip_num_slots() : 0;
        const int worker_count = qat_workers + iaa_workers + cpu_workers;
        // Slot k of every worker comes before slot k + 1 of any, so that threads beyond the worker
        // count split the slots of one worker. The backends keep all request state per slot.
        std::vector<Lane> all;
        for (int k = 0;; k++) {
            const size_t before = all.size();
            for (int w = 0; w < worker_count; w++) {
                ZipBackend backend = ZipBackend::CPU;
                int first = qat_workers + iaa_workers;
                if (w < qat_workers) {
                    backend = ZipBackend::QAT;
                    first = 0;
                } else if (w < qat_workers + iaa_workers) {
                    backend = ZipBackend::IAA;
                    first = qat_workers;
                }
                const int depth = ops(backend).queue_depth();
                if (k < depth)
                    all.push_back({backend, (w - first) * depth + k, kNoTask, w});
            }
            if (all.size() == before)
                break;
        }
        const int threads =
            static_cast<int>(std::min<size_t>(envs.IAXL_OMP_THREAD_NUM, all.size()));
        // Up to one thread per worker, each worker stays on one thread so no instance is polled
        // from two; extra threads take the slots round-robin.
        for (size_t j = 0; j < all.size(); j++)
            all[j].owner = threads <= worker_count ? all[j].owner % threads : int(j % threads);
        return LanePlan{std::move(all), threads};
    }();
    return plan;
}

// The backend slots are process-wide, so one batch drives them at a time.
static std::mutex pipeline_mutex;

template <class Next, class Submit, class Complete>
static void zip_pipeline(Next &&get_next, Submit &&submit, Complete &&complete) {
    const LanePlan &plan = lane_plan();
    std::lock_guard<std::mutex> lock(pipeline_mutex);

    auto drive = [&](int tid) {
        std::vector<Lane> lanes;
        for (const Lane &l : plan.lanes) {
            if (l.owner != tid)
                continue;
            const size_t i = get_next(l.backend);
            if (i == kNoTask)
                continue;
            submit(l.backend, l.slot, i);
            lanes.push_back({l.backend, l.slot, i, tid});
        }

        while (!lanes.empty()) {
            for (size_t s = 0; s < lanes.size();) {
                Lane &lane = lanes[s];
                void *out;
                int out_len;
                const int status =
                    ops(lane.backend).wait(lane.slot, &out, &out_len, /*non_block=*/1);
                if (status == IAXL_ZIP_PENDING) {
                    s++;
                    continue;
                }
                IAXL_CHECK(status == 0, "kv_zip: zip wait failed");
                complete(lane.backend, lane.slot, lane.item, out, out_len);

                lane.item = get_next(lane.backend);
                if (lane.item == kNoTask) {
                    lane = lanes.back();
                    lanes.pop_back();
                    continue;
                }
                submit(lane.backend, lane.slot, lane.item);
                s++;
            }
        }
    };

    parallel::run_threads(plan.threads, drive);
}

// Compression state of one tensor. Segments finishing while others are still in flight park their
// payloads in `park`, at their own offset in the tensor, and the last one to finish assembles the
// block.
struct ZipBlock {
    const char *data = nullptr; // tensor bytes
    char *park = nullptr;
    uint16_t mask = 0xFFFF; // lossy truncation
    BlockHeader header{};
    uint32_t first = 0; // position of segment 0 in the batch's segment list
    std::atomic<uint32_t> pending{0};
};

static void log_layout(size_t orig_size) {
    static std::once_flag logged;
    std::call_once(logged, [&] {
        const BlockHeader h = make_header(orig_size, 0);
        printf("[kv_zip] block_size=%u B segment_size=%u B segments=%u\n", h.orig_size,
               h.segment_size, h.segments);
        fflush(stdout);
    });
}

// A tensor to compress. Its segments park in `park`, which may be the tensor itself: a segment's
// bytes are read at submit and its payload is not longer.
struct ZipSource {
    const char *data;
    size_t size;
    char *park;
};

// Compresses every source into one block, assembled in out(i, block_size), and records the block
// sizes in out_sizes. `out` may return the source's park buffer when that buffer starts where the
// block's payloads do, since payloads only move towards the header.
template <class Out>
static void compress_blocks(const std::vector<ZipSource> &sources, int elem_size, bool bf16,
                            bool compress, Out &&out, std::vector<size_t> &out_sizes) {
    const size_t n = sources.size();
    if (n > 0)
        log_layout(sources[0].size);

    if (!compress || !envs.IAXL_KV_COMPRESSION) {
        parallel::parallel_for(n, [&](size_t i) {
            const BlockHeader h = make_header(sources[i].size, 0);
            const size_t start = payload_start(h.segments);
            char *buf = out(i, start + h.orig_size);
            memcpy(buf, &h, sizeof(h));
            auto *table = reinterpret_cast<SegmentEntry *>(buf + sizeof(h));
            for (uint32_t k = 0; k < h.segments; k++)
                table[k] = {static_cast<uint32_t>(start + size_t(k) * h.segment_size),
                            segment_length(h, k), kSegmentStored};
            memcpy(buf + start, sources[i].data, h.orig_size);
            out_sizes[i] = start + h.orig_size;
        });
        return;
    }

    std::vector<ZipBlock> blocks(n);
    std::vector<SegmentRef> segs;
    segs.reserve(n);
    const bool shuffle = data_shuffle_enabled() && bf16;
    for (size_t i = 0; i < n; i++) {
        ZipBlock &b = blocks[i];
        b.data = sources[i].data;
        b.park = sources[i].park;
        b.mask = lossy_mask(elem_size);
        const uint16_t shuffled = envs.IAXL_ZIP_EXPONENT_ONLY
                                      ? kBlockShuffled | kBlockExponentOnly
                                      : kBlockShuffled;
        b.header = make_header(sources[i].size, shuffle ? shuffled : 0);
        // Segments are prepared in the backends' input buffers. The last one is the longest.
        IAXL_CHECK(segment_length(b.header, b.header.segments - 1) <=
                       static_cast<uint32_t>(envs.IAXL_ZIP_SRC_CAP),
                   "kv_zip: segment exceeds IAXL_ZIP_SRC_CAP; raise it or set "
                   "IAXL_ZIP_SEGMENT_SIZE_KB");
        b.first = static_cast<uint32_t>(segs.size());
        b.pending.store(b.header.segments, std::memory_order_relaxed);
        for (uint32_t k = 0; k < b.header.segments; k++)
            segs.push_back({static_cast<uint32_t>(i), k});
    }
    std::vector<SegmentEntry> entries(segs.size());

    // `payload` is set when segment `last` was not parked.
    auto assemble = [&](uint32_t i, uint32_t last, const char *payload) {
        ZipBlock &b = blocks[i];
        const BlockHeader &h = b.header;
        SegmentEntry *table = &entries[b.first];
        size_t size = payload_start(h.segments);
        for (uint32_t k = 0; k < h.segments; k++)
            size += table[k].length;
        char *buf = out(i, size);
        size_t offset = payload_start(h.segments);
        for (uint32_t k = 0; k < h.segments; k++) {
            const char *src =
                (k == last && payload) ? payload : b.park + size_t(k) * h.segment_size;
            memmove(buf + offset, src, table[k].length);
            table[k].offset = static_cast<uint32_t>(offset);
            offset += table[k].length;
        }
        memcpy(buf, &h, sizeof(h));
        memcpy(buf + sizeof(h), table, h.segments * sizeof(SegmentEntry));
        out_sizes[i] = size;
    };

    std::atomic<size_t> next{0};
    zip_pipeline(
        [&](ZipBackend) {
            const size_t s = next.fetch_add(1, std::memory_order_relaxed);
            return s < segs.size() ? s : kNoTask;
        },
        [&](ZipBackend backend, int slot, size_t s) {
            const SegmentRef r = segs[s];
            const ZipBlock &b = blocks[r.block];
            const char *src = b.data + size_t(r.index) * b.header.segment_size;
            const uint32_t len = segment_length(b.header, r.index);
            char *in = static_cast<char *>(ops(backend).src_buf(slot));
            if (b.header.flags & kBlockShuffled)
                shuffle_copy(in, src, len, b.mask);
            else
                masked_copy(in, src, len, b.mask);
            const int status =
                ops(backend).compress(slot, in, static_cast<int>(zip_length(b.header, r.index)));
            IAXL_CHECK(status == 0, "kv_zip: zip compress failed");
        },
        [&](ZipBackend backend, int slot, size_t s, void *out, int out_len) {
            const SegmentRef r = segs[s];
            ZipBlock &b = blocks[r.block];
            const uint32_t len = segment_length(b.header, r.index);
            const uint32_t zlen = zip_length(b.header, r.index);
            const auto stream = static_cast<uint32_t>(out_len);
            char *in = static_cast<char *>(ops(backend).src_buf(slot));
            const char *payload = static_cast<const char *>(out);
            if (stream < zlen) {
                if (zlen < len) {
                    // Move the stream over the high half, which it replaces, before the raw rest.
                    char *p = in + zlen - stream;
                    memcpy(p, out, stream);
                    payload = p;
                }
                entries[s] = {0, stream + len - zlen,
                              backend == ZipBackend::IAA ? kSegmentIaa : 0u};
            } else {
                // Not compressible: store the prepared segment, still in the input buffer.
                entries[s] = {0, len, kSegmentStored};
                payload = in;
            }
            if (b.pending.load(std::memory_order_acquire) > 1) {
                memcpy(b.park + size_t(r.index) * b.header.segment_size, payload,
                       entries[s].length);
                if (b.pending.fetch_sub(1, std::memory_order_acq_rel) > 1)
                    return;
                payload = nullptr;
            }
            assemble(r.block, r.index, payload);
        });
}

void kv_zip_compress_batch(const kv_pool::KVPool &pool, const std::vector<int64_t> &block_indices,
                           std::vector<char *> &out_bufs, std::vector<size_t> &out_sizes,
                           std::vector<size_t> &orig_sizes, bool compress) {
    IAXL_CHECK(pool.device == kv_pool::PoolDevice::CPU, "kv_zip: pool must be in CPU memory");
    IAXL_CHECK(pool.block_bytes <= INT_MAX, "kv_zip: block byte size exceeds zip integer length range");
    const size_t n = block_indices.size();
    std::fill_n(orig_sizes.begin(), n, static_cast<size_t>(pool.block_bytes));
    // Segments park in the pool block itself, so a block takes no memory beyond its own size.
    std::vector<ZipSource> sources(n);
    for (size_t i = 0; i < n; i++) {
        char *block = pool.block_ptr(block_indices[i]);
        sources[i] = {block, static_cast<size_t>(pool.block_bytes), block};
    }
    compress_blocks(
        sources, static_cast<int>(pool.elem_size), pool.dtype == static_cast<int32_t>(at::kBFloat16),
        compress,
        [&](size_t i, size_t size) {
            char *buf = static_cast<char *>(malloc(size));
            IAXL_CHECK(buf != nullptr, "kv_zip: cache buffer allocation failed");
            out_bufs[i] = buf;
            return buf;
        },
        out_sizes);
}

// Decompresses data_ptrs[i] into dsts[i], which holds dst_sizes[i] bytes.
static void decompress_blocks(const std::vector<const char *> &data_ptrs,
                              const std::vector<char *> &dsts,
                              const std::vector<size_t> &dst_sizes) {
    const size_t n = data_ptrs.size();

    // IAA-produced segments must go back to IAA, the other compressed ones to QAT/CPU.
    std::vector<SegmentRef> stored, iaa, others;
    for (size_t i = 0; i < n; i++) {
        const BlockHeader &h = block_header(data_ptrs[i]);
        IAXL_CHECK(h.magic == kBlockMagic && h.version == kBlockVersion,
                   "kv_zip: unknown block format, e.g. a KV cache persisted by an older iaxl");
        IAXL_CHECK(h.orig_size == dst_sizes[i], "kv_zip: block size does not match destination size");
        IAXL_CHECK(h.segment_size > 0 && h.segments == segment_count(h.orig_size, h.segment_size),
                   "kv_zip: invalid segment table");
        IAXL_CHECK(!(h.flags & kBlockExponentOnly) || (h.flags & kBlockShuffled),
                   "kv_zip: exponent-only block is not shuffled");
        const SegmentEntry *table = segment_table(data_ptrs[i]);
        for (uint32_t k = 0; k < h.segments; k++) {
            const SegmentRef r{static_cast<uint32_t>(i), k};
            if (table[k].flags & kSegmentStored)
                stored.push_back(r);
            else if (table[k].flags & kSegmentIaa)
                iaa.push_back(r);
            else
                others.push_back(r);
        }
    }

    // `src` is the prepared segment, or only its first half when the segment was compressed in an
    // exponent-only block; the raw second half then ends the payload.
    auto place = [&](SegmentRef r, const void *src, size_t len) {
        const BlockHeader &h = block_header(data_ptrs[r.block]);
        const SegmentEntry &e = segment_table(data_ptrs[r.block])[r.index];
        const uint32_t full = segment_length(h, r.index);
        const bool split = !(e.flags & kSegmentStored) && (h.flags & kBlockExponentOnly);
        IAXL_CHECK(len == (split ? zip_length(h, r.index) : full),
                   "kv_zip: decompressed size does not match segment size");
        const char *head = static_cast<const char *>(src);
        const char *tail = split ? data_ptrs[r.block] + e.offset + e.length - (full - len)
                                 : head + full / 2;
        char *dst = dsts[r.block] + size_t(r.index) * h.segment_size;
        if (h.flags & kBlockShuffled)
            shuffle_copy(dst, head, tail, full);
        else
            memcpy(dst, src, full);
    };

    parallel::parallel_for(stored.size(), [&](size_t j) {
        const SegmentRef r = stored[j];
        const SegmentEntry &e = segment_table(data_ptrs[r.block])[r.index];
        place(r, data_ptrs[r.block] + e.offset, e.length);
    });

    if (iaa.empty() && others.empty())
        return;
    IAXL_CHECK(iaa.empty() || envs.IAXL_IAA_ZIP_ENABLE,
               "kv_zip: batch holds IAA-compressed segments but the IAA backend is disabled");
    IAXL_CHECK(others.empty() || envs.IAXL_QAT_ZIP_ENABLE || envs.IAXL_CPU_ZIP_ENABLE,
               "kv_zip: batch holds QAT/CPU-compressed segments but both backends are disabled");

    // IAA segments first, then the others, each range with its own cursor.
    const size_t iaa_end = iaa.size();
    std::vector<SegmentRef> segs = std::move(iaa);
    segs.insert(segs.end(), others.begin(), others.end());
    std::atomic<size_t> iaa_next{0}, other_next{iaa_end};
    zip_pipeline(
        [&](ZipBackend backend) {
            if (backend == ZipBackend::IAA) {
                const size_t s = iaa_next.fetch_add(1, std::memory_order_relaxed);
                return s < iaa_end ? s : kNoTask;
            }
            const size_t s = other_next.fetch_add(1, std::memory_order_relaxed);
            return s < segs.size() ? s : kNoTask;
        },
        [&](ZipBackend backend, int slot, size_t s) {
            const SegmentRef r = segs[s];
            const BlockHeader &h = block_header(data_ptrs[r.block]);
            const SegmentEntry &e = segment_table(data_ptrs[r.block])[r.index];
            const uint32_t raw = segment_length(h, r.index) - zip_length(h, r.index);
            IAXL_CHECK(e.length > raw, "kv_zip: invalid segment length");
            char *payload = const_cast<char *>(data_ptrs[r.block] + e.offset);
            const int status =
                ops(backend).decompress(slot, payload, static_cast<int>(e.length - raw));
            IAXL_CHECK(status == 0, "kv_zip: zip decompress failed");
        },
        [&](ZipBackend, int, size_t s, void *out, int out_len) {
            place(segs[s], out, static_cast<size_t>(out_len));
        });
}

void kv_zip_decompress_batch(const std::vector<const char *> &data_ptrs,
                             const kv_pool::KVPool &pool,
                             const std::vector<int64_t> &block_indices) {
    IAXL_CHECK(pool.device == kv_pool::PoolDevice::CPU, "kv_zip: pool must be in CPU memory");
    const size_t n = block_indices.size();
    IAXL_CHECK(data_ptrs.size() == n, "kv_zip: decompression inputs must have matching lengths");
    std::vector<char *> dsts(n);
    for (size_t i = 0; i < n; i++)
        dsts[i] = pool.block_ptr(block_indices[i]);
    decompress_blocks(data_ptrs, dsts, std::vector<size_t>(n, static_cast<size_t>(pool.block_bytes)));
}

[[noreturn]] static void invalid(const std::string &what) {
    throw std::invalid_argument("kv_zip: " + what);
}

size_t kv_zip_max_block_size(size_t orig_size) {
    if (orig_size == 0 || orig_size > INT_MAX)
        invalid("tensor size " + std::to_string(orig_size) + " is outside [1, INT_MAX]");
    return payload_start(make_header(orig_size, 0).segments) + orig_size;
}

std::vector<size_t> kv_zip_compress_into(const std::vector<ConstBytes> &srcs,
                                         const std::vector<Bytes> &dsts, int elem_size, bool bf16) {
    const size_t n = srcs.size();
    if (dsts.size() != n)
        invalid("got " + std::to_string(n) + " sources but " + std::to_string(dsts.size()) +
                " destinations");
    std::vector<ZipSource> sources(n);
    for (size_t i = 0; i < n; i++) {
        const size_t need = kv_zip_max_block_size(srcs[i].size);
        if (dsts[i].size < need)
            invalid("destination of " + std::to_string(dsts[i].size) + " bytes is below the " +
                    std::to_string(need) + " of kv_zip_max_block_size");
        const BlockHeader h = make_header(srcs[i].size, 0);
        if (envs.IAXL_KV_COMPRESSION &&
            segment_length(h, h.segments - 1) > static_cast<uint32_t>(envs.IAXL_ZIP_SRC_CAP))
            invalid("segment exceeds IAXL_ZIP_SRC_CAP; raise it or set IAXL_ZIP_SEGMENT_SIZE_KB");
        // Segments park where the payloads start, so assembling only moves them forward.
        sources[i] = {srcs[i].data, srcs[i].size, dsts[i].data + payload_start(h.segments)};
    }
    std::vector<size_t> sizes(n);
    compress_blocks(sources, elem_size, bf16, true, [&](size_t i, size_t) { return dsts[i].data; },
                    sizes);
    return sizes;
}

// Checks what decompress_blocks would abort on, and that every segment lies within the block and
// fits the backend buffers. Returns the tensor size.
static size_t check_block(const ConstBytes &src) {
    if (src.size < sizeof(BlockHeader))
        invalid("block of " + std::to_string(src.size) + " bytes is shorter than its header");
    const BlockHeader &h = block_header(src.data);
    if (h.magic != kBlockMagic || h.version != kBlockVersion)
        invalid("unknown block format");
    if (h.segment_size == 0 || h.segments != segment_count(h.orig_size, h.segment_size) ||
        src.size < payload_start(h.segments))
        invalid("invalid segment table");
    if ((h.flags & kBlockExponentOnly) && !(h.flags & kBlockShuffled))
        invalid("exponent-only block is not shuffled");
    const SegmentEntry *table = segment_table(src.data);
    for (uint32_t k = 0; k < h.segments; k++) {
        const SegmentEntry &e = table[k];
        if (e.offset < payload_start(h.segments) || uint64_t(e.offset) + e.length > src.size)
            invalid("segment " + std::to_string(k) + " lies outside the block");
        if (e.flags & kSegmentStored) {
            if (e.length != segment_length(h, k))
                invalid("stored segment " + std::to_string(k) + " has the wrong length");
            continue;
        }
        const uint32_t raw = segment_length(h, k) - zip_length(h, k);
        if (e.length <= raw || e.length - raw > static_cast<uint32_t>(envs.IAXL_ZIP_DST_CAP))
            invalid("compressed segment " + std::to_string(k) + " has an invalid length");
        if (zip_length(h, k) > static_cast<uint32_t>(envs.IAXL_ZIP_SRC_CAP))
            invalid("segment exceeds IAXL_ZIP_SRC_CAP");
        if ((e.flags & kSegmentIaa) ? !envs.IAXL_IAA_ZIP_ENABLE
                                    : !(envs.IAXL_QAT_ZIP_ENABLE || envs.IAXL_CPU_ZIP_ENABLE))
            invalid("segment " + std::to_string(k) + " needs a zip backend that is disabled");
    }
    return h.orig_size;
}

std::vector<size_t> kv_zip_decompress_into(const std::vector<ConstBytes> &srcs,
                                           const std::vector<Bytes> &dsts) {
    const size_t n = srcs.size();
    if (dsts.size() != n)
        invalid("got " + std::to_string(n) + " sources but " + std::to_string(dsts.size()) +
                " destinations");
    std::vector<const char *> data_ptrs(n);
    std::vector<char *> dst_ptrs(n);
    std::vector<size_t> sizes(n);
    for (size_t i = 0; i < n; i++) {
        sizes[i] = check_block(srcs[i]);
        if (dsts[i].size < sizes[i])
            invalid("destination of " + std::to_string(dsts[i].size) + " bytes is below the " +
                    std::to_string(sizes[i]) + " the block holds");
        data_ptrs[i] = srcs[i].data;
        dst_ptrs[i] = dsts[i].data;
    }
    decompress_blocks(data_ptrs, dst_ptrs, sizes);
    return sizes;
}

} // namespace kv_zip
