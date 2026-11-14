// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

// RDMA flavour of kv_xfer::Ops for the remote_pool daemon (always compiled, used when a
// Context is created on a PoolDevice::Remote pool): the "GPU" is a client's VRAM reached
// over RDMA through NIXL. put = RDMA READ (d2h), get = RDMA WRITE (h2d).
// Transfers are posted and polled synchronously on the TaskQueue worker, so
// copy_chunks_batch returning means the data has landed; events are plain flags.

#include <atomic>
#include <chrono>
#include <cstdio>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

#include <nixl.h>

#include "iaxl_common.h"
#include "kv_pool.h"
#include "kv_xfer.h"
#include "kv_xfer_rdma.h"

#define NIXL_CHECK(call)                                                                           \
    do {                                                                                           \
        nixl_status_t st_ = (call);                                                                \
        if (st_ != NIXL_SUCCESS) {                                                                 \
            fprintf(stderr, "NIXL error %s:%d: %s returned %s\n", __FILE__, __LINE__, #call,       \
                    nixlEnumStrings::statusStr(st_).c_str());                                      \
            abort();                                                                               \
        }                                                                                          \
    } while (0)

namespace kv_xfer {

using kv_pool::KVPool;
using kv_pool::PoolDevice;

namespace {

// NIXL state of one registered buffer; pools sharing a base (hybrid layers) share it.
struct Reg {
    struct StagingDlist {
        int64_t outer_dims = 0;
        int64_t inner_size = 0;
        nixlDlistH *handle = nullptr;
    };
    nixlDlistH *dlist = nullptr;            // Remote: VRAM descs, one per buffer
    std::vector<StagingDlist> staging;      // CPU: DRAM descs, one per page size
    int refs = 0;
};

struct Rdma {
    // Heap-allocated and never freed: destroying nixlAgent during static teardown
    // (after UCX / plugin statics are gone) segfaults; the daemon lives as long as the process.
    static Rdma &get() {
        static Rdma *r = new Rdma;
        return *r;
    }

    nixlAgent &agent() {
        IAXL_CHECK(agent_ != nullptr, "rdma_init must be called first");
        return *agent_;
    }

    std::unique_ptr<nixlAgent> agent_;
    nixl_opt_args_t opt;                     // pins every call to the UCX backend
    std::mutex mu;                           // regs bookkeeping
    std::unordered_map<uintptr_t, Reg> regs; // by base
    std::vector<nixl_reg_dlist_t> mem_regs;  // registrations stay alive for the process lifetime
};

struct XferContext {
    const KVPool *remote;
    const KVPool *local;
    nixlDlistH *remote_dlist;
    nixlDlistH *local_dlist;
};

inline XferContext *as_ctx(context_t c) { return static_cast<XferContext *>(c); }
inline std::atomic<bool> *as_ev(event_t e) { return static_cast<std::atomic<bool> *>(e); }

// kv_pool hooks: called with the registry lock held, so never call back into kv_pool here.
void on_register(const KVPool &p) {
    Rdma &r = Rdma::get();
    std::lock_guard<std::mutex> lock(r.mu);
    auto [it, inserted] = r.regs.try_emplace(p.base);
    it->second.refs++;
    if (!inserted)
        return;
    if (p.device == PoolDevice::Remote) {
        nixl_xfer_dlist_t descs(VRAM_SEG, p.outer_dims * p.num_blocks);
        for (int64_t o = 0; o < p.outer_dims; o++)
            for (int64_t b = 0; b < p.num_blocks; b++)
                descs[o * p.num_blocks + b] = nixlBasicDesc(
                    reinterpret_cast<uintptr_t>(p.block_ptr(b, o)), p.inner_size, p.dev_id);
        NIXL_CHECK(r.agent().prepXferDlist(p.peer, descs, it->second.dlist, &r.opt));
    } else {
        const bool gpu = p.device == PoolDevice::GPU;
        // `total_bytes` counts a full stride after the last page; on a packed layer
        // that tail reaches past the allocation, so register up to the last page end.
        const size_t span = (static_cast<size_t>(p.outer_dims - 1) * p.outer_block_size) +
                            (static_cast<size_t>(p.num_blocks - 1) * p.chunk_stride) +
                            static_cast<size_t>(p.inner_size);
        nixl_reg_dlist_t mem(gpu ? VRAM_SEG : DRAM_SEG);
        mem.addDesc(nixlBlobDesc(p.base, span, gpu ? p.dev_id : 0, ""));
        NIXL_CHECK(r.agent().registerMem(mem, &r.opt));
        r.mem_regs.push_back(std::move(mem));
    }
}

void on_unregister(const KVPool &p) {
    Rdma &r = Rdma::get();
    std::lock_guard<std::mutex> lock(r.mu);
    auto it = r.regs.find(p.base);
    if (it == r.regs.end() || --it->second.refs > 0)
        return;
    if (it->second.dlist)
        NIXL_CHECK(r.agent().releasedDlistH(it->second.dlist));
    for (auto &d : it->second.staging)
        NIXL_CHECK(r.agent().releasedDlistH(d.handle));
    r.regs.erase(it);
}

Reg &reg_of(Rdma &r, const KVPool &p) {
    auto it = r.regs.find(p.base);
    IAXL_CHECK(it != r.regs.end(), "pool was registered before rdma_init");
    return it->second;
}

// The CPU pool's descriptor list splits every block into the remote's outer_dims
// sub-blocks of inner_size bytes, so it is built once the first remote pool is known.
nixlDlistH *local_dlist(Rdma &r, const KVPool &local, const KVPool &remote) {
    std::lock_guard<std::mutex> lock(r.mu);
    Reg &reg = reg_of(r, local);
    const int64_t od = remote.outer_dims, is = remote.inner_size;
    // Mixed page sizes share one max-page staging pool, so a dlist is built per
    // remote geometry: its descriptors point at the first `is` bytes of a block.
    IAXL_CHECK(local.inner_size >= is, "staging block is smaller than the remote page");
    for (const auto &d : reg.staging)
        if (d.outer_dims == od && d.inner_size == is)
            return d.handle;
    nixl_xfer_dlist_t descs(DRAM_SEG, local.num_blocks * od);
    for (int64_t b = 0; b < local.num_blocks; b++)
        for (int64_t o = 0; o < od; o++)
            descs[b * od + o] =
                nixlBasicDesc(reinterpret_cast<uintptr_t>(local.block_ptr(b)) + o * is, is, 0);
    nixlDlistH *handle = nullptr;
    NIXL_CHECK(r.agent().prepXferDlist(NIXL_INIT_AGENT, descs, handle, &r.opt));
    reg.staging.push_back({od, is, handle});
    fprintf(stderr, "[kv_xfer/rdma] pool %s: %ld blocks x %ld descs of %ld B\n",
            local.name.c_str(), local.num_blocks, od, is);
    return handle;
}

} // namespace

// ---- kv_xfer_rdma.h --------------------------------------------------------

void rdma_init(const std::string &name, int listen_port) {
    Rdma &r = Rdma::get();
    IAXL_CHECK(r.agent_ == nullptr, "rdma_init called twice");
    // Listen thread serves metadata; STRICT lets the main thread poll notifs while
    // TaskQueue workers post/poll transfers.
    nixlAgentConfig cfg(false, true, listen_port, nixl_thread_sync_t::NIXL_THREAD_SYNC_STRICT);
    r.agent_ = std::make_unique<nixlAgent>(name, cfg);
    nixl_mem_list_t mems;
    nixl_b_params_t params;
    NIXL_CHECK(r.agent_->getPluginParams("UCX", mems, params));
    nixlBackendH *ucx = nullptr;
    NIXL_CHECK(r.agent_->createBackend("UCX", params, ucx));
    r.opt.backends.push_back(ucx);
    kv_pool::add_pool_hooks({on_register, on_unregister});
    fprintf(stderr, "[kv_xfer/rdma] agent %s listening on port %d\n", name.c_str(), listen_port);
}

void rdma_wait_peer(const std::string &peer, double timeout_s) {
    nixlAgent &a = Rdma::get().agent();
    nixl_xfer_dlist_t empty(DRAM_SEG);
    auto deadline = std::chrono::steady_clock::now() + std::chrono::duration<double>(timeout_s);
    while (a.checkRemoteMD(peer, empty) != NIXL_SUCCESS) {
        IAXL_CHECK(std::chrono::steady_clock::now() < deadline, "timed out waiting for peer metadata");
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

void rdma_remove_peer(const std::string &peer) { NIXL_CHECK(Rdma::get().agent().invalidateRemoteMD(peer)); }

void rdma_send_notif(const std::string &peer, const std::string &msg) {
    Rdma &r = Rdma::get();
    NIXL_CHECK(r.agent().genNotif(peer, msg, &r.opt));
}

std::vector<std::pair<std::string, std::string>> rdma_get_notifs() {
    nixl_notifs_t map;
    NIXL_CHECK(Rdma::get().agent().getNotifs(map));
    std::vector<std::pair<std::string, std::string>> out;
    for (auto &[peer, msgs] : map)
        for (auto &m : msgs)
            out.emplace_back(peer, std::move(m));
    return out;
}

// ---- kv_xfer::Ops (RDMA flavour) ---------------------------------------------

namespace rdma {

event_t event_acquire() { return new std::atomic<bool>(false); }
void event_release(event_t event) { delete as_ev(event); }

void event_synchronize(event_t event) {
    auto *f = as_ev(event);
    while (!f->load(std::memory_order_acquire))
        std::this_thread::yield();
}

void context_destroy(context_t ctx) { delete as_ctx(ctx); }

unsigned long long context_stream_id(context_t) { return 0; }
bool context_same_stream(context_t) { return true; }

void context_record_event(context_t, event_t event) { as_ev(event)->store(true, std::memory_order_release); }
void context_work_wait_event(context_t, event_t event) { event_synchronize(event); }
void context_cur_wait_event(context_t, event_t) {}
void context_work_wait_cur(context_t) {}
void context_sync_cur(context_t) {}

void copy_chunks_batch(context_t ctx, const std::vector<int64_t> &chunk_indices, const std::vector<char *> &cpu_ptrs,
                       bool h2d) {
    Rdma &r = Rdma::get();
    XferContext *x = as_ctx(ctx);
    const KVPool &remote = *x->remote;
    const KVPool &local = *x->local;
    const size_t n = chunk_indices.size();
    if (n == 0)
        return;

    const int64_t od = remote.outer_dims;
    std::vector<int> local_idx(n * od), remote_idx(n * od);
    for (size_t i = 0, k = 0; i < n; i++) {
        const int64_t lb = local.block_index(cpu_ptrs[i]);
        for (int64_t o = 0; o < od; o++, k++) {
            local_idx[k] = static_cast<int>(lb * od + o);
            remote_idx[k] = static_cast<int>(o * remote.num_blocks + chunk_indices[i]);
        }
    }

    static std::once_flag noted;
    std::call_once(noted, [&] {
        fprintf(stderr, "[kv_xfer] copy_chunks_batch: using RDMA backend (%ld B x %ld per block)\n",
                remote.inner_size, od);
    });

    nixlAgent &a = r.agent();
    nixlXferReqH *req = nullptr;
    NIXL_CHECK(a.makeXferReq(h2d ? NIXL_WRITE : NIXL_READ, x->local_dlist, local_idx, x->remote_dlist, remote_idx,
                             req, &r.opt));
    nixl_status_t st = a.postXferReq(req);
    while (st == NIXL_IN_PROG)
        st = a.getXferStatus(req);
    NIXL_CHECK(st);
    NIXL_CHECK(a.releaseXferReq(req));
}

void copy_chunk(context_t ctx, char *cpu_base, int64_t chunk_index, bool h2d) {
    copy_chunks_batch(ctx, {chunk_index}, {cpu_base}, h2d);
}

} // namespace rdma

const Ops &rdma_ops() {
    static const Ops ops{rdma::event_acquire,         rdma::event_release,          rdma::event_synchronize,
                         rdma::context_destroy,       rdma::context_stream_id,      rdma::context_same_stream,
                         rdma::copy_chunk,            rdma::copy_chunks_batch,      rdma::context_record_event,
                         rdma::context_work_wait_event, rdma::context_cur_wait_event, rdma::context_work_wait_cur,
                         rdma::context_sync_cur,      nullptr};
    return ops;
}

context_t rdma_context_create(const KVPool &remote, const KVPool &local) {
    IAXL_CHECK(remote.device == PoolDevice::Remote, "rdma_context_create: pool is not remote");
    Rdma &r = Rdma::get();
    XferContext *x = new XferContext{&remote, &local, nullptr, nullptr};
    {
        std::lock_guard<std::mutex> lock(r.mu);
        x->remote_dlist = reg_of(r, remote).dlist;
    }
    x->local_dlist = local_dlist(r, local, remote);
    return x;
}

} // namespace kv_xfer
