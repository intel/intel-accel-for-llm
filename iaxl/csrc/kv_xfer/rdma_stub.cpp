// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

// DEVICE=cpu builds do not link NIXL; the remote_pool RDMA entry points fail with a clear error.

#include <stdexcept>

#include "kv_xfer_rdma.h"

namespace kv_xfer {

[[noreturn]] static void rdma_unavailable() {
    throw std::runtime_error("IAXL RDMA (remote_pool) needs NIXL, which DEVICE=cpu builds do not "
                             "link; rebuild with DEVICE=cuda or DEVICE=xpu");
}

const Ops &rdma_ops() { rdma_unavailable(); }
context_t rdma_context_create(char *, int64_t, int64_t, int64_t, int64_t) { rdma_unavailable(); }

void rdma_init(const std::string &, int) { rdma_unavailable(); }
void rdma_wait_peer(const std::string &, double) { rdma_unavailable(); }
void rdma_remove_peer(const std::string &) { rdma_unavailable(); }
void rdma_register_mem(uintptr_t, size_t) { rdma_unavailable(); }
void rdma_register_local(uintptr_t, size_t, int64_t) { rdma_unavailable(); }
uintptr_t rdma_register_remote(const std::string &, uintptr_t, const std::vector<int64_t> &,
                               int64_t, int, int) {
    rdma_unavailable();
}
void rdma_unregister_remote(uintptr_t) { rdma_unavailable(); }
void rdma_send_notif(const std::string &, const std::string &) { rdma_unavailable(); }
std::vector<std::pair<std::string, std::string>> rdma_get_notifs() { rdma_unavailable(); }

} // namespace kv_xfer
