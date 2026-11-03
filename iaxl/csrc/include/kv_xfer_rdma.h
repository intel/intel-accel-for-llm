// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

// Daemon-side RDMA backend (always compiled, enabled via IAXL_RDMA_ENABLE). One
// nixlAgent per process; the data plane reads/writes the client's GPU memory through it.

#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "kv_pool.h"
#include "kv_xfer.h"

namespace kv_xfer {

// Data-plane entry points used by Context::create for a PoolDevice::Remote pool.
const Ops &rdma_ops();
context_t rdma_context_create(const kv_pool::KVPool &remote, const kv_pool::KVPool &local);

// Creates the agent and installs the kv_pool hooks: CPU/GPU pools are registered with
// NIXL, Remote pools get a VRAM descriptor list on the peer.
void rdma_init(const std::string &name, int listen_port);
void rdma_wait_peer(const std::string &peer, double timeout_s);
void rdma_remove_peer(const std::string &peer);

void rdma_send_notif(const std::string &peer, const std::string &msg);
std::vector<std::pair<std::string, std::string>> rdma_get_notifs();

} // namespace kv_xfer
