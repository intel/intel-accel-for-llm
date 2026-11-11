// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>

#include <ATen/ATen.h>

namespace kv_zip {

// IAXL_KV_LOSSY_TRUNC as an AND mask for every 16-bit word of a tensor; 0xFFFF keeps all bits.
uint16_t lossy_mask(int element_size);

// Pluggable lossy codec interface. torch_ext ships weak stubs (lossy.cpp) that stand for "no
// codec loaded"; a codec library defines the same symbols strongly and takes them over when
// it is loaded ahead of torch_ext, e.g. LD_PRELOAD=_lib/libiaxl_kv_lossy.so. The default
// visibility keeps the references interposable despite the module's -fvisibility=hidden.
#define IAXL_LOSSY_API __attribute__((visibility("default")))

// Compress side: encode `t` (contiguous) into `out`, which holds `out_size` bytes, and return
// the payload length. The codec aborts (IAXL_CHECK) on a tensor it cannot handle or a payload
// that does not fit; the stubs abort with "no lossy codec loaded".
IAXL_LOSSY_API size_t lossy_serialize(const at::Tensor &t, void *out, size_t out_size);

// Decompress side: rebuild `t` in place from the payload `src` of `src_size` bytes.
IAXL_LOSSY_API void lossy_deserialize(const void *src, size_t src_size, const at::Tensor &t);

} // namespace kv_zip
