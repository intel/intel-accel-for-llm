// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "lossy.h"

#include "env.h"
#include "iaxl_common.h"

namespace kv_zip {

uint16_t lossy_mask(int element_size) {
    const int bits = envs.IAXL_KV_LOSSY_TRUNC;
    if (bits <= 0)
        return 0xFFFF;
    if (element_size == 2)
        return static_cast<uint16_t>(0xFFFF << bits);
    if (element_size == 1) {
        const auto byte = static_cast<uint8_t>(0xFF << bits);
        return static_cast<uint16_t>(byte << 8 | byte);
    }
    return 0xFFFF;
}

// "No codec loaded" stubs; a preloaded codec library overrides them (see lossy.h).
#define IAXL_LOSSY_STUB __attribute__((weak)) IAXL_LOSSY_API

#define IAXL_LOSSY_STUB_MSG                                                                        \
    "IAXL_KV_LOSSY_ENABLE is set but no lossy codec is loaded; "                                  \
    "LD_PRELOAD _lib/libiaxl_kv_lossy.so"

IAXL_LOSSY_STUB size_t lossy_serialize(const at::Tensor &, void *, size_t) {
    IAXL_CHECK(false, IAXL_LOSSY_STUB_MSG);
    return 0;
}

IAXL_LOSSY_STUB void lossy_deserialize(const void *, size_t, const at::Tensor &) {
    IAXL_CHECK(false, IAXL_LOSSY_STUB_MSG);
}

} // namespace kv_zip
