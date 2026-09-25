// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#ifndef DSA_MEMCPY_H
#define DSA_MEMCPY_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

int dsa_memcpy(void *dest, const void *src, size_t n);
int dsa_memcpy_batch(void *const dest[], const void *const src[], const size_t n[], size_t count);

// One asynchronous host copy. Must stay at a fixed address from submit until poll reports done.
typedef struct {
    __attribute__((aligned(32))) uint8_t comp[32]; // struct dsa_completion_record
    int64_t start_ns;
    uint32_t polls;
    int wq;
} dsa_copy_job;

// 0 = queued, 1 = work queue full (copy this one on the CPU), -1 = DSA unavailable.
int dsa_copy_submit(dsa_copy_job *job, void *dest, const void *src, size_t n);

// 1 = copied, 0 = still in flight, -1 = failed (dest is undefined; redo it on the CPU).
int dsa_copy_poll(dsa_copy_job *job);

#ifdef __cplusplus
}
#endif

#endif
