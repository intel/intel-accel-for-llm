// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#ifndef DSA_GD_H
#define DSA_GD_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define GD_GPU_PAGE_SHIFT 16
#define GD_GPU_PAGE_SIZE (1UL << GD_GPU_PAGE_SHIFT)
#define GD_GPU_PAGE_MASK (~(GD_GPU_PAGE_SIZE - 1))

typedef struct gd_ctx gd_ctx;

gd_ctx *dsa_gd_open(void);

void dsa_gd_close(gd_ctx *ctx);

void dsa_gd_reset(gd_ctx *ctx);

int dsa_gd_register(gd_ctx *ctx, uint64_t dptr, size_t nbytes);

int dsa_gd_unregister(gd_ctx *ctx, uint64_t dptr, size_t nbytes);

int dsa_gd_copy(gd_ctx *ctx, uint64_t dst, int dst_cuda, uint64_t src, int src_cuda, size_t nbytes);

int dsa_gd_dsa_copy(gd_ctx *ctx, uint64_t dst, int dst_cuda, uint64_t src, int src_cuda,
                    size_t nbytes);

int dsa_gd_dsa_copy_batch(gd_ctx *ctx, const uint64_t *dst, const int *dst_cuda,
                          const uint64_t *src, const int *src_cuda, const size_t *nbytes,
                          size_t count);

int dsa_gd_default_register(uint64_t dptr, size_t nbytes);
int dsa_gd_default_unregister(uint64_t dptr, size_t nbytes);
int dsa_gd_default_copy(uint64_t dst, int dst_cuda, uint64_t src, int src_cuda, size_t nbytes);
int dsa_gd_default_dsa_copy(uint64_t dst, int dst_cuda, uint64_t src, int src_cuda, size_t nbytes);
int dsa_gd_default_dsa_copy_batch(const uint64_t *dst, const int *dst_cuda, const uint64_t *src,
                                  const int *src_cuda, const size_t *nbytes, size_t count);

int dsa_gd_default_gpu_bar_addr(uint64_t gpu_ptr, size_t nbytes, void **cpu_out);

void dsa_gd_default_reset(void);

#ifdef __cplusplus
}
#endif

#endif
