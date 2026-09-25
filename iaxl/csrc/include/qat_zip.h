// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#ifndef QAT_ZIP_H
#define QAT_ZIP_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

int qat_zip_init(void);

int qat_zip_num_slots(void);

int qat_zip_queue_depth(void);

int qat_zip_src_cap(void);

int qat_zip_compress(int slot, void *src, int len);
int qat_zip_decompress(int slot, void *src, int len);

// Slot-owned DMA input buffer (src_cap bytes); fill it, then submit with compress_staged.
void *qat_zip_input_buf(int slot);
int qat_zip_compress_staged(int slot, int len);

// Non-blocking completion check: 1 done, 0 in flight, -1 invalid slot.
int qat_zip_poll(int slot);

int qat_zip_wait(int slot, void **dest, int *len);

void qat_zip_shutdown(void);

#ifdef __cplusplus
}
#endif

#endif
