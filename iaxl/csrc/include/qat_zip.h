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

// Returns 0 on completion, -1 on failure, or IAXL_ZIP_PENDING if non_block and still in flight.
int qat_zip_wait(int slot, void **dest, int *len, int non_block);

void qat_zip_shutdown(void);

#ifdef __cplusplus
}
#endif

#endif
