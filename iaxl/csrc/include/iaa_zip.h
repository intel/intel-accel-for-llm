// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#ifndef IAA_ZIP_H
#define IAA_ZIP_H

#ifdef __cplusplus
extern "C" {
#endif

int iaa_zip_init(void);

int iaa_zip_num_slots(void);

int iaa_zip_queue_depth(void);

int iaa_zip_src_cap(void);

int iaa_zip_compress(int slot, void *src, int len);
int iaa_zip_decompress(int slot, void *src, int len);

// Returns 0 on completion, -1 on failure, or IAXL_ZIP_PENDING if non_block and still in flight.
int iaa_zip_wait(int slot, void **dest, int *len, int non_block);

void iaa_zip_shutdown(void);

#ifdef __cplusplus
}
#endif

#endif
