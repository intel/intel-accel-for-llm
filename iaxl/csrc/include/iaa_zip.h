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

// Slot-owned input buffer (src_cap bytes); fill it, then submit with compress_staged.
void *iaa_zip_input_buf(int slot);
int iaa_zip_compress_staged(int slot, int len);

// Non-blocking completion check: 1 done, 0 in flight, -1 invalid slot.
int iaa_zip_poll(int slot);

int iaa_zip_wait(int slot, void **dest, int *len);

void iaa_zip_shutdown(void);

#ifdef __cplusplus
}
#endif

#endif
