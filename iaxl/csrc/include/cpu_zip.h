// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#ifndef CPU_ZIP_H
#define CPU_ZIP_H

#ifdef __cplusplus
extern "C" {
#endif


int cpu_zip_init(void);

int cpu_zip_num_slots(void);

int cpu_zip_queue_depth(void);

int cpu_zip_src_cap(void);

// Non-zero when this build's compressed output stays inside the 4 KB window IAA can decode.
int cpu_zip_iaa_decodable(void);

int cpu_zip_compress(int slot, void *src, int len);
int cpu_zip_decompress(int slot, void *src, int len);

// Slot-owned input buffer (src_cap bytes); fill it, then submit with compress_staged.
void *cpu_zip_input_buf(int slot);
int cpu_zip_compress_staged(int slot, int len);

// Synchronous backend: 1 once a result is ready, -1 for an invalid or idle slot.
int cpu_zip_poll(int slot);

int cpu_zip_wait(int slot, void **dest, int *len);

void cpu_zip_shutdown(void);

#ifdef __cplusplus
}
#endif

#endif