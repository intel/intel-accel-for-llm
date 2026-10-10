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

// Requests finish inside submit, so non_block is accepted only to match the QAT/IAA API.
int cpu_zip_wait(int slot, void **dest, int *len, int non_block);

void cpu_zip_shutdown(void);

#ifdef __cplusplus
}
#endif

#endif