// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <stdio.h>
#include <stdlib.h>

#include <zlib.h>

#include "cpu_zip.h"
#include "env.h"

typedef struct {
    unsigned char *in;
    unsigned char *out;
    int len;
    int ready;
} CpuZipSlot;

static CpuZipSlot *g_slots;
static int g_initialized;
static int g_slot_count;
static int g_src_cap;
static int g_dst_cap;
static int g_buf_cap;

int cpu_zip_init(void) {
    if (g_initialized)
        return 0;

    g_slot_count = envs.IAXL_CPU_ZIP_THREADS;
    if (g_slot_count < 0)
        return -1;
    g_src_cap = envs.IAXL_ZIP_SRC_CAP;
    g_dst_cap = envs.IAXL_ZIP_DST_CAP;
    g_buf_cap = g_src_cap > g_dst_cap ? g_src_cap : g_dst_cap;

    if (g_slot_count == 0) {
        g_initialized = 1;
        printf("[cpu_zip] disabled: threads=0\n");
        return 0;
    }

    g_slots = calloc((size_t)g_slot_count, sizeof(*g_slots));
    if (!g_slots)
        return -1;

    for (int slot = 0; slot < g_slot_count; slot++) {
        g_slots[slot].in = malloc((size_t)g_src_cap);
        g_slots[slot].out = malloc((size_t)g_buf_cap);
        if (!g_slots[slot].in || !g_slots[slot].out) {
            cpu_zip_shutdown();
            return -1;
        }
    }

    g_initialized = 1;
    printf("[cpu_zip] config: threads=%d slots=%d src_cap=%d B dst_cap=%d B depth=1\n",
           g_slot_count, g_slot_count, g_src_cap, g_dst_cap);
    return 0;
}

int cpu_zip_num_slots(void) { return g_slot_count; }
int cpu_zip_queue_depth(void) { return 1; }
int cpu_zip_src_cap(void) { return g_src_cap; }

// Intel QPL only decodes a 4 KB history window, so shrink ours whenever an IAA worker may be the
// one to decompress this chunk. kv_zip tags each block with the answer and routes accordingly.
static int compress_window_bits(void) { return envs.IAXL_IAA_ZIP_ENABLE ? -12 : -MAX_WBITS; }

int cpu_zip_iaa_decodable(void) { return compress_window_bits() >= -12; }

int cpu_zip_compress(int slot, void *src, int len) {
    if (!g_slots || slot < 0 || slot >= g_slot_count || !src || len <= 0 || len > g_src_cap)
        return -1;

    CpuZipSlot *state = &g_slots[slot];
    z_stream stream = {0};
    state->ready = 0;
    if (deflateInit2(&stream, Z_DEFAULT_COMPRESSION, Z_DEFLATED, compress_window_bits(), 8,
                     Z_DEFAULT_STRATEGY) != Z_OK)
        return -1;
    stream.next_in = src;
    stream.avail_in = (uInt)len;
    stream.next_out = state->out;
    stream.avail_out = (uInt)g_dst_cap;
    int status = deflate(&stream, Z_FINISH);
    int out_len = (int)stream.total_out;
    deflateEnd(&stream);
    if (status != Z_STREAM_END)
        return -1;

    state->len = out_len;
    state->ready = 1;
    return 0;
}

int cpu_zip_decompress(int slot, void *src, int len) {
    if (!g_slots || slot < 0 || slot >= g_slot_count || !src || len <= 0 || len > g_dst_cap)
        return -1;

    CpuZipSlot *state = &g_slots[slot];
    z_stream stream = {0};
    state->ready = 0;
    if (inflateInit2(&stream, -MAX_WBITS) != Z_OK)
        return -1;
    stream.next_in = src;
    stream.avail_in = (uInt)len;
    stream.next_out = state->out;
    stream.avail_out = (uInt)g_src_cap;
    int status = inflate(&stream, Z_FINISH);
    int out_len = (int)stream.total_out;
    inflateEnd(&stream);
    if (status != Z_STREAM_END)
        return -1;

    state->len = out_len;
    state->ready = 1;
    return 0;
}

int cpu_zip_wait(int slot, void **dest, int *len) {
    if (!g_slots || slot < 0 || slot >= g_slot_count || !g_slots[slot].ready)
        return -1;

    if (dest)
        *dest = g_slots[slot].out;
    if (len)
        *len = g_slots[slot].len;
    return 0;
}

void *cpu_zip_input_buf(int slot) {
    if (!g_slots || slot < 0 || slot >= g_slot_count)
        return NULL;
    return g_slots[slot].in;
}

int cpu_zip_compress_staged(int slot, int len) {
    return cpu_zip_compress(slot, cpu_zip_input_buf(slot), len);
}

int cpu_zip_poll(int slot) {
    if (!g_slots || slot < 0 || slot >= g_slot_count || !g_slots[slot].ready)
        return -1;
    return 1;
}

void cpu_zip_shutdown(void) {
    if (g_slots) {
        for (int slot = 0; slot < g_slot_count; slot++) {
            free(g_slots[slot].in);
            free(g_slots[slot].out);
        }
        free(g_slots);
    }
    g_slots = NULL;
    g_initialized = 0;
    g_slot_count = 0;
    g_src_cap = 0;
    g_dst_cap = 0;
    g_buf_cap = 0;
}