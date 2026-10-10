// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cpu_zip.h"
#include "env.h"

int main(void) {
    static const unsigned char input[] =
        "cpu_zip synchronous compression and decompression round-trip data";

    setenv("IAXL_CPU_ZIP_THREADS", "3", 1);
    setenv("IAXL_ZIP_SRC_CAP", "4096", 1);
    setenv("IAXL_ZIP_DST_CAP", "128", 1);
    envs_init();

    if (cpu_zip_init() != 0 || cpu_zip_num_slots() != 3 || cpu_zip_queue_depth() != 1 ||
        cpu_zip_src_cap() != 4096) {
        fprintf(stderr, "[test] invalid cpu_zip configuration\n");
        return 1;
    }

    void *compressed;
    int compressed_len;
    if (cpu_zip_wait(0, NULL, NULL, 0) == 0 ||
        cpu_zip_compress(0, (void *)input, (int)sizeof(input)) != 0 ||
        cpu_zip_wait(0, &compressed, &compressed_len, 0) != 0) {
        fprintf(stderr, "[test] synchronous compression failed\n");
        cpu_zip_shutdown();
        return 1;
    }

    unsigned char *compressed_copy = malloc((size_t)compressed_len);
    if (!compressed_copy) {
        cpu_zip_shutdown();
        return 1;
    }
    memcpy(compressed_copy, compressed, (size_t)compressed_len);

    void *decompressed;
    int decompressed_len;
    if (cpu_zip_decompress(1, compressed_copy, compressed_len) != 0 ||
        cpu_zip_wait(1, &decompressed, &decompressed_len, 0) != 0 ||
        decompressed_len != (int)sizeof(input) ||
        memcmp(decompressed, input, sizeof(input)) != 0) {
        fprintf(stderr, "[test] synchronous decompression failed\n");
        free(compressed_copy);
        cpu_zip_shutdown();
        return 1;
    }

    free(compressed_copy);
    cpu_zip_shutdown();

    setenv("IAXL_CPU_ZIP_THREADS", "0", 1);
    envs_init();
    if (cpu_zip_init() != 0 || cpu_zip_num_slots() != 0) {
        fprintf(stderr, "[test] zero-thread cpu_zip configuration failed\n");
        return 1;
    }
    cpu_zip_shutdown();

    printf("[test] cpu_zip round-trip passed\n");
    return 0;
}