// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "env.h"
#include "iaa_zip.h"

static void usage(const char *prog) {
    printf("Usage: %s\n\n"
           "Environment overrides (applied inside iaa_zip_init):\n"
           "  IAXL_IAA_DEVICES                   comma separated NUMA nodes or 'auto'\n"
           "  IAXL_IAA_ZIP_INSTANCES_PER_DEVICE  instances (== threads) per IAA device\n"
           "  IAXL_IAA_ZIP_QUEUE_DEPTH           in-flight requests per instance\n"
           "  IAXL_ZIP_SRC_CAP / IAXL_ZIP_DST_CAP  block capacities in bytes\n",
           prog);
}

int main(int argc, char **argv) {
    if (argc > 1 && (!strcmp(argv[1], "-h") || !strcmp(argv[1], "--help"))) {
        usage(argv[0]);
        return 0;
    }
    setenv("IAXL_ZIP_SRC_CAP", "262144", 1);
    setenv("IAXL_ZIP_DST_CAP", "262144", 1);
    envs_init();

    if (iaa_zip_init() != 0) {
        fprintf(stderr, "[test] iaa_zip_init failed\n");
        return 1;
    }

    const int depth = iaa_zip_queue_depth();
    const int slots = iaa_zip_num_slots();
    const int block = iaa_zip_src_cap();
    printf("[test] slots=%d queue_depth=%d block=%d B\n", slots, depth, block);

    int status = 1;
    unsigned char *input = malloc((size_t)block);
    unsigned char *compressed = NULL;
    if (!input)
        goto out;
    for (int i = 0; i < block; i++)
        input[i] = (unsigned char)(i % 251);

    // Keep every slot busy at once to exercise the asynchronous submit/wait path.
    for (int s = 0; s < slots; s++) {
        if (iaa_zip_compress(s, input, block) != 0) {
            fprintf(stderr, "[test] compression submit failed on slot %d\n", s);
            goto out;
        }
    }

    int compressed_len = 0;
    for (int s = 0; s < slots; s++) {
        void *out;
        int len;
        if (iaa_zip_wait(s, &out, &len, 0) != 0 || len <= 0) {
            fprintf(stderr, "[test] compression failed on slot %d\n", s);
            goto out;
        }
        if (s == 0) {
            compressed = malloc((size_t)len);
            if (!compressed)
                goto out;
            memcpy(compressed, out, (size_t)len);
            compressed_len = len;
        }
    }

    for (int s = 0; s < slots; s++) {
        if (iaa_zip_decompress(s, compressed, compressed_len) != 0) {
            fprintf(stderr, "[test] decompression submit failed on slot %d\n", s);
            goto out;
        }
    }

    for (int s = 0; s < slots; s++) {
        void *out;
        int len;
        if (iaa_zip_wait(s, &out, &len, 0) != 0 || len != block ||
            memcmp(out, input, (size_t)block) != 0) {
            fprintf(stderr, "[test] decompression mismatch on slot %d\n", s);
            goto out;
        }
    }

    printf("[test] iaa_zip round-trip passed: %d B -> %d B (ratio %.2fx)\n", block, compressed_len,
           (double)block / (double)compressed_len);
    status = 0;

out:
    free(input);
    free(compressed);
    iaa_zip_shutdown();
    return status;
}
