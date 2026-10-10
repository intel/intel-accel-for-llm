// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <omp.h>

#include "qat_zip.h"
#include "env.h"

#define PERF_ITERS 20000

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void usage(const char *prog) {
    printf("Usage: %s [data-file] [devices]\n"
           "\n"
           "QAT DEFLATE compress/decompress throughput test.\n"
           "\n"
           "Positional arguments:\n"
           "  data-file   test-data file, cycled to fill each block (default: data.txt)\n"
           "  devices     comma-separated device indices to use, e.g. 0,1 (default: 0)\n"
           "\n"
           "Options:\n"
           "  -h, --help  show this help and exit\n"
           "\n"
           "Environment overrides (applied inside qat_zip_init):\n"
           "  IAXL_QAT_DEVICES                   comma-separated device indices (default 0)\n"
           "  IAXL_QAT_ZIP_INSTANCES_PER_DEVICE  instances (== threads) per device\n"
           "  IAXL_ZIP_SRC_CAP                   source block size in bytes\n"
           "  IAXL_ZIP_DST_CAP                   compressed-output cap in bytes\n"
           "  IAXL_QAT_ZIP_QUEUE_DEPTH           in-flight requests per instance\n",
           prog);
}

static unsigned char *load_data_file(const char *path, size_t *out_size) {
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "[test] cannot open %s\n", path);
        return NULL;
    }
    fseek(fp, 0, SEEK_END);
    long sz = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    if (sz <= 0) {
        fclose(fp);
        fprintf(stderr, "[test] %s is empty\n", path);
        return NULL;
    }
    unsigned char *buf = (unsigned char *)malloc((size_t)sz);
    if (!buf) {
        fclose(fp);
        return NULL;
    }
    if (fread(buf, 1, (size_t)sz, fp) != (size_t)sz) {
        fclose(fp);
        free(buf);
        fprintf(stderr, "[test] read failed\n");
        return NULL;
    }
    fclose(fp);
    *out_size = (size_t)sz;
    printf("[data] loaded %s (%zu bytes)\n", path, *out_size);
    return buf;
}

static void fill_cycled(unsigned char *buf, size_t len, const unsigned char *data,
                        size_t data_len) {
    for (size_t off = 0; off < len;) {
        size_t n = len - off;
        if (n > data_len)
            n = data_len;
        memcpy(buf + off, data, n);
        off += n;
    }
}

int main(int argc, char **argv) {
    for (int i = 1; i < argc; i++)
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            usage(argv[0]);
            return 0;
        }

    const char *data_path = (argc > 1) ? argv[1] : "data.txt";
    const char *devices = (argc > 2) ? argv[2] : NULL;

    if (devices)
        setenv("IAXL_QAT_DEVICES", devices, 1);
    envs_init();

    size_t data_len = 0;
    unsigned char *data = load_data_file(data_path, &data_len);
    if (!data)
        return 1;

    qat_zip_init();

    int qd = qat_zip_queue_depth();
    int block = qat_zip_src_cap();
    int inst_count = qat_zip_num_slots() / qd;

    unsigned char *src = (unsigned char *)malloc((size_t)block);
    if (!src) {
        free(data);
        qat_zip_shutdown();
        return 1;
    }
    fill_cycled(src, (size_t)block, data, data_len);

    void *cptr;
    int clen;
    if (qat_zip_compress(0, src, block) != 0 || qat_zip_wait(0, &cptr, &clen, 0) != 0) {
        fprintf(stderr, "[verify] compress failed\n");
        free(src);
        free(data);
        qat_zip_shutdown();
        return 1;
    }

    unsigned char *cblock = (unsigned char *)malloc((size_t)clen);
    int cblock_len = clen;
    memcpy(cblock, cptr, (size_t)clen);

    void *dptr;
    int dlen;
    if (qat_zip_decompress(0, cblock, cblock_len) != 0 || qat_zip_wait(0, &dptr, &dlen, 0) != 0) {
        fprintf(stderr, "[verify] decompress failed\n");
        free(cblock);
        free(src);
        free(data);
        qat_zip_shutdown();
        return 1;
    }
    if (dlen != block || memcmp(src, dptr, (size_t)block) != 0) {
        fprintf(stderr, "[verify] round-trip verification FAILED\n");
        free(cblock);
        free(src);
        free(data);
        qat_zip_shutdown();
        return 1;
    }
    printf("[verify] round-trip OK: %d -> %d -> %d bytes (ratio %.3fx)\n", block, clen, dlen,
           (double)block / (double)clen);

    for (int t = 0; t < inst_count; t++)
        for (int k = 0; k < 100; k++) {
            int slot = t * qd;
            void *p;
            int l;
            qat_zip_compress(slot, src, block);
            qat_zip_wait(slot, &p, &l, 0);
        }

    double t0 = now_sec();
#pragma omp parallel for schedule(static, 1) num_threads(inst_count)
    for (int t = 0; t < inst_count; t++) {
        int base = t * qd;
        int depth = qd < PERF_ITERS ? qd : PERF_ITERS;
        for (int s = 0; s < depth; s++)
            qat_zip_compress(base + s, src, block);
        int submitted = depth, completed = 0, s = 0;
        while (completed < PERF_ITERS) {
            void *p;
            int l;
            qat_zip_wait(base + s, &p, &l, 0);
            completed++;
            if (submitted < PERF_ITERS) {
                qat_zip_compress(base + s, src, block);
                submitted++;
            }
            s = (s + 1) % depth;
        }
    }
    double c_sec = now_sec() - t0;

    t0 = now_sec();
#pragma omp parallel for schedule(static, 1) num_threads(inst_count)
    for (int t = 0; t < inst_count; t++) {
        int base = t * qd;
        int depth = qd < PERF_ITERS ? qd : PERF_ITERS;
        for (int s = 0; s < depth; s++)
            qat_zip_decompress(base + s, cblock, cblock_len);
        int submitted = depth, completed = 0, s = 0;
        while (completed < PERF_ITERS) {
            void *p;
            int l;
            qat_zip_wait(base + s, &p, &l, 0);
            completed++;
            if (submitted < PERF_ITERS) {
                qat_zip_decompress(base + s, cblock, cblock_len);
                submitted++;
            }
            s = (s + 1) % depth;
        }
    }
    double d_sec = now_sec() - t0;

    double total_gb = (double)block * PERF_ITERS * inst_count / 1e9;
    printf("[perf] instances=%d  iters/instance=%d  chunk=%d KB\n", inst_count, PERF_ITERS,
           block / 1024);
    printf("[perf] compress:   %.2f GB/s  (%.3f s)\n", total_gb / c_sec, c_sec);
    printf("[perf] decompress: %.2f GB/s  (%.3f s)\n", total_gb / d_sec, d_sec);

    free(cblock);
    free(src);
    free(data);
    qat_zip_shutdown();
    return 0;
}
