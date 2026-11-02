// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

// DDR-to-DDR throughput of dsa_memcpy_batch vs dsa_memcpy_batch_async + dsa_memcpy_batch_wait.
// The block list is split into --calls API calls the way kv_xfer issues one call per chunk:
// the sync API drains the WQ at the end of every call, the async API only at the final wait.

#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>

#include "env.h"

int dsa_memcpy_batch(void *const dest[], const void *const src[], const size_t n[], size_t count);
int dsa_memcpy_batch_async(void *const dest[], const void *const src[], const size_t n[],
                           size_t count);
int dsa_memcpy_batch_wait(void);

struct blocks {
    void **dest;
    const void **src;
    size_t *n, count, total;
    unsigned char *sbuf, *dbuf;
};

static double now_sec(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

static int copy_all(const struct blocks *b, size_t calls, int async) {
    size_t per_call = (b->count + calls - 1) / calls, done;

    for (done = 0; done < b->count; done += per_call) {
        size_t cnt = b->count - done < per_call ? b->count - done : per_call;
        int rc = async ? dsa_memcpy_batch_async(b->dest + done, b->src + done, b->n + done, cnt)
                       : dsa_memcpy_batch(b->dest + done, b->src + done, b->n + done, cnt);

        if (rc)
            return rc;
    }
    return async ? dsa_memcpy_batch_wait() : 0;
}

/* Returns average GB/s over iters, or a negative value on failure. */
static double measure(const struct blocks *b, size_t calls, int async, int iters) {
    const char *name = async ? "dsa_memcpy_batch_async" : "dsa_memcpy_batch";
    double t0, elapsed;
    int it;

    memset(b->dbuf, 0, b->total);
    if (copy_all(b, calls, async)) {
        fprintf(stderr, "%s: copy failed\n", name);
        return -1;
    }
    if (memcmp(b->dbuf, b->sbuf, b->total)) {
        fprintf(stderr, "%s: verification FAILED\n", name);
        return -1;
    }

    t0 = now_sec();
    for (it = 0; it < iters; it++) {
        if (copy_all(b, calls, async)) {
            fprintf(stderr, "%s: copy failed\n", name);
            return -1;
        }
    }
    elapsed = (now_sec() - t0) / iters;

    printf("  %-24s %6zu %10.3f %8.2f\n", name, calls, elapsed * 1e3, (double)b->total / elapsed / 1e9);
    return (double)b->total / elapsed / 1e9;
}

static int blocks_init(struct blocks *b, size_t total, size_t block) {
    size_t i;

    b->count = total / block;
    b->total = b->count * block;
    if (posix_memalign((void **)&b->sbuf, 4096, b->total) ||
        posix_memalign((void **)&b->dbuf, 4096, b->total)) {
        perror("posix_memalign");
        return -1;
    }
    if (mlock(b->sbuf, b->total) || mlock(b->dbuf, b->total))
        perror("mlock (continuing)");

    for (i = 0; i < b->total; i += 64)
        b->sbuf[i] = (unsigned char)(i * 131 + 7);

    b->dest = malloc(b->count * sizeof(*b->dest));
    b->src = malloc(b->count * sizeof(*b->src));
    b->n = malloc(b->count * sizeof(*b->n));
    for (i = 0; i < b->count; i++) {
        b->dest[i] = b->dbuf + i * block;
        b->src[i] = b->sbuf + i * block;
        b->n[i] = block;
    }
    return 0;
}

static void usage(const char *prog) {
    fprintf(stderr,
            "usage: %s [-s total_MiB] [-b block_KiB] [-c calls] [-i iters]\n"
            "  -s  bytes copied per iteration            (default 512 MiB)\n"
            "  -b  bytes per block / descriptor          (default 64 KiB)\n"
            "  -c  API calls the block list is split into (default 64)\n"
            "  -i  timed iterations per API              (default 5)\n"
            "WQs come from $IAXL_DSA_WQS (default wq0.0).\n",
            prog);
}

int main(int argc, char **argv) {
    size_t total = 512ull << 20, block = 64ull << 10, calls = 64;
    int iters = 5, opt;
    struct blocks b;
    double sync_gbps, async_gbps;

    while ((opt = getopt(argc, argv, "s:b:c:i:h")) != -1) {
        switch (opt) {
        case 's':
            total = strtoull(optarg, NULL, 0) << 20;
            break;
        case 'b':
            block = strtoull(optarg, NULL, 0) << 10;
            break;
        case 'c':
            calls = strtoull(optarg, NULL, 0);
            break;
        case 'i':
            iters = atoi(optarg);
            break;
        default:
            usage(argv[0]);
            return opt == 'h' ? 0 : 1;
        }
    }
    if (!total || !block || block % 8 || !calls || iters < 1 || blocks_init(&b, total, block))
        return 1;
    if (calls > b.count)
        calls = b.count;

    printf("DDR2DDR: %zu MiB as %zu x %zu KiB blocks, WQs=%s, %d iters\n", b.total >> 20, b.count,
           block >> 10, envs.IAXL_DSA_WQS(), iters);
    printf("  %-24s %6s %10s %8s\n", "API", "calls", "avg(ms)", "GB/s");

    if (measure(&b, 1, 0, iters) < 0 || measure(&b, 1, 1, iters) < 0)
        return 1;
    sync_gbps = measure(&b, calls, 0, iters);
    async_gbps = measure(&b, calls, 1, iters);
    if (sync_gbps < 0 || async_gbps < 0)
        return 1;

    printf("async/sync speedup at %zu calls: %.2fx\n", calls, async_gbps / sync_gbps);
    return 0;
}
