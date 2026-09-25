// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <unistd.h>
#include <sched.h>
#include <time.h>
#include <sys/mman.h>
#include <linux/idxd.h>
#include <x86intrin.h>
#include <omp.h>

#include "dsa_memcpy.h"
#include "env.h"
#include "iaxl_common.h"

#if !defined(DSA_WAIT_BUSYPOLL) && !defined(DSA_WAIT_UMWAIT) && !defined(DSA_WAIT_YIELD) &&        \
    !defined(DSA_WAIT_TPAUSE)
#define DSA_WAIT_BUSYPOLL
#endif

#define C01_STATE 1
#define C02_STATE 0
#define UMWAIT_DELAY 100000u
#define TPAUSE_DELAY 1000u
#define DSA_COMPLETION_TIMEOUT_NS (10LL * 1000 * 1000 * 1000)
#define DSA_TIMEOUT_CHECK_INTERVAL 4096u

#define DSA_WQS_ENV "IAXL_DSA_WQS"
#define DSA_WQS_DEFAULT "wq0.0"
#define DSA_MAX_WQ 64
#define DSA_PORTAL_SIZE 0x1000
#define DSA_MAX_XFER 2147483648
#define DSA_ALIGN 8u
#define ENQCMD_MAX_RETRIES 1000000u
/* Batches kept in flight per WQ so the engine never idles while descriptors are refilled. */
#define DSA_BATCH_DEPTH 8
// Entries per WQ left to the synchronous copy paths, which do not take credits: the batch
// pipeline (serialised by g_batch_mutex) plus single dsa_memcpy() callers.
#define DSA_SYNC_RESERVE (DSA_BATCH_DEPTH + 8)

static char g_wq_name[DSA_MAX_WQ][32];
static void *g_wq_portal[DSA_MAX_WQ];
// A dedicated WQ silently drops a MOVDIR64B beyond its size, so async copies hold a credit.
static int g_wq_credits[DSA_MAX_WQ];
static unsigned int g_next_wq;
static size_t g_num_wq;
static pthread_mutex_t g_init_mutex = PTHREAD_MUTEX_INITIALIZER;

static size_t g_max_xfer = DSA_MAX_XFER;

static size_t g_max_batch = 1;

// Without BOF the engine aborts on the first untouched page (fresh malloc) instead of blocking.
static uint32_t g_desc_flags = IDXD_OP_FLAG_CRAV | IDXD_OP_FLAG_RCR;

/* Descriptor buffers, one set of DSA_BATCH_DEPTH slots per WQ, reused by every batch call. */
static struct dsa_hw_desc *g_subs[DSA_MAX_WQ];
static struct dsa_completion_record *g_comps[DSA_MAX_WQ];
static struct dsa_completion_record *g_bcomps[DSA_MAX_WQ];
// The buffers above are shared, so batch calls from different threads take turns.
static pthread_mutex_t g_batch_mutex = PTHREAD_MUTEX_INITIALIZER;

static inline void dsa_wait_pause(const volatile uint8_t *comp) {
#if defined(DSA_WAIT_YIELD)
    (void)comp;
    sched_yield();
#elif defined(DSA_WAIT_UMWAIT)
    _umonitor((void *)comp);
    _umwait(C02_STATE, _rdtsc() + UMWAIT_DELAY);
#elif defined(DSA_WAIT_TPAUSE)
    (void)comp;
    _tpause(C02_STATE, _rdtsc() + TPAUSE_DELAY);
#else
    (void)comp;
    _mm_pause();
#endif
}

static inline void dsa_check_timeout(const struct timespec *start, unsigned int *iterations) {
    struct timespec now;
    int64_t elapsed_ns;

    if (++*iterations != DSA_TIMEOUT_CHECK_INTERVAL)
        return;

    IAXL_CHECK(clock_gettime(CLOCK_MONOTONIC, &now) == 0,
               "dsa: failed to read completion timeout clock");
    elapsed_ns = (int64_t)(now.tv_sec - start->tv_sec) * 1000000000LL +
                 (int64_t)(now.tv_nsec - start->tv_nsec);
    IAXL_CHECK(elapsed_ns < DSA_COMPLETION_TIMEOUT_NS,
               "dsa: completion poll timed out after 10 seconds");
    *iterations = 0;
}

static inline void dsa_wait_completion(const volatile uint8_t *comp) {
    struct timespec start;
    unsigned int iterations = 0;

    IAXL_CHECK(clock_gettime(CLOCK_MONOTONIC, &start) == 0,
               "dsa: failed to read completion timeout clock");

    while (*comp == 0) {
        dsa_wait_pause(comp);
        dsa_check_timeout(&start, &iterations);
    }
}

/* Polls every in-flight batch and returns the index in slots[] of the first one that finishes. */
static size_t dsa_wait_any(const volatile struct dsa_completion_record *bcomp, const size_t *slots,
                           size_t count) {
    struct timespec start;
    unsigned int iterations = 0;

    IAXL_CHECK(clock_gettime(CLOCK_MONOTONIC, &start) == 0,
               "dsa: failed to read completion timeout clock");

    for (;;) {
        size_t k;

        for (k = 0; k < count; k++) {
            if (bcomp[slots[k]].status)
                return k;
        }
        dsa_wait_pause(&bcomp[slots[0]].status);
        dsa_check_timeout(&start, &iterations);
    }
}

static inline unsigned char enqcmd(struct dsa_hw_desc *desc, volatile void *reg) {
    unsigned char retry;

    asm volatile(".byte 0xf2, 0x0f, 0x38, 0xf8, 0x02\t\n"
                 "setz %0\t\n"
                 : "=r"(retry)
                 : "a"(reg), "d"(desc));
    return retry;
}

static inline void movdir64b(struct dsa_hw_desc *desc, volatile void *reg) {
    asm volatile(".byte 0x66, 0x0f, 0x38, 0xf8, 0x02\t\n" : : "a"(reg), "d"(desc));
}

static size_t dsa_read_wq_attr(const char *wq, const char *attr, size_t fallback) {
    char path[96];
    char buf[32];
    unsigned long long val;
    int fd;
    ssize_t r;

    snprintf(path, sizeof(path), "/sys/bus/dsa/devices/%s/%s", wq, attr);
    fd = open(path, O_RDONLY);
    if (fd < 0)
        return fallback;

    r = read(fd, buf, sizeof(buf) - 1);
    close(fd);
    if (r <= 0)
        return fallback;

    buf[r] = '\0';
    val = strtoull(buf, NULL, 0);
    if (val == 0)
        return fallback;

    return (size_t)val;
}

static int dsa_submit_batch(void *portal, struct dsa_hw_desc *sub,
                            struct dsa_completion_record *comp,
                            struct dsa_completion_record *bcomp, void *const dest[],
                            const void *const src[], const size_t n[], size_t first, size_t cnt);

static int dsa_submit(struct dsa_hw_desc *desc, void *portal) {
#ifdef DSA_WQ_SHARED
    unsigned int retries = 0;

    while (enqcmd(desc, portal)) {
        if (++retries > ENQCMD_MAX_RETRIES) {
            fprintf(stderr, "enqcmd retries exhausted\n");
            return -1;
        }
        _mm_pause();
    }
#else
    movdir64b(desc, portal);
#endif
    return 0;
}

static size_t dsa_parse_wqs(void) {
    const char *env = envs.IAXL_DSA_WQS();
    char buf[DSA_MAX_WQ * 32];
    char *tok, *save;
    size_t count = 0;

    snprintf(buf, sizeof(buf), "%s", env);
    for (tok = strtok_r(buf, ", \t", &save); tok && count < DSA_MAX_WQ;
         tok = strtok_r(NULL, ", \t", &save)) {
        snprintf(g_wq_name[count], sizeof(g_wq_name[count]), "%s", tok);
        count++;
    }

    return count;
}

static int dsa_init(void) {
    void *portals[DSA_MAX_WQ] = {NULL};
    struct dsa_hw_desc *subs[DSA_MAX_WQ] = {NULL};
    struct dsa_completion_record *comps[DSA_MAX_WQ] = {NULL};
    struct dsa_completion_record *bcomps[DSA_MAX_WQ] = {NULL};
    size_t num_wq, w;
    int ret = -1;

    pthread_mutex_lock(&g_init_mutex);
    if (g_num_wq) {
        ret = 0;
        goto out;
    }

    num_wq = dsa_parse_wqs();
    if (num_wq == 0) {
        fprintf(stderr, "no WQ configured in $" DSA_WQS_ENV "\n");
        goto out;
    }

    g_max_xfer = dsa_read_wq_attr(g_wq_name[0], "max_transfer_size", DSA_MAX_XFER);
    g_max_batch = dsa_read_wq_attr(g_wq_name[0], "max_batch_size", 1);
    if (dsa_read_wq_attr(g_wq_name[0], "block_on_fault", 0))
        g_desc_flags |= IDXD_OP_FLAG_BOF;

    for (w = 0; w < num_wq; w++) {
        char path[64];
        int fd;
        void *portal;

        snprintf(path, sizeof(path), "/dev/dsa/%s", g_wq_name[w]);
        fd = open(path, O_RDWR);
        if (fd < 0) {
            fprintf(stderr, "open %s failed: %s\n", path, strerror(errno));
            goto rollback;
        }

        portal = mmap(NULL, DSA_PORTAL_SIZE, PROT_WRITE, MAP_SHARED | MAP_POPULATE, fd, 0);
        close(fd);
        if (portal == MAP_FAILED) {
            fprintf(stderr, "mmap portal %s failed: %s\n", path, strerror(errno));
            goto rollback;
        }

        portals[w] = portal;
    }

    for (w = 0; w < num_wq; w++) {
        if (posix_memalign((void **)&subs[w], 64,
                           DSA_BATCH_DEPTH * g_max_batch * sizeof(*subs[w])) ||
            posix_memalign((void **)&comps[w], 32,
                           DSA_BATCH_DEPTH * g_max_batch * sizeof(*comps[w])) ||
            posix_memalign((void **)&bcomps[w], 32, DSA_BATCH_DEPTH * sizeof(*bcomps[w]))) {
            fprintf(stderr, "dsa_init: descriptor alloc failed\n");
            goto rollback;
        }
    }

    for (w = 0; w < num_wq; w++) {
        size_t size = dsa_read_wq_attr(g_wq_name[w], "size", 2 * DSA_SYNC_RESERVE);

        g_wq_portal[w] = portals[w];
        g_subs[w] = subs[w];
        g_comps[w] = comps[w];
        g_bcomps[w] = bcomps[w];
        g_wq_credits[w] = size > DSA_SYNC_RESERVE ? (int)(size - DSA_SYNC_RESERVE) : 1;
    }
    __atomic_store_n(&g_num_wq, num_wq, __ATOMIC_RELEASE);

    fprintf(stderr,
            "dsa_init: mapped %zu WQ(s) from $" DSA_WQS_ENV " (first %s), "
            "max_transfer_size=%zu, max_batch_size=%zu, block_on_fault=%d\n",
            g_num_wq, g_wq_name[0], g_max_xfer, g_max_batch,
            (g_desc_flags & IDXD_OP_FLAG_BOF) != 0);

    ret = 0;
    goto out;

rollback:
    for (w = 0; w < num_wq; w++) {
        if (portals[w])
            munmap(portals[w], DSA_PORTAL_SIZE);
        free(subs[w]);
        free(comps[w]);
        free(bcomps[w]);
    }
out:
    pthread_mutex_unlock(&g_init_mutex);
    return ret;
}

int dsa_memcpy(void *dest, const void *src, size_t n) {

    if (((uintptr_t)dest | (uintptr_t)src | (uintptr_t)n) & (DSA_ALIGN - 1)) {
        fprintf(stderr, "dsa_memcpy: unaligned dest=%p src=%p n=%zu (need %u-byte)\n", dest, src, n,
                DSA_ALIGN);
        return -1;
    }
#if 0

	{
		volatile uint64_t *d = (volatile uint64_t *)dest;
		const uint64_t *s = (const uint64_t *)src;
		size_t words = n / sizeof(uint64_t);
		size_t i;

		for (i = 0; i < words; i++)
			d[i] = s[i];
		__builtin_ia32_sfence();
		return 0;
	}
#else
    struct dsa_completion_record comp __attribute__((aligned(32)));
    struct dsa_hw_desc desc;
    size_t done = 0;

    if (dsa_init())
        return -1;

    while (done < n) {
        size_t len = n - done;
        unsigned int retries = 0;

        if (len > g_max_xfer)
            len = g_max_xfer;

        memset(&desc, 0, sizeof(desc));
        desc.opcode = DSA_OPCODE_MEMMOVE;

        desc.flags = g_desc_flags;
        desc.completion_addr = (uint64_t)&comp;
        desc.src_addr = (uint64_t)src + done;
        desc.dst_addr = (uint64_t)dest + done;
        desc.xfer_size = (uint32_t)len;
        comp.status = 0;

        __builtin_ia32_sfence();

#ifdef DSA_WQ_SHARED

        while (enqcmd(&desc, g_wq_portal[0])) {
            if (++retries > ENQCMD_MAX_RETRIES) {
                fprintf(stderr, "enqcmd retries exhausted\n");
                return -1;
            }
            _mm_pause();
        }
#else
        (void)retries;

        movdir64b(&desc, g_wq_portal[0]);
#endif

        dsa_wait_completion(&comp.status);

        if (comp.status != DSA_COMP_SUCCESS) {
            fprintf(stderr, "dsa op failed, status=0x%x\n", comp.status);
            return -1;
        }

        done += len;
    }

    return 0;
#endif
}

/* Fills one batch of descriptors and hands it to the WQ without waiting. */
static int dsa_submit_batch(void *portal, struct dsa_hw_desc *sub,
                            struct dsa_completion_record *comp,
                            struct dsa_completion_record *bcomp, void *const dest[],
                            const void *const src[], const size_t n[], size_t first, size_t cnt) {
    struct dsa_hw_desc bdesc;
    size_t j;

    memset(sub, 0, cnt * sizeof(*sub));
    for (j = 0; j < cnt; j++) {
        sub[j].opcode = DSA_OPCODE_MEMMOVE;

        sub[j].flags = g_desc_flags;
        sub[j].completion_addr = (uint64_t)&comp[j];
        sub[j].src_addr = (uint64_t)src[first + j];
        sub[j].dst_addr = (uint64_t)dest[first + j];
        sub[j].xfer_size = (uint32_t)n[first + j];
        comp[j].status = 0;
    }

    bcomp->status = 0;

    if (cnt == 1) {

        sub[0].completion_addr = (uint64_t)bcomp;
        __builtin_ia32_sfence();
        return dsa_submit(&sub[0], portal);
    }

    memset(&bdesc, 0, sizeof(bdesc));
    bdesc.opcode = DSA_OPCODE_BATCH;
    // BOF is only legal on the sub-descriptors; the batch descriptor rejects it.
    bdesc.flags = IDXD_OP_FLAG_CRAV | IDXD_OP_FLAG_RCR;
    bdesc.desc_list_addr = (uint64_t)sub;
    bdesc.desc_count = (uint32_t)cnt;
    bdesc.completion_addr = (uint64_t)bcomp;

    __builtin_ia32_sfence();
    return dsa_submit(&bdesc, portal);
}

static int64_t monotonic_ns(void) {
    struct timespec now;
    IAXL_CHECK(clock_gettime(CLOCK_MONOTONIC, &now) == 0, "dsa: failed to read the clock");
    return (int64_t)now.tv_sec * 1000000000LL + now.tv_nsec;
}

int dsa_copy_submit(dsa_copy_job *job, void *dest, const void *src, size_t n) {
    struct dsa_completion_record *comp = (struct dsa_completion_record *)job->comp;
    struct dsa_hw_desc desc;
    size_t num_wq = __atomic_load_n(&g_num_wq, __ATOMIC_ACQUIRE);

    _Static_assert(sizeof(job->comp) == sizeof(struct dsa_completion_record),
                   "dsa_copy_job completion record size mismatch");
    if (num_wq == 0) {
        if (dsa_init())
            return -1;
        num_wq = g_num_wq;
    }
    if (n == 0 || n > g_max_xfer)
        return 1;

    int wq = -1;
    unsigned int first = __atomic_fetch_add(&g_next_wq, 1, __ATOMIC_RELAXED);
    for (size_t k = 0; k < num_wq; k++) {
        int w = (int)((first + k) % num_wq);
        if (__atomic_sub_fetch(&g_wq_credits[w], 1, __ATOMIC_ACQUIRE) >= 0) {
            wq = w;
            break;
        }
        __atomic_add_fetch(&g_wq_credits[w], 1, __ATOMIC_RELEASE);
    }
    if (wq < 0)
        return 1;

    memset(&desc, 0, sizeof(desc));
    desc.opcode = DSA_OPCODE_MEMMOVE;
    desc.flags = g_desc_flags;
    desc.completion_addr = (uint64_t)comp;
    desc.src_addr = (uint64_t)src;
    desc.dst_addr = (uint64_t)dest;
    desc.xfer_size = (uint32_t)n;
    memset(comp, 0, sizeof(*comp));
    job->wq = wq;
    job->polls = 0;
    job->start_ns = monotonic_ns();

    __builtin_ia32_sfence();
    if (dsa_submit(&desc, g_wq_portal[wq])) {
        __atomic_add_fetch(&g_wq_credits[wq], 1, __ATOMIC_RELEASE);
        return 1;
    }
    return 0;
}

int dsa_copy_poll(dsa_copy_job *job) {
    struct dsa_completion_record *comp = (struct dsa_completion_record *)job->comp;
    // The engine writes the record only after the copied data is globally visible.
    const uint8_t status = __atomic_load_n(&comp->status, __ATOMIC_ACQUIRE);

    if (status == DSA_COMP_NONE) {
        if (++job->polls % DSA_TIMEOUT_CHECK_INTERVAL == 0)
            IAXL_CHECK(monotonic_ns() - job->start_ns < DSA_COMPLETION_TIMEOUT_NS,
                       "dsa: async copy timed out after 10 seconds");
        return 0;
    }
    __atomic_add_fetch(&g_wq_credits[job->wq], 1, __ATOMIC_RELEASE);
    if (status == DSA_COMP_SUCCESS)
        return 1;
    fprintf(stderr, "dsa async copy failed, status=0x%x fault_addr=0x%llx\n", status,
            (unsigned long long)comp->fault_addr);
    return -1;
}

int dsa_memcpy_batch(void *const dest[], const void *const src[], const size_t n[], size_t count) {
    size_t per_batch, i, nbatches, nthreads;
    int ok = 1;

    if (count == 0)
        return 0;

    if (dsa_init())
        return -1;

    nthreads = g_num_wq;

    for (i = 0; i < count; i++) {
        if (((uintptr_t)dest[i] | (uintptr_t)src[i] | (uintptr_t)n[i]) & (DSA_ALIGN - 1)) {
            fprintf(stderr,
                    "dsa_memcpy_batch: unaligned entry %zu "
                    "dest=%p src=%p n=%zu (need %u-byte)\n",
                    i, dest[i], src[i], n[i], DSA_ALIGN);
            return -1;
        }
        if (n[i] > g_max_xfer) {
            fprintf(stderr,
                    "dsa_memcpy_batch: entry %zu size %zu exceeds "
                    "max_transfer_size %zu\n",
                    i, n[i], g_max_xfer);
            return -1;
        }
    }

    per_batch = g_max_batch;
    nbatches = (count + per_batch - 1) / per_batch;

    pthread_mutex_lock(&g_batch_mutex);
#pragma omp parallel num_threads(nthreads) reduction(&& : ok)
    {
        int tid = omp_get_thread_num();
        void *portal = g_wq_portal[tid];
        struct dsa_completion_record *bcomp = g_bcomps[tid];
        size_t free_slots[DSA_BATCH_DEPTH], busy_slots[DSA_BATCH_DEPTH];
        size_t slot_cnt[DSA_BATCH_DEPTH];
        size_t next = (size_t)tid;
        size_t nfree = DSA_BATCH_DEPTH, inflight = 0, k;

        for (k = 0; k < DSA_BATCH_DEPTH; k++)
            free_slots[k] = k;

        while (next < nbatches || inflight) {
            while (nfree && next < nbatches) {
                size_t slot = free_slots[nfree - 1];
                size_t done = next * per_batch;
                size_t cnt = count - done;

                if (cnt > per_batch)
                    cnt = per_batch;

                if (dsa_submit_batch(portal, g_subs[tid] + slot * per_batch,
                                     g_comps[tid] + slot * per_batch, &bcomp[slot], dest, src, n,
                                     done, cnt)) {
                    ok = 0;
                    next = nbatches; /* drain what is in flight, submit no more */
                    break;
                }
                slot_cnt[slot] = cnt;
                nfree--;
                busy_slots[inflight++] = slot;
                next += nthreads;
            }

            if (!inflight)
                break;

            k = dsa_wait_any(bcomp, busy_slots, inflight);
            if (bcomp[busy_slots[k]].status != DSA_COMP_SUCCESS) {
                const size_t slot = busy_slots[k];
                const struct dsa_completion_record *comp = g_comps[tid] + slot * per_batch;
                unsigned int sub_status = 0;
                size_t sub_index = 0, j;

                for (j = 0; j < slot_cnt[slot] && slot_cnt[slot] > 1; j++) {
                    if (comp[j].status != DSA_COMP_SUCCESS && comp[j].status != DSA_COMP_NONE) {
                        sub_status = comp[j].status;
                        sub_index = j;
                        break;
                    }
                }
                fprintf(stderr,
                        "dsa batch failed, status=0x%x (first failed sub %zu status=0x%x)\n",
                        bcomp[slot].status, sub_index, sub_status);
                ok = 0;
            }
            free_slots[nfree++] = busy_slots[k];
            busy_slots[k] = busy_slots[--inflight];
        }
    }
    pthread_mutex_unlock(&g_batch_mutex);

    return ok ? 0 : -1;
}

#ifdef DSA_MEMCPY_TEST
#include <stdlib.h>

int main(void) {
    size_t n = 4 * 1024 * 1024;
    unsigned char *src = NULL, *dst = NULL;
    size_t i;

    src = malloc(n);
    dst = malloc(n);
    if (!src || !dst) {
        perror("malloc");
        return 1;
    }

    if (mlock(src, n) || mlock(dst, n))
        perror("mlock");

    for (i = 0; i < n; i++) {
        src[i] = (unsigned char)(i * 131 + 7);
        dst[i] = 0;
    }

    if (dsa_memcpy(dst, src, n)) {
        fprintf(stderr, "dsa_memcpy failed\n");
        return 1;
    }

    if (memcmp(dst, src, n) != 0) {
        fprintf(stderr, "verification FAILED\n");
        return 1;
    }

    printf("dsa_memcpy OK: %zu bytes copied and verified\n", n);

    {
        const size_t nblk = 64;
        const size_t blk = n / nblk;
        unsigned char *dst2 = malloc(n);
        void *dests[nblk];
        const void *srcs[nblk];
        size_t sizes[nblk];

        if (!dst2 || (blk & (DSA_ALIGN - 1))) {
            fprintf(stderr, "batch test setup failed\n");
            free(dst2);
            return 1;
        }
        if (mlock(dst2, n))
            perror("mlock");
        memset(dst2, 0, n);

        for (i = 0; i < nblk; i++) {
            srcs[i] = src + i * blk;
            dests[i] = dst2 + i * blk;
            sizes[i] = blk;
        }

        if (dsa_memcpy_batch(dests, srcs, sizes, nblk)) {
            fprintf(stderr, "dsa_memcpy_batch failed\n");
            free(dst2);
            return 1;
        }

        if (memcmp(dst2, src, nblk * blk) != 0) {
            fprintf(stderr, "batch verification FAILED\n");
            free(dst2);
            return 1;
        }

        printf("dsa_memcpy_batch OK: %zu blocks x %zu bytes copied "
               "and verified\n",
               nblk, blk);
        free(dst2);
    }

    free(src);
    free(dst);
    return 0;
}
#endif
