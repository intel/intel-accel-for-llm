// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "dsa_gd.h"

#include <stdio.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>

#include "gdrapi.h"

#ifdef __cplusplus
extern "C" {
#endif
extern int dsa_memcpy(void *dest, const void *src, size_t n);
extern int dsa_memcpy_batch(void *const dest[], const void *const src[], const size_t n[],
                            size_t count);
#ifdef __cplusplus
}
#endif

struct mapping {
    gdr_mh_t mh;
    void *bar;
    size_t map_size;
    uint64_t va;
    uint64_t base;
    struct mapping *next;
};

struct gd_ctx {
    gdr_t g;
    struct mapping *maps;
};

#define CU_POINTER_ATTRIBUTE_SYNC_MEMOPS 6
static void set_sync_memops(uint64_t dptr) {
    static void *cuda;
    static int (*set_attr)(const void *, int, unsigned long long);
    static int tried;

    if (!tried) {
        tried = 1;
        cuda = dlopen("libcuda.so.1", RTLD_NOW | RTLD_GLOBAL);
        if (cuda)
            set_attr = (int (*)(const void *, int, unsigned long long))dlsym(
                cuda, "cuPointerSetAttribute");
    }
    if (set_attr) {
        int flag = 1;
        set_attr(&flag, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS, dptr);
    }
}

gd_ctx *dsa_gd_open(void) {
    gd_ctx *ctx = (gd_ctx *)calloc(1, sizeof(*ctx));

    if (!ctx)
        return NULL;
    ctx->g = gdr_open();
    if (!ctx->g) {
        fprintf(stderr, "gdr_open() failed (is gdrdrv loaded?)\n");
        free(ctx);
        return NULL;
    }
    return ctx;
}

void dsa_gd_close(gd_ctx *ctx) {
    struct mapping *m;

    if (!ctx)
        return;
    m = ctx->maps;
    while (m) {
        struct mapping *next = m->next;

        gdr_unmap(ctx->g, m->mh, m->bar, m->map_size);
        gdr_unpin_buffer(ctx->g, m->mh);
        free(m);
        m = next;
    }
    if (ctx->g)
        gdr_close(ctx->g);
    free(ctx);
}

void dsa_gd_reset(gd_ctx *ctx) {
    struct mapping *m;

    if (!ctx)
        return;
    m = ctx->maps;
    while (m) {
        struct mapping *next = m->next;

        gdr_unmap(ctx->g, m->mh, m->bar, m->map_size);
        gdr_unpin_buffer(ctx->g, m->mh);
        free(m);
        m = next;
    }
    ctx->maps = NULL;
}

static struct mapping *find_enclosing(gd_ctx *ctx, uint64_t dptr, size_t n) {
    for (struct mapping *m = ctx->maps; m; m = m->next)
        if (m->va <= dptr && dptr + n <= m->va + m->map_size)
            return m;
    return NULL;
}

static struct mapping *mapping_for(gd_ctx *ctx, uint64_t dptr, size_t n) {
    struct mapping *m = find_enclosing(ctx, dptr, n);
    uint64_t base, offset;
    size_t map_size;
    gdr_mh_t mh;
    void *bar = NULL;
    gdr_info_v2_t info;
    int rc;

    if (m)
        return m;

    base = dptr & GD_GPU_PAGE_MASK;
    offset = dptr - base;
    map_size = (offset + n + GD_GPU_PAGE_SIZE - 1) & GD_GPU_PAGE_MASK;

    for (m = ctx->maps; m; m = m->next)
        if (m->base == base && m->map_size == map_size)
            return m;

    rc = gdr_pin_buffer(ctx->g, base, map_size, 0, 0, &mh);
    if (rc != 0) {
        fprintf(stderr, "gdr_pin_buffer failed rc=%d base=0x%lx size=%zu\n", rc,
                (unsigned long)base, map_size);
        return NULL;
    }
    rc = gdr_map(ctx->g, mh, &bar, map_size);
    if (rc != 0) {
        fprintf(stderr, "gdr_map failed rc=%d\n", rc);
        gdr_unpin_buffer(ctx->g, mh);
        return NULL;
    }
    rc = gdr_get_info_v2(ctx->g, mh, &info);
    if (rc != 0) {
        fprintf(stderr, "gdr_get_info_v2 failed rc=%d\n", rc);
        gdr_unmap(ctx->g, mh, bar, map_size);
        gdr_unpin_buffer(ctx->g, mh);
        return NULL;
    }
    set_sync_memops(base);

    m = (struct mapping *)calloc(1, sizeof(*m));
    if (!m) {
        gdr_unmap(ctx->g, mh, bar, map_size);
        gdr_unpin_buffer(ctx->g, mh);
        return NULL;
    }
    m->mh = mh;
    m->bar = bar;
    m->map_size = map_size;
    m->va = info.va;
    m->base = base;
    m->next = ctx->maps;
    ctx->maps = m;
    return m;
}

int dsa_gd_register(gd_ctx *ctx, uint64_t dptr, size_t nbytes) {
    return mapping_for(ctx, dptr, nbytes) ? 0 : -1;
}

/* Releases the mapping that dsa_gd_register(ctx, dptr, nbytes) created; -1 if there is none. */
int dsa_gd_unregister(gd_ctx *ctx, uint64_t dptr, size_t nbytes) {
    uint64_t base = dptr & GD_GPU_PAGE_MASK;
    size_t map_size = (dptr - base + nbytes + GD_GPU_PAGE_SIZE - 1) & GD_GPU_PAGE_MASK;
    struct mapping *prev = NULL;
    struct mapping *m = ctx->maps;

    while (m && !(m->base == base && m->map_size == map_size)) {
        prev = m;
        m = m->next;
    }
    if (!m)
        return -1;
    if (prev)
        prev->next = m->next;
    else
        ctx->maps = m->next;
    gdr_unmap(ctx->g, m->mh, m->bar, m->map_size);
    gdr_unpin_buffer(ctx->g, m->mh);
    free(m);
    return 0;
}

static int resolve(gd_ctx *ctx, uint64_t ptr, int is_cuda, size_t n, struct mapping **out_map,
                   uint64_t *cpu_addr) {
    if (is_cuda) {
        struct mapping *m = mapping_for(ctx, ptr, n);

        if (!m)
            return -1;
        *out_map = m;
        *cpu_addr = (uint64_t)(uintptr_t)m->bar + (ptr - m->va);
    } else {
        *out_map = NULL;
        *cpu_addr = ptr;
    }
    return 0;
}

int dsa_gd_copy(gd_ctx *ctx, uint64_t dst, int dst_cuda, uint64_t src, int src_cuda,
                size_t nbytes) {
    struct mapping *dm, *sm;
    uint64_t daddr, saddr;
    int rc;

    if (resolve(ctx, dst, dst_cuda, nbytes, &dm, &daddr))
        return -1;
    if (resolve(ctx, src, src_cuda, nbytes, &sm, &saddr))
        return -1;

    if (dm && !sm) {
        rc = gdr_copy_to_mapping(dm->mh, (void *)(uintptr_t)daddr, (const void *)(uintptr_t)saddr,
                                 nbytes);
        if (rc) {
            fprintf(stderr, "gdr_copy_to_mapping failed rc=%d\n", rc);
            return -1;
        }
    } else if (sm && !dm) {
        rc = gdr_copy_from_mapping(sm->mh, (void *)(uintptr_t)daddr, (const void *)(uintptr_t)saddr,
                                   nbytes);
        if (rc) {
            fprintf(stderr, "gdr_copy_from_mapping failed rc=%d\n", rc);
            return -1;
        }
    } else {
        fprintf(stderr, "dsa_gd_copy: exactly one side must be CUDA\n");
        return -1;
    }
    return 0;
}

int dsa_gd_dsa_copy(gd_ctx *ctx, uint64_t dst, int dst_cuda, uint64_t src, int src_cuda,
                    size_t nbytes) {
    struct mapping *dm, *sm;
    uint64_t daddr, saddr;

    if (resolve(ctx, dst, dst_cuda, nbytes, &dm, &daddr))
        return -1;
    if (resolve(ctx, src, src_cuda, nbytes, &sm, &saddr))
        return -1;
    return dsa_memcpy((void *)(uintptr_t)daddr, (const void *)(uintptr_t)saddr, nbytes);
}

int dsa_gd_dsa_copy_batch(gd_ctx *ctx, const uint64_t *dst, const int *dst_cuda,
                          const uint64_t *src, const int *src_cuda, const size_t *nbytes,
                          size_t count) {
    void **dest_c = NULL, **src_c = NULL;
    size_t *n_c = NULL;
    int ret = -1;

    if (count == 0)
        return 0;

    dest_c = (void **)calloc(count, sizeof(*dest_c));
    src_c = (void **)calloc(count, sizeof(*src_c));
    n_c = (size_t *)calloc(count, sizeof(*n_c));
    if (!dest_c || !src_c || !n_c)
        goto out;

    for (size_t i = 0; i < count; i++) {
        struct mapping *dm, *sm;
        uint64_t daddr, saddr;

        if (resolve(ctx, dst[i], dst_cuda[i], nbytes[i], &dm, &daddr))
            goto out;
        if (resolve(ctx, src[i], src_cuda[i], nbytes[i], &sm, &saddr))
            goto out;
        dest_c[i] = (void *)(uintptr_t)daddr;
        src_c[i] = (void *)(uintptr_t)saddr;
        n_c[i] = nbytes[i];
    }

    ret = dsa_memcpy_batch((void *const *)dest_c, (const void *const *)src_c, n_c, count);
out:
    free(dest_c);
    free(src_c);
    free(n_c);
    return ret;
}

static gd_ctx *g_default;
static pthread_mutex_t g_default_mutex = PTHREAD_MUTEX_INITIALIZER;

static void default_close(void) {
    pthread_mutex_lock(&g_default_mutex);
    if (g_default) {
        dsa_gd_close(g_default);
        g_default = NULL;
    }
    pthread_mutex_unlock(&g_default_mutex);
}

static gd_ctx *default_ctx_locked(void) {
    if (!g_default) {
        g_default = dsa_gd_open();
        if (g_default)
            atexit(default_close);
    }
    return g_default;
}

int dsa_gd_default_register(uint64_t dptr, size_t nbytes) {
    gd_ctx *ctx;
    int ret;

    pthread_mutex_lock(&g_default_mutex);
    ctx = default_ctx_locked();
    ret = ctx ? dsa_gd_register(ctx, dptr, nbytes) : -1;
    pthread_mutex_unlock(&g_default_mutex);
    return ret;
}

int dsa_gd_default_unregister(uint64_t dptr, size_t nbytes) {
    int ret;

    pthread_mutex_lock(&g_default_mutex);
    ret = g_default ? dsa_gd_unregister(g_default, dptr, nbytes) : -1;
    pthread_mutex_unlock(&g_default_mutex);
    return ret;
}

int dsa_gd_default_copy(uint64_t dst, int dst_cuda, uint64_t src, int src_cuda, size_t nbytes) {
    gd_ctx *ctx;
    int ret;

    pthread_mutex_lock(&g_default_mutex);
    ctx = default_ctx_locked();
    ret = ctx ? dsa_gd_copy(ctx, dst, dst_cuda, src, src_cuda, nbytes) : -1;
    pthread_mutex_unlock(&g_default_mutex);
    return ret;
}

int dsa_gd_default_dsa_copy(uint64_t dst, int dst_cuda, uint64_t src, int src_cuda, size_t nbytes) {
    gd_ctx *ctx;
    int ret;

    pthread_mutex_lock(&g_default_mutex);
    ctx = default_ctx_locked();
    ret = ctx ? dsa_gd_dsa_copy(ctx, dst, dst_cuda, src, src_cuda, nbytes) : -1;
    pthread_mutex_unlock(&g_default_mutex);
    return ret;
}

int dsa_gd_default_dsa_copy_batch(const uint64_t *dst, const int *dst_cuda, const uint64_t *src,
                                  const int *src_cuda, const size_t *nbytes, size_t count) {
    gd_ctx *ctx;
    int ret;

    pthread_mutex_lock(&g_default_mutex);
    ctx = default_ctx_locked();
    ret = ctx ? dsa_gd_dsa_copy_batch(ctx, dst, dst_cuda, src, src_cuda, nbytes, count) : -1;
    pthread_mutex_unlock(&g_default_mutex);
    return ret;
}

int dsa_gd_default_gpu_bar_addr(uint64_t gpu_ptr, size_t nbytes, void **cpu_out) {
    gd_ctx *ctx;
    struct mapping *m;
    uint64_t addr;
    int ret = -1;

    pthread_mutex_lock(&g_default_mutex);
    ctx = default_ctx_locked();
    if (ctx && cpu_out && !resolve(ctx, gpu_ptr, 1, nbytes, &m, &addr)) {
        *cpu_out = (void *)(uintptr_t)addr;
        ret = 0;
    }
    pthread_mutex_unlock(&g_default_mutex);
    return ret;
}

void dsa_gd_default_reset(void) {
    pthread_mutex_lock(&g_default_mutex);
    if (g_default)
        dsa_gd_reset(g_default);
    pthread_mutex_unlock(&g_default_mutex);
}
