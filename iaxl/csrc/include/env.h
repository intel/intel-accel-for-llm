// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdlib.h>
#include <strings.h>
#ifndef __cplusplus
#include <stdbool.h>
#endif

static inline int env_int(const char *name, int fallback) {
    const char *v = getenv(name);
    if (!v || !*v)
        return fallback;
    int x = atoi(v);
    return x > 0 ? x : fallback;
}

static inline int env_nonnegative_int(const char *name, int fallback) {
    const char *v = getenv(name);
    if (!v || !*v)
        return fallback;
    int x = atoi(v);
    return x >= 0 ? x : fallback;
}

static inline const char *env_str(const char *name, const char *fallback) {
    const char *v = getenv(name);
    return (v && *v) ? v : fallback;
}

static inline int env_bool(const char *name, int fallback) {
    const char *v = getenv(name);
    if (!v || !*v)
        return fallback;
    if (!strcasecmp(v, "true") || !strcasecmp(v, "yes") || !strcasecmp(v, "on"))
        return 1;
    if (!strcasecmp(v, "false") || !strcasecmp(v, "no") || !strcasecmp(v, "off"))
        return 0;
    return atoi(v) != 0;
}

struct Envs {

    int IAXL_ZIP_SRC_CAP;
    int IAXL_ZIP_DST_CAP;

    bool IAXL_QAT_ZIP_ENABLE;
    bool IAXL_IAA_ZIP_ENABLE;
    bool IAXL_CPU_ZIP_ENABLE;
    int IAXL_QAT_INSTANCE_NUM;
    const char *(*IAXL_QAT_DEVICES)(void);
    int IAXL_QAT_ZIP_INSTANCES_PER_DEVICE;
    int IAXL_QAT_ZIP_QUEUE_DEPTH;
    int IAXL_IAA_INSTANCE_NUM;
    const char *(*IAXL_IAA_DEVICES)(void);
    int IAXL_IAA_ZIP_INSTANCES_PER_DEVICE;
    int IAXL_IAA_ZIP_QUEUE_DEPTH;
    int IAXL_CPU_ZIP_THREADS;
    // Threads that drive QAT/IAA instances; each poller multiplexes instances/pollers devices.
    int IAXL_QAT_POLL_THREADS;
    int IAXL_IAA_POLL_THREADS;
    int IAXL_OMP_THREAD_NUM;

    bool IAXL_KV_COMPRESSION;
    int IAXL_KV_LOSSY_TRUNC;
    bool IAXL_KV_DATA_SHUFFLE;
    int IAXL_CACHE_CACHEGROUP_SIZE;
    int IAXL_CACHE_CACHEGROUP_NUM;

    bool IAXL_DSA_GD_ENABLE;
    bool IAXL_DSA_GD_RESET_ON_DESTROY;
    // Host-to-host DSA memcpy for the pure-copy segments of the CPU inference path.
    bool IAXL_DSA_MEMCPY_ENABLE;
    // Batches smaller than this stay on memcpy: DSA submit+completion latency exceeds the copy.
    size_t IAXL_DSA_MEMCPY_MIN_BYTES;
    const char *(*IAXL_DSA_WQS)(void);

    // CPU list ("32-35,40") for every IAXL native thread; empty leaves the inherited mask.
    const char *IAXL_CPU_AFFINITY;

    bool IAXL_DEBUG_LOG;
    const char *IAXL_PROFILE_MODE;
};

#ifdef __cplusplus
extern "C" {
#endif

extern struct Envs envs;

void envs_init(void);

// Pins the calling thread to IAXL_CPU_AFFINITY. Returns the CPU count of that set, 0 when the
// variable is unset, or -1 when the list could not be applied.
int iaxl_apply_thread_affinity(void);
int iaxl_affinity_cpu_count(void);

#ifdef __cplusplus
}
#endif
