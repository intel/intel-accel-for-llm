// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "env.h"

static const char *qat_devices_get(void) {
    static const char *cached = NULL;
    if (!cached)
        cached = env_str("IAXL_QAT_DEVICES", "0");
    return cached;
}

static const char *iaa_devices_get(void) {
    static const char *cached = NULL;
    if (!cached)
        cached = env_str("IAXL_IAA_DEVICES", "auto");
    return cached;
}

static const char *dsa_wqs_get(void) {
    static const char *cached = NULL;
    if (!cached)
        cached = env_str("IAXL_DSA_WQS", "wq0.0");
    return cached;
}

struct Envs envs = {
    .IAXL_QAT_DEVICES = qat_devices_get,
    .IAXL_IAA_DEVICES = iaa_devices_get,
    .IAXL_DSA_WQS = dsa_wqs_get,
};

static int available_cpu_count(void) {
    cpu_set_t set;
    CPU_ZERO(&set);
    if (sched_getaffinity(0, sizeof(set), &set) == 0)
        return CPU_COUNT(&set);
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    return n > 0 ? (int)n : 1;
}

// Accepts "0-3,8,10-11"; returns the CPU count or -1 on a malformed list.
static int parse_cpu_list(const char *list, cpu_set_t *set) {
    CPU_ZERO(set);
    if (!list || !*list)
        return 0;
    const char *p = list;
    while (*p) {
        char *end;
        long lo = strtol(p, &end, 10);
        if (end == p || lo < 0 || lo >= CPU_SETSIZE)
            return -1;
        long hi = lo;
        if (*end == '-') {
            p = end + 1;
            hi = strtol(p, &end, 10);
            if (end == p || hi < lo || hi >= CPU_SETSIZE)
                return -1;
        }
        for (long c = lo; c <= hi; c++)
            CPU_SET((int)c, set);
        p = end;
        if (*p == ',')
            p++;
        else if (*p)
            return -1;
    }
    return CPU_COUNT(set);
}

int iaxl_affinity_cpu_count(void) {
    cpu_set_t set;
    return parse_cpu_list(envs.IAXL_CPU_AFFINITY, &set);
}

int iaxl_apply_thread_affinity(void) {
    cpu_set_t set;
    const int count = parse_cpu_list(envs.IAXL_CPU_AFFINITY, &set);
    if (count <= 0)
        return count;
    if (sched_setaffinity(0, sizeof(set), &set) != 0) {
        static int warned = 0;
        if (!warned) {
            warned = 1;
            fprintf(stderr, "[iaxl] WARNING: sched_setaffinity(IAXL_CPU_AFFINITY=%s) failed\n",
                    envs.IAXL_CPU_AFFINITY);
        }
        return -1;
    }
    return count;
}

// Pollers default to one per instance (the historical layout) and never exceed the instances.
static int poll_threads(const char *name, int instances) {
    if (instances <= 0)
        return 0;
    int pollers = env_int(name, instances);
    return pollers > instances ? instances : pollers;
}

__attribute__((constructor(101))) void envs_init(void) {

    envs.IAXL_ZIP_SRC_CAP = env_int("IAXL_ZIP_SRC_CAP", 256 * 1024);
    envs.IAXL_ZIP_DST_CAP = env_int("IAXL_ZIP_DST_CAP", 256 * 1024);

    envs.IAXL_QAT_ZIP_ENABLE = env_bool("IAXL_QAT_ZIP_ENABLE", 1);
    envs.IAXL_IAA_ZIP_ENABLE = env_bool("IAXL_IAA_ZIP_ENABLE", 0);
    envs.IAXL_CPU_ZIP_ENABLE = env_bool("IAXL_CPU_ZIP_ENABLE", 1);
    envs.IAXL_QAT_INSTANCE_NUM = env_nonnegative_int("IAXL_QAT_INSTANCE_NUM", 4);
    envs.IAXL_IAA_INSTANCE_NUM = env_nonnegative_int("IAXL_IAA_INSTANCE_NUM", 4);

    envs.IAXL_QAT_ZIP_INSTANCES_PER_DEVICE = env_int("IAXL_QAT_ZIP_INSTANCES_PER_DEVICE", 4);
    envs.IAXL_QAT_ZIP_QUEUE_DEPTH = env_int("IAXL_QAT_ZIP_QUEUE_DEPTH", 4);
    envs.IAXL_IAA_ZIP_INSTANCES_PER_DEVICE = env_int("IAXL_IAA_ZIP_INSTANCES_PER_DEVICE", 4);
    envs.IAXL_IAA_ZIP_QUEUE_DEPTH = env_int("IAXL_IAA_ZIP_QUEUE_DEPTH", 4);
    envs.IAXL_CPU_ZIP_THREADS = env_nonnegative_int("IAXL_CPU_ZIP_THREADS", 4);
    if (!envs.IAXL_QAT_ZIP_ENABLE)
        envs.IAXL_QAT_INSTANCE_NUM = 0;
    if (!envs.IAXL_IAA_ZIP_ENABLE)
        envs.IAXL_IAA_INSTANCE_NUM = 0;
    if (!envs.IAXL_CPU_ZIP_ENABLE)
        envs.IAXL_CPU_ZIP_THREADS = 0;
    envs.IAXL_QAT_POLL_THREADS = poll_threads("IAXL_QAT_POLL_THREADS", envs.IAXL_QAT_INSTANCE_NUM);
    envs.IAXL_IAA_POLL_THREADS = poll_threads("IAXL_IAA_POLL_THREADS", envs.IAXL_IAA_INSTANCE_NUM);
    const int zip_workers = envs.IAXL_QAT_POLL_THREADS + envs.IAXL_IAA_POLL_THREADS +
                            envs.IAXL_CPU_ZIP_THREADS;
#ifdef CPU_SUPPORT
    envs.IAXL_OMP_THREAD_NUM = zip_workers > 0 ? zip_workers : 1;
#else
    envs.IAXL_OMP_THREAD_NUM =
        env_int("OMP_NUM_THREADS", zip_workers);
    if (envs.IAXL_OMP_THREAD_NUM < 1)
        envs.IAXL_OMP_THREAD_NUM = 1;
#endif

    envs.IAXL_KV_COMPRESSION = env_bool("IAXL_KV_COMPRESSION", 1);
    if (envs.IAXL_KV_COMPRESSION && !envs.IAXL_QAT_ZIP_ENABLE && !envs.IAXL_IAA_ZIP_ENABLE &&
        !envs.IAXL_CPU_ZIP_ENABLE) {
        fprintf(stderr, "[iaxl] ERROR: IAXL_KV_COMPRESSION=1 requires at least one zip backend; "
                        "enable IAXL_QAT_ZIP_ENABLE/IAXL_IAA_ZIP_ENABLE/IAXL_CPU_ZIP_ENABLE "
                        "or set IAXL_KV_COMPRESSION=0\n");
        abort();
    }
    envs.IAXL_KV_LOSSY_TRUNC = env_int("IAXL_KV_LOSSY_TRUNC", 0);
    envs.IAXL_KV_DATA_SHUFFLE = env_bool("IAXL_KV_DATA_SHUFFLE", 0);
    envs.IAXL_CACHE_CACHEGROUP_SIZE = env_int("IAXL_CACHE_CACHEGROUP_SIZE", 100);
    envs.IAXL_CACHE_CACHEGROUP_NUM = env_int("IAXL_CACHE_CACHEGROUP_NUM", 100000);

    envs.IAXL_DSA_GD_ENABLE = env_bool("IAXL_DSA_GD_ENABLE", 0);
    envs.IAXL_DSA_GD_RESET_ON_DESTROY = env_bool("IAXL_DSA_GD_RESET_ON_DESTROY", 0);
    envs.IAXL_DSA_MEMCPY_ENABLE = env_bool("IAXL_DSA_MEMCPY_ENABLE", 0);
    envs.IAXL_DSA_MEMCPY_MIN_BYTES =
        (size_t)env_nonnegative_int("IAXL_DSA_MEMCPY_MIN_BYTES", 1024 * 1024);
    envs.IAXL_CPU_AFFINITY = env_str("IAXL_CPU_AFFINITY", "");

    envs.IAXL_DEBUG_LOG = env_bool("IAXL_DEBUG_LOG", 0);
    envs.IAXL_PROFILE_MODE = env_str("IAXL_PROFILE_MODE", "disabled");

    static int printed = 0;
    if (!printed) {
        printed = 1;

        int cpus = available_cpu_count();
        const int pinned = iaxl_affinity_cpu_count();
        if (pinned < 0)
            fprintf(stderr, "[iaxl] WARNING: ignoring malformed IAXL_CPU_AFFINITY=%s\n",
                    envs.IAXL_CPU_AFFINITY);
        else if (pinned > 0)
            cpus = pinned;

#ifdef _OPENMP

        int omp_threads = omp_get_max_threads();
        if (cpus < envs.IAXL_OMP_THREAD_NUM) {
            fprintf(stderr,
                    "[iaxl] WARNING: only %d CPU(s) available to IAXL threads, "
                "but compression uses %d OpenMP workers (omp_max_threads=%d); compression "
                "workers will be oversubscribed and throughput may degrade.\n",
                cpus, envs.IAXL_OMP_THREAD_NUM, omp_threads);
        }
#endif

         printf("[iaxl] config: qat_zip=%s iaa_zip=%s cpu_zip=%s qat_instances=%d "
             "qat_pollers=%d iaa_instances=%d iaa_pollers=%d cpu_zip_threads=%d "
             "omp_threads=%d cpus=%d affinity=%s "
               "compression=%s data_shuffle=%s lossy_trunc=%d dsa_gd=%s "
               "dsa_gd_reset=%s dsa_memcpy=%s dsa_memcpy_min_bytes=%zu "
               "profile=%s\n",
               envs.IAXL_QAT_ZIP_ENABLE ? "ON" : "OFF",
               envs.IAXL_IAA_ZIP_ENABLE ? "ON" : "OFF",
               envs.IAXL_CPU_ZIP_ENABLE ? "ON" : "OFF", envs.IAXL_QAT_INSTANCE_NUM,
               envs.IAXL_QAT_POLL_THREADS, envs.IAXL_IAA_INSTANCE_NUM, envs.IAXL_IAA_POLL_THREADS,
               envs.IAXL_CPU_ZIP_THREADS,
               envs.IAXL_OMP_THREAD_NUM, cpus, *envs.IAXL_CPU_AFFINITY ? envs.IAXL_CPU_AFFINITY : "inherit",
               envs.IAXL_KV_COMPRESSION ? "ON" : "OFF",
               envs.IAXL_KV_DATA_SHUFFLE ? "ON" : "OFF", envs.IAXL_KV_LOSSY_TRUNC,
               envs.IAXL_DSA_GD_ENABLE ? "ON" : "OFF",
               envs.IAXL_DSA_GD_RESET_ON_DESTROY ? "ON" : "OFF",
               envs.IAXL_DSA_MEMCPY_ENABLE ? "ON" : "OFF", envs.IAXL_DSA_MEMCPY_MIN_BYTES,
               envs.IAXL_PROFILE_MODE);
    }
}
