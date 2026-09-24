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

__attribute__((constructor(101))) void envs_init(void) {

    envs.IAXL_ZIP_SRC_CAP = env_int("IAXL_ZIP_SRC_CAP", 256 * 1024);
    envs.IAXL_ZIP_DST_CAP = env_int("IAXL_ZIP_DST_CAP", 256 * 1024);
    /* Split QAT jobs larger than this many bytes (0 = disabled). */
    envs.IAXL_ZIP_SEGMENT = env_int("IAXL_ZIP_SEGMENT", 0);

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
    envs.IAXL_OMP_THREAD_NUM =
        env_int("OMP_NUM_THREADS", envs.IAXL_QAT_INSTANCE_NUM + envs.IAXL_IAA_INSTANCE_NUM +
                                       envs.IAXL_CPU_ZIP_THREADS);
    if (envs.IAXL_OMP_THREAD_NUM < 1)
        envs.IAXL_OMP_THREAD_NUM = 1;

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

    envs.IAXL_DEBUG_LOG = env_bool("IAXL_DEBUG_LOG", 0);
    envs.IAXL_PROFILE_MODE = env_str("IAXL_PROFILE_MODE", "disabled");

    static int printed = 0;
    if (!printed) {
        printed = 1;

        int cpus = available_cpu_count();

#ifdef _OPENMP

        int omp_threads = omp_get_max_threads();
        if (cpus < envs.IAXL_OMP_THREAD_NUM) {
            fprintf(stderr,
                    "[iaxl] WARNING: only %d CPU(s) available to this process, "
                "but OMP_NUM_THREADS=%d (omp_max_threads=%d); compression "
                "workers will be oversubscribed and throughput may degrade.\n",
                cpus, envs.IAXL_OMP_THREAD_NUM, omp_threads);
        }
#endif

         printf("[iaxl] config: qat_zip=%s iaa_zip=%s cpu_zip=%s qat_instances=%d "
             "iaa_instances=%d cpu_zip_threads=%d "
             "omp_threads=%d cpus=%d "
               "compression=%s data_shuffle=%s lossy_trunc=%d dsa_gd=%s "
               "dsa_gd_reset=%s "
               "profile=%s\n",
               envs.IAXL_QAT_ZIP_ENABLE ? "ON" : "OFF",
               envs.IAXL_IAA_ZIP_ENABLE ? "ON" : "OFF",
               envs.IAXL_CPU_ZIP_ENABLE ? "ON" : "OFF", envs.IAXL_QAT_INSTANCE_NUM,
               envs.IAXL_IAA_INSTANCE_NUM,
               envs.IAXL_CPU_ZIP_THREADS,
               envs.IAXL_OMP_THREAD_NUM, cpus, envs.IAXL_KV_COMPRESSION ? "ON" : "OFF",
               envs.IAXL_KV_DATA_SHUFFLE ? "ON" : "OFF", envs.IAXL_KV_LOSSY_TRUNC,
               envs.IAXL_DSA_GD_ENABLE ? "ON" : "OFF",
               envs.IAXL_DSA_GD_RESET_ON_DESTROY ? "ON" : "OFF", envs.IAXL_PROFILE_MODE);
    }
}
