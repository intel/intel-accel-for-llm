// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

// Parallel helpers that run on OpenMP or, with IAXL_USE_OMP=0, on iaxl's own thread pool, which
// stays off the OpenMP runtime and settings shared with vLLM and torch. Both use
// IAXL_OMP_THREAD_NUM threads.

#pragma once

#include <omp.h>

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>

#include "env.h"
#include "iaxl_common.h"

namespace parallel {

// Persistent threads; the caller of run() acts as tid 0.
class ThreadPool {
  public:
    explicit ThreadPool(int size);

    // Calls fn(tid) for every tid in [0, n) and returns once all of them finish.
    void run(int n, const std::function<void(int)> &fn);

    // Shared pool of IAXL_OMP_THREAD_NUM threads, created on first use.
    static ThreadPool &instance();

  private:
    void loop(int tid);

    const int size_;
    std::mutex run_mutex_;
    std::mutex mutex_;
    std::condition_variable wake_, done_;
    const std::function<void(int)> *fn_ = nullptr;
    int team_ = 0;
    int pending_ = 0;
    uint64_t generation_ = 0;
};

// Calls fn(tid) for every tid in [0, n) concurrently; a one-thread run stays on the caller.
template <class Fn> void run_threads(int n, Fn &&fn) {
    if (n <= 0)
        return;
    if (n == 1) {
        fn(0);
    } else if (!envs.IAXL_USE_OMP) {
        ThreadPool::instance().run(n, fn);
    } else {
#pragma omp parallel num_threads(n)
        {
            IAXL_CHECK(omp_get_num_threads() == n,
                       "run_threads: OpenMP team is smaller than requested, check OMP_THREAD_LIMIT");
            fn(omp_get_thread_num());
        }
    }
}

// Calls body(i) for every i in [0, n), with up to IAXL_OMP_THREAD_NUM threads claiming items.
template <class Body> void parallel_for(size_t n, Body &&body) {
    std::atomic<size_t> next{0};
    const int threads = static_cast<int>(std::min<size_t>(n, envs.IAXL_OMP_THREAD_NUM));
    run_threads(threads, [&](int) {
        for (size_t i = next++; i < n; i = next++)
            body(i);
    });
}

} // namespace parallel
