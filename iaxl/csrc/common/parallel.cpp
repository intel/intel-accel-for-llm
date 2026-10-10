// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "parallel.h"

#include <pthread.h>

#include <string>
#include <thread>
#include <utility>

namespace parallel {

// Pool whose work the current thread is running; a nested run on it would deadlock.
static thread_local const ThreadPool *t_pool = nullptr;

ThreadPool::ThreadPool(int size) : size_(size) {
    for (int tid = 1; tid < size; tid++)
        std::thread([this, tid] { loop(tid); }).detach();
}

void ThreadPool::run(int n, const std::function<void(int)> &fn) {
    IAXL_CHECK(n >= 1 && n <= size_, "ThreadPool: team size out of range");
    IAXL_CHECK(t_pool != this, "ThreadPool: nested run on the same pool");
    std::lock_guard<std::mutex> serial(run_mutex_);
    {
        std::lock_guard<std::mutex> lock(mutex_);
        fn_ = &fn;
        team_ = n;
        pending_ = n - 1;
        generation_++;
    }
    wake_.notify_all();
    const ThreadPool *outer = std::exchange(t_pool, this);
    fn(0);
    t_pool = outer;
    std::unique_lock<std::mutex> lock(mutex_);
    done_.wait(lock, [this] { return pending_ == 0; });
}

void ThreadPool::loop(int tid) {
    pthread_setname_np(pthread_self(), ("POOL-" + std::to_string(tid)).c_str());
    t_pool = this;
    uint64_t seen = 0;
    std::unique_lock<std::mutex> lock(mutex_);
    for (;;) {
        wake_.wait(lock, [&] { return generation_ != seen; });
        seen = generation_;
        if (tid >= team_)
            continue;
        const std::function<void(int)> *fn = fn_;
        lock.unlock();
        (*fn)(tid);
        lock.lock();
        if (--pending_ == 0)
            done_.notify_one();
    }
}

ThreadPool &ThreadPool::instance() {
    // Never destroyed: work may still be running on it while the process exits.
    static ThreadPool *pool = new ThreadPool(envs.IAXL_OMP_THREAD_NUM);
    return *pool;
}

} // namespace parallel
