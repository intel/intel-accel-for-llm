// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <array>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <future>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>
#include <pthread.h>
#include <sched.h>

#include "env.h"
#include "iaxl_common.h"

class TaskQueue {
  public:
    // Levels, most urgent first; the worker takes the front task of the most urgent non-empty one.
    enum {
        PRIORITY_P0 = 0,
        PRIORITY_P1,
        PRIORITY_P2,
        PRIORITY_P3,
        NUM_PRIORITIES,
        PRIORITY_HIGH = PRIORITY_P0,
        PRIORITY_LOW = PRIORITY_P3
    };

    explicit TaskQueue(const char *name) : name_(name) {}

    ~TaskQueue() { shutdown(); }

    void init() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (state_ == State::RUNNING)
            return;
        if (state_ != State::CREATED) {
            throw std::runtime_error("TaskQueue cannot be restarted after shutdown");
        }

        worker_ = std::thread([this]() {
            pthread_setname_np(pthread_self(), name_);
            bind_cpus();
            worker_loop();
        });
        state_ = State::RUNNING;
    }

    // `bundle_id` (0 = none) is only read by set_bundles_priority(), `sort_key` (-1 = none) only by
    // sort_priority_level(); a level is otherwise FIFO.
    template <typename F>
    std::future<void> submit(F &&func, int priority = PRIORITY_HIGH, uint64_t bundle_id = 0,
                             int64_t sort_key = -1) {
        check_priority(priority);
        auto task = std::make_shared<std::packaged_task<void()>>(std::forward<F>(func));
        auto future = task->get_future();

        Item item;
        item.fn = [task]() { (*task)(); };
        item.bundle_id = bundle_id;
        item.sort_key = sort_key;

        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (state_ != State::RUNNING) {
                throw std::runtime_error("TaskQueue submit rejected: queue is not running");
            }
            tasks_[priority].push_back(std::move(item));
        }
        cv_.notify_one();

        return future;
    }

    // Moves waiting tasks of `bundle_ids` from less urgent levels to the tail of `priority`, in order.
    size_t set_bundles_priority(const std::vector<uint64_t> &bundle_ids, int priority) {
        check_priority(priority);
        size_t moved = 0;
        std::lock_guard<std::mutex> lock(mutex_);
        for (int level = priority + 1; level < NUM_PRIORITIES; ++level) {
            std::deque<Item> kept;
            for (Item &item : tasks_[level]) {
                if (item.bundle_id != 0 &&
                    std::find(bundle_ids.begin(), bundle_ids.end(), item.bundle_id) !=
                        bundle_ids.end()) {
                    tasks_[priority].push_back(std::move(item));
                    ++moved;
                } else {
                    kept.push_back(std::move(item));
                }
            }
            tasks_[level].swap(kept);
        }
        return moved;
    }

    // Stable-sorts the waiting tasks of level `priority` by sort_key; other levels are untouched.
    void sort_priority_level(int priority) {
        check_priority(priority);
        std::lock_guard<std::mutex> lock(mutex_);
        std::stable_sort(tasks_[priority].begin(), tasks_[priority].end(),
                         [](const Item &a, const Item &b) { return a.sort_key < b.sort_key; });
    }

    void shutdown() {
        {
            std::unique_lock<std::mutex> lock(mutex_);
            if (state_ == State::CREATED) {
                state_ = State::STOPPED;
                return;
            }
            if (state_ == State::STOPPED)
                return;
            if (state_ == State::STOPPING) {
                state_cv_.wait(lock, [this]() { return state_ == State::STOPPED; });
                return;
            }
            state_ = State::STOPPING;
        }
        cv_.notify_one();

        if (worker_.joinable()) {
            worker_.join();
        }

        {
            std::lock_guard<std::mutex> lock(mutex_);
            state_ = State::STOPPED;
        }
        state_cv_.notify_all();
    }

    // Pins to IAXL_CPU_OMP_THREADS_BIND ("0-7,16"); threads spawned afterwards inherit it.
    static void bind_cpus() {
        const char *spec = env_str("IAXL_CPU_OMP_THREADS_BIND", nullptr);
        if (!spec)
            return;
        cpu_set_t cpus;
        CPU_ZERO(&cpus);
        for (const char *p = spec;; ++p) {
            char *end;
            long first = strtol(p, &end, 10);
            long last = first;
            if (end != p && *end == '-')
                last = strtol(p = end + 1, &end, 10);
            IAXL_CHECK(end != p && 0 <= first && first <= last && last < CPU_SETSIZE &&
                           (*end == ',' || *end == '\0'),
                       "invalid IAXL_CPU_OMP_THREADS_BIND");
            for (long cpu = first; cpu <= last; ++cpu)
                CPU_SET(cpu, &cpus);
            if (*end == '\0')
                break;
            p = end;
        }
        IAXL_CHECK(pthread_setaffinity_np(pthread_self(), sizeof(cpus), &cpus) == 0,
                   "cannot bind to IAXL_CPU_OMP_THREADS_BIND");
    }

  private:
    enum class State { CREATED, RUNNING, STOPPING, STOPPED };

    struct Item {
        std::function<void()> fn;
        uint64_t bundle_id = 0;
        int64_t sort_key = -1;
    };

    static void check_priority(int priority) {
        if (priority < 0 || priority >= NUM_PRIORITIES) {
            throw std::invalid_argument("TaskQueue priority must be in [PRIORITY_P0, PRIORITY_P3]");
        }
    }

    bool all_empty() const {
        for (const auto &level : tasks_) {
            if (!level.empty())
                return false;
        }
        return true;
    }

    void worker_loop() {
        while (true) {
            Item item;

            {
                std::unique_lock<std::mutex> lock(mutex_);
                cv_.wait(lock, [this]() { return state_ != State::RUNNING || !all_empty(); });

                if (state_ == State::STOPPING && all_empty()) {
                    return;
                }

                for (auto &level : tasks_) {
                    if (!level.empty()) {
                        item = std::move(level.front());
                        level.pop_front();
                        break;
                    }
                }
            }

            item.fn();
        }
    }

    const char *name_;
    std::thread worker_;
    std::mutex mutex_;
    std::condition_variable cv_;
    std::condition_variable state_cv_;
    std::array<std::deque<Item>, NUM_PRIORITIES> tasks_;
    State state_ = State::CREATED;
};
