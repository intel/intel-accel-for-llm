// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <optional>
#include <string>
#include <sys/prctl.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <pthread.h>
#include <iostream>

#include "env.h"

#if defined(ENABLE_NVTX) && ENABLE_NVTX
#include <nvtx3/nvToolsExt.h>
#endif

#ifdef ENABLE_TORCH_PROFILER
#include <ATen/record_function.h>
#endif

namespace profiler {

enum class ProfileMode { Disabled = 0, Nvtx = 1, Full = 2 };

namespace detail {

#define KVCLIP_UNLIKELY(x) __builtin_expect(!!(x), 0)
#define KVCLIP_LIKELY(x) __builtin_expect(!!(x), 1)

inline ProfileMode resolve_profile_mode() {
    const char *env = envs.IAXL_PROFILE_MODE;
    if (env == nullptr || env[0] == '\0') {
        return ProfileMode::Disabled;
    }

    if (std::strcmp(env, "nvtx") == 0) {
#if defined(ENABLE_NVTX) && ENABLE_NVTX
        return ProfileMode::Nvtx;
#else
        return ProfileMode::Disabled;
#endif
    }

    if (std::strcmp(env, "full") == 0) {
#ifdef ENABLE_TORCH_PROFILER
        return ProfileMode::Full;
#else
        return ProfileMode::Disabled;
#endif
    }

    return ProfileMode::Disabled;
}

inline ProfileMode &get_mode() {
    static ProfileMode mode = resolve_profile_mode();
    return mode;
}

inline bool is_profiling_enabled() {
    static bool enabled = (get_mode() != ProfileMode::Disabled);
    return enabled;
}

#if defined(ENABLE_NVTX) && ENABLE_NVTX
inline void nvtx_range_push(const char *name, uint32_t color = 0xFF00FF00) {
    nvtxEventAttributes_t attr = {0};
    attr.version = NVTX_VERSION;
    attr.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    attr.colorType = NVTX_COLOR_ARGB;
    attr.color = color;
    attr.messageType = NVTX_MESSAGE_TYPE_ASCII;
    attr.message.ascii = name;
    nvtxRangePushEx(&attr);
}

inline void nvtx_range_pop() { nvtxRangePop(); }

inline void nvtx_name_thread(const char *name) { nvtxNameOsThreadA(pthread_self(), name); }

inline void nvtx_mark(const char *name) { nvtxMarkA(name); }
#else
inline void nvtx_range_push(const char *, uint32_t = 0) {}
inline void nvtx_range_pop() {}
inline void nvtx_name_thread(const char *) {}
inline void nvtx_mark(const char *) {}
#endif

#ifdef ENABLE_TORCH_PROFILER

#include <memory>
#include <stack>

inline std::stack<std::unique_ptr<at::RecordFunction>> &get_record_function_stack() {
    thread_local std::stack<std::unique_ptr<at::RecordFunction>> stack;
    return stack;
}

inline void torch_profiler_push(const char *name) {
    auto rf = std::make_unique<at::RecordFunction>(at::RecordScope::USER_SCOPE);
    rf->before(name);
    get_record_function_stack().push(std::move(rf));
}

inline void torch_profiler_pop() {
    auto &stack = get_record_function_stack();
    if (!stack.empty()) {
        stack.pop();
    }
}
#else
inline void torch_profiler_push(const char *) {}
inline void torch_profiler_pop() {}
#endif

} // namespace detail

class ProfileScope {
  public:
    explicit ProfileScope(const char *name, uint32_t color = 0xFF00FF00) : active_(false) {
        start(name, color);
    }

    explicit ProfileScope(const std::string &name, uint32_t color = 0xFF00FF00)
        : ProfileScope(name.c_str(), color) {}

    ~ProfileScope() {
        if (KVCLIP_UNLIKELY(active_)) {
            stop();
        }
    }

    ProfileScope(const ProfileScope &) = delete;
    ProfileScope &operator=(const ProfileScope &) = delete;
    ProfileScope(ProfileScope &&) = delete;
    ProfileScope &operator=(ProfileScope &&) = delete;

  private:
    void start(const char *name, uint32_t color) {
        auto mode = detail::get_mode();
        if (mode == ProfileMode::Nvtx) {
            detail::nvtx_range_push(name, color);
            active_ = true;
        } else if (mode == ProfileMode::Full) {
            detail::torch_profiler_push(name);
            active_ = true;
        }
    }

    void stop() {
        auto mode = detail::get_mode();
        if (mode == ProfileMode::Nvtx) {
            detail::nvtx_range_pop();
        } else if (mode == ProfileMode::Full) {
            detail::torch_profiler_pop();
        }
        active_ = false;
    }

    bool active_;
};

inline void profile_start(const char *name, uint32_t color = 0xFF00FF00) {
    if (KVCLIP_UNLIKELY(detail::is_profiling_enabled())) {
        auto mode = detail::get_mode();
        if (mode == ProfileMode::Nvtx) {
            detail::nvtx_range_push(name, color);
        } else if (mode == ProfileMode::Full) {
            detail::torch_profiler_push(name);
        }
    }
}

inline void profile_end() {
    if (KVCLIP_UNLIKELY(detail::is_profiling_enabled())) {
        auto mode = detail::get_mode();
        if (mode == ProfileMode::Nvtx) {
            detail::nvtx_range_pop();
        } else if (mode == ProfileMode::Full) {
            detail::torch_profiler_pop();
        }
    }
}

inline void profile_mark(const char *name) {
    if (KVCLIP_UNLIKELY(detail::is_profiling_enabled())) {
        auto mode = detail::get_mode();
        if (mode == ProfileMode::Nvtx) {
            detail::nvtx_mark(name);
        }
    }
}

inline void set_thread_name(const char *name) {

    char truncated[16];
    std::strncpy(truncated, name, 15);
    truncated[15] = '\0';

    pid_t pid = getpid();
    pid_t tid = syscall(SYS_gettid);
    /*
    std::cerr << "[set_thread_name] pid=" << pid << " tid=" << tid << " name=\"" << truncated
              << "\"" << std::endl;
    */
    prctl(PR_SET_NAME, truncated, 0, 0, 0);

    if (detail::is_profiling_enabled()) {
        auto mode = detail::get_mode();
        if (mode == ProfileMode::Nvtx) {
            detail::nvtx_name_thread(truncated);
        }
    }
}

inline void set_thread_name(const std::string &name) { set_thread_name(name.c_str()); }

inline bool is_profiling_enabled() { return detail::is_profiling_enabled(); }

inline ProfileMode get_profile_mode() { return detail::get_mode(); }

inline const char *get_profile_mode_string() {
    switch (detail::get_mode()) {
    case ProfileMode::Nvtx:
        return "nvtx";
    case ProfileMode::Full:
        return "full";
    default:
        return "disabled";
    }
}

} // namespace profiler

namespace profiler {

struct Metrics {
    // bytes / ns accumulated per operation; gbps() = bytes / ns (decimal GB/s).
    struct Counter {
        std::atomic<uint64_t> bytes{0};
        std::atomic<uint64_t> ns{0};

        void add(uint64_t b, uint64_t t) {
            bytes.fetch_add(b, std::memory_order_relaxed);
            ns.fetch_add(t, std::memory_order_relaxed);
        }
        void reset() {
            bytes.store(0, std::memory_order_relaxed);
            ns.store(0, std::memory_order_relaxed);
        }
        double gbps() const {
            uint64_t t = ns.load(std::memory_order_relaxed);
            return t ? static_cast<double>(bytes.load(std::memory_order_relaxed)) / t : 0.0;
        }
    };

    Counter compress, decompress;
    // Time the xfer path blocks (copy submission on the H2D/D2H thread + xfer_wait).
    Counter h2d, d2h;

    void reset() {
        compress.reset();
        decompress.reset();
        h2d.reset();
        d2h.reset();
    }
};

inline Metrics g_metrics;

inline bool g_metrics_enabled = false;

struct MetricsTimer {
    bool enabled = false;
    std::chrono::steady_clock::time_point t0;
};

} // namespace profiler

using profiler::g_metrics;

#define METRICS_TIMER_START(var)                                                                   \
    ::profiler::MetricsTimer var;                                                                  \
    var.enabled = ::profiler::g_metrics_enabled;                                                   \
    if (KVCLIP_UNLIKELY(var.enabled))                                                              \
    var.t0 = std::chrono::steady_clock::now()

// Adds `bytes` and the time since METRICS_TIMER_START(var) to g_metrics.<counter>.
#define METRICS_ADD(var, counter, bytes)                                                           \
    do {                                                                                           \
        if (KVCLIP_UNLIKELY((var).enabled)) {                                                      \
            ::profiler::g_metrics.counter.add(                                                     \
                static_cast<uint64_t>(bytes),                                                      \
                static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(        \
                                          std::chrono::steady_clock::now() - (var).t0)             \
                                          .count()));                                              \
        }                                                                                          \
    } while (0)

#define METRICS_ADD_COMPRESS(var, bytes) METRICS_ADD(var, compress, bytes)
#define METRICS_ADD_DECOMPRESS(var, bytes) METRICS_ADD(var, decompress, bytes)
#define METRICS_ADD_XFER(var, is_h2d, bytes)                                                       \
    do {                                                                                           \
        if (is_h2d)                                                                                \
            METRICS_ADD(var, h2d, bytes);                                                          \
        else                                                                                       \
            METRICS_ADD(var, d2h, bytes);                                                          \
    } while (0)

#ifdef KVCLIP_PROFILER_DISABLED

#define PROFILE_SCOPE(name) ((void)0)
#define PROFILE_SCOPE_FMT(fmt, ...) ((void)0)
#define PROFILE_SCOPE_COLOR(name, color) ((void)0)
#define PROFILE_START(name) ((void)0)
#define PROFILE_START_FMT(fmt, ...) ((void)0)
#define PROFILE_END() ((void)0)
#define PROFILE_MARK(name) ((void)0)
#define PROFILE_SET_THREAD_NAME(name) ((void)0)

#else

#define PROFILE_SCOPE(name)                                                                        \
    std::optional<::profiler::ProfileScope> _profile_scope_##__LINE__;                             \
    if (KVCLIP_UNLIKELY(::profiler::detail::is_profiling_enabled()))                               \
    _profile_scope_##__LINE__.emplace(name)

#define PROFILE_SCOPE_FMT(fmt, ...)                                                                \
    std::optional<::profiler::ProfileScope> _profile_scope_##__LINE__;                             \
    if (KVCLIP_UNLIKELY(::profiler::detail::is_profiling_enabled())) {                             \
        char _ps_buf[256];                                                                         \
        snprintf(_ps_buf, sizeof(_ps_buf), fmt, ##__VA_ARGS__);                                    \
        _profile_scope_##__LINE__.emplace(_ps_buf);                                                \
    }

#define PROFILE_SCOPE_COLOR(name, color)                                                           \
    std::optional<::profiler::ProfileScope> _profile_scope_##__LINE__;                             \
    if (KVCLIP_UNLIKELY(::profiler::detail::is_profiling_enabled()))                               \
    _profile_scope_##__LINE__.emplace(name, color)

#define PROFILE_START(name) ::profiler::profile_start(name)

#define PROFILE_START_FMT(fmt, ...)                                                                \
    if (KVCLIP_UNLIKELY(::profiler::detail::is_profiling_enabled())) {                             \
        char _ps_buf[256];                                                                         \
        snprintf(_ps_buf, sizeof(_ps_buf), fmt, ##__VA_ARGS__);                                    \
        ::profiler::profile_start(_ps_buf);                                                        \
    }
#define PROFILE_END() ::profiler::profile_end()
#define PROFILE_MARK(name) ::profiler::profile_mark(name)
#define PROFILE_SET_THREAD_NAME(name) ::profiler::set_thread_name(name)

#endif
