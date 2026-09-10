// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "context.h"
#include "kv_pool.h"
#include "kv_zip.h"

using namespace profiler;

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {

    py::enum_<GpuTransferDirection>(m, "GpuTransferDirection")
        .value("H2D", GpuTransferDirection::H2D)
        .value("D2H", GpuTransferDirection::D2H);

    py::class_<Context>(m, "Context")
        .def_static(
            "create",
            [](const torch::Tensor &tensor, int chunk_dim, GpuTransferDirection direction,
               const std::string &name, py::object work_stream) -> Context {
                kv_xfer::stream_t gpu_work_stream = nullptr;
                if (!work_stream.is_none()) {
                    gpu_work_stream = kv_xfer::extract_stream(work_stream);
                }
                return Context::create(tensor, chunk_dim, direction, name, gpu_work_stream);
            },
            "Create transfer context from GPU tensor.\n"
            "work_stream: GPU stream for memcpy (put_stream or get_stream). None = use current "
            "stream.",
            py::arg("tensor"), py::arg("chunk_dim"),
            py::arg("direction") = GpuTransferDirection::H2D, py::arg("name") = "gpu_xfer",
            py::arg("work_stream") = py::none())
        .def("xfer_wait_cur_stream", &Context::xfer_wait_cur_stream,
             "Make work_stream wait for cur_stream's pending work (GPU-side, async).\n"
             "sync_cur_stream: if True, additionally CPU-blocking wait until cur_stream's\n"
             "  pending work completes (default False).",
             py::arg("sync_cur_stream") = false, py::call_guard<py::gil_scoped_release>())
        .def("xfer_wait_stream",
             static_cast<bool (Context::*)(py::object)>(&Context::xfer_wait_stream),
             "Wait for stream to complete current work before starting transfers.\n"
             "CUDA: async (returns True), XPU: sync blocking (returns False)",
             py::arg("cur_stream"))
        .def("xfer_chunk", &Context::xfer_chunk, "Transfer single chunk (async)",
             py::arg("cpu_tensor"), py::arg("chunk_idx"), py::call_guard<py::gil_scoped_release>())
        .def("xfer_chunks_batch", &Context::xfer_chunks_batch, "Transfer batch of chunks (async)",
             py::arg("chunk_indices"), py::arg("cpu_tensors"),
             py::call_guard<py::gil_scoped_release>())
        .def("xfer_finish", &Context::xfer_finish, "Record GPU event after all transfers (async)",
             py::call_guard<py::gil_scoped_release>())
        .def("xfer_wait", &Context::xfer_wait, "Wait for raw transfers to complete",
             py::call_guard<py::gil_scoped_release>())
        .def("xfer_is_complete", &Context::xfer_is_complete,
             "Check if raw transfers complete (non-blocking)",
             py::call_guard<py::gil_scoped_release>())
        .def("zip_to_mem", &Context::zip_to_mem,
             "Compress CPU tensors and store in cache (async via omp_queue).\n"
             "Call zip_wait() to block until done.",
             py::arg("cache"), py::arg("label"), py::arg("tensor_key"), py::arg("chunk_labels"),
             py::arg("cpu_tensors"), py::arg("compress") = true,
             py::call_guard<py::gil_scoped_release>())
        .def("unzip_from_mem", &Context::unzip_from_mem,
             "Retrieve from cache, decompress, and H2D transfer (async).\n"
             "Call unzip_wait() + xfer_wait() to block until done.",
             py::arg("cache"), py::arg("label"), py::arg("tensor_key"), py::arg("chunk_labels"),
             py::arg("chunk_indices"), py::arg("cpu_tensors"),
             py::call_guard<py::gil_scoped_release>())
        .def("zip_wait", &Context::zip_wait, "Wait for zip_to_mem to complete",
             py::call_guard<py::gil_scoped_release>())
        .def("zip_is_complete", &Context::zip_is_complete,
             "Check if zip_to_mem complete (non-blocking)",
             py::call_guard<py::gil_scoped_release>())
        .def("unzip_wait", &Context::unzip_wait, "Wait for unzip OMP work to complete",
             py::call_guard<py::gil_scoped_release>())
        .def("unzip_is_complete", &Context::unzip_is_complete,
             "Check if unzip OMP work complete (non-blocking)",
             py::call_guard<py::gil_scoped_release>());

    py::class_<kv_pool::Storage>(m, "Storage", "Persistent storage handler")
        .def(py::init<const std::string &>(), "Create storage handler with a persist directory.",
             py::arg("persist_dir"))
        .def(
            "save",
            [](kv_pool::Storage &self, const std::string &full_label, py::bytes data) {
                char *buffer;
                Py_ssize_t length;
                if (PyBytes_AsStringAndSize(data.ptr(), &buffer, &length) == -1) {
                    throw py::error_already_set();
                }
                py::gil_scoped_release release;
                return self.save(full_label, buffer, length);
            },
            "Save a data buffer to disk for the given label.", py::arg("full_label"),
            py::arg("data"))
        .def(
            "load",
            [](kv_pool::Storage &self, const std::string &full_label) {
                auto [buffer, size] = self.load(full_label);
                if (buffer) {
                    py::bytes data(buffer, size);
                    free(buffer);
                    return data;
                } else {
                    return py::bytes();
                }
            },
            "load data from disk for the given label and return the allocated buffer.",
            py::arg("full_label"));

    py::class_<kv_pool::Mem>(m, "Mem",
                             "High-performance cache with GIL-free operations.\n\n"
                             "Designed for millions of entries with O(1) lookup/insert.\n"
                             "Data is stored as raw C pointers, no Python objects involved.\n"
                             "All operations release GIL for maximum concurrency.")
        .def(py::init<size_t, kv_pool::Storage *, kv_pool::Record *>(),
             "Create cache with capacity in bytes (0 = unlimited) and optional storage handler and "
             "record handler",
             py::arg("capacity_bytes") = 0, py::arg("storage") = nullptr,
             py::arg("record") = nullptr)

        .def(
            "put",
            [](kv_pool::Mem &self, const std::vector<std::string> &keys,
               const std::vector<py::bytes> &data_list) {
                PROFILE_SCOPE("put_pyapi");
                IAXL_CHECK(keys.size() == data_list.size(),
                           "Mem.put: keys and data must have the same length");

                try {

                    std::vector<char *> data_ptrs;
                    std::vector<size_t> sizes;
                    data_ptrs.reserve(keys.size());
                    sizes.reserve(keys.size());

                    for (const auto &data : data_list) {
                        char *ptr = nullptr;
                        Py_ssize_t size = 0;
                        IAXL_CHECK(PyBytes_AsStringAndSize(data.ptr(), &ptr, &size) == 0,
                                   "Mem.put: failed to read Python bytes");
                        IAXL_CHECK(size > 0, "Mem.put: data must not be empty");

                        char *copy = (char *)malloc(size);
                        IAXL_CHECK(copy != nullptr, "Mem.put: data allocation failed");
                        memcpy(copy, ptr, size);

                        data_ptrs.push_back(copy);
                        sizes.push_back(size);
                    }

                    {
                        py::gil_scoped_release release;
                        self.put(keys, std::move(data_ptrs), sizes);
                    }
                } catch (...) {
                    IAXL_CHECK(false, "Mem.put: unexpected exception");
                }
            },
            "Put batch of key-value pairs (copies py::bytes to C memory)", py::arg("keys"),
            py::arg("data_list"))

        .def(
            "get",
            [](kv_pool::Mem &self, const std::vector<std::string> &keys) {
                PROFILE_SCOPE("get_pyapi");
                py::list py_results;

                {
                    py::gil_scoped_release release;
                    self.lock();
                }

                try {

                    auto results = self.get_unlocked(keys);
                    for (size_t i = 0; i < keys.size(); i++) {
                        const auto &[ptr, size] = results[i];
                        if (ptr) {
                            py_results.append(py::bytes(ptr, size));
                        } else {

                            throw std::runtime_error("Cache key not found: " + keys[i]);
                        }
                    }
                } catch (...) {
                    self.unlock();
                    throw;
                }

                {
                    py::gil_scoped_release release;
                    self.unlock();
                }

                return py_results;
            },
            "Get batch of values, returns list of bytes. Throws if any key not found.",
            py::arg("keys"))

        .def(
            "has",
            [](kv_pool::Mem &self, const std::vector<std::string> &keys) {
                std::vector<bool> results;
                {
                    py::gil_scoped_release release;
                    results = self.has(keys);
                }
                return results;
            },
            "Check existence of keys", py::arg("keys"))

        .def(
            "remove",
            [](kv_pool::Mem &self, const std::string &key) {
                size_t freed;
                {
                    py::gil_scoped_release release;
                    freed = self.remove(key);
                }
                return freed;
            },
            "Remove single key, returns bytes freed", py::arg("key"))

        .def(
            "evict_to_size",
            [](kv_pool::Mem &self, size_t target_bytes) {
                size_t evicted;
                {
                    py::gil_scoped_release release;
                    evicted = self.evict_to_size(target_bytes);
                }
                return evicted;
            },
            "Evict oldest entries until cache size <= target_bytes (respects evict_enabled)",
            py::arg("target_bytes"))

        .def(
            "force_evict_to_size",
            [](kv_pool::Mem &self, size_t target_bytes) {
                size_t evicted;
                {
                    py::gil_scoped_release release;
                    evicted = self.force_evict_to_size(target_bytes);
                }
                return evicted;
            },
            "Force evict oldest entries (ignores evict_enabled flag)", py::arg("target_bytes"))

        .def(
            "persist_groups",
            [](kv_pool::Mem &self, size_t max_groups) {
                py::gil_scoped_release release;
                return self.persist_groups(max_groups);
            },
            "Persist oldest unpersisted groups, returns list of (group_key, bytes_written)",
            py::arg("max_groups"))

        .def(
            "evict_groups",
            [](kv_pool::Mem &self, size_t max_groups) {
                py::gil_scoped_release release;
                return self.evict_groups(max_groups);
            },
            "Evict oldest LRU groups, returns list of (group_key, bytes_freed) for evicted groups",
            py::arg("max_groups"))

        .def(
            "set_evict_enabled",
            [](kv_pool::Mem &self, bool enabled) { self.set_evict_enabled(enabled); },
            "Enable/disable eviction (for safe pointer access periods)", py::arg("enabled"))
        .def("is_evict_enabled", &kv_pool::Mem::is_evict_enabled, "Check if eviction is enabled")

        .def(
            "lock",
            [](kv_pool::Mem &self) {
                py::gil_scoped_release release;
                self.lock();
            },
            "Acquire cache lock (for thread-safe pointer access)")
        .def(
            "unlock",
            [](kv_pool::Mem &self) {
                py::gil_scoped_release release;
                self.unlock();
            },
            "Release cache lock")

        .def(
            "acquire_deletion_guard",
            [](kv_pool::Mem &self) {
                py::gil_scoped_release release;
                self.acquire_deletion_guard();
            },
            "Acquire deletion guard (blocks eviction/removal, allows concurrent reads)")
        .def(
            "release_deletion_guard",
            [](kv_pool::Mem &self) {
                py::gil_scoped_release release;
                self.release_deletion_guard();
            },
            "Release deletion guard")
        .def("is_deletion_guarded", &kv_pool::Mem::is_deletion_guarded,
             "Check if any deletion guard is held")

        .def_property_readonly("size", &kv_pool::Mem::size, "Number of entries in cache")
        .def_property_readonly("group_count", &kv_pool::Mem::group_count,
                               "Number of groups in cache")
        .def_property_readonly("current_bytes", &kv_pool::Mem::current_bytes,
                               "Current cache size in bytes")
        .def_property_readonly("capacity_bytes", &kv_pool::Mem::capacity_bytes,
                               "Maximum cache capacity in bytes")
        .def("set_capacity_bytes", &kv_pool::Mem::set_capacity_bytes, "Set maximum cache capacity",
             py::arg("capacity"))
        .def_property_readonly("hits", &kv_pool::Mem::hits, "Total cache hits")
        .def_property_readonly("hits_in_storage", &kv_pool::Mem::hits_in_storage,
                               "Total cache hits that were loaded from storage")
        .def_property_readonly("misses", &kv_pool::Mem::misses, "Total cache misses")
        .def_property_readonly("puts", &kv_pool::Mem::puts, "Total put operations")
        .def_property_readonly("evictions", &kv_pool::Mem::evictions, "Total entries evicted")
        .def_property_readonly("total_unzip_bytes", &kv_pool::Mem::total_unzip_bytes,
                               "Total uncompressed (pre-compression) bytes accumulated")
        .def_property_readonly("total_zip_bytes", &kv_pool::Mem::total_zip_bytes,
                               "Total compressed bytes accumulated")
        .def_property_readonly("compression_ratio", &kv_pool::Mem::compression_ratio,
                               "Compression ratio (original / compressed)")
        .def_property_readonly("unpersisted_count", &kv_pool::Mem::unpersisted_count,
                               "Number of entries not yet persisted to disk")
        .def("reset_stats", &kv_pool::Mem::reset_stats, "Reset hit/miss/put/eviction counters")
        .def("clear", &kv_pool::Mem::clear, "Clear all entries")

        .def(
            "get_unpersisted",
            [](kv_pool::Mem &self, size_t max_count) {
                py::gil_scoped_release release;
                PROFILE_SCOPE("get_unpersisted_pyapi");
                self.lock();
                try {
                    auto entries = self.get_unpersisted_unlocked(max_count);

                    py::gil_scoped_acquire acquire;
                    py::list result;
                    for (const auto &[key, data, size] : entries) {
                        if (data && size > 0) {
                            result.append(py::make_tuple(key, py::bytes(data, size)));
                        }
                    }

                    self.unlock();
                    return result;
                } catch (...) {
                    self.unlock();
                    throw;
                }
            },
            "Get oldest unpersisted entries as list of (key, data) tuples", py::arg("max_count"))

        .def(
            "get_lru_oldest",
            [](kv_pool::Mem &self, size_t max_count) {
                py::gil_scoped_release release;
                PROFILE_SCOPE("get_lru_oldest_pyapi");
                self.lock();
                try {
                    auto entries = self.get_lru_oldest_unlocked(max_count);

                    py::gil_scoped_acquire acquire;
                    py::list result;
                    for (const auto &[key, data, size] : entries) {
                        if (data && size > 0) {
                            result.append(py::make_tuple(key, py::bytes(data, size)));
                        }
                    }

                    self.unlock();
                    return result;
                } catch (...) {
                    self.unlock();
                    throw;
                }
            },
            "Get oldest LRU entries as list of (key, data) tuples (for eviction)",
            py::arg("max_count"))

        .def(
            "mark_persisted",
            [](kv_pool::Mem &self, const std::vector<std::string> &keys) {
                py::gil_scoped_release release;
                self.mark_persisted(keys);
            },
            "Mark entries as persisted (remove from unpersisted list)", py::arg("keys"));

    py::class_<kv_pool::Record>(
        m, "Record",
        "Async SQLite record writer using background C++ thread.\n\n"
        "Thread-safe: submit() and sync() can be called from any thread.\n"
        "Uses TaskQueue for async execution (same pattern as GPU transfers).\n"
        "Safe for use in daemon processes where Python multiprocessing fails.\n"
        "Constructor opens database and starts worker thread immediately.")
        .def(py::init<const std::string &, bool>(),
             "Create Record with SQLite database path (opens DB and starts worker)",
             py::arg("sqlite_path"), py::arg("cleanup_unpersisted") = false)
        .def(
            "submit",
            [](kv_pool::Record &self, const std::string &label,
               const std::vector<std::string> &chunk_labels) {
                py::gil_scoped_release release;
                self.submit(label, chunk_labels);
            },
            "Submit group_keys to record (async, thread-safe)", py::arg("label"),
            py::arg("chunk_labels"))
        .def(
            "remove",
            [](kv_pool::Record &self, const std::vector<std::string> &chunk_labels) {
                py::gil_scoped_release release;
                self.remove(chunk_labels);
            },
            "Remove chunk records from SQLite (async, batched)", py::arg("chunk_labels"))
        .def(
            "is_persisted",
            [](kv_pool::Record &self, const std::vector<std::string> &chunk_labels) {
                py::gil_scoped_release release;
                return self.is_persisted(chunk_labels);
            },
            "Check if chunks are persisted in SQLite (syncs pending writes first).",
            py::arg("chunk_labels"))
        .def(
            "sync",
            [](kv_pool::Record &self) {
                py::gil_scoped_release release;
                self.sync();
            },
            "Wait for all pending tasks to complete (thread-safe)")
        .def(
            "has",
            [](kv_pool::Record &self, const std::string &label,
               const std::vector<std::string> &chunk_labels) {
                py::gil_scoped_release release;
                return self.has(label, chunk_labels);
            },
            "Check if chunk groups exist in SQLite (syncs pending writes first).\n"
            "Returns one bool per chunk_label.",
            py::arg("label"), py::arg("chunk_labels"))
        .def(
            "mark_persisted",
            [](kv_pool::Record &self, const std::vector<std::string> &chunk_labels) {
                py::gil_scoped_release release;
                self.mark_persisted(chunk_labels);
            },
            "Mark chunks as persisted after successful disk save (async, batched)",
            py::arg("chunk_labels"))
        .def(
            "shutdown",
            [](kv_pool::Record &self) {
                py::gil_scoped_release release;
                self.shutdown();
            },
            "Shutdown worker thread and close database (waits for pending tasks)")
        .def_property_readonly(
            "pending_count", [](kv_pool::Record &self) { return self.pending_count(); },
            "Number of pending record requests");

    // GPU-less compress/decompress entry points that bypass Context entirely: they operate
    // directly on already-CPU-resident tensors and share the same QAT/IAA/CPU zip task pool and
    // native Mem cache used by the local (GPU-attached) KVStore path. Intended for the remote NIXL
    // cache daemon (iaxl.remote), which receives KV bytes via RDMA into CPU staging buffers and has
    // no GPU/Context of its own.
    m.def(
        "zip_compress_to_mem",
        [](kv_pool::Mem &mem, const std::vector<std::string> &full_chunk_labels,
           const std::vector<torch::Tensor> &cpu_tensors, bool compress) {
            IAXL_CHECK(full_chunk_labels.size() == cpu_tensors.size(),
                       "zip_compress_to_mem: labels and tensors must have the same length");
            py::gil_scoped_release release;
            const size_t n = cpu_tensors.size();
            std::vector<char *> out_bufs(n);
            std::vector<size_t> out_sizes(n), orig_sizes(n);
            kv_zip::kv_zip_compress_batch(cpu_tensors, out_bufs, out_sizes, orig_sizes, compress);
            mem.put(full_chunk_labels, std::move(out_bufs), out_sizes, orig_sizes);
        },
        "Compress a batch of contiguous CPU tensors (shared QAT/IAA/CPU zip pool, same code path\n"
        "as the local KVStore) and insert the results directly into a native Mem cache. No\n"
        "Context/GPU transfer is involved.",
        py::arg("mem"), py::arg("full_chunk_labels"), py::arg("cpu_tensors"),
        py::arg("compress") = true);

    m.def(
        "zip_decompress_from_mem",
        [](kv_pool::Mem &mem, const std::vector<std::string> &full_chunk_labels,
           const std::vector<torch::Tensor> &cpu_tensors) {
            IAXL_CHECK(full_chunk_labels.size() == cpu_tensors.size(),
                       "zip_decompress_from_mem: labels and tensors must have the same length");
            py::gil_scoped_release release;
            mem.acquire_deletion_guard();
            try {
                auto results = mem.get(full_chunk_labels);
                std::vector<const char *> data_ptrs(results.size());
                for (size_t i = 0; i < results.size(); i++) {
                    IAXL_CHECK(results[i].first != nullptr,
                               ("zip_decompress_from_mem: cache key missing: " +
                                full_chunk_labels[i])
                                   .c_str());
                    data_ptrs[i] = results[i].first;
                }
                kv_zip::kv_zip_decompress_batch(data_ptrs, cpu_tensors);
            } catch (...) {
                mem.release_deletion_guard();
                throw;
            }
            mem.release_deletion_guard();
        },
        "Fetch compressed blobs for full_chunk_labels from a native Mem cache and decompress\n"
        "directly into the given CPU tensors (shared QAT/IAA/CPU zip pool, same code path as the\n"
        "local KVStore). Raises if any key is missing.",
        py::arg("mem"), py::arg("full_chunk_labels"), py::arg("cpu_tensors"));

    m.def(
        "metrics_set_enabled", [](bool enable) { profiler::g_metrics_enabled = enable; },
        "Enable or disable compression/decompression throughput metrics", py::arg("enable") = true);

    m.def(
        "metrics_is_enabled", []() { return profiler::g_metrics_enabled; },
        "Return whether compression/decompression metrics collection is enabled");

    m.def(
        "metrics_reset", []() { profiler::g_metrics.reset(); },
        "Reset accumulated compression/decompression metrics");

    m.def(
        "metrics_read",
        []() {
            py::dict d;
            d["enabled"] = profiler::g_metrics_enabled;
            d["compress_bytes"] =
                profiler::g_metrics.compress_bytes.load(std::memory_order_relaxed);
            d["compress_ns"] = profiler::g_metrics.compress_ns.load(std::memory_order_relaxed);
            d["compress_gbps"] = profiler::g_metrics.compress_gbps();
            d["decompress_bytes"] =
                profiler::g_metrics.decompress_bytes.load(std::memory_order_relaxed);
            d["decompress_ns"] = profiler::g_metrics.decompress_ns.load(std::memory_order_relaxed);
            d["decompress_gbps"] = profiler::g_metrics.decompress_gbps();
            return d;
        },
        "Read accumulated compression/decompression throughput metrics (GB/s is decimal)");
}
