// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "kv_pool.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

static int g_failures = 0;

#define CHECK(cond, msg)                                                                           \
    do {                                                                                           \
        if (!(cond)) {                                                                             \
            std::fprintf(stderr, "[FAIL] %s (%s:%d)\n", (msg), __FILE__, __LINE__);                \
            ++g_failures;                                                                          \
        } else {                                                                                   \
            std::printf("[ok]   %s\n", (msg));                                                     \
        }                                                                                          \
    } while (0)

static char *dup_buf(const std::string &s) {
    char *p = static_cast<char *>(std::malloc(s.size()));
    std::memcpy(p, s.data(), s.size());
    return p;
}

static bool buf_eq(const char *p, size_t n, const std::string &s) {
    return p != nullptr && n == s.size() && std::memcmp(p, s.data(), n) == 0;
}

int main(int argc, char **argv) {
    const std::string dir = (argc > 1) ? argv[1] : "/tmp/kv_pool_test_dir";
    const std::string db_path = (argc > 2) ? argv[2] : "/tmp/kv_pool_test.db";

    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
    std::remove(db_path.c_str());
    std::remove((db_path + "-wal").c_str());
    std::remove((db_path + "-shm").c_str());

    {
        kv_pool::Storage storage(dir);
        const std::string label = "kv:abc123:0";
        const std::string payload = "hello-storage";
        CHECK(storage.save(label, payload.data(), payload.size()), "storage.save() ok");
        auto [buf, n] = storage.load(label);
        CHECK(buf_eq(buf, n, payload), "storage.load() returns saved bytes");
        std::free(buf);
    }

    {
        kv_pool::Mem mem(0);
        const std::vector<std::string> keys = {"kv:0:k", "kv:0:v", "kv:1:k"};
        const std::string d0 = "AAAA", d1 = "BBBBBB", d2 = "CC";

        std::vector<char *> ptrs = {dup_buf(d0), dup_buf(d1), dup_buf(d2)};
        std::vector<size_t> sizes = {d0.size(), d1.size(), d2.size()};
        mem.put(keys, std::move(ptrs), sizes);

        CHECK(mem.size() == 3, "size() == 3 after put");
        CHECK(mem.group_count() == 2, "group_count() == 2 (kv:0, kv:1)");
        CHECK(mem.current_bytes() == d0.size() + d1.size() + d2.size(),
              "current_bytes() sums buffers");

        auto got = mem.get(keys);
        CHECK(buf_eq(got[0].first, got[0].second, d0), "get() key0 matches");
        CHECK(buf_eq(got[1].first, got[1].second, d1), "get() key1 matches");
        CHECK(buf_eq(got[2].first, got[2].second, d2), "get() key2 matches");
        CHECK(mem.hits() == 3, "hits() == 3 after 3 gets");

        auto present = mem.has({"kv:0:k", "kv:9:x"});
        CHECK(present.size() == 2 && present[0] && !present[1], "has() reports presence");

        size_t freed = mem.remove("kv:0:k");
        CHECK(freed == d0.size() + d1.size(), "remove() frees whole group bytes");
        CHECK(mem.group_count() == 1, "group_count() == 1 after remove");
        CHECK(mem.size() == 1, "size() == 1 after remove");

        auto miss = mem.get({"kv:0:k"});
        CHECK(miss[0].first == nullptr, "get() after remove is a miss");
        CHECK(mem.misses() >= 1, "misses() incremented");
    }

    {
        kv_pool::Mem mem(0);
        std::vector<std::string> keys;
        std::vector<char *> ptrs;
        std::vector<size_t> sizes;
        for (int i = 0; i < 4; ++i) {
            keys.push_back("kv:" + std::to_string(i) + ":k");
            std::string d(100, 'x');
            ptrs.push_back(dup_buf(d));
            sizes.push_back(d.size());
        }
        mem.put(keys, std::move(ptrs), sizes);
        CHECK(mem.current_bytes() == 400, "current_bytes() == 400 before evict");

        size_t evicted = mem.evict_to_size(150);
        CHECK(evicted >= 250, "evict_to_size() freed oldest groups");
        CHECK(mem.current_bytes() <= 150, "current_bytes() <= target after evict");
        CHECK(mem.evictions() >= 1, "evictions() incremented");
    }

    {
        kv_pool::Storage storage(dir);
        kv_pool::Record record(db_path, true);
        kv_pool::Mem mem(0, &storage, &record);

        const std::vector<std::string> keys = {"kv:aa:k", "kv:bb:k"};
        const std::string da = "payload-A", db = "payload-B";
        std::vector<char *> ptrs = {dup_buf(da), dup_buf(db)};
        std::vector<size_t> sizes = {da.size(), db.size()};

        record.submit("kv", {"aa", "bb"});
        record.sync();
        mem.put(keys, std::move(ptrs), sizes);
        CHECK(mem.unpersisted_count() == 2, "unpersisted_count() == 2 after put");

        auto persisted = mem.persist_groups(10);
        CHECK(persisted.size() == 2, "persist_groups() persisted 2 groups");
        CHECK(mem.unpersisted_count() == 0, "unpersisted_count() == 0 after persist");

        size_t evicted = mem.force_evict_to_size(0);
        CHECK(evicted == da.size() + db.size(), "force_evict_to_size(0) cleared memory");
        CHECK(mem.group_count() == 0, "group_count() == 0 after full evict");

        auto reloaded = mem.get(keys, true);
        CHECK(buf_eq(reloaded[0].first, reloaded[0].second, da), "get() reloads key0 from storage");
        CHECK(buf_eq(reloaded[1].first, reloaded[1].second, db), "get() reloads key1 from storage");
        CHECK(mem.hits_in_storage() == 2, "hits_in_storage() == 2 after reload");
    }

    // Groups are recorded only once all entries are stored, and over-budget puts evict LRU
    // groups and drop them from the Record.
    {
        std::remove(db_path.c_str());
        kv_pool::Storage storage(dir);
        kv_pool::Record record(db_path, true);
        kv_pool::Mem mem(1000, &storage, &record);
        mem.set_entries_per_group(2);
        mem.set_evict_watermark(0.5);

        auto put_layer = [&](const std::string &chunk, const std::string &layer) {
            std::string d(100, 'x');
            std::vector<char *> ptrs = {dup_buf(d)};
            mem.put({"kv:" + chunk + ":" + layer}, std::move(ptrs), {d.size()});
        };

        put_layer("c0", "k");
        CHECK(!record.has("kv", {"c0"})[0], "partially stored group is not recorded");
        put_layer("c0", "v");
        CHECK(record.has("kv", {"c0"})[0], "group recorded once all entries are stored");

        for (int i = 1; i <= 4; ++i) {
            put_layer("c" + std::to_string(i), "k");
            put_layer("c" + std::to_string(i), "v");
        }
        CHECK(mem.current_bytes() == 1000, "pool exactly at capacity is not evicted");
        CHECK(mem.evictions() == 0, "no eviction while within capacity");

        put_layer("c5", "k");
        CHECK(mem.current_bytes() <= 500, "over-budget put evicts down to the watermark");
        CHECK(mem.evictions() == 3, "the three least-recently-used groups were evicted");
        CHECK(mem.evicted_bytes() == 600, "evicted_bytes() counts freed bytes");
        auto rec = record.has("kv", {"c0", "c1", "c2", "c3", "c4", "c5"});
        CHECK(!rec[0] && !rec[1] && !rec[2], "evicted groups were removed from the Record");
        CHECK(rec[3] && rec[4], "surviving complete groups stay recorded");
        CHECK(!rec[5], "group written by the evicting put is kept but not yet recorded");
        auto present = mem.has({"kv:c5:k", "kv:c0:k"});
        CHECK(present[0] && !present[1], "the just-written group survives eviction");
    }

    // A single put larger than the whole pool must not overflow it either.
    {
        std::remove(db_path.c_str());
        kv_pool::Record record(db_path, true);
        kv_pool::Mem mem(250, nullptr, &record);
        mem.set_entries_per_group(1);
        mem.set_evict_watermark(0.9);
        std::vector<std::string> keys;
        std::vector<char *> ptrs;
        std::vector<size_t> sizes;
        std::string d(100, 'w');
        for (int i = 0; i < 5; ++i) {
            keys.push_back("kv:b" + std::to_string(i) + ":k");
            ptrs.push_back(dup_buf(d));
            sizes.push_back(d.size());
        }
        mem.put(keys, std::move(ptrs), sizes);
        CHECK(mem.current_bytes() <= 250, "oversized put is trimmed to capacity");
        auto present = mem.has(keys);
        CHECK(!present[0] && !present[1] && !present[2] && present[3] && present[4],
              "oversized put keeps its newest groups");
        auto rec = record.has("kv", {"b0", "b1", "b2", "b3", "b4"});
        CHECK(!rec[0] && !rec[1] && !rec[2] && rec[3] && rec[4],
              "Record matches the groups kept from the oversized put");
    }

    // Pinned groups survive every eviction path, and the over-budget put still evicts others.
    {
        std::remove(db_path.c_str());
        kv_pool::Record record(db_path, true);
        kv_pool::Mem mem(300, nullptr, &record);
        mem.set_entries_per_group(1);
        mem.set_evict_watermark(1.0);

        auto put_group = [&](const std::string &chunk) {
            std::string d(100, 'y');
            std::vector<char *> ptrs = {dup_buf(d)};
            mem.put({"kv:" + chunk + ":k"}, std::move(ptrs), {d.size()});
        };
        put_group("p0");
        put_group("p1");
        put_group("p2");

        std::vector<kv_pool::Mem::PinHandle> p0_pins;
        auto got = mem.get_pinned({"kv:p0:k", "kv:zz:k"}, p0_pins);
        CHECK(got[0].first != nullptr && got[1].first == nullptr, "get_pinned() hit and miss");
        mem.get({"kv:p1:k", "kv:p2:k"});  // make the pinned p0 the LRU-oldest group again

        put_group("p3");
        auto present = mem.has({"kv:p0:k", "kv:p1:k", "kv:p2:k", "kv:p3:k"});
        CHECK(present[0], "pinned LRU-oldest group is skipped by over-budget eviction");
        CHECK(!present[1] && present[2] && present[3], "the next unpinned LRU group is evicted");

        CHECK(mem.evict_groups(10).size() == 2, "evict_groups() evicts only unpinned groups");
        CHECK(mem.force_evict_to_size(0) == 0, "force_evict_to_size() skips pinned groups");
        CHECK(mem.remove("kv:p0:k") == 0, "remove() refuses a pinned group");
        CHECK(record.has("kv", {"p0"})[0], "pinned group is still recorded");

        CHECK(p0_pins.size() == 1, "get_pinned() returns one handle per hit");
        mem.unpin(p0_pins);
        CHECK(mem.evict_groups(10).size() == 1, "group is evictable after unpin()");
        CHECK(!record.has("kv", {"p0"})[0], "manual eviction removes the Record entry");
        CHECK(mem.group_count() == 0, "pool is empty");
    }

    // Persisted groups keep their Record entry on eviction and reload from storage.
    {
        std::remove(db_path.c_str());
        kv_pool::Storage storage(dir);
        kv_pool::Record record(db_path, true);
        kv_pool::Mem mem(250, &storage, &record);
        mem.set_entries_per_group(1);

        std::string d(100, 'z');
        std::vector<char *> ptrs = {dup_buf(d)};
        mem.put({"kv:s0:k"}, std::move(ptrs), {d.size()});
        CHECK(mem.persist_groups(10).size() == 1, "group persisted");
        for (const char *chunk : {"s1", "s2"}) {
            std::vector<char *> p = {dup_buf(d)};
            mem.put({std::string("kv:") + chunk + ":k"}, std::move(p), {d.size()});
        }
        CHECK(!mem.has({"kv:s0:k"})[0], "persisted group evicted from DRAM");
        CHECK(record.has("kv", {"s0"})[0], "persisted group stays recorded after eviction");
        std::vector<kv_pool::Mem::PinHandle> s0_pins;
        auto reloaded = mem.get_pinned({"kv:s0:k"}, s0_pins, true);
        CHECK(buf_eq(reloaded[0].first, reloaded[0].second, d), "evicted group reloads from storage");
        mem.unpin(s0_pins);
        CHECK(mem.current_bytes() <= 250, "reload kept the pool within capacity");

        // A key repeated in one reload batch must not leave a pointer to a replaced buffer.
        mem.force_evict_to_size(0);
        std::vector<kv_pool::Mem::PinHandle> dup_pins;
        auto dup = mem.get_pinned({"kv:s0:k", "kv:s0:k"}, dup_pins, true);
        CHECK(dup[0].first == dup[1].first && buf_eq(dup[0].first, dup[0].second, d),
              "duplicate keys in a reload resolve to the same live buffer");
        CHECK(dup_pins.size() == 2 && dup_pins[0] == dup_pins[1],
              "duplicate keys pin the same group twice");
        mem.unpin(dup_pins);
        CHECK(mem.evict_groups(10).size() == 1, "both duplicate pins are released");
    }

    // Only complete groups are persisted; group pins hold across a load and unpin re-enforces.
    {
        std::remove(db_path.c_str());
        kv_pool::Storage storage(dir);
        kv_pool::Record record(db_path, true);
        kv_pool::Mem mem(300, &storage, &record);
        mem.set_entries_per_group(2);

        auto put_entry = [&](const std::string &label) {
            std::string d(100, 'q');
            std::vector<char *> ptrs = {dup_buf(d)};
            mem.put({label}, std::move(ptrs), {d.size()});
        };
        put_entry("kv:g0:k");
        CHECK(mem.persist_groups(10).empty(), "incomplete group is not persisted");
        put_entry("kv:g0:v");
        CHECK(mem.persist_groups(10).size() == 1, "complete group is persisted");

        auto pinned = mem.pin_groups({"kv:g0", "kv:none"});
        CHECK(pinned[0] == kv_pool::Mem::kGroupComplete && pinned[1] == kv_pool::Mem::kGroupAbsent,
              "pin_groups() reports complete and absent groups");
        put_entry("kv:gx:k");
        auto partial = mem.pin_groups({"kv:gx"});
        CHECK(partial[0] == kv_pool::Mem::kGroupIncomplete, "pin_groups() reports incomplete groups");
        mem.unpin_groups({"kv:gx"});
        mem.remove("kv:gx:k");
        put_entry("kv:g1:k");
        put_entry("kv:g1:v");
        auto present = mem.has({"kv:g0:k", "kv:g1:k"});
        CHECK(present[0] && !present[1] && mem.current_bytes() <= 300,
              "pinned group survives; the over-budget put drops its own group instead");
        mem.set_capacity_bytes(100);
        mem.unpin_groups({"kv:g0"});
        CHECK(mem.current_bytes() <= 100 && !mem.has({"kv:g0:k"})[0],
              "unpin_groups() evicts what the pin held back");
        CHECK(record.has("kv", {"g0"})[0], "evicted persisted group stays recorded");

        mem.set_capacity_bytes(1000);
        put_entry("kv:g2:k");
        put_entry("kv:g2:v");
        CHECK(record.has("kv", {"g2"})[0], "complete group recorded");
        CHECK(mem.remove("kv:g2:k") == 200, "remove() drops a whole group");
        CHECK(!record.has("kv", {"g2"})[0], "remove() also drops the Record row");
    }

    std::filesystem::remove_all(dir, ec);
    std::remove(db_path.c_str());
    std::remove((db_path + "-wal").c_str());
    std::remove((db_path + "-shm").c_str());

    if (g_failures == 0) {
        std::printf("\nAll kv_pool checks passed.\n");
        return 0;
    }
    std::fprintf(stderr, "\n%d kv_pool check(s) failed.\n", g_failures);
    return 1;
}
