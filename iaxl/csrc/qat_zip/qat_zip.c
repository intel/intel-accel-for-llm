// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "cpa.h"
#include "cpa_dc.h"
#include "icp_sal_user.h"
#include "icp_sal_poll.h"
#include "qae_mem.h"

#include "env.h"
#include "iaxl_common.h"
#include "qat_zip.h"

#define MAX_DEVICES 8
#define MAX_INSTANCES_PER_DEVICE 64
#define MAX_INSTANCES (MAX_DEVICES * MAX_INSTANCES_PER_DEVICE)
#define MAX_RAW_INSTANCES 256
#define MAX_QUEUE_DEPTH 4

#define DEFAULT_INSTANCES_PER_DEVICE 4
#define DEFAULT_SRC_CAP (256 * 1024)
#define DEFAULT_DST_CAP (256 * 1024)
#define DEFAULT_DEVICES "0"

static int g_instances_per_device = DEFAULT_INSTANCES_PER_DEVICE;
static uint32_t g_src_cap = DEFAULT_SRC_CAP;
static uint32_t g_dst_cap = DEFAULT_DST_CAP;
static int g_queue_depth = MAX_QUEUE_DEPTH;
static uint32_t g_buf_cap = DEFAULT_DST_CAP;
/* Split compression requests larger than this many bytes (0 = disabled). */
static int g_segment = 0;

typedef struct {
    void *src_meta;
    void *dst_meta;
    CpaFlatBuffer fsrc, fdst;
    CpaBufferList lsrc, ldst;
    CpaDcRqResults res;
    CpaDcOpData op;
    int done;
    unsigned char *in;
    unsigned char *out;
    /* Multi-segment state used when a request exceeds g_segment. A block is
     * submitted as several FLUSH_FULL requests plus a final FLUSH_FINAL one,
     * and the segment outputs are concatenated into one DEFLATE stream. */
    int seg_active;
    int seg_total;
    int seg_count;
    int seg_next;
    const unsigned char *seg_src;
    unsigned char *seg_acc;
    size_t seg_acc_cap;
    size_t seg_acc_len;
} Slot;

typedef struct {
    CpaInstanceHandle inst;
    CpaDcSessionHandle sess;
    int node;
    int pcie_bus;
    int accel_id;
    Slot slot[MAX_QUEUE_DEPTH];
} Instance;

static Instance g_inst[MAX_INSTANCES];
static int g_inst_count = 0;

typedef struct {
    CpaInstanceHandle inst;
    int pcie_bus;
    int accel_id;
    int node;
} RawInst;

static RawInst g_all[MAX_RAW_INSTANCES];
static int g_all_count = 0;

static int g_dev_bus[MAX_DEVICES];
static int g_dev_node[MAX_DEVICES];
static int g_dev_accel[MAX_DEVICES];
static int g_dev_count = 0;

static void dc_callback(void *cb_param, CpaStatus status) {
    (void)status;
    *(int *)cb_param = 1;
}

static int submit_op(Instance *d, int si, int compress, void *src, uint32_t src_len, void *dst,
                     uint32_t dst_cap, CpaDcFlush flush) {
    Slot *sl = &d->slot[si];
    sl->fsrc.dataLenInBytes = src_len;
    sl->fsrc.pData = src;
    sl->fdst.dataLenInBytes = dst_cap;
    sl->fdst.pData = dst;
    sl->lsrc.numBuffers = 1;
    sl->lsrc.pBuffers = &sl->fsrc;
    sl->lsrc.pPrivateMetaData = sl->src_meta;
    sl->ldst.numBuffers = 1;
    sl->ldst.pBuffers = &sl->fdst;
    sl->ldst.pPrivateMetaData = sl->dst_meta;
    memset(&sl->res, 0, sizeof(sl->res));
    memset(&sl->op, 0, sizeof(sl->op));
    sl->op.flushFlag = flush;

    if (compress)
        sl->op.compressAndVerify = CPA_TRUE;

    sl->done = 0;
    CpaStatus s = compress ? cpaDcCompressData2(d->inst, d->sess, &sl->lsrc, &sl->ldst, &sl->op,
                                                &sl->res, &sl->done)
                           : cpaDcDecompressData2(d->inst, d->sess, &sl->lsrc, &sl->ldst, &sl->op,
                                                  &sl->res, &sl->done);
    return (s == CPA_STATUS_SUCCESS) ? 0 : -1;
}

static int wait_op(Instance *d, int si, uint32_t *produced) {
    Slot *sl = &d->slot[si];
    while (!sl->done) {
        icp_sal_DcPollInstance(d->inst, 0);
    }
    if (sl->res.status != CPA_DC_OK)
        return -1;
    if (produced)
        *produced = sl->res.produced;
    return 0;
}

static void qat_start(void) {
    if (icp_sal_userStartMultiProcess("SHIM", CPA_FALSE) == CPA_STATUS_SUCCESS)
        return;
    if (icp_sal_userStartMultiProcess("SSL", CPA_FALSE) == CPA_STATUS_SUCCESS)
        return;
    if (icp_sal_userStart("DEFAULT") == CPA_STATUS_SUCCESS)
        return;
    IAXL_CHECK(0, "qat_zip: failed to start QAT user space");
}

static void discover_all(void) {
    Cpa16U total = 0;
    IAXL_CHECK(cpaDcGetNumInstances(&total) == CPA_STATUS_SUCCESS,
               "qat_zip: cpaDcGetNumInstances failed");
    IAXL_CHECK(total > 0, "qat_zip: no QAT instances found");
    IAXL_CHECK(total <= MAX_RAW_INSTANCES,
               "qat_zip: driver reported more instances than supported");

    CpaInstanceHandle *handles = calloc(total, sizeof(*handles));
    IAXL_CHECK(handles != NULL, "qat_zip: instance handle allocation failed");
    IAXL_CHECK(cpaDcGetInstances(total, handles) == CPA_STATUS_SUCCESS,
               "qat_zip: cpaDcGetInstances failed");

    fprintf(stderr, "[discovery] found %d QAT instance(s)\n", (int)total);
    for (int i = 0; i < total; i++) {
        CpaInstanceInfo2 info = {0};
        IAXL_CHECK(cpaDcInstanceGetInfo2(handles[i], &info) == CPA_STATUS_SUCCESS,
                   "qat_zip: cpaDcInstanceGetInfo2 failed");

        int bus = (info.physInstId.busAddress >> 8) & 0xFF;

        if (g_all_count < MAX_RAW_INSTANCES) {
            g_all[g_all_count].inst = handles[i];
            g_all[g_all_count].pcie_bus = bus;
            g_all[g_all_count].accel_id = info.physInstId.acceleratorId;
            g_all[g_all_count].node = info.nodeAffinity;
            g_all_count++;
        }

        int known = 0;
        for (int j = 0; j < g_dev_count; j++)
            if (g_dev_bus[j] == bus) {
                known = 1;
                break;
            }
        if (!known && g_dev_count < MAX_DEVICES) {
            int d = g_dev_count++;
            g_dev_bus[d] = bus;
            g_dev_node[d] = info.nodeAffinity;
            g_dev_accel[d] = info.physInstId.acceleratorId;
            fprintf(stderr, "  device[%d]: PCIe bus=0x%02x  accel_id=%d  NUMA=%d\n", d, bus,
                    g_dev_accel[d], g_dev_node[d]);
        }
    }
    free(handles);
}

static int parse_selection(const char *env, int discovered, int *sel, int max_sel) {
    if (!env || !*env)
        env = DEFAULT_DEVICES;

    int n = 0;
    char buf[256];
    strncpy(buf, env, sizeof(buf) - 1);
    buf[sizeof(buf) - 1] = '\0';

    for (char *tok = strtok(buf, ","); tok && n < max_sel; tok = strtok(NULL, ",")) {
        int idx = atoi(tok);
        IAXL_CHECK(idx >= 0 && idx < discovered, "qat_zip: selected device index is out of range");
        sel[n++] = idx;
    }
    IAXL_CHECK(n > 0, "qat_zip: no valid device selected");
    return n;
}

static void instance_init(Instance *d, const RawInst *r) {
    CpaDcSessionSetupData sd = {0};

    sd.compLevel = CPA_DC_L12;
    sd.compType = CPA_DC_DEFLATE;
    sd.huffType = CPA_DC_HT_FULL_DYNAMIC;
    sd.sessDirection = CPA_DC_DIR_COMBINED;
    sd.sessState = CPA_DC_STATELESS;
    sd.checksum = CPA_DC_CRC32;
    // gen4 always compresses with a 32 KB history window: cpaDcQueryCapabilities
    // reports no smaller size and a CPA_DC_WINSIZE_4K session still emits 32 KB
    // distances, which is why IAA cannot decompress QAT streams.
    sd.windowSize = CPA_DC_WINSIZE_32K;

    d->inst = r->inst;
    d->node = r->node;
    d->pcie_bus = r->pcie_bus;
    d->accel_id = r->accel_id;

    IAXL_CHECK(cpaDcSetAddressTranslation(d->inst, qaeVirtToPhysNUMA) == CPA_STATUS_SUCCESS,
               "qat_zip: cpaDcSetAddressTranslation failed");
    IAXL_CHECK(cpaDcStartInstance(d->inst, 0, NULL) == CPA_STATUS_SUCCESS,
               "qat_zip: cpaDcStartInstance failed");

    Cpa32U sess_size = 0, ctx_size = 0, meta_size = 0;
    IAXL_CHECK(cpaDcGetSessionSize(d->inst, &sd, &sess_size, &ctx_size) == CPA_STATUS_SUCCESS,
               "qat_zip: cpaDcGetSessionSize failed");
    IAXL_CHECK(cpaDcBufferListGetMetaSize(d->inst, 1, &meta_size) == CPA_STATUS_SUCCESS,
               "qat_zip: cpaDcBufferListGetMetaSize failed");

    d->sess = qaeMemAllocNUMA(sess_size, d->node, 64);
    IAXL_CHECK(d->sess != NULL, "qat_zip: session allocation failed");

    IAXL_CHECK(cpaDcInitSession(d->inst, d->sess, &sd, NULL, dc_callback) == CPA_STATUS_SUCCESS,
               "qat_zip: cpaDcInitSession failed");

    for (int si = 0; si < MAX_QUEUE_DEPTH; si++) {
        Slot *sl = &d->slot[si];
        sl->src_meta = qaeMemAllocNUMA(meta_size, d->node, 64);
        sl->dst_meta = qaeMemAllocNUMA(meta_size, d->node, 64);
        sl->in = qaeMemAllocNUMA(g_buf_cap, d->node, 64);
        sl->out = qaeMemAllocNUMA(g_buf_cap, d->node, 64);
        IAXL_CHECK(sl->src_meta && sl->dst_meta && sl->in && sl->out,
                   "qat_zip: slot buffer allocation failed");
    }
}

static void instance_teardown(Instance *d) {
    cpaDcRemoveSession(d->inst, d->sess);
    cpaDcStopInstance(d->inst);
    qaeMemFreeNUMA(&d->sess);
    for (int si = 0; si < MAX_QUEUE_DEPTH; si++) {
        Slot *sl = &d->slot[si];
        qaeMemFreeNUMA(&sl->src_meta);
        qaeMemFreeNUMA(&sl->dst_meta);
        qaeMemFreeNUMA((void **)&sl->in);
        qaeMemFreeNUMA((void **)&sl->out);
        free(sl->seg_acc);
        sl->seg_acc = NULL;
    }
}

int qat_zip_init(void) {
    g_instances_per_device = envs.IAXL_QAT_ZIP_INSTANCES_PER_DEVICE;
    g_src_cap = (uint32_t)envs.IAXL_ZIP_SRC_CAP;
    g_dst_cap = (uint32_t)envs.IAXL_ZIP_DST_CAP;
    g_queue_depth = envs.IAXL_QAT_ZIP_QUEUE_DEPTH;

    if (g_instances_per_device < 1)
        g_instances_per_device = 1;
    if (g_instances_per_device > MAX_INSTANCES_PER_DEVICE)
        g_instances_per_device = MAX_INSTANCES_PER_DEVICE;
    if (g_queue_depth < 1)
        g_queue_depth = 1;
    if (g_queue_depth > MAX_QUEUE_DEPTH)
        g_queue_depth = MAX_QUEUE_DEPTH;

    g_buf_cap = g_src_cap > g_dst_cap ? g_src_cap : g_dst_cap;

    g_segment = envs.IAXL_ZIP_SEGMENT;
    if (g_segment < 0)
        g_segment = 0;
    if (g_segment > 0) {
        IAXL_CHECK((uint32_t)g_segment <= g_src_cap,
                   "qat_zip: IAXL_ZIP_SEGMENT must not exceed IAXL_ZIP_SRC_CAP");
        IAXL_CHECK((uint32_t)g_segment <= g_dst_cap,
                   "qat_zip: IAXL_ZIP_SEGMENT must not exceed IAXL_ZIP_DST_CAP");
    }

    const char *dev_sel = envs.IAXL_QAT_DEVICES();

    fprintf(stderr,
            "[config] instances/device=%d  src_cap=%u B  dst_cap=%u B  queue_depth=%d  "
            "devices=%s\n",
            g_instances_per_device, g_src_cap, g_dst_cap, g_queue_depth, dev_sel);

    qat_start();
    discover_all();

    int sel[MAX_DEVICES];
    int nsel = parse_selection(dev_sel, g_dev_count, sel, MAX_DEVICES);

    for (int s = 0; s < nsel; s++) {
        int bus = g_dev_bus[sel[s]];
        int got = 0;
        for (int k = 0; k < g_all_count && got < g_instances_per_device; k++) {
            if (g_all[k].pcie_bus != bus)
                continue;
            Instance *in = &g_inst[g_inst_count];
            instance_init(in, &g_all[k]);
            fprintf(stderr,
                    "[select] instance[%d] on device[%d]: PCIe bus=0x%02x  accel_id=%d  NUMA=%d\n",
                    g_inst_count, sel[s], in->pcie_bus, in->accel_id, in->node);
            g_inst_count++;
            got++;
        }
        if (got < g_instances_per_device)
            fprintf(stderr, "[select] device[%d] only provided %d instance(s) (wanted %d)\n",
                    sel[s], got, g_instances_per_device);
    }
    IAXL_CHECK(g_inst_count > 0, "qat_zip: no usable instance");
    fprintf(stderr, "[qat_zip] init done: %d instance(s) on devices=%s\n", g_inst_count, dev_sel);
    return 0;
}

int qat_zip_num_slots(void) { return g_inst_count * g_queue_depth; }
int qat_zip_queue_depth(void) { return g_queue_depth; }
int qat_zip_src_cap(void) { return (int)g_src_cap; }

static int submit_slot(int slot, int compress, void *src, int len, CpaDcFlush flush) {
    if (slot < 0 || slot >= g_inst_count * g_queue_depth)
        return -1;
    if (!src || len <= 0)
        return -1;
    uint32_t input_cap = compress ? g_src_cap : g_dst_cap;
    uint32_t output_cap = compress ? g_dst_cap : g_src_cap;
    IAXL_CHECK((uint32_t)len <= input_cap, "qat_zip: input length exceeds configured capacity");

    Instance *in = &g_inst[slot / g_queue_depth];
    int si = slot % g_queue_depth;
    Slot *sl = &in->slot[si];

    memcpy(sl->in, src, (size_t)len);
    return submit_op(in, si, compress, sl->in, (uint32_t)len, sl->out, output_cap, flush);
}

/* Submit segment `s` of a multi-segment block on `slot`. Every segment but the
 * last ends the current DEFLATE block (FLUSH_FULL); the last ends the stream
 * (FLUSH_FINAL), so the concatenated outputs form a single valid DEFLATE stream. */
static int submit_segment(Instance *in, int si, int s) {
    Slot *sl = &in->slot[si];
    const int off = s * g_segment;
    int len = sl->seg_total - off;
    if (len > g_segment)
        len = g_segment;
    const CpaDcFlush flush = (s == sl->seg_count - 1) ? CPA_DC_FLUSH_FINAL : CPA_DC_FLUSH_FULL;
    const int slot = (int)(in - g_inst) * g_queue_depth + si;
    return submit_slot(slot, 1, (void *)(sl->seg_src + off), len, flush);
}

int qat_zip_compress(int slot, void *src, int len) {
    if (slot < 0 || slot >= g_inst_count * g_queue_depth || !src || len <= 0)
        return -1;
    if (g_segment > 0 && len > g_segment) {
        Instance *in = &g_inst[slot / g_queue_depth];
        int si = slot % g_queue_depth;
        Slot *sl = &in->slot[si];
        const int nsegs = (len + g_segment - 1) / g_segment;
        /* An incompressible stream can fall back to stored blocks (len + 5 bytes
         * per 65535) and QAT closes an internal block roughly every 16 KB, so
         * 128 B of slack per segment and per 16 KB is ample. */
        const size_t need =
            (size_t)len + ((size_t)nsegs + (size_t)len / 16384 + 1) * 128 + 128;
        if (sl->seg_acc_cap < need) {
            unsigned char *p = realloc(sl->seg_acc, need);
            if (!p)
                return -1;
            sl->seg_acc = p;
            sl->seg_acc_cap = need;
        }
        sl->seg_active = 1;
        sl->seg_total = len;
        sl->seg_count = nsegs;
        sl->seg_next = 1;
        sl->seg_src = (const unsigned char *)src;
        sl->seg_acc_len = 0;
        return submit_segment(in, si, 0);
    }
    return submit_slot(slot, 1, src, len, CPA_DC_FLUSH_FINAL);
}

int qat_zip_decompress(int slot, void *src, int len) {
    return submit_slot(slot, 0, src, len, CPA_DC_FLUSH_FINAL);
}

int qat_zip_wait(int slot, void **dest, int *len) {
    if (slot < 0 || slot >= g_inst_count * g_queue_depth)
        return -1;

    Instance *in = &g_inst[slot / g_queue_depth];
    int si = slot % g_queue_depth;
    Slot *sl = &in->slot[si];

    if (sl->seg_active) {
        for (;;) {
            uint32_t produced = 0;
            if (wait_op(in, si, &produced) != 0) {
                sl->seg_active = 0;
                return -1;
            }
            if (sl->seg_acc_len + produced > sl->seg_acc_cap) {
                sl->seg_active = 0;
                return -1;
            }
            memcpy(sl->seg_acc + sl->seg_acc_len, sl->out, produced);
            sl->seg_acc_len += produced;
            if (sl->seg_next >= sl->seg_count) {
                if (dest)
                    *dest = sl->seg_acc;
                if (len)
                    *len = (int)sl->seg_acc_len;
                sl->seg_active = 0;
                return 0;
            }
            if (submit_segment(in, si, sl->seg_next) != 0) {
                sl->seg_active = 0;
                return -1;
            }
            sl->seg_next++;
        }
    }

    uint32_t produced = 0;
    if (wait_op(in, si, &produced) != 0)
        return -1;
    if (dest)
        *dest = sl->out;
    if (len)
        *len = (int)produced;
    return 0;
}

void qat_zip_shutdown(void) {
    for (int i = 0; i < g_inst_count; i++)
        instance_teardown(&g_inst[i]);
    icp_sal_userStop();
    g_inst_count = 0;
    g_all_count = 0;
    g_dev_count = 0;
}
