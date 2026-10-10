// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <zlib.h>

#include "cpu_zip.h"
#include "env.h"
#include "iaa_zip.h"
#include "qat_zip.h"

static int raw_deflate(const unsigned char *src, int src_len, unsigned char *dst, int *dst_len,
                       int window_bits) {
    z_stream stream = {0};
    if (deflateInit2(&stream, Z_DEFAULT_COMPRESSION, Z_DEFLATED, window_bits, 8,
                     Z_DEFAULT_STRATEGY) != Z_OK)
        return -1;
    stream.next_in = (Bytef *)src;
    stream.avail_in = (uInt)src_len;
    stream.next_out = dst;
    stream.avail_out = (uInt)*dst_len;
    int rc = deflate(&stream, Z_FINISH);
    *dst_len = (int)stream.total_out;
    deflateEnd(&stream);
    return rc == Z_STREAM_END ? 0 : -1;
}

static int raw_inflate(const unsigned char *src, int src_len, unsigned char *dst, int *dst_len) {
    z_stream stream = {0};
    if (inflateInit2(&stream, -MAX_WBITS) != Z_OK)
        return -1;
    stream.next_in = (Bytef *)src;
    stream.avail_in = (uInt)src_len;
    stream.next_out = dst;
    stream.avail_out = (uInt)*dst_len;
    int rc = inflate(&stream, Z_FINISH);
    *dst_len = (int)stream.total_out;
    inflateEnd(&stream);
    return rc == Z_STREAM_END ? 0 : -1;
}

static int verify(const char *name, const unsigned char *expected, int expected_len, void *actual,
                  int actual_len) {
    if (actual_len != expected_len || memcmp(actual, expected, (size_t)expected_len) != 0) {
        fprintf(stderr, "[compat] %s FAILED: expected %d bytes, got %d\n", name, expected_len,
                actual_len);
        return -1;
    }
    printf("[compat] %s passed\n", name);
    return 0;
}

int main(int argc, char **argv) {
    static const int input_len = 64 * 1024;
    const char *devices = argc > 1 ? argv[1] : NULL;
    int status = 1;
    int failures = 0;
    unsigned char *input = NULL;
    unsigned char *qat_data = NULL;
    unsigned char *cpu_data = NULL;
    unsigned char *cpu4k_data = NULL;
    unsigned char *iaa_data = NULL;
    unsigned char *raw_data = NULL;

    if (devices)
        setenv("IAXL_QAT_DEVICES", devices, 1);
    setenv("IAXL_CPU_ZIP_THREADS", "1", 1);
    setenv("IAXL_IAA_ZIP_ENABLE", "1", 1);
    setenv("IAXL_IAA_ZIP_INSTANCES_PER_DEVICE", "1", 1);
    setenv("IAXL_IAA_ZIP_QUEUE_DEPTH", "1", 1);
    setenv("IAXL_ZIP_SRC_CAP", "262144", 1);
    setenv("IAXL_ZIP_DST_CAP", "262144", 1);
    envs_init();

    input = malloc((size_t)input_len);
    if (!input)
        goto out;
    // A 16 KB pseudo-random pattern repeated: matches sit far beyond a 4 KB window, so
    // this catches compressors that emit distances Intel QPL cannot decode.
    unsigned int seed = 12345;
    for (int i = 0; i < 16 * 1024; i++) {
        seed = seed * 1103515245u + 12345u;
        input[i] = (unsigned char)(seed >> 16);
    }
    for (int i = 16 * 1024; i < input_len; i++)
        input[i] = input[i - 16 * 1024];

    if (qat_zip_init() != 0 || cpu_zip_init() != 0 || iaa_zip_init() != 0) {
        fprintf(stderr, "[compat] backend initialization failed\n");
        goto out;
    }

    void *output;
    int output_len;
    int rc = qat_zip_compress(0, input, input_len);
    if (rc != 0) {
        fprintf(stderr, "[compat] QAT compression submission failed: %d\n", rc);
        goto out;
    }
    rc = qat_zip_wait(0, &output, &output_len, 0);
    if (rc != 0 || output_len <= 0) {
        fprintf(stderr, "[compat] QAT compression completion failed: status=%d length=%d\n", rc,
                output_len);
        goto out;
    }
    if (!output) {
        fprintf(stderr, "[compat] QAT compression failed\n");
        goto out;
    }
    qat_data = malloc((size_t)output_len);
    if (!qat_data)
        goto out;
    memcpy(qat_data, output, (size_t)output_len);
    int qat_len = output_len;

    rc = cpu_zip_decompress(0, qat_data, qat_len);
    if (rc != 0) {
        fprintf(stderr, "[compat] QAT compress -> CPU decompress rejected: %d\n", rc);
        failures++;
    } else {
        rc = cpu_zip_wait(0, &output, &output_len, 0);
        if (rc != 0 ||
            verify("QAT compress -> CPU decompress", input, input_len, output, output_len) != 0)
            failures++;
    }

    rc = cpu_zip_compress(0, input, input_len);
    if (rc != 0 || (rc = cpu_zip_wait(0, &output, &output_len, 0)) != 0 || output_len <= 0) {
        fprintf(stderr, "[compat] CPU compression failed: status=%d length=%d\n", rc, output_len);
        goto out;
    }
    cpu_data = malloc((size_t)output_len);
    if (!cpu_data)
        goto out;
    memcpy(cpu_data, output, (size_t)output_len);
    int cpu_len = output_len;

    rc = qat_zip_decompress(0, cpu_data, cpu_len);
    if (rc != 0) {
        fprintf(stderr, "[compat] CPU compress -> QAT decompress submission failed: %d\n", rc);
        failures++;
    } else {
        rc = qat_zip_wait(0, &output, &output_len, 0);
        if (rc != 0) {
            fprintf(stderr, "[compat] CPU compress -> QAT decompress completion failed: %d\n", rc);
            failures++;
        } else if (verify("CPU compress -> QAT decompress", input, input_len, output, output_len) !=
                   0) {
            failures++;
        }
    }

    if (failures == 0) {
        printf("[compat] QAT and CPU compressed streams are mutually compatible\n");
    } else {
        fprintf(stderr, "[compat] QAT and CPU compressed streams are not mutually compatible\n");
    }

    rc = iaa_zip_compress(0, input, input_len);
    if (rc != 0 || (rc = iaa_zip_wait(0, &output, &output_len, 0)) != 0 || output_len <= 0) {
        fprintf(stderr, "[compat] IAA compression failed: status=%d length=%d\n", rc, output_len);
        goto out;
    }
    iaa_data = malloc((size_t)output_len);
    if (!iaa_data)
        goto out;
    memcpy(iaa_data, output, (size_t)output_len);
    int iaa_len = output_len;

    rc = cpu_zip_decompress(0, iaa_data, iaa_len);
    if (rc != 0 || (rc = cpu_zip_wait(0, &output, &output_len, 0)) != 0 ||
        verify("IAA compress -> CPU decompress", input, input_len, output, output_len) != 0) {
        fprintf(stderr, "[compat] IAA compress -> CPU decompress failed: %d\n", rc);
        failures++;
    }

    rc = qat_zip_decompress(0, iaa_data, iaa_len);
    if (rc != 0 || (rc = qat_zip_wait(0, &output, &output_len, 0)) != 0 ||
        verify("IAA compress -> QAT decompress", input, input_len, output, output_len) != 0) {
        fprintf(stderr, "[compat] IAA compress -> QAT decompress failed: %d\n", rc);
        failures++;
    }

    // The invariant kv_zip is built on: QAT streams must never reach an IAA worker.
    rc = iaa_zip_decompress(0, qat_data, qat_len);
    if (rc == 0)
        rc = iaa_zip_wait(0, &output, &output_len, 0);
    if (rc == 0) {
        fprintf(stderr, "[compat] IAA unexpectedly decoded a QAT stream\n");
        failures++;
    } else {
        printf("[compat] IAA rejects QAT streams, as kv_zip assumes\n");
    }

    rc = iaa_zip_decompress(0, cpu_data, cpu_len);
    if (rc == 0)
        rc = iaa_zip_wait(0, &output, &output_len, 0);
    if (rc == 0) {
        fprintf(stderr, "[compat] IAA unexpectedly decoded a 32 KB-window CPU stream\n");
        failures++;
    } else {
        printf("[compat] IAA rejects 32 KB-window CPU streams, as kv_zip assumes\n");
    }

    // Without QAT the CPU backend shrinks to a 4 KB window so IAA can take over
    // decompression; emulate that configuration for one round.
    envs.IAXL_QAT_ZIP_ENABLE = false;
    rc = cpu_zip_compress(0, input, input_len);
    if (rc == 0)
        rc = cpu_zip_wait(0, &output, &output_len, 0);
    envs.IAXL_QAT_ZIP_ENABLE = true;
    if (rc != 0 || output_len <= 0) {
        fprintf(stderr, "[compat] 4 KB-window CPU compression failed: %d\n", rc);
        goto out;
    }
    cpu4k_data = malloc((size_t)output_len);
    if (!cpu4k_data)
        goto out;
    memcpy(cpu4k_data, output, (size_t)output_len);
    int cpu4k_len = output_len;

    rc = iaa_zip_decompress(0, cpu4k_data, cpu4k_len);
    if (rc != 0 || (rc = iaa_zip_wait(0, &output, &output_len, 0)) != 0 ||
        verify("CPU compress (4 KB window) -> IAA decompress", input, input_len, output,
               output_len) != 0) {
        fprintf(stderr, "[compat] 4 KB CPU compress -> IAA decompress failed: %d\n", rc);
        failures++;
    }

    int raw_cap = envs.IAXL_ZIP_SRC_CAP > envs.IAXL_ZIP_DST_CAP ? envs.IAXL_ZIP_SRC_CAP
                                                                : envs.IAXL_ZIP_DST_CAP;
    raw_data = malloc((size_t)raw_cap);
    if (!raw_data)
        goto out;
    int raw_len = envs.IAXL_ZIP_SRC_CAP;
    if (raw_inflate(qat_data, qat_len, raw_data, &raw_len) != 0 ||
        verify("QAT compress -> raw zlib inflate", input, input_len, raw_data, raw_len) != 0)
        goto out;

    raw_len = envs.IAXL_ZIP_DST_CAP;
    if (raw_deflate(input, input_len, raw_data, &raw_len, -MAX_WBITS) != 0) {
        fprintf(stderr, "[compat] raw zlib deflate failed\n");
        goto out;
    }
    raw_len = envs.IAXL_ZIP_SRC_CAP;
    if (raw_inflate(iaa_data, iaa_len, raw_data, &raw_len) != 0 ||
        verify("IAA compress -> raw zlib inflate", input, input_len, raw_data, raw_len) != 0)
        goto out;

    // Intel QPL only decodes a 4 KB history window, so the reference stream must use one.
    raw_len = envs.IAXL_ZIP_DST_CAP;
    if (raw_deflate(input, input_len, raw_data, &raw_len, -12) != 0) {
        fprintf(stderr, "[compat] raw zlib deflate failed\n");
        goto out;
    }
    rc = qat_zip_decompress(0, raw_data, raw_len);
    if (rc != 0 || (rc = qat_zip_wait(0, &output, &output_len, 0)) != 0 ||
        verify("raw zlib deflate -> QAT decompress", input, input_len, output, output_len) != 0) {
        fprintf(stderr, "[compat] raw zlib -> QAT failed: %d\n", rc);
        goto out;
    }
    rc = iaa_zip_decompress(0, raw_data, raw_len);
    if (rc != 0 || (rc = iaa_zip_wait(0, &output, &output_len, 0)) != 0 ||
        verify("raw zlib deflate -> IAA decompress", input, input_len, output, output_len) != 0) {
        fprintf(stderr, "[compat] raw zlib -> IAA failed: %d\n", rc);
        goto out;
    }
    printf("[compat] QAT, IAA and CPU all use raw DEFLATE and interoperate via zlib raw mode\n");
    printf("[compat] note: compatibility is one-way. IAA decodes at most a 4 KB history window, "
           "so QAT and CPU decompress anything but IAA only decodes IAA and 4 KB-window CPU "
           "streams\n");
    if (failures == 0)
        status = 0;

out:
    free(raw_data);
    free(iaa_data);
    free(cpu4k_data);
    free(cpu_data);
    free(qat_data);
    free(input);
    iaa_zip_shutdown();
    cpu_zip_shutdown();
    qat_zip_shutdown();
    return status;
}