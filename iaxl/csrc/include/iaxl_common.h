// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#ifndef IAXL_COMMON_H
#define IAXL_COMMON_H

#include <stdio.h>
#include <stdlib.h>

// Returned by a zip backend's non-blocking wait while the request is still in flight.
#define IAXL_ZIP_PENDING 1

#define IAXL_CHECK(condition, message)                                                             \
    do {                                                                                           \
        if (!(condition)) {                                                                        \
            fprintf(stderr, "IAXL check failed: %s (%s) at %s:%d\n", #condition, (message),        \
                    __FILE__, __LINE__);                                                           \
            abort();                                                                               \
        }                                                                                          \
    } while (0)

#endif