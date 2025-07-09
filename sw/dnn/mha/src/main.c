// Copyright 2023 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Luca Colagrande <colluca@iis.ee.ethz.ch>

// #include "blas.h"
#include "dnn.h" // Includes mha.h and snrt.h

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wreorder-init-list"
#include "data.h"
#pragma clang diagnostic pop



int main() {
    if (snrt_is_compute_core() && snrt_cluster_idx() < num_heads) {
        mha_layer(*layers[snrt_cluster_idx()], num_heads, (uint32_t *)W_O);
    }
    return 0;
}
