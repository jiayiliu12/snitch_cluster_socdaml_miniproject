// Copyright 2023 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Luca Colagrande <colluca@iis.ee.ethz.ch>

#include "blas.h"
#include "dnn.h"

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wreorder-init-list"
#include "mha_data.h"
#pragma clang diagnostic pop



int main() {
    if (snrt_cluster_idx()==0) {
        mha_layer(layer_0);
    }
    if (snrt_cluster_idx()==1) {
        mha_layer(layer_1);
    }
    if (snrt_cluster_idx()==2) {
        mha_layer(layer_2);
    } 
    return 0;
}
