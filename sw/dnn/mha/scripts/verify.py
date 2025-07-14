#!/usr/bin/env python3
# Copyright 2023 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Luca Colagrande <colluca@iis.ee.ethz.ch>

import numpy as np
import sys
from datagen import exact_flexfloat_golden_model
import pyflexfloat as ff

from snitch.util.sim.verif_utils import Verifier
from snitch.util.sim.data_utils import ctype_from_precision_t, ff_desc_from_precision_t


class MhaVerifier(Verifier):

    OUTPUT_UIDS = ['O']
    # ERR_THRESHOLD = {4: 1e-2, 2: 1e-1, 1: 3e-1} # 4: 1e-6, 8e-3, 1: 3e-1
    ERR_THRESHOLD = {4: 1e-6, 2: 8e-3, 1: 3e-1} # 4: 1e-6, 8e-3, 1: 3e-1

    def __init__(self):
        super().__init__()
        self.layer_struct = {
            'num_heads': 'I',
            'L': 'I',
            'S': 'I',
            'd': 'I',
            'B_r': 'I',
            'B_c': 'I',
            'dtype': 'I',
            'baseline': 'I',
            'gemm_implementation': 'I',
            'Q': 'I',
            'K': 'I',
            'V': 'I',
            'O': 'I',
            'W_O': 'I'
        }
        self.layer = self.get_input_from_symbol('layer', self.layer_struct)
        self.L = self.layer['L']
        self.S = self.layer['S']
        self.d = self.layer['d']
        self.B_r = self.layer['B_r']
        self.B_c = self.layer['B_c']
        self.prec = self.layer['dtype']
        self.num_heads = self.layer['num_heads']
        self.W_O = self.layer['W_O']    
        self.Q = [self.get_input_from_symbol('Q_' + str(head), ctype_from_precision_t(self.prec)) for head in range(self.num_heads)]

    def get_actual_results(self):
        print(f"@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ num_heads = {self.num_heads}")
        # print(f"@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ W_O = {self.W_O!r}")
        # print(f"@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ Q = {self.Q!r}")
        print(f"@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ prec = {self.prec}")

        return self.get_output_from_symbol(self.OUTPUT_UIDS[0], ctype_from_precision_t(self.prec))

    def get_expected_results(self):
        O = []
        for head in range(self.num_heads):
            Q = self.get_input_from_symbol('Q_' + str(head), ctype_from_precision_t(self.prec))
            K = self.get_input_from_symbol('K_' + str(head), ctype_from_precision_t(self.prec))
            V = self.get_input_from_symbol('V_' + str(head), ctype_from_precision_t(self.prec))
            
            # convert Q, K, V to float using ff.FlexFloat.__float__
            Q_f = np.array([q.__float__() for q in Q])
            K_f = np.array([k.__float__() for k in K])
            V_f = np.array([v.__float__() for v in V])
            
            # Q = torch.from_numpy(Q.reshape(self.L, self.d))
            # V = torch.from_numpy(V.reshape(self.S, self.d))
            # K = torch.from_numpy(K.reshape(self.S, self.d))
            ff_desc = ff_desc_from_precision_t(self.prec)
            Q = ff.array(Q_f.reshape(self.L, self.d), ff_desc)
            V = ff.array(V_f.reshape(self.S, self.d), ff_desc)
            K = ff.array(K_f.reshape(self.S, self.d), ff_desc)
            # return torch_golden_model(Q, K, V).detach().numpy().flatten()
            # return exact_golden_model(Q, K, V, self.B_r, self.B_c).flatten()
            O.append(exact_flexfloat_golden_model(Q, K, V, self.B_r, self.B_c, ff_desc))

        W_O = self.get_input_from_symbol('W_O', ctype_from_precision_t(self.prec))
        W_O_f = np.array([w.__float__() for w in W_O])
        W_O = ff.array(W_O_f.reshape(self.d*self.num_heads, self.d), ff_desc_from_precision_t(self.prec))

        O = np.concatenate(O, axis=1) @ W_O

        return O.flatten()

    def check_results(self, *args):
        return super().check_results(*args, rtol=self.ERR_THRESHOLD[self.prec])


if __name__ == "__main__":
    sys.exit(MhaVerifier().main())
