"""
Copyright (c) 2024 Massachusetts Institute of Technology 
SPDX-License-Identifier: BSD-2-Clause
"""
from fable_functions import *
from circuit_files import *
import numpy as np
import scipy as sp
import random
import time
import cirq
import pytest

@pytest.mark.skip
class TestFableFunctions:
    def test_image_creation(self):
        # Moving the Qiskit import down here so it won't get flagged by pytest while we are running the general testing
        import qiskit
        n=5
        [i1,i2,v] = gen_sparse(n,2)
        #A1 = sparse_to_mat(i1,i2,v,2**n)
        #A3 = ls_block(A1)

        #print(i1)
        #print(i2)
        #print(v)

        t0 = time.time()
        LS = LS_FALBE_full(i1,i2,v,n)
        t1 = time.time()

        A = cirq.unitary(LS)
        A = np.real(A)
        #A2 = (2**n) * A[0:2**n,0:2**n]

        #print(l2norm(A3-A2))
        #print(l2norm(A1-A3))

        print(t1-t0)

        m = np.max(A)
        A = A/m

        im = Image.fromarray(np.uint8(cm.gist_earth(A)*255))
        im.save()
