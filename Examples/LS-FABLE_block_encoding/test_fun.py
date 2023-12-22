"""
DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.

This material is based upon work supported by the Under Secretary of Defense for
Research and Engineering under Air Force Contract No. FA8702-15-D-0001. Any opinions,
findings, conclusions or recommendations expressed in this material are those of the
author(s) and do not necessarily reflect the views of the Under Secretary of Defense
for Research and Engineering.

© 2023 Massachusetts Institute of Technology.

The software/firmware is provided to you on an As-Is basis

Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS Part
252.227-7013 or 7014 (Feb 2014). Notwithstanding any copyright notice, U.S. Government
rights in this work are defined by DFARS 252.227-7013 or DFARS 252.227-7014 as detailed
above. Use of this work other than as specifically authorized by the U.S. Government
may violate any copyrights that exist in this work.
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
