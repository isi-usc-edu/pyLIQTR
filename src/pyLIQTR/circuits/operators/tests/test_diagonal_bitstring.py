"""
DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.

This material is based upon work supported by the Under Secretary of Defense for
Research and Engineering under Air Force Contract No. FA8702-15-D-0001. Any opinions,
findings, conclusions or recommendations expressed in this material are those of the
author(s) and do not necessarily reflect the views of the Under Secretary of Defense
for Research and Engineering.

© 2022 Massachusetts Institute of Technology.

The software/firmware is provided to you on an As-Is basis

Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS Part
252.227-7013 or 7014 (Feb 2014). Notwithstanding any copyright notice, U.S. Government
rights in this work are defined by DFARS 252.227-7013 or DFARS 252.227-7014 as detailed
above. Use of this work other than as specifically authorized by the U.S. Government
may violate any copyrights that exist in this work.
"""

import cirq
import pytest
import qualtran
import warnings
import numpy as np

from ..diagonal_bitstring import DiagonalBitstring
from qualtran._infra.gate_with_registers import get_named_qubits
from pyLIQTR.utils.circuit_decomposition import circuit_decompose_multi
from qualtran.cirq_interop.t_complexity_protocol import t_complexity

class TestDiagonalBitstring:
    """
    Test functionality of DiagonalBitstring by comparing
    circuit generated to analytical unitaries as well as
    T counts at different depth decompositions.

    TODO: unoptimized (n, K) flag is not currently tested
    """

    @pytest.fixture(scope = "class")
    def getVector(self):
        return ((5, 7), (15, 31), (8, (2, 3)))


    @pytest.fixture(scope = "class")
    def getUnitary(self, getVector):

        anc_reg = qualtran.Register(name = "ancilla", bitsize = 1)
        data_reg0 = qualtran.Register(name = "data", bitsize = getVector[0][0] - 1)
        data_reg1 = qualtran.Register(name = "data", bitsize = getVector[0][1] - 1)
        
        signature0 = qualtran.Signature([anc_reg, data_reg0])
        signature1 = qualtran.Signature([anc_reg, data_reg1])
        registers0 = get_named_qubits(signature0)
        registers1 = get_named_qubits(signature1)

        DBString0 = DiagonalBitstring(getVector[0][0] - 1, (0, getVector[1][0]))
        DBString1 = DiagonalBitstring(getVector[0][1] - 1, (1, getVector[1][1]))

        return [cirq.Circuit(DBString0.on_registers(**registers0)).unitary(),
                cirq.Circuit(DBString1.on_registers(**registers1)).unitary()]


    def testUnitary0(self, getUnitary, getVector):

        bases0 = 2**getVector[0][0]
        bases1 = 2**getVector[0][1]

        test0 = []
        test1 = []
        for row in range(bases0):
            test0.append([])
            for col in range(bases0):
                if col == row and col != getVector[1][0] and col != getVector[1][0]+bases0/2:
                    test0[row].append(1)
                elif (col == row + bases0/2 and col == getVector[1][0]+bases0/2) or (col == row - bases0/2 and col == getVector[1][0]):
                    test0[row].append(1)
                else:
                    test0[row].append(0)
        for row in range(bases1):
            test1.append([])
            for col in range(bases1):
                if (col == row + bases1/2 or col == row - bases1/2) and col != getVector[1][1] and col != getVector[1][1]+bases1/2:
                    test1[row].append(1)
                elif col == row and (col == getVector[1][1] or col == getVector[1][1]+bases1/2):
                    test1[row].append(1)
                else:
                    test1[row].append(0)

        np.testing.assert_allclose(test0, getUnitary[0])
        np.testing.assert_allclose(test1, getUnitary[1])


    def testTCount(self, getVector):

        anc_reg = qualtran.Register(name = "ancilla", bitsize = 1)
        data_reg = qualtran.Register(name = "data", bitsize = getVector[2][0] - 1)

        signature = qualtran.Signature([anc_reg, data_reg])
        registers = get_named_qubits(signature)

        DBString = DiagonalBitstring(getVector[2][0] - 1, getVector[2][1], True)

        topLevel = cirq.Circuit(DBString.on_registers(**registers))
        decomposeLevel = circuit_decompose_multi(topLevel, 1)

        assert(t_complexity(topLevel).t == t_complexity(decomposeLevel).t)