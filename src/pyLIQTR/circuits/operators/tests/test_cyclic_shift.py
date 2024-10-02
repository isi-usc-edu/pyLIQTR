"""
Copyright (c) 2024 Massachusetts Institute of Technology 
SPDX-License-Identifier: BSD-2-Clause
"""


import cirq
import pytest
import numpy as np

from ..cyclic_shift import CyclicShiftPermutation
from qualtran._infra.gate_with_registers import get_named_qubits

class TestTranslation:
    """
    Test functionality of cyclicShift by comparing
    circuit generated to analytical unitaries.
    
    TODO: not nearly enough test coverage
    """

    @pytest.fixture(scope = "class")
    def getVector(self):
        return ((5, 7), ("left", "right"))


    @pytest.fixture(scope = "class")
    def getUnitary(self, getVector):

        shiftL = CyclicShiftPermutation(getVector[0][0], getVector[1][0], (False, 0), False)
        shiftR = CyclicShiftPermutation(getVector[0][1], getVector[1][1], (False, 0), False)

        registersL = get_named_qubits(shiftL.signature)
        registersR = get_named_qubits(shiftR.signature)

        return [cirq.Circuit(shiftL.on_registers(**registersL)).unitary(),
                cirq.Circuit(shiftR.on_registers(**registersR)).unitary()]

    def testUnitaryCS(self, getUnitary, getVector):

        basesL = 2**getVector[0][0]
        basesR = 2**getVector[0][1]

        left_test = []
        right_test = []
        for row in range(basesL):
            left_test.append([])
            for col in range(basesL):
                if col == basesL - 1 + row:
                    left_test[row].append(1)
                elif col == row - 1:
                    left_test[row].append(1)
                else:
                    left_test[row].append(0)
        
        for row in range(basesR):
            right_test.append([])
            for col in range(basesR):
                if row == basesR - 1 + col:
                    right_test[row].append(1)
                elif col == row + 1:
                    right_test[row].append(1)
                else:
                    right_test[row].append(0)

        np.testing.assert_allclose(left_test, getUnitary[0])
        np.testing.assert_allclose(right_test, getUnitary[1])