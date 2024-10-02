"""
Copyright (c) 2024 Massachusetts Institute of Technology 
SPDX-License-Identifier: BSD-2-Clause
"""
import pytest
import numpy as np
import cirq

from pyLIQTR.circuits.pyLOperator import pyLOperator as po


class TestPyLOperator:
    @pytest.fixture(scope="class")
    def single_gate(self):
        qubits = cirq.LineQubit.range(1)
        single_gate = cirq.X.on(qubits[0])

        yield single_gate
        del single_gate

    def test_base_class_pyloperator_creation(self, single_gate):
        """This test will simply try and create a single gate operator without using the child class that instantiates the abstract methods"""
        with pytest.raises(TypeError):
            # should raise the following error [TypeError: Can't instantiate abstract class pyLOperator with abstract methods _num_qubits_, _qid_shape_, get_resouces, num_qubits]
            plo = po(single_gate)
