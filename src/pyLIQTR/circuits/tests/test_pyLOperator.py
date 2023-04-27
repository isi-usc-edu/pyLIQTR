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
