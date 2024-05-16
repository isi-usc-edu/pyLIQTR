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
import pytest
import numpy as np
import cirq

"""
Some testing ideas to implement:
create valid circuit [should pass]
attempt to create valid circuit [should fail]
add valid gate to circuit [pass/fail]
add invalid (not sure what this would be just yet) gate to circuit [fail]
add multiple gates to circuit [pass]
add large number of gates to circuit [pass]
can be printed [small/large]
valid circuit can be decomposed into gate(s) [pass]
empty circuit can be decomposed into gate(s) [fail?]
"""
from pyLIQTR.circuits.pyLCircuit import pyLCircuit as circuit
from pyLIQTR.gate_decomp.cirq_transforms import get_approximate_t_depth
from pyLIQTR.utils.resource_analysis import estimate_resources

class TestPylCircuit:
    @pytest.fixture(scope="class")
    def single_gate(self):
        qubits = cirq.LineQubit.range(1)
        single_gate = cirq.X.on(qubits[0])

        yield single_gate
        del single_gate

    @pytest.fixture(scope="class")
    def ten_gates(self):
        qubits = cirq.LineQubit.range(10)
        ten_gates = []
        for q in qubits:
            ten_gates.append(cirq.X.on(q))

        yield ten_gates
        del ten_gates

    @pytest.fixture(scope="class")
    def hundred_gates(self):
        qubits = cirq.LineQubit.range(100)
        hundred_gates = cirq.X.on(qubits)

        yield hundred_gates
        del hundred_gates

    @pytest.fixture(scope="class")
    def thousand_gates(self):
        qubits = cirq.LineQubit.range(1000)
        thousand_gates = cirq.X.on(qubits)

        yield thousand_gates
        del thousand_gates

    def test_pylcircuit_init(self):
        print("")

    @pytest.mark.skip()
    def test_single_gate_depth(self, single_gate):
        t_depth = get_approximate_t_depth(single_gate)
        assert t_depth == 109

    def test_single_gate_resources(self, single_gate):
        resources = estimate_resources(single_gate)
        assert resources is not None
        assert len(resources) == 3
        assert resources['LogicalQubits'] == 1
        assert resources['T'] == 0
        assert resources['Clifford'] == 1

    @pytest.mark.skip()
    def test_ten_gate_resources(self, ten_gates):
        resources = []
        for c in ten_gates:
            resources.append(estimate_resources(c))
        assert resources is not None
        assert len(resources) == 3
        assert resources['LogicalQubits'] == 1
        assert resources['T'] == 0
        assert resources['Clifford'] == 1

    @pytest.mark.skip()
    def test_hundred_gate_resources(self, hundred_gates):
        resources = estimate_resources(hundred_gates)
        assert resources is not None
        assert len(resources) == 3
        assert resources['LogicalQubits'] == 1
        assert resources['T'] == 0
        assert resources['Clifford'] == 1

    @pytest.mark.skip()
    def test_thousand_gate_resources(self, thousand_gates):
        resources = estimate_resources(thousand_gates)
        assert resources is not None
        assert len(resources) == 3
        assert resources['LogicalQubits'] == 1
        assert resources['T'] == 0
        assert resources['Clifford'] == 1
    
