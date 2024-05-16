import inspect
from typing import List
import pytest
import cirq

import pyLIQTR.circuits.operators.prepare as prep


def get_common_gate_ops() -> List[cirq.GateOperation]:
    """Generate a list of GateOperations from cirq's list of common ops."""
    # Get cirq's common gate names
    gate_names = [
        name for name, c in inspect.getmembers(cirq) if isinstance(c, cirq.Gate)
    ]

    ops = []
    for gate_name in gate_names:
        # Gate class and gate's required num of qubits
        clss = getattr(cirq, gate_name)
        num_q = clss.num_qubits()

        # Create gate operation and store in list of ops
        ops.append(clss(*[cirq.LineQubit(i) for i in range(num_q)]))
    return ops


class TestPrepare:
    @pytest.mark.parametrize("op", get_common_gate_ops())
    def test_prepare_comparison_to_common_ops(self, op):
        """Should be False when comparing the prepare operation to common cirq operations.
        Tests issue #9.
        """
        # Create arbitrary prepare operation
        qbs = [cirq.LineQubit(i) for i in range(2)]  # arbitrary
        alphas = [0.1, 0.2, 0.3, 0.4]  # arbitrary
        op1 = prep.Prepare(qbs, alphas).on(*qbs)
        op2 = op
        assert op1 != op2
