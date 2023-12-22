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
import cirq


@pytest.fixture(scope="class")
def test_circuit_4_qubits(request):
    q0, q1, q2, q3 = cirq.LineQubit.range(4)
    test_circuit_4_qubits = cirq.Circuit()
    test_circuit_4_qubits.append(
        [
            cirq.CX(q0, q1),
            cirq.T(q2),
            cirq.S(q3),
            cirq.T(q0),
            cirq.CCX(q3, q2, q1),
            cirq.CCZ(q0, q1, q2),
            cirq.rz(0.431).on(q3),
            cirq.X(q0),
            cirq.CX(q2, q1),
            cirq.Z(q3),
        ]
    )
    request.cls.test_circuit_4_qubits = test_circuit_4_qubits

@pytest.fixture(scope="class")
def test_circuit_no_rz(request):
    q0, q1, q2, q3 = cirq.LineQubit.range(4)
    test_circuit_no_rz = cirq.Circuit()
    test_circuit_no_rz.append(
        [
            cirq.CX(q0, q1),
            cirq.T(q2),
            cirq.S(q3),
            cirq.T(q0),
            cirq.CCX(q3, q2, q1),
            cirq.CCZ(q0, q1, q2),
            cirq.X(q0),
            cirq.CX(q2, q1),
            cirq.Z(q3),
        ]
    )
    request.cls.test_circuit_no_rz = test_circuit_no_rz
