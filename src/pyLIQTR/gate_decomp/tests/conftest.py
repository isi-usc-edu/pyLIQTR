"""
Copyright (c) 2024 Massachusetts Institute of Technology 
SPDX-License-Identifier: BSD-2-Clause
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
