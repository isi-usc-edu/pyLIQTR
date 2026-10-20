import cirq.protocols
import pytest
import numpy as np

import cirq
from cirq.circuits.circuit import Circuit
from cirq.ops.greedy_qubit_manager import GreedyQubitManager
from cirq.ops.qubit_manager import SimpleQubitManager
from cirq.ops.named_qubit import NamedQubit
from qualtran.bloqs.state_preparation.prepare_uniform_superposition import PrepareUniformSuperposition
from cirq.protocols.inverse_protocol import inverse
from cirq.protocols.decompose_protocol import decompose, DecompositionContext

def assert_inverse_preparation(decomposed, control, target):
    circuit = cirq.Circuit(decomposed)
    extra = sorted(circuit.all_qubits() - set(control + target))
    qubits = control + target + extra
    initial = np.zeros(2**len(qubits), dtype=np.complex128)
    for value in range(3):
        initial[value * 2**len(extra)] = 1 / np.sqrt(3)
    result = cirq.Simulator(dtype=np.complex128).simulate(
        circuit, qubit_order=qubits, initial_state=initial
    )
    assert abs(result.final_state_vector[0]) == pytest.approx(1, abs=1e-12)
    assert all(len(op.qubits) == len(set(op.qubits)) for op in decomposed)

class TestGreedyQubitManager:
    def test_gqm(self):
        n=3
        target = NamedQubit.range((n-1).bit_length(),prefix='target')
        control = [NamedQubit('control')]
        circuit_error = inverse(Circuit(PrepareUniformSuperposition(n,cvs=(0,)).on_registers(ctrl=control,target=target)))

        gqm = GreedyQubitManager(prefix="_ancilla", maximize_reuse=True)
        decomposed_circuit = decompose(circuit_error,context = DecompositionContext(gqm))
        assert_inverse_preparation(decomposed_circuit, control, list(target))
        
    def test_sqm(self):
        n=3
        target = NamedQubit.range((n-1).bit_length(),prefix='target')
        control = [NamedQubit('control')]
        circuit_error = inverse(Circuit(PrepareUniformSuperposition(n,cvs=(0,)).on_registers(ctrl=control,target=target)))

        gqm = SimpleQubitManager(prefix="_ancilla")
        decomposed_circuit = decompose(circuit_error,context = DecompositionContext(gqm))
        assert_inverse_preparation(decomposed_circuit, control, list(target))