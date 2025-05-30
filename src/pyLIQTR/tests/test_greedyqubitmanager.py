import cirq.protocols
import pytest

import cirq
from cirq.circuits.circuit import Circuit
from cirq.ops.greedy_qubit_manager import GreedyQubitManager
from cirq.ops.qubit_manager import SimpleQubitManager
from cirq.ops.named_qubit import NamedQubit
from qualtran.bloqs.state_preparation.prepare_uniform_superposition import PrepareUniformSuperposition
from cirq.protocols.inverse_protocol import inverse
from cirq.protocols.decompose_protocol import decompose, DecompositionContext

class TestGreedyQubitManager:
    def test_gqm(self):
        n=3
        target = NamedQubit.range((n-1).bit_length(),prefix='target')
        control = [NamedQubit('control')]
        circuit_error = inverse(Circuit(PrepareUniformSuperposition(n,cvs=(0,)).on_registers(ctrl=control,target=target)))

        # TODO: This was asserting a ValueError: duplicate qids, but I'm not sure why it's
        #       supposed to be broken
        gqm = GreedyQubitManager(prefix="_ancilla", maximize_reuse=True)
        decomposed_circuit = decompose(circuit_error, context=DecompositionContext(gqm))
        assert len(decomposed_circuit) == 290
        
    def test_sqm(self):
        n=3
        target = NamedQubit.range((n-1).bit_length(),prefix='target')
        control = [NamedQubit('control')]
        circuit_error = inverse(Circuit(PrepareUniformSuperposition(n,cvs=(0,)).on_registers(ctrl=control,target=target)))

        gqm = SimpleQubitManager(prefix="_ancilla")
        decomposed_circuit = decompose(circuit_error,context = DecompositionContext(gqm))
        assert len(decomposed_circuit) == 290  # TODO: verify