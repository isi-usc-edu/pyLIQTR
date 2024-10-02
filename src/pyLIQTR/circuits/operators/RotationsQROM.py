"""
Copyright (c) 2024 Massachusetts Institute of Technology 
SPDX-License-Identifier: BSD-2-Clause
"""
import cirq
from typing import Tuple, Callable
from numpy.typing import NDArray
from qualtran._infra.gate_with_registers import total_bits
from qualtran.bloqs.basic_gates import CNOT

from pyLIQTR.circuits.operators.QROMwithMeasurementUncompute import QROMwithMeasurementUncompute

class RotationsQROM(QROMwithMeasurementUncompute):
    '''
    Modify QROM for case when data[l] is an array of length N with elements 0 or 1. When the selection register stores index l, data[l] is output on the target register (which has N qubits).
    '''

    def _load_nth_data(
        self,
        selection_idx: Tuple[int, ...],
        gate: Callable[[cirq.Qid], cirq.Operation],
        **target_regs: NDArray[cirq.Qid],
    ) -> cirq.OP_TREE:
        for i, d in enumerate(self.data):
            target = target_regs.get(f'target{i}_', ())
            for q, bit in zip(target, d[selection_idx]):
                if int(bit):
                    yield gate(q)

    def _circuit_diagram_info_(self, _) -> cirq.CircuitDiagramInfo:
        wire_symbols = ["@"] * self.num_controls
        wire_symbols += ["In"] * total_bits(self.selection_registers)
        for i, target in enumerate(self.target_registers):
            wire_symbols += [f"RotationsQROM_{i}"] * target.total_bits()
        return cirq.CircuitDiagramInfo(wire_symbols=wire_symbols)

    def nth_operation_callgraph(self, **kwargs: int):
        selection_idx = tuple(kwargs[reg.name] for reg in self.selection_registers)
        return {(CNOT(), sum( int(sum(d[selection_idx])) for d in self.data ))}