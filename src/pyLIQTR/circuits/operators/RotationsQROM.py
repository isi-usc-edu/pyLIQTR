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
import cirq
from cirq._compat import cached_property
from typing import Tuple, Callable
from numpy.typing import NDArray
from qualtran._infra.gate_with_registers import total_bits

from pyLIQTR.circuits.operators.QROMwithMeasurementUncompute import QROMwithMeasurementUncompute

class RotationsQROM(QROMwithMeasurementUncompute):
    '''
    Modify QROM for case when data[l] is an array of length N with elements 0 or 1. When the selection register stores index l, data[l] is output on the target register (which has N qubits).
    '''

    def _load_nth_data(
        self,
        selection_idx: Tuple[int, ...],
        gate: Callable[[cirq.Qid], cirq.Operation],
        **target_regs: NDArray[cirq.Qid],  # type: ignore[type-var]
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