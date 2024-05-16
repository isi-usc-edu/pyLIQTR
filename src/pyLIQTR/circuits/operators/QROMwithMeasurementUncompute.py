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
from typing import Optional
from qualtran._infra.gate_with_registers import split_qubits
from qualtran.bloqs.qrom import QROM

from pyLIQTR.utils.global_ancilla_manager import gam as gam
from pyLIQTR.circuits.operators.FixupTableQROM import FixupTableQROM

class QROMwithMeasurementUncompute(QROM):
    '''
    This is an extension of qualtran's QROM class that adds a method to carry out measurement based uncomputation.
    '''

    def measurement_uncompute(
        self, context: Optional[cirq.DecompositionContext] = None, measurement_key: Optional[str] = 'qrom_data_measurement',**quregs
    ) -> cirq.OP_TREE:
        '''
        This is a method that implements the phase fixup measurement based uncomputation described in Appendix C of https://arxiv.org/pdf/1902.02134.pdf.

        Note, if this method is used twice in the same circuit, different measurement_key strings should be used.
        '''

        if context is None:
            context = cirq.DecompositionContext(gam)
        
        original_select_key = [reg.name for reg in self.selection_registers]
        original_target_key = [reg.name for reg in self.target_registers]
        if len(original_select_key) > 1:
            raise RuntimeError("Measurement based uncomputation only implemented for qrom with single select register.")
        elif len(original_target_key) > 1:
            raise RuntimeError("Measurement based uncomputation only implemented for qrom with single target register.")
        else:
            original_select_key = original_select_key[0]
            original_target_key = original_target_key[0]

        original_select = split_qubits(self.selection_registers,quregs['selection'])[original_select_key]
        new_select = original_select[:-1]
        q_bit = original_select[-1]
        u_bit = context.qubit_manager.qalloc(1)
        data_reg = split_qubits(self.target_registers,quregs['data'])[original_target_key]

        # prep q and u
        yield [
            cirq.X(*u_bit),
            cirq.CNOT(q_bit,*u_bit),
            cirq.H(q_bit),
            cirq.H(*u_bit)
        ]

        # measure data in x basis
        yield [
            cirq.H.on_each(*data_reg),
            cirq.measure(list(data_reg),key=measurement_key)
        ]

        # compute and apply fixup table
        fixup = FixupTableQROM(data_to_uncompute=self.data[0],measurement_key=measurement_key,max_uncompute_bits=len(data_reg))
        yield fixup.on_registers(selection=new_select,u=u_bit,q=q_bit)

        # data qubits are now free, reset to |0>
        yield [cirq.reset(q) for q in data_reg]

        # undo prep q and u
        yield [
            cirq.H(q_bit),
            cirq.H(*u_bit),
            cirq.CNOT(q_bit,*u_bit),
            cirq.X(*u_bit)
        ]

        context.qubit_manager.qfree(u_bit)