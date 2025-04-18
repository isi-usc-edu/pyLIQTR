"""
Copyright (c) 2024 Massachusetts Institute of Technology 
SPDX-License-Identifier: BSD-2-Clause
"""
from typing import Tuple, Set

import attrs
import cirq
import numpy as np
from functools import cached_property
from numpy.typing import NDArray

from qualtran import GateWithRegisters, Signature
from qualtran.bloqs.mcmt.and_bloq import And, MultiAnd
from qualtran.bloqs.arithmetic import LessThanConstant
from qualtran.bloqs.mcmt.multi_control_multi_target_pauli import MultiControlPauli
from qualtran.bloqs.basic_gates import XGate, Hadamard, CZPowGate, CNOT, Ry, OnEach

class FlaggedPrepareUniformSuperposition(GateWithRegisters):
    '''
    prepares uniform superposition over d basis states according to procedure outlined in https://arxiv.org/abs/2011.03494 Appendix A page 39 with the added step of flagging success. Toffoli cost nominally :math:`3 \\log_2 d - 3\\eta + 2 b_r - 9`, where :math:`2^\\eta` is a factor of :math:`d` and :math:`b_r` is the rotation gate precision bits. However, here we use the arbitrary rotation gate that is compiled directly to Clifford+T instead of the phase gradient rotation method, so there is no dependence on :math:`b_r`.

    See https://arxiv.org/abs/2408.03145 Appendix B for detailed walkthrough.

    :param int d: number of basis states in the superposition.
    '''
    def __init__(self,d:int):
        self.d = d 
        self.nd_bits = (d-1).bit_length()

    @cached_property
    def signature(self) -> Signature:
        return Signature.build(
            target=self.nd_bits,
            less_than_ancilla=1,
            rot_ancilla=1,
            success=1
        )

    def get_eta_and_logL(self):
        '''finds :math:`\\eta` and :math:`\\log_2 L = \\log_2 d-\\eta`'''
        d, eta = self.d, 0
        while d > 1 and d % 2 == 0:
            eta += 1
            d = d // 2
        l, logL = int(d), self.nd_bits - eta
        return l, logL

    def __repr__(self):
        return f'{self.__class__.__name__}(d={self.d})'

    def decompose_from_registers(
        self,
        *,
        context: cirq.DecompositionContext,
        **quregs: NDArray[cirq.Qid],  
    ) -> cirq.OP_TREE:
        target = quregs['target']
        less_than,success_ancilla, rot_ancilla = quregs['less_than_ancilla'],quregs['success'], quregs['rot_ancilla']

        l, logL = self.get_eta_and_logL()
        logL_qubits = target[:logL]
        
        if not len(logL_qubits):
            yield OnEach(self.nd_bits, Hadamard()).on_registers(q=target) #Hadamard().on_each(*target) 
            yield XGate().on(*success_ancilla)
            return

        yield OnEach(self.nd_bits+1, Hadamard()).on_registers(q=[*target,*rot_ancilla]) #Hadamard().on_each(*target,*rot_ancilla) ####### TODO

        # iii. Inequality test between target register and l
        yield LessThanConstant(logL, l).on_registers(x=logL_qubits, target=less_than)

        # iv. Rotate ancilla qubit
        rotation_angle = 2*np.arcsin(np.sqrt(2**(logL)/(4*l)))
        yield Ry(angle=rotation_angle)(*rot_ancilla)

        # v. Reflection on result of inequality test and ancilla
        yield XGate().on(*rot_ancilla)
        yield CZPowGate().on(*less_than,*rot_ancilla)
        yield XGate().on(*rot_ancilla)

        # vi. Invert rotation
        yield Ry(angle=-rotation_angle)(*rot_ancilla)
        ## and invert inequality test
        yield LessThanConstant(logL, l).on_registers(x=logL_qubits, target=less_than)

        # vii. Repeat Hadamards
        yield OnEach(logL+1, Hadamard()).on_registers(q=[*logL_qubits,*rot_ancilla])#Hadamard().on_each(*logL_qubits,*rot_ancilla) ######### TODO

        # viii. Reflect about zero state 
        yield XGate().on(*rot_ancilla)
        yield MultiControlPauli(cvs=(0,)*logL,target_gate=cirq.Z).on_registers(controls=[[q] for q in logL_qubits],target=rot_ancilla)
        yield XGate().on(*rot_ancilla)

        # ix. Repeat Hadamards 
        yield OnEach(logL+1, Hadamard()).on_registers(q=[*logL_qubits,*rot_ancilla])#Hadamard().on_each(*logL_qubits,*rot_ancilla) ####### TODO

        # x. Repeat inequality test
        yield LessThanConstant(logL, l).on_registers(x=logL_qubits, target=less_than)

        ## success flagged on inequality test
        yield CNOT().on(*less_than,*success_ancilla)

        # Invert inequality test
        yield LessThanConstant(logL, l).on_registers(x=logL_qubits, target=less_than)

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        l, logL = self.get_eta_and_logL()
        if not logL:
            return{(OnEach(self.nd_bits, Hadamard()), 1),(XGate(),1)}
        rotation_angle = 2*np.arcsin(np.sqrt(2**(logL)/(4*l)))
        return {(OnEach(self.nd_bits+1, Hadamard()), 1),(OnEach(logL+1, Hadamard()),2), (LessThanConstant(logL,l),4),(Ry(angle=rotation_angle),1),(Ry(angle=-rotation_angle),1),(XGate(),4),(CZPowGate(),1),(MultiControlPauli(cvs=(0,)*logL,target_gate=cirq.Z),1), (CNOT(),1)}