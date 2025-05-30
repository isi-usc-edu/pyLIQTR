"""
Copyright (c) 2024 Massachusetts Institute of Technology 
SPDX-License-Identifier: BSD-2-Clause
"""
import qualtran as qt
import cirq
import numpy as np
from qualtran import Register, GateWithRegisters, Signature, QBit, BoundedQUInt, QAny
import qualtran.cirq_interop as qt_ci
from functools import cached_property
from typing import List, Tuple, Sequence, Optional
from numpy.typing import NDArray

class FermionicSelect_LinearT(GateWithRegisters):

    '''
    Implements circuit from Fig . 14 of https://arxiv.org/pdf/1805.03662.pdf using cirq_ft gates. Performs the operation

    SELECT|theta,U,V,p,alpha,q,beta>|Psi> = (-1)^theta|theta,U,V,p,alpha,q,beta> \times 
                                                Z_{p,alpha}|Psi>                  U ∧ ¬V ∧ ((p,alpha)=(q,beta))
                                                Z_{p,alpha}Z_{q,beta}|Psi>       ¬U ∧ V ∧ ((p,alpha)!=(q,beta))
                                                X_{p,alpha}ZX_{q,alpha}|Psi>     ¬U ∧ ¬V ∧ (p<q) ∧ (alpha=beta)
                                                Y_{q,alpha}ZY_{p,alpha}|Psi>     ¬U ∧ ¬V ∧ (p>q) ∧ (alpha=beta)
                                                undefined                        otherwise

    Args:
        M_vals: Number of grid points (orbitals) along each spatial dimension.
        control_val: # optional, set to 1 for a controlled select operator
    '''
    def __init__(self, M_vals: NDArray[np.int_], control_val: Optional[int] = None):

        self.__M_vals = M_vals.tolist() # convert numpy int to python int
        self.__control_val = control_val

        self.__N = 2*int(np.prod(M_vals)) # total number of spin orbitals. Factor of 2 for up and down spin
        self.__Np_bits = int(sum(np.ceil(np.log2(M_vals)))) # number of bits needed to represent all p values

        super(FermionicSelect_LinearT, self)

    @cached_property
    def control_registers(self) -> Tuple[Register]:
        registers = () if self.__control_val is None else (Register('control', QBit()),)
        return registers

    @cached_property
    def selection_registers(self) -> Tuple[Register]:
        p_reg = Register(name='p',dtype=BoundedQUInt(bitsize=self.__Np_bits,iteration_length=int(self.__N/2)))
        q_reg = Register(name='q',dtype=BoundedQUInt(bitsize=self.__Np_bits,iteration_length=int(self.__N/2)))
        theta_reg = Register(name="theta",dtype=QBit())
        U_reg = Register(name="U",dtype=BoundedQUInt(1,2))
        V_reg = Register(name="V",dtype=BoundedQUInt(1,2))
        a_reg = Register(name="a",dtype=BoundedQUInt(1,2))
        b_reg = Register(name="b",dtype=BoundedQUInt(1,2))
        return (theta_reg,U_reg,V_reg,p_reg,a_reg,q_reg,b_reg)

    @cached_property
    def target_registers(self) -> Tuple[Register]:
        return (Register('target', QAny(self.__N)), )

    @cached_property
    def signature(self) -> Signature:
        return Signature(
            [*self.control_registers, *self.selection_registers, *self.target_registers]
        )

    def decompose_from_registers(
        self,
        *,
        context: cirq.DecompositionContext,
        **quregs: NDArray[cirq.Qid],  # type:ignore[type-var]
    ) -> cirq.OP_TREE:
        theta = quregs['theta']
        p, q = quregs['p'], quregs['q']
        U, V, a, b = quregs['U'], quregs['V'], quregs['a'], quregs['b']
        control, target = quregs.get('control', ()), quregs['target']

        qIterationLength = self.signature.get_left('q').dtype.iteration_length

        yield qt.bloqs.multiplexers.selected_majorana_fermion.SelectedMajoranaFermion(
            selection_regs=(
                Register('a',BoundedQUInt(1,2)),
                Register('p',BoundedQUInt(self.signature.get_left('p').total_bits(),iteration_length=int(self.__N/2))),
            ),
            control_regs=self.control_registers,
            target_gate=cirq.Y, 
        ).on_registers(control=control, p=p, a=a,target=target)

        yield qt.bloqs.basic_gates.swap.CSwap.make_on(ctrl=V, x=p, y=q)
        yield qt.bloqs.basic_gates.swap.CSwap.make_on(ctrl=V, x=a, y=b)

        q_selection_regs = (
                    self.signature.get_left('b'),
                    self.signature.get_left('q'),
            )
        yield qt.bloqs.multiplexers.selected_majorana_fermion.SelectedMajoranaFermion(
            selection_regs=q_selection_regs,
            control_regs=self.control_registers, 
            target_gate=cirq.X
        ).on_registers(control=control, q=q, b=b, target=target)

        yield qt.bloqs.basic_gates.swap.CSwap.make_on(ctrl=V, x=a, y=b)
        yield qt.bloqs.basic_gates.swap.CSwap.make_on(ctrl=V, x=p, y=q)

        yield cirq.S(*control) if control else cirq.global_phase_operation(1j)  # Fix errant i from XY=iZ (for U/V terms, combined with Zs below to fix -1), or fix -i from ZY = -iX or XZ = -iY (for T terms)
        yield cirq.Z(*theta).controlled_by(*control)  # Take care of overall sign from coefficient
        yield cirq.Z(*U).controlled_by(*control)  # Fix errant -1 from multiple pauli applications
        yield cirq.Z(*V).controlled_by(*control)  # Fix errant -1 from multiple pauli applications
        
        target_qubits_for_apply_to_lth_gate = [
            target[np.ravel_multi_index((b, q), (2, qIterationLength))]
            for q in range(qIterationLength)
            for b  in range(2)
        ]

        yield qt.bloqs.multiplexers.apply_gate_to_lth_target.ApplyGateToLthQubit(
            selection_regs=q_selection_regs,
            nth_gate=lambda *_: cirq.Z,
            control_regs=Register('control', QAny(1 + qt_ci.total_bits(self.control_registers))),
        ).on_registers(
            q=q, b=b, control=[*V, *control], target=target_qubits_for_apply_to_lth_gate
        )

    def __repr__(self) -> str:
        return f'pyLIQTR.FermionicSelect_LinearT({self.__M_vals}, {self.__control_val})'