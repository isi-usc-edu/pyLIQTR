"""
Copyright (c) 2025 Massachusetts Institute of Technology 
SPDX-License-Identifier: BSD-2-Clause
"""
from functools import cached_property
from typing import Iterator, Tuple

import cirq
import numpy as np
from numpy.typing import NDArray

from qualtran.bloqs.basic_gates import CSwap
from qualtran.bloqs.mcmt.and_bloq import MultiAnd
from qualtran.bloqs.state_preparation.prepare_uniform_superposition import PrepareUniformSuperposition
from qualtran.symbolics.math_funcs import acos, ssqrt
from qualtran.bloqs.chemistry.hubbard_model.qubitization.prepare_hubbard import PrepareHubbard
from pyLIQTR.circuits.operators.AddMod import Add
from qualtran.bloqs.mcmt import And
from qualtran import QInt, Register, QAny, Side

class PrepareHubbardPYL(PrepareHubbard):
    r"""Modifies the PrepareHubbard decomposition to have a fully implemented modular addition.

    Args:
        x_dim: the number of sites along the x axis.
        y_dim: the number of sites along the y axis.
        t: coefficient for hopping terms in the Hubbard model hamiltonian.
        u: coefficient for single body Z term and two-body ZZ terms in the Hubbard model
            hamiltonian.

    Registers:
        control: A control bit for the entire gate.
        U: Whether we're applying the single-site part of the potential.
        V: Whether we're applying the pairwise part of the potential.
        p_x: First set of site indices, x component.
        p_y: First set of site indices, y component.
        alpha: First set of sites' spin indicator.
        q_x: Second set of site indices, x component.
        q_y: Second set of site indices, y component.
        beta: Second set of sites' spin indicator.
        target: The system register to apply the select operation.
        temp: Temporary Work space.
        add_ancilla: Ancilla used for addition.

    References:
        [Encoding Electronic Spectra in Quantum Circuits with Linear T Complexity](https://arxiv.org/abs/1805.03662).
        Section V. and Fig. 20.
    """

    @cached_property
    def junk_registers(self) -> Tuple[Register, ...]:
        return (Register('temp', QAny(2)),
                Register('add_ancilla', QInt((self.x_dim - 1).bit_length()),side=Side.RIGHT))
    
    def decompose_from_registers(
        self,
        *,
        context: cirq.DecompositionContext,
        **quregs: NDArray[cirq.Qid],
    ) -> Iterator[cirq.OP_TREE]:
        p_x, p_y, q_x, q_y = quregs['p_x'], quregs['p_y'], quregs['q_x'], quregs['q_y']
        U, V, alpha, beta = quregs['U'], quregs['V'], quregs['alpha'], quregs['beta']
        temp = quregs['temp']
        add_ancilla = quregs['add_ancilla']

        N = self.x_dim * self.y_dim * 2

        yield cirq.Ry(rads=2 * acos(ssqrt(self.t * N / self.l1_norm_of_coeffs))).on(*V)
        yield cirq.Ry(rads=2 * np.arccos(np.sqrt(1 / 5))).on(*U).controlled_by(*V)

        yield PrepareUniformSuperposition(self.x_dim).on_registers(target=p_x)
        yield PrepareUniformSuperposition(self.y_dim).on_registers(target=p_y)

        yield cirq.H.on_each(*temp)

        yield cirq.CNOT(*U, *V)

        yield cirq.X(*beta)

        yield from [cirq.X(*V), cirq.H(*alpha).controlled_by(*V), cirq.CX(*V, *beta), cirq.X(*V)]

        yield cirq.CNOT.on_each([*zip([*p_x, *p_y, *alpha], [*q_x, *q_y, *beta])])

        yield CSwap.make_on(ctrl=temp[:1], x=q_x, y=q_y)

        # add 1 Mod x_dim controlled on U and V
        ## flip add_ancilla lsb to 1 controlled on U and V
        yield And(cv1=0,cv2=0).on_registers(ctrl=[U,V],target=add_ancilla[-1])
        ## add in place, drops carry bit such that its mod 2**bitsize. Output stored on q_x
        yield Add(a_dtype=QInt(bitsize=len(q_x))).on_registers(a=add_ancilla,b=q_x)
        ## flip add_ancilla lsb controlled on U and V
        yield And(cv1=0,cv2=0,uncompute=True).on_registers(ctrl=[U,V],target=add_ancilla[-1])

        yield CSwap.make_on(ctrl=temp[:1], x=q_x, y=q_y)

        and_target = context.qubit_manager.qalloc(1)
        and_anc = context.qubit_manager.qalloc(1)
        yield MultiAnd(cvs=(0, 0, 1)).on_registers(
            ctrl=np.array([U, V, temp[-1:]]), junk=np.array([and_anc]), target=and_target
        )
        yield CSwap.make_on(ctrl=and_target, x=[*p_x, *p_y, *alpha], y=[*q_x, *q_y, *beta])
        yield MultiAnd(cvs=(0, 0, 1)).adjoint().on_registers(
            ctrl=np.array([U, V, temp[-1:]]), junk=np.array([and_anc]), target=and_target
        )
        context.qubit_manager.qfree([*and_anc, *and_target])


class PrepareHubbardPYL_invert_workaround(PrepareHubbard):
    r"""Modifies the PrepareHubbard decomposition to have a fully implemented modular addition.

    This is a temporary work around for inverting the PrepareHubbard due to a known issue with the qualtran version we are using. 
    This will be deprecated when we support the new qualtran version.

    Args:
        x_dim: the number of sites along the x axis.
        y_dim: the number of sites along the y axis.
        t: coefficient for hopping terms in the Hubbard model hamiltonian.
        u: coefficient for single body Z term and two-body ZZ terms in the Hubbard model
            hamiltonian.

    Registers:
        control: A control bit for the entire gate.
        U: Whether we're applying the single-site part of the potential.
        V: Whether we're applying the pairwise part of the potential.
        p_x: First set of site indices, x component.
        p_y: First set of site indices, y component.
        alpha: First set of sites' spin indicator.
        q_x: Second set of site indices, x component.
        q_y: Second set of site indices, y component.
        beta: Second set of sites' spin indicator.
        target: The system register to apply the select operation.
        temp: Temporary Work space.
        add_ancilla: Ancilla used for addition.

    References:
        [Encoding Electronic Spectra in Quantum Circuits with Linear T Complexity](https://arxiv.org/abs/1805.03662).
        Section V. and Fig. 20.
    """

    @cached_property
    def junk_registers(self) -> Tuple[Register, ...]:
        return (Register('temp', QAny(2)),
                Register('add_ancilla', QInt((self.x_dim - 1).bit_length()),side=Side.RIGHT))
    
    def decompose_from_registers(
        self,
        *,
        context: cirq.DecompositionContext,
        **quregs: NDArray[cirq.Qid],
    ) -> Iterator[cirq.OP_TREE]:
        p_x, p_y, q_x, q_y = quregs['p_x'], quregs['p_y'], quregs['q_x'], quregs['q_y']
        U, V, alpha, beta = quregs['U'], quregs['V'], quregs['alpha'], quregs['beta']
        temp = quregs['temp']
        add_ancilla = quregs['add_ancilla']

        N = self.x_dim * self.y_dim * 2




        and_target = context.qubit_manager.qalloc(1)
        and_anc = context.qubit_manager.qalloc(1)
        yield MultiAnd(cvs=(0, 0, 1)).on_registers(
            ctrl=np.array([U, V, temp[-1:]]), junk=np.array([and_anc]), target=and_target
        )
        yield CSwap.make_on(ctrl=and_target, x=[*p_x, *p_y, *alpha], y=[*q_x, *q_y, *beta])
        yield MultiAnd(cvs=(0, 0, 1)).adjoint().on_registers(
            ctrl=np.array([U, V, temp[-1:]]), junk=np.array([and_anc]), target=and_target
        )
        context.qubit_manager.qfree([*and_anc, *and_target])

        yield CSwap.make_on(ctrl=temp[:1], x=q_x, y=q_y)
        ## flip add_ancilla lsb controlled on U and V
        yield And(cv1=0,cv2=0).on_registers(ctrl=[U,V],target=add_ancilla[-1])
        ## add in place, drops carry bit such that its mod 2**bitsize. Output stored on q_x

        yield Add(a_dtype=QInt(bitsize=len(q_x))).on_registers(a=add_ancilla,b=q_x)
        # add 1 Mod x_dim controlled on U and V
        ## flip add_ancilla lsb to 1 controlled on U and V
        yield And(cv1=0,cv2=0, uncompute=True).on_registers(ctrl=[U,V],target=add_ancilla[-1])
        yield CSwap.make_on(ctrl=temp[:1], x=q_x, y=q_y)
        yield cirq.CNOT.on_each([*zip([*p_x, *p_y, *alpha], [*q_x, *q_y, *beta])])
        yield from [cirq.X(*V), cirq.H(*alpha).controlled_by(*V), cirq.CX(*V, *beta), cirq.X(*V)]
        yield cirq.X(*beta)
        yield cirq.CNOT(*U, *V)
        yield cirq.H.on_each(*temp)
        yield PrepareUniformSuperposition(self.y_dim).on_registers(target=p_y)
        yield PrepareUniformSuperposition(self.x_dim).on_registers(target=p_x)
        yield cirq.Ry(rads=2 * np.arccos(np.sqrt(1 / 5))).on(*U).controlled_by(*V)
        yield cirq.Ry(rads=2 * acos(ssqrt(self.t * N / self.l1_norm_of_coeffs))).on(*V)
        
        

        

        

        


        
        

        

