"""
Copyright (c) 2024 Massachusetts Institute of Technology 
SPDX-License-Identifier: BSD-2-Clause
"""
import cirq
import ctypes
from warnings import warn
from functools import cached_property
from typing import Set, List
from numpy.typing import NDArray

import numpy as np
from attrs import evolve, frozen

from qualtran import GateWithRegisters, QAny, Register, Signature, QBit, BoundedQUInt
from qualtran.bloqs.basic_gates import XGate
from qualtran.bloqs.data_loading import QROM
from qualtran.symbolics.math_funcs import bit_length

from pyLIQTR.circuits.operators.QROMwithMeasurementUncompute import QROMwithMeasurementUncompute
from pyLIQTR.utils.utils import float_array_to_fixed_width_int


@frozen
class PrepareZetaState(GateWithRegisters):
    """Prepare the superpostion over :math:`\\ell` weighted by :math:`\\zeta_\\ell` with states corresponding to :math:`n_R`-bit representations of the nuclear position vector :math:`R_\\ell`. Note, :math:`\\ell \\in \\[0,L-1\\]` where :math:`L` is the number of atoms in the system. The nuclear position vector is only needed in the case of :math:`U`, so this preparation is controlled on the qubit selecting between :math:`U` and :math:`V`.

    Registers:

    .. line-block::
        control: Qubit to control preparation on.
        select: Register used to select value of :math:`R_\\ell`.
        Rl: Register to store the value of the nuclear coordinates :math:`R_\\ell`.

    References:
        `Fault-Tolerant Quantum Simulations of Chemistry in First Quantization <https://arxiv.org/abs/2105.12767>`_ 
        page 23-24, last 3 paragraphs before Section IID.

    :param int num_atoms: The number of atoms.
    :param int lambda_zeta: The sum of nuclear charges.
    :num_bits_nuc_pos: The number of bits of precision for representing the nuclear coordinates :math:`R_\\ell`.
    :param int eta: The number of electrons.
    :param bool is_adjoint: Flag to do the adjoint of this gate or not. Default is False.
    """
    num_atoms: int
    lambda_zeta: int
    num_bits_nuc_pos: int
    eta: int = 10
    is_adjoint: bool = False

    @cached_property
    def signature(self) -> Signature:
        n_eta_zeta = bit_length(self.eta + 2 * self.lambda_zeta - 1)
        return Signature([Register("control", QBit()),
            Register("select", BoundedQUInt(bitsize=n_eta_zeta)),
            Register("Rl", QAny(bitsize=self.num_bits_nuc_pos), shape=(3,))])

    @cached_property
    def atomic_numbers(self) -> List:
        '''
        Returns a list of the atomic numbers :math:`\\zeta_\\ell`. Currently assumes atomic number is divided evenly between all atoms. In reality this information should be provided from the ProblemInstance.
        '''
        atomic_num, remainder = divmod(self.lambda_zeta,self.num_atoms)
        return [atomic_num]*(self.num_atoms-1) + [remainder+atomic_num]

    @cached_property
    def R_l_data(self):
        '''
        Returns the nuclear data positions as a list of :math:`n_R`-bit integer approximations. For each value of :math:`R_\\ell` there are :math:`\\zeta_\\ell` copies such that the resulting amplitude will be proportional to :math:`\\sqrt\\zeta_\\ell`. Currently these positions are selected randomly but they should be changed in the future to be read in from the ProblemInstance.
        '''
        # limit bit width based on system maximal value for signed longs. This is needed since qrom uses numpy int dtype rather than native python dtype
        max_width = int(np.log2((1 << (8 * ctypes.sizeof(ctypes.c_long) - 1)) - 1))
        if 3*self.num_bits_nuc_pos<max_width:
            width = 3*self.num_bits_nuc_pos
        else:
            warn(f"Desired precision {3*self.num_bits_nuc_pos} is beyond system spec {max_width}. Using system spec instead. This will result in lower Clifford gate count.",stacklevel=2)
            width = max_width
        # get data as random floats bounded by omega**(1/3)/2
        # multiply by 2pi/omega**(1/3)
        # convert to integer
        # should be length num_atoms, each element num_bits_nuc_pos, 3d
        np.random.seed(0)
        Rl = 2*np.pi*np.random.uniform(low=0.0, high=0.5, size=(self.num_atoms,3))
        as_ints = np.array([float_array_to_fixed_width_int(f,width_bits=width) for f in Rl])
        # for ell in L, we need to output zeta_ell (atomic number ell) copies of Rl. This will give a state with relative amplitude sqrt(zeta_ell) assuming input selection register is in equal superposition over lambda_zeta states
        R_l_data = []
        for l,pos in enumerate(as_ints):
            R_l_data += [pos]*self.atomic_numbers[l] 
        return R_l_data

    def adjoint(self) -> 'Bloq':
        return evolve(self, is_adjoint=not self.is_adjoint)

    @cached_property
    def qrom_gate(self):
        return QROMwithMeasurementUncompute.build_from_data(self.R_l_data,target_bitsizes=(3*self.num_bits_nuc_pos,),num_controls=2)

    def decompose_from_registers(
        self,
        *,
        context: cirq.DecompositionContext,
        **quregs: NDArray[cirq.Qid],
    ) -> cirq.OP_TREE:

        control = quregs['control']
        full_superposition = quregs['select']
        data_output = quregs['Rl'].flatten()

        # isolate qrom selection register
        n_lz = bit_length(self.lambda_zeta-1)
        qrom_control = full_superposition[0]
        qrom_select = full_superposition[1:n_lz+1]

        # control such that register Rl is non zero only for U
        yield XGate().on(qrom_control)
        if not self.is_adjoint:
            yield self.qrom_gate.on_registers(selection=qrom_select,target0_=data_output, control=[*control,*[qrom_control]],context=context)
        else:
            yield self.qrom_gate.measurement_uncompute(selection=qrom_select,data=data_output, control=[*control,*[qrom_control]], context=context)
        yield XGate().on(qrom_control)

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        if self.is_adjoint:
            return self.decompose_bloq().build_call_graph(ssa)
        else:
            return {(self.qrom_gate, 1),(XGate(),2)}
