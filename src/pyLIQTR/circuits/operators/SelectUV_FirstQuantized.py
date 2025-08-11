"""
Copyright (c) 2024 Massachusetts Institute of Technology 
SPDX-License-Identifier: BSD-2-Clause
"""
import cirq
from functools import cached_property
from typing import Set
from numpy.typing import NDArray

from qualtran import Signature, Register,Side,QBit,QAny, GateWithRegisters, QUInt
from qualtran.bloqs.basic_gates import XGate

from pyLIQTR.circuits.operators.SignedIntegerToTwosComplement import SignedIntegerToTwosComplement
from pyLIQTR.circuits.operators.ApplyNuclearPhase import ApplyNuclearPhase
from pyLIQTR.circuits.operators.ControlledCopy import ControlledCopy
from pyLIQTR.circuits.operators.ControlledAddOrSubtract import ControlledAddOrSubtract

class SelectUVFirstQuantization(GateWithRegisters):
    """SELECT for the U and V operators for the first quantized chemistry Hamiltonian.

    This does not include the controlled swaps from p_i and q_j system registers
    into ancilla registers and back again. Hence there is no system register.

    Registers:

    References:
        `Fault-Tolerant Quantum Simulations of Chemistry in First Quantization <https://arxiv.org/abs/2105.12767>`_

    :param int num_bits_p: The number of bits to represent each dimension of the momentum register.
    :param int eta: The number of electrons.
    :param int num_atoms: The number of atoms.
    :param int num_bits_nuc_pos: The number of bits to store each component of the
        nuclear positions. :math:`n_R` in the reference.
    """
    def __init__(self,num_bits_p:int,eta: int,num_atoms: int,num_bits_nuc_pos: int, bphi: int=None):

        self.num_bits_p = num_bits_p
        self.eta = eta
        self.num_atoms = num_atoms
        self.num_bits_nuc_pos = num_bits_nuc_pos
        if bphi is None:
            self.bphi = self.num_bits_nuc_pos
        else:
            self.bphi = bphi

    def pretty_name(self) -> str:
        return r'SEL UV'

    @cached_property
    def signature(self) -> Signature:
        n_nu = self.num_bits_p + 1
        return Signature(
            [
                Register("flag_tuv", QBit()),
                Register("uv", QBit(),side=Side.RIGHT),
                Register("Rl", QAny(bitsize=self.num_bits_nuc_pos), shape=(3,)),
                Register("nu", QAny(bitsize=n_nu), shape=(3,)),
                Register("phi", QAny(bitsize=self.bphi)),
                Register("p", QAny(bitsize=self.num_bits_p), shape=(3,)),
                Register("q", QAny(bitsize=self.num_bits_p), shape=(3,)),
                Register("overflow", QAny(bitsize=2), shape=(3,))
            ]
        )
    
    def decompose_from_registers(
        self,
        *,
        context: cirq.DecompositionContext,
        **quregs: NDArray[cirq.Qid],
    ) -> cirq.OP_TREE:

        flag_tuv = quregs['flag_tuv']
        flag_uv = quregs['uv']
        Rl = quregs['Rl']
        nu = quregs['nu']
        phi = quregs['phi']
        p = quregs['p']
        q = quregs['q']
        overflow = quregs['overflow']
        ancilla = context.qubit_manager.qalloc(3*(self.num_bits_p+1))
        n_nu = self.num_bits_p + 1
        anc_x = ancilla[:n_nu]
        anc_y = ancilla[n_nu:2*n_nu]
        anc_z = ancilla[2*n_nu:]
        ancilla_divided = [anc_x, anc_y, anc_z]

        # for arithmetic with twos complement, reg[0] is msb

        # convert x,y,z components of p and q to twos complement
        for dim in range(len(p)):
            yield SignedIntegerToTwosComplement(self.num_bits_p).on_registers(x=p[dim])
            yield SignedIntegerToTwosComplement(self.num_bits_p).on_registers(x=q[dim])

        # for adding into p
        ## control copy of nu components into ancilla register
        for dim in range(len(nu)):
            yield ControlledCopy(bitsize=n_nu,control_val=0).on_registers(control=flag_tuv,original=nu[dim],target=ancilla_divided[dim])

        ## add controlled on sign of nu
        ## done in place, result stored in p
        for dim in range(len(p)):
            nu_sign_bit = ancilla_divided[dim][0]
            overflow_bit = overflow[dim][0]
            # a reg is nu ancilla, b reg is p
            yield ControlledAddOrSubtract(a_dtype=QUInt(n_nu-1),b_dtype=QUInt(self.num_bits_p+1)).on_registers(ctrl=nu_sign_bit,a=ancilla_divided[dim][1:],b=[overflow_bit,*p[dim]])

        ## erase ancilla
        for dim in range(len(nu)):
            yield ControlledCopy(bitsize=n_nu,control_val=0).adjoint().on_registers(control=flag_tuv,original=nu[dim],target=ancilla_divided[dim])

        # for subtracting into q
        ## control copy of nu components into ancilla register
        for dim in range(len(nu)):
            yield ControlledCopy(bitsize=n_nu,control_val=0).on_registers(control=flag_uv,original=nu[dim],target=ancilla_divided[dim])

        ## subtract controlled on sign of nu
        ## done in place, result stored in q
        for dim in range(len(p)):
            nu_sign_bit = ancilla_divided[dim][0]
            overflow_bit = overflow[dim][1]
            # a reg is nu ancilla, b reg is q
            yield XGate().on(nu_sign_bit)
            yield ControlledAddOrSubtract(a_dtype=QUInt(n_nu-1),b_dtype=QUInt(self.num_bits_p+1)).on_registers(ctrl=nu_sign_bit,a=ancilla_divided[dim][1:],b=[overflow_bit,*q[dim]])
            yield XGate().on(nu_sign_bit)

        ## erase ancilla
        for dim in range(len(nu)):
            yield ControlledCopy(bitsize=n_nu,control_val=0).adjoint().on_registers(control=flag_uv,original=nu[dim],target=ancilla_divided[dim])

        # convert back to signed integer
        for dim in range(len(p)):
            overflow_bit_p = overflow[dim][0]
            overflow_bit_q = overflow[dim][1]
            yield SignedIntegerToTwosComplement(self.num_bits_p+1).adjoint().on_registers(x=[overflow_bit_p,*p[dim]])
            yield SignedIntegerToTwosComplement(self.num_bits_p+1).adjoint().on_registers(x=[overflow_bit_q,*q[dim]])

        # apply nuclear phase
        phi_nr = phi[:self.num_bits_nuc_pos]
        yield ApplyNuclearPhase(self.num_bits_p, self.num_bits_nuc_pos).on_registers(Rl=Rl,nu=nu,phi=phi_nr,tuv=flag_tuv,uv=flag_uv)

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        n_nu = self.num_bits_p+1
        int_conv = (SignedIntegerToTwosComplement(self.num_bits_p), 6)
        copies = (ControlledCopy(bitsize=n_nu,control_val=0),6)
        erase_copies = (ControlledCopy(bitsize=n_nu,control_val=0).adjoint(),6)
        flips = (XGate(),6)
        add_sub = (ControlledAddOrSubtract(a_dtype=QUInt(n_nu-1),b_dtype=QUInt(self.num_bits_p+1)), 6)
        # + 1 due to bitsize increase from addition of nu
        inv_int_conv = (SignedIntegerToTwosComplement(self.num_bits_p + 1).adjoint(), 6)
        phase = (ApplyNuclearPhase(self.num_bits_p, self.num_bits_nuc_pos), 1)
        return {int_conv,copies,erase_copies,flips,add_sub,inv_int_conv,phase}