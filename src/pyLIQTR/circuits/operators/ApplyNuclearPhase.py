"""
Copyright (c) 2024 Massachusetts Institute of Technology 
SPDX-License-Identifier: BSD-2-Clause
"""
import cirq
from functools import cached_property
from typing import Optional, Tuple
from numpy.typing import NDArray

from attrs import frozen

from qualtran import GateWithRegisters, QAny, Register, Signature, QBit
from qualtran.bloqs.basic_gates import Toffoli, CNOT, CZPowGate
from qualtran.bloqs.mcmt import MultiTargetCNOT
from qualtran.drawing import Text, WireSymbol

from pyLIQTR.circuits.operators.ControlledCopy import ControlledCopy
from pyLIQTR.circuits.operators.AddIntoPhaseGrad import AddIntoPhaseGradwithDecompose

@frozen
class ApplyNuclearPhase(GateWithRegisters):
    """Apply the phase factor :math:`-e^{-ik_\\nu\\cdot R_\\ell}` to the state.

    Registers:

    .. line-block::
        Rl: A register storing the value of :math:`R_\\ell`.
        nu: The momentum transfer register.
        phi: The phase gradient state register.
        tuv: Qubit to select between T and U+V
        uv: Qubit to select between U and V.

    References:
        `Fault-Tolerant Quantum Simulations of Chemistry in First Quantization <https://arxiv.org/abs/2105.12767>`_
        pg 25, paragraph 2.

    :param int num_bits_p: Number of bits for the momentum register.
    :param int num_bits_nuc: Number of bits of precision for the nuclear positions :math:`R_\\ell`.
    """

    num_bits_p: int
    num_bits_nuc: int

    @cached_property
    def signature(self) -> Signature:
        return Signature(
            [
                Register("Rl", QAny(bitsize=self.num_bits_nuc), shape=(3,)),
                Register("nu", QAny(bitsize=self.num_bits_p+1), shape=(3,)),
                Register("phi", QAny(bitsize=self.num_bits_nuc)),
                Register("tuv", QBit()),
                Register("uv", QBit()),
            ]
        )

    def wire_symbol(self, reg: Optional[Register], idx: Tuple[int, ...] = tuple()) -> 'WireSymbol':
        if reg is None:
            return Text(r'-e^(-k_ν⋅R_l)')
        return super().wire_symbol(reg, idx)

    def decompose_from_registers(
        self,
        *,
        context: cirq.DecompositionContext,
        **quregs: NDArray[cirq.Qid],
    ) -> cirq.OP_TREE:

        Rl = quregs["Rl"]
        nu = quregs["nu"]
        phase_gradient_state = quregs["phi"]
        flag_tuv = quregs['tuv']
        flag_uv = quregs["uv"]
        ancilla = context.qubit_manager.qalloc(self.num_bits_nuc)

        # overall factor of minus 1 controlled on selecting U
        yield CZPowGate(exponent=1.0,global_shift=0.0).on(*flag_uv,*flag_tuv)

        # loop over each dimension
        for dim in range(3):
            Rl_i = Rl[dim][1:] # 0th element is sign bit, 1st element is msb
            nu_i = nu[dim][1:][::-1] # reverse order so msb is last element in list
            product_sign_bit = nu[dim][0]

            # multiply sign bits
            yield CNOT().on_registers(ctrl=Rl[dim][0],target=product_sign_bit)
            # apply CNOT to phi register so that product_sign_bit=|1> will yield subtraction, |0> addition
            yield MultiTargetCNOT(self.num_bits_nuc).on_registers(control=product_sign_bit,targets=phase_gradient_state)

            # loop over nu components starting with lsb
            for k,nu_bit in enumerate(nu_i[:self.num_bits_nuc]):
                if k == 0:
                    target_bitsize = self.num_bits_nuc-1
                    bits_to_multiply = Rl_i
                    product_target = ancilla[1:]
                else:
                    target_bitsize = self.num_bits_nuc-k
                    bits_to_multiply = Rl_i[k-1:]
                    product_target = ancilla[k:]

                multiply_out_of_place = ControlledCopy(bitsize=target_bitsize)
                
                yield multiply_out_of_place.on_registers(control=nu_bit,original=bits_to_multiply,target=product_target)

                # add in to phase gradient
                add_bitsize = self.num_bits_nuc-k
                yield AddIntoPhaseGradwithDecompose(x_bitsize=add_bitsize,phase_bitsize=add_bitsize).on_registers(x=ancilla[k:],phase_grad=phase_gradient_state[:add_bitsize])

                # uncompute product
                yield (multiply_out_of_place.on_registers(control=nu_bit,original=bits_to_multiply,target=product_target))**(-1)

            # un-multiply sign bits
            yield CNOT().on_registers(ctrl=Rl[dim][0],target=product_sign_bit)
            # un-apply CNOT to phi register
            yield MultiTargetCNOT(self.num_bits_nuc).on_registers(control=product_sign_bit,targets=phase_gradient_state)

        context.qubit_manager.qfree(ancilla)