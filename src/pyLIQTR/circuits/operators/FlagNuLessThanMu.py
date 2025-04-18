"""
Copyright (c) 2024 Massachusetts Institute of Technology 
SPDX-License-Identifier: BSD-2-Clause
"""
import cirq
from functools import cached_property
from numpy.typing import NDArray
from typing import Set, Optional, Tuple
from qualtran import GateWithRegisters, Signature, QAny, QBit, Register, Side
from qualtran.bloqs.basic_gates import CNOT, Toffoli, XGate
from qualtran.bloqs.mcmt import MultiAnd, And
from qualtran.drawing import WireSymbol
from qualtran.drawing.musical_score import Text

class FlagNuLessThanMu(GateWithRegisters):
    """Gate to flag if all components of :math:`\\nu` are smaller in absolute value than :math:`2^{\mu-2}`. The register :math:`|\\mu\\rangle` should be prepared in a unary-encoded superposition prior to input. This can be done using :class:`PrepareMuUnaryEncoded`. The register :math:`|\\nu\\rangle` should also be prepared in a superposition state using :class:`PrepareNuSuperposition`.

    Registers:

    .. line-block::
        mu: the register containing the state of :math:`\\mu`.
        nu: the register containing the state of :math:`\\nu`
        flag_nu_lt_mu: a flag bit for when all :math:`|\\nu_i| < 2^{\\mu-2}`
        and_ancilla: ancilla qubits used for the AND ladder.

    References:
        `Fault-Tolerant Quantum Simulations of Chemistry in First Quantization <https://arxiv.org/abs/2105.12767>`_
        page 21, Eq 81.

    :param int bitsize: the number of bits in the mu register, :math:`n_p` in the reference.
    """

    def __init__(self,bitsize:int):
        self.bitsize = bitsize

    @cached_property
    def signature(self) -> Signature:
        return Signature(
            [
                Register("mu", QAny(self.bitsize)),
                Register("nu", QAny(self.bitsize + 1), shape=(3,)),
                Register("flag_nu_lt_mu", QBit(), side=Side.RIGHT),
                Register("and_ancilla",QBit(),shape=(self.bitsize,2), side=Side.RIGHT)
            ]
        )

    def wire_symbol(
        self, reg: Optional['Register'], idx: Tuple[int, ...] = tuple()
    ) -> 'WireSymbol':
        if reg is None:
            return Text(r'|ν|<2^(μ−2)')
        return super().wire_symbol(reg, idx)

    def decompose_from_registers(
        self,
        *,
        context: cirq.DecompositionContext,
        **quregs: NDArray[cirq.Qid],
    ) -> cirq.OP_TREE:

        mu_reg = quregs['mu']
        nu_reg = quregs['nu']
        flag = quregs['flag_nu_lt_mu']
        ancillas = quregs['and_ancilla']
        
        # convert mu to one-hot unary
        for bit in range(self.bitsize-1):
            yield CNOT().on(mu_reg[bit+1],mu_reg[bit])

        #for each value of mu, flag (on zero) when all abs(nu_i) < 2^(mu-1) excluding when all abs(nu_i) < 2^(mu-2)
        # ie, if flag=|1>, all abs(nu_i) < 2^(mu-2)
        for i in range(self.bitsize):
            # yield MultiAnd(cvs=(1,0,0,0)).on_registers(ctrl=[[mu_reg[i]],[nu_reg[0][i]],[nu_reg[1][i]],[nu_reg[2][i]]],junk=ancillas[i],target=flag)
            yield And(cv1=1,cv2=0).on_registers(ctrl=[[mu_reg[i]],[nu_reg[0][i]]],target=ancillas[i][0])
            yield And(cv1=1,cv2=0).on_registers(ctrl=[ancillas[i][0],[nu_reg[1][i]]],target=ancillas[i][1])
            
            yield XGate().on(nu_reg[2][i])
            yield Toffoli().on_registers(ctrl=[ancillas[i][1],[nu_reg[2][i]]],target=flag)
            yield XGate().on(nu_reg[2][i])

            # MultiAnd error since And assumes target is in 0 state so it can't be used multiple times, BUT for our case we know only one bit of the mu register should be 1 so should technically be able to use. Using Toffoli until error is resolved.

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        # return {(CNOT(),self.bitsize-1),(MultiAnd(cvs=(1,0,0,0)), self.bitsize)}   
        return {(CNOT(),self.bitsize-1),(And(cv1=1,cv2=0), 2*self.bitsize),(Toffoli(),self.bitsize),(XGate(),2*self.bitsize)}   