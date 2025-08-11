"""
Copyright (c) 2024 Massachusetts Institute of Technology 
SPDX-License-Identifier: BSD-2-Clause
"""
import numpy as np
from functools import cached_property
from typing import Set, Dict
from qualtran import Bloq, Signature, Register, QIntOnesComp, QInt, QUInt,Side, BloqBuilder, SoquetT, QAny
from qualtran.bloqs.basic_gates import CNOT
from qualtran.bloqs.arithmetic.addition import Add

class SignedIntegerToTwosComplement(Bloq):
    """Convert a register storing the signed integer representation to two's complement inplace.

    Registers:
        x: represents signed integer (ones' complement) on input and is output in two's complement.

    References:
        [Fault-Tolerant Quantum Simulations of Chemistry in First Quantization](https://arxiv.org/abs/2105.12767).
        page 24, 4th paragraph from the bottom.

    :param int bitsize: size of the register.
    """

    def __init__(self,bitsize:int):
        self.bitsize = bitsize

    @cached_property
    def signature(self) -> Signature:
        return Signature(
            [
                Register('x', QAny(self.bitsize))
            ]
        )

    def build_composite_bloq(
        self, bb: BloqBuilder, x: SoquetT
    ) -> Dict[str, 'SoquetT']:

        xs = bb.split(x)
        sign_bit = xs[0]

        for i in range(1,self.bitsize):
            # note, there are alternate multi target CNOT constructions which lower clifford depth but increase count: https://arxiv.org/pdf/1812.00954 App B.1
            sign_bit, xs[i] = bb.add(CNOT(),ctrl=sign_bit, target=xs[i])
        
        sign_bit, new_x = bb.add(Add(a_dtype=QUInt(1),b_dtype=QUInt(self.bitsize-1)),a=sign_bit,b=bb.join(xs[1:]))
        new_xs = bb.split(new_x)

        return {'x':bb.join(np.insert(new_xs,0,sign_bit))}

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        return {(Add(a_dtype=QUInt(1),b_dtype=QUInt(self.bitsize-1)), 1),(CNOT(),self.bitsize-1)}

    def _num_qubits_(self) -> int:
        return self.bitsize