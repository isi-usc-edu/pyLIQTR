"""
DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.

This material is based upon work supported by the Under Secretary of Defense for
Research and Engineering under Air Force Contract No. FA8702-15-D-0001. Any opinions,
findings, conclusions or recommendations expressed in this material are those of the
author(s) and do not necessarily reflect the views of the Under Secretary of Defense
for Research and Engineering.

© 2022 Massachusetts Institute of Technology.

The software/firmware is provided to you on an As-Is basis

Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS Part
252.227-7013 or 7014 (Feb 2014). Notwithstanding any copyright notice, U.S. Government
rights in this work are defined by DFARS 252.227-7013 or DFARS 252.227-7014 as detailed
above. Use of this work other than as specifically authorized by the U.S. Government
may violate any copyrights that exist in this work.
"""

import cirq
from typing import Tuple, Sequence
from cirq._compat import cached_property
from qualtran import Signature, GateWithRegisters
from pyLIQTR.circuits.operators.cyclic_shift import CyclicShiftPermutation

class Translation(GateWithRegisters):
    """
    Implements
    __________
        A translation operator which is of the form:

        \[L_{n,k}=\sum_{j=0}^{2^n-k-1}|k+j\rangle\langle j|+\sum_{j=0}^{k-1}|j\rangle\langle2^n-k-j|\]

        \[R_{n,k}=\sum_{j=0}^{2^n-k-1}|j\rangle\langle k+j|+\sum_{j=0}^{k-1}|2^n-k-j\rangle\langle j|\]

        where the above unitaries differ from cyclicShiftPermutation in that they can permute any length permutation
        with less than exponential scaling gates, i.e., using the recursive relationship:
        
        \[L_{k,n}=\prod_{j=0}^{n-1}b_kL_{2^j,j}\otimes I^{\otimes n-j-1}\] 
        
        where $b_k\in\{0,1\}$ and $k=\sum_jb_k2^j$.

    Parameters
    __________
        magnitude : int
            how many qubits to be targeted 
        length : Tuple[int, bool]
            how many basis elements to permute
            can also pass (exponent, False) if you are permuting by a power of two
        direction : str
            either l(eft) or r(ight) for the two seperate cyclic shift matrices
        control : Tuple[bool, int]
            whether or not to control entire translation by some int number of registers
        optimize : bool
            decomposing cyclic shift into And ladders improves T complexity

    Raises
    ______
        valueError 
            If input direction not valid
            If magnitude < 2
            If length invalid
    """

    def __init__(self, magnitude: int, length: Tuple[int, bool], direction: str, control: Tuple[bool, int], optimize : bool):

        if not ((direction == "left") or (direction == "right")):
            if not ((direction == "l") or (direction == "r")):
                raise ValueError("direction must be l(eft) or r(ight)")

        if magnitude < 1:
            raise ValueError("magnitude must be greater than or equal to one qubit")
        if length[1]:
            if length[0] < 1 or length[0] >= 2**magnitude:
                raise ValueError("invalid length")
        else:
            if length[0] > magnitude and length[1]:
                raise ValueError("invalid length")
        
        self.control = control
        self.optimize = optimize

        self.__dir = direction
        self.__mag = magnitude
        self.__len = length


    @cached_property
    def vector(self) -> Tuple[int, int, str]:
        return (self.__mag, self.__len, self.__dir)
    

    @cached_property # don't forget to cache this
    def signature(self) -> Signature:
        if self.control[0]:
            return Signature.build(data = self.vector[0], control = self.control[1])
        else:
            return Signature.build(data = self.vector[0])


    def _circuit_diagram_info_(self, _) -> cirq.CircuitDiagramInfo:
        wire_symbols = []
        for reg in self.signature:
            if reg.name == "control":
                wire_symbols += ["control"] * reg.total_bits()
            else:
                if self.vector[1][1]:
                    wire_symbols += ["Translation" + str(self.vector[2][0]).upper()
                                 + str(self.vector[1][0])] * reg.total_bits()
                else:
                    wire_symbols += ["Translation" + str(self.vector[2][0]).upper()
                                 + "2**" + str(self.vector[1][0])] * reg.total_bits()
        return cirq.CircuitDiagramInfo(wire_symbols = wire_symbols)


    def __repr__(self) -> str:
        return f"Translation"


    def decompose_from_registers(
        self, *, context: cirq.DecompositionContext, **quregs: Sequence[cirq.Qid]
    ) -> cirq.OP_TREE:

        data = quregs["data"]
        if self.control[0]:
            control = quregs["control"]

        if self.vector[1][1]:
            for idx, bit in enumerate(f'{self.vector[1][0]:0{len(list(data))}b}'[::-1]):
                if int(bit) == 1:
                    if not self.control[0]:
                        yield CyclicShiftPermutation(len(data) - idx, self.vector[2], (False, 0), self.optimize).on_registers(data = data[0:len(data)-idx])
                    elif self.control[0]:
                        yield CyclicShiftPermutation(len(data) - idx, self.vector[2], (True, self.control[1]), self.optimize).on_registers(data = data[0:len(data)-idx], control = control)
        elif not self.vector[1][1]:
            register = self.vector[1][0]
            if not self.control[0]:
                yield CyclicShiftPermutation(len(data)-register, self.vector[2], (False, 0), self.optimize).on_registers(data = data[0:(len(data)-register)])
            elif self.control[0]:
                yield CyclicShiftPermutation(len(data)-register, self.vector[2], (True, self.control[1]), self.optimize).on_registers(data = data[0:(len(data)-register)], control = control)