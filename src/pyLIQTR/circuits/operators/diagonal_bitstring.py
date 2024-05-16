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
import warnings
import numpy as np
from typing import Sequence, Tuple
from qualtran.bloqs.and_bloq import And 
from cirq._compat import cached_property
from qualtran import Signature, GateWithRegisters
from qualtran.cirq_interop.t_complexity_protocol import TComplexity
from pyLIQTR.utils.global_ancilla_manager import GlobalQubitManager as manager
from qualtran.bloqs.multi_control_multi_target_pauli import MultiControlPauli as mcmtp

class DiagonalBitstring(GateWithRegisters):
    """
    Implements
    __________
        A block encoding of a diagonal matrix whose diagonal is a bitstring.
        Currently only supports all 1/0s with one 0/1 and specific use case for Carleman.
        In principle O(logn) MCX gates are needed for arbitrary bitstrings.

    Parameters
    __________
        magnitude : int
            2^magnitude basis states in block encoded diagonal
        ancilla : cirq.Qid
            which register will act as ancilla 
        flag : Tuple[int]
            (0/1, j) - one 0/1 at index j with others 1/0s
            (n, K) - \sum_{j=1}^K 2^(nj) ones followed by 
                     2(2^(nK))-\sum_{j=1}^K 2^(nj) zeros

    Raises
    ______
        valueError
            Invalid flag
            Invalid magnitude
        Warning
            flag[0] == 1 not supported for n and K use case
    """

    def __init__(self, magnitude: int, flag: Tuple[int, int], optimize: bool = False):

        if (flag[0] == 0 or flag[0] == 1) and flag[1] >= 2**magnitude:
            raise ValueError("The flag you entered must have second index less than 2**magnitude+1")
        
        if (flag[0] != 0 and flag[0] != 1) and (magnitude < 1 + flag[0] * flag[1]):
            raise ValueError("Incompatible flag and magnitude values")
        
        if flag[0] == 1:
            warnings.warn("flag (n, K) not supported with n=1")
        
        if magnitude < 1:
            raise ValueError("The magnitude must be positive")

        self.__flag = flag
        self.__mag = magnitude
        self.optimize = optimize


    @cached_property
    def vector(self) -> Tuple[int, Tuple[int, int]]:
        return (self.__mag, self.__flag)


    @cached_property
    def signature(self) -> Signature:
        return Signature.build(ancilla = 1, data = self.vector[0])


    def _circuit_diagram_info_(self, _) -> cirq.CircuitDiagramInfo:
        wire_symbols = []
        for reg in self.signature:
            if reg.name == "ancilla":
                wire_symbols += ["ancilla"] 
            else:
                wire_symbols += ["DBString"] * reg.total_bits()
        return cirq.CircuitDiagramInfo(wire_symbols = wire_symbols)
    

    def __repr__(self) -> str:
        return f"Diagonal Bitstring"
    

    def _t_complexity_(self) -> TComplexity:
        if self.optimize and self.vector[0] > 1:
            return TComplexity(t=4*self.vector[1][0]*self.vector[1][1],
                            clifford=7*self.vector[1][0]*self.vector[1][1]+3*self.vector[1][1]+4)


    def decompose_from_registers(
        self, *, context: cirq.DecompositionContext, **quregs: Sequence[cirq.Qid]
        ) -> cirq.OP_TREE:

        data = quregs["data"]
        anc = quregs["ancilla"]
        qbits = list(anc) + list(data)

        if not self.vector[1][0] == 0:
            yield cirq.X.on(anc[0])
        
        if self.vector[1][0] == 0 or self.vector[1][0] == 1:
            open_reg = f'{self.vector[1][1]:0{len(qbits) - 1}b}'
            for idx, qbit in enumerate(range(len(open_reg))):
                if int(open_reg[qbit]) == 0:
                    yield cirq.X.on(data[idx])
            cvs = tuple((1, ) * len(data))
            yield mcmtp(cvs, target_gate=cirq.X).on_registers(target=anc, controls=data)
            for idx, qbit in enumerate(range(len(open_reg))):
                if int(open_reg[qbit]) == 0:
                    yield cirq.X.on(data[idx])
        else:                    
            for kdx, qbit in enumerate(range(self.vector[1][1]*self.vector[1][0] + 1)):
                if not kdx % self.vector[1][0]:
                     yield cirq.X.on(data[kdx])
            if not self.optimize:
                for jdx in range(self.vector[1][1] + 1):
                    temp_bitstring = ""
                    for idx in range(jdx * self.vector[1][0] + 1):
                        if not idx % self.vector[1][0]:
                            temp_bitstring += "0"
                        else:
                            temp_bitstring += "1"
                    cvs = tuple((1, ) * len(temp_bitstring))
                    yield mcmtp(cvs, target_gate=cirq.X).on_registers(target=anc, controls=data[:len(temp_bitstring)])
            elif self.optimize:
                qm = manager()
                clean = np.array(qm.qalloc(n=(self.vector[1][0]*self.vector[1][1])))
                yield cirq.CX.on(data[0], anc[0])
                yield And().on(*data[:2], clean[0])

                for nkdx in range(len(clean)-1):
                    yield And().on(clean[nkdx], data[nkdx+2], clean[nkdx+1])
                    if not (nkdx % self.vector[1][1]):
                        yield cirq.CX.on(clean[nkdx+1], anc[0])
                for nkdx in range(len(clean)-1)[::-1]:
                    yield And(uncompute=True).on(clean[nkdx], data[nkdx+2], clean[nkdx+1])
                
                yield And(uncompute=True).on(*data[:2], clean[0])
                qm.qfree(qubits=[*clean])
            for kdx, qbit in enumerate(range(self.vector[1][1]*self.vector[1][0] + 1)):
                if not kdx % self.vector[1][0]:
                     yield cirq.X.on(data[kdx])