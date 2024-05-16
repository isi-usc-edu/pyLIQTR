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
import numpy as np
from typing import Tuple, Sequence
from cirq._compat import cached_property
from qualtran import Signature, GateWithRegisters
from qualtran.bloqs.and_bloq import And 
from qualtran.cirq_interop.t_complexity_protocol import TComplexity
from pyLIQTR.utils.global_ancilla_manager import GlobalQubitManager as manager
from qualtran.bloqs.multi_control_multi_target_pauli import MultiControlPauli as mcmtp

class CyclicShiftPermutation(GateWithRegisters):
    """
    Implements
    __________
        A cyclic shift operator from figure 2 of https://arxiv.org/pdf/2203.10236.pdf which is of the form:

        \[L_n=\sum_{j=0}^{2^n-2}|1+j\rangle\langle j|+|0\rangle\langle2^n-1|\]

        \[R_n=\sum_{j=0}^{2^n-2}|j\rangle\langle 1+j|+|2^n-1\rangle\langle 0|\]

    Parameters
    __________
        magnitude : int
            how many qubits to be targeted 
        direction : str
            either l(eft) or r(ight) for the two seperate cyclic shift matrices
        control : Tuple[bool, int]
            whether or not to control entire translation by some int number of registers
        optimize : bool
            less T gates when decomposed into component compute/uncompute Ands

    Raises
    ______
        valueError 
            If input direction not valid
            If magnitude < 1
    """

    def __init__(self, magnitude: int, direction: str, control: Tuple[bool, int], optimize: bool):

        if not ((direction == "left") or (direction == "right")):
            if not ((direction == "l") or (direction == "r")):
                raise ValueError("direction must be l(eft) or r(ight)")

        if magnitude < 1:
            raise ValueError("magnitude must be greater than or equal to one qubit")

        self.control = control
        self.optimize = optimize
        self.__dir = direction
        self.__mag = magnitude


    @cached_property
    def vector(self) -> Tuple[int, str]:
        return (self.__mag, self.__dir)
    

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
                wire_symbols += ["CyclicShift" + str(self.vector[1][0]).upper()] * reg.total_bits()
        return cirq.CircuitDiagramInfo(wire_symbols = wire_symbols)


    def __repr__(self) -> str:
        return f"Cyclic Shift"


    def decompose_from_registers(
        self, *, context: cirq.DecompositionContext, **quregs: Sequence[cirq.Qid]
    ) -> cirq.OP_TREE:

        data = quregs["data"]
        if self.control[0]:
            control = quregs["control"]
            all = list(data) + list(control)
        elif not self.control[0]:
            all = data

        data_dict = {}
        for idx, qbit in enumerate(data):
            data_dict[qbit] = "data" + str(idx)

        if not self.optimize or self.vector[0] <= 2:
            if self.vector[1][0] == "r":
                for reg in data[1:len(list(data))]:   
                    if self.control[0]:
                        yield mcmtp((1,) * self.control[1], target_gate=cirq.X).on_registers(target=reg, controls=control)
                    elif not self.control[0]:
                        yield cirq.X.on(reg)
            for jdx, qbit in enumerate(data[::-1][:-1]):
                cvs0 = tuple((1, ) * int(data_dict[qbit].split("data")[1]))
                cvs1 = tuple((1, ) * (int(data_dict[qbit].split("data")[1]) + self.control[1]))
                if len(cvs0) > 0:
                    if self.control[0]:
                        controls = list(data[jdx:][::-1][:-1]) + list(control)
                        yield mcmtp(cvs1, target_gate=cirq.X).on_registers(target=data[jdx:][::-1][-1], controls=controls)
                    elif not self.control[0]:
                        yield mcmtp(cvs0, target_gate=cirq.X).on_registers(target=data[jdx:][::-1][-1], controls=data[jdx:][::-1][:-1])
                elif len(cvs0) == 0:
                    yield cirq.X.on(data[jdx:][::-1]) 
            if self.vector[1][0] == "l" or self.vector[0] == 1:
                if self.control[0]:
                    yield mcmtp((1,) * self.control[1], target_gate=cirq.X).on_registers(target=data[-1], controls=control)
                elif not self.control[0]:
                    yield cirq.X.on(data[-1])
            elif self.vector[1][0] == "r":
                for reg in data[1:len(list(data))-1]:   
                    if self.control[0]:
                        yield mcmtp((1,) * self.control[1], target_gate=cirq.X).on_registers(target=reg, controls=control)
                    elif not self.control[0]:
                        yield cirq.X.on(reg)
        elif self.optimize and self.vector[0] > 2:
            qm = manager()
            clean = np.array(qm.qalloc(n=(self.vector[0]+self.control[1]-2)))

            if self.vector[1][0] == "r":
                if self.control[1] <= 1:
                    yield cirq.CX.on(*all[-2:][::-1])
                if not self.control[0]:
                    yield cirq.X.on(all[-1])
            yield And().on(*all[-2:][::-1], clean[0])
            for idx in range(len(clean)-1):
                if self.vector[1][0] == "r" and (idx+self.vector[0]-2 > self.control[1]):
                    yield cirq.CX.on(clean[idx], all[-3-idx])
                yield And().on(clean[idx], all[-3-idx], clean[idx+1])
            yield cirq.CX.on(clean[-1], all[0])
            for jdx in range(len(clean)-1)[::-1]:
                yield And(uncompute=True).on(clean[jdx], all[-3-jdx], clean[jdx+1])
                if self.vector[1][0] == "l" and (jdx >= self.control[1]-2):
                    yield cirq.CX.on(clean[jdx], all[-3-jdx])
            yield And(uncompute=True).on(*all[-2:][::-1], clean[0])
            if self.vector[1][0] == "l":
                if self.control[1] <= 1:
                    yield cirq.CX.on(*all[-2:][::-1])
                if not self.control[0]:
                    yield cirq.X.on(all[-1])

            qm.qfree(qubits=[*clean])