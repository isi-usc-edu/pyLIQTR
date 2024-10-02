"""
Copyright (c) 2024 Massachusetts Institute of Technology 
SPDX-License-Identifier: BSD-2-Clause
"""


from typing import Tuple
from functools import cached_property
from qualtran import GateWithRegisters
from qualtran.cirq_interop.t_complexity_protocol import TComplexity
from pyLIQTR.ProblemInstances.ProblemInstance import ProblemInstance
from pyLIQTR.circuits.operators.ExampleFMatrices import F0_operator, F1_operator, F2_operator

class FOperators(ProblemInstance):

    def __init__(self, magnitude : int = 2, truncation : int = 4,
                 ancilla : Tuple[int] = (2, 2, 1, 2), scaling : Tuple[int] = (1, 1, 1),
                 t_cost : Tuple[TComplexity] = (TComplexity(t=0,clifford=0),)*3):
        
        self.magnitude = magnitude
        self.truncation = truncation

        self.ancilla = ancilla 
        self.scaling = scaling
        self.t_cost = t_cost
    

    @cached_property
    def vector(self) -> Tuple[int]:
        return (self.magnitude, self.truncation)
    

    def __str__(self):
        return None
    

    def n_qubits(self):
        return None


    def F0_oracle(self) -> GateWithRegisters:
        return F0_operator(self.vector[0], self.ancilla[0], self.scaling[0], self.t_cost[0])
    

    def F1_oracle(self) -> GateWithRegisters:
        return F1_operator(self.vector[0], self.ancilla[1], self.scaling[1], self.t_cost[1])
    

    def F2_oracle(self) -> GateWithRegisters:
        return F2_operator(self.vector[0], *self.ancilla[-2:], self.scaling[2], self.t_cost[2])