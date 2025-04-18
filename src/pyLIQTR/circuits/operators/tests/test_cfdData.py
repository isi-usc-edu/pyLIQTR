"""
Copyright (c) 2024 Massachusetts Institute of Technology 
SPDX-License-Identifier: BSD-2-Clause
"""


import cirq
import pytest
import qualtran

from ..cfdData import F1_operator, F2_operator, F3_operator
from pyLIQTR.utils.circuit_decomposition import circuit_decompose_multi
from pyLIQTR.scheduler.scheduler import schedule_circuit
from pyLIQTR.utils.global_ancilla_manager import gam
from pyLIQTR.utils.printing import openqasm

class TestFMatrices:
    """
    call graph and circuit casting functionality
    """

    @pytest.fixture(scope = "class")
    def getMiscStruct(self):

        data = [qualtran.Register("data", qualtran.QAny(bitsize = 6)), 
                qualtran.Register("data", qualtran.QAny(bitsize = 27)),
                qualtran.Register("data", qualtran.QAny(bitsize = 40))]
        ancilla = [qualtran.Register("ancilla", qualtran.QAny(bitsize = 20)),
                   qualtran.Register("ancilla", qualtran.QAny(bitsize = 13)),
                   qualtran.Register("ancilla", qualtran.QAny(bitsize = 10))]
        signature = [qualtran.Signature([data[0], ancilla[0]]),
                     qualtran.Signature([data[1], ancilla[1]]),
                     qualtran.Signature([data[2], ancilla[2]])]
        registers = [qualtran._infra.gate_with_registers.get_named_qubits(signature[0]),
                     qualtran._infra.gate_with_registers.get_named_qubits(signature[1]),
                     qualtran._infra.gate_with_registers.get_named_qubits(signature[2])]
        return registers

    def testCallGraph(self):
        call_graph0, _0 = F1_operator().call_graph(generalizer=qualtran.resource_counting.generalizers.ignore_split_join)
        call_graph1, _1 = F2_operator().call_graph(generalizer=qualtran.resource_counting.generalizers.ignore_split_join)
        call_graph2, _2 = F3_operator().call_graph(generalizer=qualtran.resource_counting.generalizers.ignore_split_join)

        assert(True)

    def testCircuits(self, getMiscStruct):
        circuit0 = cirq.Circuit(F1_operator().on_registers(**getMiscStruct[0]))
        circuit1 = cirq.Circuit(F2_operator().on_registers(**getMiscStruct[1]))
        circuit2 = cirq.Circuit(F3_operator().on_registers(**getMiscStruct[2]))

        circuit_decompose0 = circuit_decompose_multi(circuit0, 10)
        circuit_decompose0 = circuit_decompose_multi(circuit1, 10)
        circuit_decompose0 = circuit_decompose_multi(circuit2, 10)

        assert(True)

    def testScheduler(self, getMiscStruct):
        res0 = schedule_circuit(cirq.Circuit(F1_operator().on_registers(**getMiscStruct[0])), full_profile = True, decomp_level = "Full")
        res1 = schedule_circuit(cirq.Circuit(F2_operator().on_registers(**getMiscStruct[1])), full_profile = True, decomp_level = "Full")
        res2 = schedule_circuit(cirq.Circuit(F3_operator().on_registers(**getMiscStruct[2])), full_profile = True, decomp_level = "Full")
                                
        assert(True)

    def testQASM(self, getMiscStruct):
        context = cirq.DecompositionContext(gam)

        qasm0 = openqasm(cirq.Circuit(F1_operator().on_registers(**getMiscStruct[0])), rotation_allowed = True, context = context)
        qasm1 = openqasm(cirq.Circuit(F2_operator().on_registers(**getMiscStruct[1])), rotation_allowed = True, context = context)
        qasm2 = openqasm(cirq.Circuit(F3_operator().on_registers(**getMiscStruct[2])), rotation_allowed = True, context = context)

        assert(True)