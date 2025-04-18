"""
Copyright (c) 2024 Massachusetts Institute of Technology 
SPDX-License-Identifier: BSD-2-Clause
"""
import pytest
import cirq
import numpy as np

from qualtran.cirq_interop.testing import assert_circuit_inp_out_cirqsim, GateHelper
from qualtran import QUInt
from pyLIQTR.utils.printing import openqasm
from pyLIQTR.circuits.operators.FlagNuLessThanMu import FlagNuLessThanMu
from pyLIQTR.utils.resource_analysis import estimate_resources
from pyLIQTR.circuits.operators.tests.utils import ignore_bookkeeping

class TestFlagNuLessThanMu:

    @pytest.mark.parametrize("bitsize",[2])
    def test_FlagNuLessThanMu(self,bitsize):
        '''
        Tests the gate carries out the expected operation by comparing simulation output with the expected output for different input values.
        '''
        gate = FlagNuLessThanMu(bitsize=bitsize)
        helper = GateHelper(gate=gate)
        # circuit
        circuit = helper.circuit
        for mu in range(1,bitsize+1):
            # define input state so mu is unary encoded and all other bits are 0
            mu_binary = [1 if i<mu else 0 for i in range(bitsize)]
            mu_unary = [1 if i==mu-1 else 0 for i in range(bitsize)]
            for nu_x in [[0]*(bitsize+1),[1]*(bitsize+1)]:
                input_state = mu_binary + nu_x + [0]*(2*(bitsize+1)+2*bitsize+1)
                # simulate
                result = cirq.Simulator(dtype=np.complex128).simulate(helper.circuit, qubit_order=[*helper.quregs['mu'],*helper.quregs['nu'].flatten(),*helper.quregs['and_ancilla'].flatten(),*helper.quregs['flag_nu_lt_mu']],initial_state=input_state)
                # isolate mu register and success flag
                dirac_actual_sub_state = cirq.dirac_notation(cirq.sub_state_vector(result.final_state_vector,keep_indices=[*range(bitsize)]+[len(input_state)-1]))
                string_actual = dirac_actual_sub_state.split(')')[-1][1:-1]
                # define expected ouput to have mu one hot unary and flag on/off depending on nu_x
                if any([1==bit for bit in nu_x]):
                    should_be = "".join(str(x) for x in mu_unary+[0])
                else:
                    should_be = "".join(str(x) for x in mu_unary+[1])
                # assert equivalence
                assert string_actual == should_be, f'\n{string_actual=}\n{should_be=}\n{input_state=}'

    @pytest.mark.parametrize("bitsize",[2])
    def test_FlagNuLessThanMu_inverse(self,bitsize):
        '''
        Tests the gate inverse correctly uncomputes by checking all qubits return to initial state when gate*(gate)**(-1) is applied.
        '''
        # initialilze operator
        gate = FlagNuLessThanMu(bitsize=bitsize)
        helper = GateHelper(gate=gate)
        operation = helper.operation
        # circuit
        circuit = cirq.Circuit(operation)
        circuit.append(operation**(-1))
        # output states equal input which is |0>
        # verify simulated results
        qubits = circuit.all_qubits()
        init_state = [0]*len(qubits)
        assert_circuit_inp_out_cirqsim(circuit,qubit_order=[*qubits],inputs=init_state,outputs=init_state)

    @pytest.mark.parametrize("bitsize",[3])
    def test_FlagNuLessThanMu_decomposes(self,bitsize):
        '''
        Tests gate decomposition existence.
        '''
        # initialilze operator
        gate = FlagNuLessThanMu(bitsize=bitsize)
        helper = GateHelper(gate=gate)
        operation = helper.operation
        # check decompose_once raises no error
        decomposed_once = cirq.decompose_once(operation)
        # check decompose returns decomposition not equal to operation itself
        decomposed = cirq.decompose(operation)
        assert([operation] != decomposed)

    def test_FlagNuLessThanMu_qasm(self):
        '''
        Tests qasm printing functionality.
        '''
        bitsize = 3
        gate = FlagNuLessThanMu(bitsize=bitsize)
        helper = GateHelper(gate=gate)
        operation = helper.operation

        qasm = openqasm(operation,rotation_allowed=True)
        assert qasm is not None
        for line in qasm:
            pass

    def test_FlagNuLessThanMu_resources(self):
        '''
        Tests resources are counted correctly.
        '''
        bitsize=3
        gate = FlagNuLessThanMu(bitsize=bitsize)
        resources = estimate_resources(gate)

        cliffs_for_zero_controls = 6*bitsize
        cliffs_for_and = 9
        num_CNOTS = bitsize-1
        num_ands = 2*bitsize

        assert resources['T'] == 4*3*bitsize # matches reference
        assert resources['Clifford'] == num_CNOTS + num_ands*(cliffs_for_and) + cliffs_for_zero_controls # clifford count doesn't include the Toffolis with qt v0.4.0
        assert resources['LogicalQubits'] == 3*(bitsize + 1) + bitsize + 1 + 2*bitsize
        assert gate.t_complexity() == gate.decompose_bloq().t_complexity()

    @pytest.mark.parametrize("bitsize",[2,6])
    def test_FlagNuLessThanMu_callgraph(self,bitsize):
        '''
        Tests ability to generate call_graph and that call_graph counts match decomposition counts.
        '''
        gate = FlagNuLessThanMu(bitsize=bitsize)
        call_graph, _ = gate.call_graph()

        call_graph_counts = gate.bloq_counts()
        cbloq = gate.decompose_bloq()
        decomp_counts = cbloq.bloq_counts(generalizer=ignore_bookkeeping)

        assert call_graph_counts == decomp_counts