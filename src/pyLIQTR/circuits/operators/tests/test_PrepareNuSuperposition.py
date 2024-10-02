"""
Copyright (c) 2024 Massachusetts Institute of Technology 
SPDX-License-Identifier: BSD-2-Clause
"""
import pytest
import cirq
import numpy as np

from qualtran.cirq_interop.testing import assert_circuit_inp_out_cirqsim, GateHelper
from pyLIQTR.utils.global_ancilla_manager import gam as gam
from pyLIQTR.utils.printing import openqasm
from pyLIQTR.circuits.operators.PrepareNuSuperposition import PrepareNuSuperposition
from pyLIQTR.utils.resource_analysis import estimate_resources

class TestPrepareNuSuperposition:

    @pytest.mark.parametrize("bitsize,states",[(2,[(2,0b000000000000),(3,0b010000000000)]),(3,[(2,0b0000000000000000),(3,0b0010000000000000),(4,0b0110000000000000)])])
    def test_PrepareNuSuperposition(self,bitsize,states):
        '''
        Tests the gate carries out the expected operation by comparing simulation output with the expected output for different input values.
        '''
        gate = PrepareNuSuperposition(bitsize=bitsize,prepare_catalytic_state=True)
        helper = GateHelper(gate=gate)
        operation = helper.operation
        # circuit
        circuit = cirq.Circuit(operation)
        # verify simulation results for different input states of mu
        for mu,state in states:
            ## set up expected output state
            coeff = 1/np.sqrt(2**mu)
            limit = 2**(mu-1)-1
            halfway = 2**(bitsize)
            expected_out = np.zeros(2**(bitsize+1),dtype=np.complex128)
            ### positive values of nu correspond to first half of state
            expected_out[:limit+1] = coeff
            ### negative values of nu correspond to second half of state
            expected_out[halfway:halfway+limit+1] = coeff
            ## run simulation
            result = cirq.Simulator(dtype=np.complex128).simulate(helper.circuit, qubit_order=[*helper.quregs['mu'],*helper.quregs['nu'].flatten(),*helper.quregs['catalytic']],initial_state=state)
            ## isoluate nu register for each dimension
            output_state_x = cirq.sub_state_vector(result.final_state_vector,keep_indices=[*range(bitsize,(bitsize+1)+bitsize)])
            output_state_y = cirq.sub_state_vector(result.final_state_vector,keep_indices=[*range((bitsize+1)+bitsize,2*(bitsize+1)+bitsize)])
            output_state_z = cirq.sub_state_vector(result.final_state_vector,keep_indices=[*range(2*(bitsize+1)+bitsize,3*(bitsize+1)+bitsize)])
            ## assert equivalence up to global phase
            cirq.testing.assert_allclose_up_to_global_phase(expected_out, output_state_x, atol=1e-6)
            cirq.testing.assert_allclose_up_to_global_phase(expected_out, output_state_y, atol=1e-6)
            cirq.testing.assert_allclose_up_to_global_phase(expected_out, output_state_z, atol=1e-6)          
                

    def test_PrepareNuSuperposition_inverse(self):
        '''
        Tests the gate inverse correctly uncomputes by checking all qubits return to initial state when gate*(gate)**(-1) is applied.
        '''
        # initialilze operator
        gate = PrepareNuSuperposition(bitsize=3,prepare_catalytic_state=True)
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

    def test_PrepareNuSuperposition_decomposes(self):
        '''
        Tests gate decomposition existence.
        '''
        # initialilze operator
        gate = PrepareNuSuperposition(bitsize=3, prepare_catalytic_state=True)
        helper = GateHelper(gate=gate)
        operation = helper.operation
        # check decompose_once raises no error
        decomposed_once = cirq.decompose_once(operation)
        # check decompose returns decomposition not equal to operation itself
        decomposed = cirq.decompose(operation)
        assert([operation] != decomposed)

    def test_PrepareNuSuperposition_qasm(self):
        '''
        Tests qasm printing functionality.
        '''
        context = cirq.DecompositionContext(gam)
        bitsize = 3
        gate = PrepareNuSuperposition(bitsize=bitsize, prepare_catalytic_state=True)
        helper = GateHelper(gate=gate)
        operation = helper.operation

        qasm = openqasm(operation,rotation_allowed=True,context=context)
        assert qasm is not None
        for line in qasm:
            pass

    def test_PrepareNuSuperposition_resources(self):
        '''
        Tests resources are counted correctly.
        '''
        bitsize=3
        gate = PrepareNuSuperposition(bitsize=bitsize)
        gate_w_cat_prep = PrepareNuSuperposition(bitsize=bitsize,prepare_catalytic_state=True)
        resources = estimate_resources(gate)
        resources_w_cat_prep = estimate_resources(gate_w_cat_prep)

        cliffs_per_CH = 31
        num_CH = 3*(bitsize-1)

        assert resources['T'] == 4*num_CH #matches paper
        assert resources['Clifford'] == 2*3 + cliffs_per_CH*num_CH
        assert resources['LogicalQubits'] == bitsize + 3*(bitsize+1) + 1
        assert gate.t_complexity() == gate.decompose_bloq().t_complexity()

        assert resources_w_cat_prep['T'] == 4*num_CH + 1 #matches paper
        assert resources_w_cat_prep['Clifford'] == 2*3 + cliffs_per_CH*num_CH + 1
        assert resources_w_cat_prep['LogicalQubits'] == bitsize + 3*(bitsize+1) + 1
        assert gate_w_cat_prep.t_complexity() == gate_w_cat_prep.decompose_bloq().t_complexity()