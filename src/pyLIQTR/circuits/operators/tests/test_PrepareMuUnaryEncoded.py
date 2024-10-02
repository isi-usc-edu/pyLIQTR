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
from pyLIQTR.circuits.operators.PrepareMuUnaryEncoded import PrepareMuUnaryEncoded
from pyLIQTR.utils.resource_analysis import estimate_resources

class TestPrepareMuUnaryEncoded:

    def test_PrepareMuUnaryEncoded(self):
        '''
        Tests the gate carries out the expected operation by comparing simulation output with the expected output for different input values.
        '''
        bitsize=4
        gate = PrepareMuUnaryEncoded(bitsize=bitsize,prepare_catalytic_state=True)
        helper = GateHelper(gate=gate)
        operation = helper.operation
        # circuit
        circuit = cirq.Circuit(operation)
        # verify simulation results
        result = cirq.Simulator(dtype=np.complex128).simulate(circuit, qubit_order=[*helper.quregs['mu'],*helper.quregs['catalytic']])
        output_state = cirq.sub_state_vector(result.final_state_vector,keep_indices=[*range(bitsize)])
        ## set up expected output state
        coeffs = [1/np.sqrt(2**bitsize)]+ [1/np.sqrt(2**(bitsize+2))*2**(m/2) for m in range(2,bitsize+2)]
        expected_out = np.array([coeffs[0]]+[coeffs[int(np.log2(i+1))]+0j if (i+1).bit_count()==1 else 0j for i in range(1,2**bitsize)])
        ## assert equivalence up to global phase
        cirq.testing.assert_allclose_up_to_global_phase(expected_out, output_state, atol=1e-6)
          
                

    def test_PrepareMuUnaryEncoded_inverse(self):
        '''
        Tests the gate inverse correctly uncomputes by checking all qubits return to initial state when gate*(gate)**(-1) is applied.
        '''
        # initialilze operator
        gate = PrepareMuUnaryEncoded(bitsize=3)
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

    def test_PrepareMuUnaryEncoded_decomposes(self):
        '''
        Tests gate decomposition existence.
        '''
        # initialilze operator
        gate = PrepareMuUnaryEncoded(bitsize=3)
        helper = GateHelper(gate=gate)
        operation = helper.operation
        # check decompose_once raises no error
        decomposed_once = cirq.decompose_once(operation)
        # check decompose returns decomposition not equal to operation itself
        decomposed = cirq.decompose(operation)
        assert([operation] != decomposed)

    def test_PrepareMuUnaryEncoded_qasm(self):
        '''
        Tests qasm printing functionality.
        '''
        context = cirq.DecompositionContext(gam)
        bitsize = 3
        gate = PrepareMuUnaryEncoded(bitsize=bitsize)
        helper = GateHelper(gate=gate)
        operation = helper.operation

        qasm = openqasm(operation,rotation_allowed=True,context=context)
        assert qasm is not None
        for line in qasm:
            pass

    def test_PrepareMuUnaryEncoded_resources(self):
        '''
        Tests resources are counted correctly.
        '''
        bitsize=3
        gate = PrepareMuUnaryEncoded(bitsize=bitsize)
        resources = estimate_resources(gate)

        cliffs_per_CH = 31
        num_CH = bitsize-1

        assert resources['T'] == 4*num_CH #matches paper
        assert resources['Clifford'] == bitsize+1 + cliffs_per_CH*num_CH
        assert resources['LogicalQubits'] == bitsize+1
        assert gate.t_complexity() == gate.decompose_bloq().t_complexity()