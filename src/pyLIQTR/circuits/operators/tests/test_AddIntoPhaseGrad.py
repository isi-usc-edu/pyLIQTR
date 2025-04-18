"""
Copyright (c) 2024 Massachusetts Institute of Technology 
SPDX-License-Identifier: BSD-2-Clause
"""
import pytest
import cirq
import numpy as np

from qualtran.cirq_interop.testing import assert_circuit_inp_out_cirqsim, GateHelper
from qualtran import QUInt, DecomposeNotImplementedError
from qualtran.bloqs.rotations.phase_gradient import PhaseGradientState
from pyLIQTR.circuits.operators.tests.utils import get_partial_state
from pyLIQTR.utils.printing import openqasm
from pyLIQTR.circuits.operators.AddIntoPhaseGrad import AddIntoPhaseGradwithDecompose
from pyLIQTR.utils.resource_analysis import estimate_resources
from pyLIQTR.circuits.operators.tests.utils import ignore_bookkeeping

class TestAddIntoPhaseGrad:

    @pytest.mark.parametrize("sign,ell,x_bitsize,phase_bitsize",[(1,5,3,3),(-1,2,3,3),(1,3,2,2),(1,1,1,1)])
    def test_AddIntoPhaseGrad(self,sign,x_bitsize,phase_bitsize,ell):
        '''
        Tests the gate carries out the expected operation by comparing simulation output with the expected output for different input values.
        '''
        gate = AddIntoPhaseGradwithDecompose(x_bitsize=x_bitsize,phase_bitsize=phase_bitsize,sign=sign)
        helper = GateHelper(gate=gate)
        # create registers
        x_reg = helper.quregs['x']
        phi = helper.quregs['phase_grad']

        # get binary representation of ell
        ell_bits = QUInt(bitsize=x_bitsize).to_bits(ell)

        circuit = cirq.Circuit()
        # prepare x register
        for i,bit in enumerate(x_reg):
            if ell_bits[i]:
                circuit.append(cirq.X.on(bit))

        # prepare phase gradient state
        create_phase_grad_state = PhaseGradientState(bitsize=phase_bitsize)
        circuit.append(create_phase_grad_state.on(*phi))

        # add into phase gradient state
        circuit.append(helper.operation)

        # simulate
        res = cirq.Simulator(dtype=np.complex128).simulate(circuit,qubit_order=[*x_reg,*phi])
        # isolate final state of x
        partial_state_x = get_partial_state(res.final_state_vector,keep_indices=[*range(x_bitsize)])
        # isolate final state of phase gradient register
        partial_state_phi = get_partial_state(res.final_state_vector,keep_indices=np.arange(x_bitsize,x_bitsize+phase_bitsize))

        # expected output for phase gradient state (phi) based on analytic form 
        phi_pref = 1/np.sqrt(2**phase_bitsize)
        expected_phi_out=np.round(phi_pref*np.array([np.exp(-2j*np.pi*(n-sign*ell)/(2**(phase_bitsize))) for n in range(2**phase_bitsize)]),8)
        # expected output for x is input
        expected_x_out = cirq.to_valid_state_vector(ell,num_qubits=x_bitsize)

        full_expected_out = np.kron(expected_x_out,expected_phi_out)

        # assert circuit final state vectors match expected outputs
        np.testing.assert_allclose(res.final_state_vector-full_expected_out, 0+0.j, atol=1e-07)

    def test_AddIntoPhaseGrad_RaisesErrors(self):
        '''
        Tests the gate raises ValueError for b_bitsize > a_bitsize.
        '''
        with pytest.raises(DecomposeNotImplementedError, match='Decomposition not implemented for'):
            gate = AddIntoPhaseGradwithDecompose(x_bitsize=3,phase_bitsize=2)
        with pytest.raises(DecomposeNotImplementedError, match='Decomposition not implemented for'):
            gate = AddIntoPhaseGradwithDecompose(x_bitsize=4,phase_bitsize=5)
        with pytest.raises(DecomposeNotImplementedError, match='right_shift!=0'):
            gate = AddIntoPhaseGradwithDecompose(x_bitsize=3,phase_bitsize=3,right_shift=1)
        with pytest.raises(DecomposeNotImplementedError, match='right_shift!=0'):
            gate = AddIntoPhaseGradwithDecompose(x_bitsize=3,phase_bitsize=3,sign=0)
        with pytest.raises(DecomposeNotImplementedError, match='right_shift!=0'):
            gate = AddIntoPhaseGradwithDecompose(x_bitsize=3,phase_bitsize=3,controlled_by=1)

    @pytest.mark.parametrize("sign,ell,x_bitsize,phase_bitsize",[(1,4,3,3),(-1,5,3,3),(1,2,2,2)])
    def test_AddIntoPhaseGrad_inverse(self,sign,ell,x_bitsize,phase_bitsize):
        '''
        Tests the gate inverse correctly uncomputes by checking all qubits return to initial state when gate*(gate)**(-1) is applied.
        '''
        gate = AddIntoPhaseGradwithDecompose(x_bitsize=x_bitsize,phase_bitsize=phase_bitsize,sign=sign)
        helper = GateHelper(gate=gate)
        # create registers
        x_reg = helper.quregs['x']
        phi = helper.quregs['phase_grad']

        # get binary representation of ell
        ell_bits = QUInt(bitsize=x_bitsize).to_bits(ell)

        circuit = cirq.Circuit()
        # prepare x register
        for i,bit in enumerate(x_reg):
            if ell_bits[i]:
                circuit.append(cirq.X.on(bit))

        # prepare phase gradient state
        create_phase_grad_state = PhaseGradientState(bitsize=phase_bitsize)
        circuit.append(create_phase_grad_state.on(*phi))

        # add(or subtract) into phase gradient state 
        circuit.append(helper.operation)

        # invert circuit
        circuit.append(helper.operation**(-1))
        circuit.append(create_phase_grad_state.on(*phi)**(-1))

        for i,bit in enumerate(x_reg):
            if ell_bits[i]:
                circuit.append(cirq.X.on(bit))

        # assert circuit final state vectors match expected outputs
        qubits = circuit.all_qubits()
        init_state = [0]*len(qubits)
        assert_circuit_inp_out_cirqsim(circuit,qubit_order=[*qubits],inputs=init_state,outputs=init_state)

    @pytest.mark.parametrize("bitsize,sign",[(3,1),(3,-1),(2,1),(1,1)])
    def test_AddIntoPhaseGrad_decomposes(self,bitsize,sign):
        '''
        Tests gate decomposition existence.
        '''
        # initialilze operator
        gate = AddIntoPhaseGradwithDecompose(x_bitsize=bitsize, phase_bitsize=bitsize,sign=sign)
        helper = GateHelper(gate=gate)
        operation = helper.operation
        # check decompose_once raises no error
        decomposed_once = cirq.decompose_once(operation)
        # check decompose returns decomposition not equal to operation itself
        decomposed = cirq.decompose(operation)
        assert([operation] != decomposed)

    @pytest.mark.parametrize("sign",[1,-1])
    def test_AddIntoPhaseGrad_qasm(self,sign):
        '''
        Tests qasm printing functionality.
        '''
        bitsize = 3
        gate = AddIntoPhaseGradwithDecompose(x_bitsize=bitsize, phase_bitsize=bitsize,sign=sign)
        helper = GateHelper(gate=gate)
        operation = helper.operation

        qasm = openqasm(operation,rotation_allowed=True)
        assert qasm is not None
        for line in qasm:
            pass

    @pytest.mark.parametrize("sign,bitsize",[(1,3),(-1,3),(1,2)])
    def test_AddIntoPhaseGrad_resources(self,sign,bitsize):
        '''
        Tests resources are counted correctly.
        '''
        gate = AddIntoPhaseGradwithDecompose(x_bitsize=bitsize,phase_bitsize=bitsize,sign=sign)
        resources = estimate_resources(gate)
        
        assert resources['T'] == 4*(bitsize-2) # matches reference
        assert resources['LogicalQubits'] == 2*bitsize # doesn't count ancilla
        assert gate.t_complexity() == gate.decompose_bloq().t_complexity()

    @pytest.mark.parametrize("bitsize,sign",[(3,1),(6,-1),(1,1),(2,1),(2,-1)])
    def test_AddIntoPhaseGrad_callgraph(self,bitsize,sign):
        '''
        Tests ability to generate call_graph and that call_graph counts match decomposition counts.
        '''
        gate = AddIntoPhaseGradwithDecompose(x_bitsize=bitsize,phase_bitsize=bitsize,sign=sign)
        call_graph, _ = gate.call_graph()

        call_graph_counts = gate.bloq_counts()
        cbloq = gate.decompose_bloq()
        decomp_counts = cbloq.bloq_counts(generalizer=ignore_bookkeeping)

        assert call_graph_counts == decomp_counts