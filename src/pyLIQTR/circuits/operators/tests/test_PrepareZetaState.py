"""
Copyright (c) 2024 Massachusetts Institute of Technology 
SPDX-License-Identifier: BSD-2-Clause
"""
import pytest
import cirq
import numpy as np
import ctypes

from qualtran.cirq_interop.testing import GateHelper
from pyLIQTR.utils.global_ancilla_manager import gam as gam
from pyLIQTR.utils.printing import openqasm
from pyLIQTR.circuits.operators.PrepareZetaState import PrepareZetaState
from pyLIQTR.utils.resource_analysis import estimate_resources
from pyLIQTR.utils.circuit_decomposition import generator_decompose

class TestPrepareZetaState:    

    @pytest.mark.parametrize("eta, L, nR, lz, is_adjoint",[(19,23,3,23,False),(33,5,4,50,False),(33,5,4,50,True)])
    def test_PrepareZetaState_decomposes(self,eta,L,nR,lz,is_adjoint):
        '''
        Tests gate decomposition existence.
        '''
        # initialilze operator
        gate = PrepareZetaState(num_atoms=L,num_bits_nuc_pos=nR,lambda_zeta=lz,eta=eta,is_adjoint=is_adjoint)
        helper = GateHelper(gate=gate)
        operation = helper.operation
        # check decompose_once raises no error
        decomposed_once = cirq.decompose_once(operation)
        # check decompose returns decomposition not equal to operation itself
        decomposed = cirq.Circuit(generator_decompose(operation))
        assert(helper.circuit != decomposed)

    @pytest.mark.parametrize("L, nR, lz, is_adjoint",[(5,4,50,False),(5,4,50,True)])
    def test_PrepareZetaState_qasm(self,L,nR,lz,is_adjoint):
        '''
        Tests qasm printing functionality.
        '''
        context = cirq.DecompositionContext(gam)
        gate = PrepareZetaState(num_atoms=L,num_bits_nuc_pos=nR,lambda_zeta=lz,is_adjoint=is_adjoint)
        helper = GateHelper(gate=gate)
        operation = helper.operation

        qasm = openqasm(operation,rotation_allowed=True,context=context)
        assert qasm is not None
        for line in qasm:
            pass

    @pytest.mark.parametrize("eta, L, nR, lz, is_adjoint",[(19,23,5,23,False),(33,3,4,50,False)])
    def test_PrepareZetaState_resources(self,eta, L, nR, lz, is_adjoint):
        '''
        Tests resources are consistent with reference and decomposition.
        '''
        gate = PrepareZetaState(num_atoms=L,num_bits_nuc_pos=nR,lambda_zeta=lz,eta=eta,is_adjoint=is_adjoint)
        helper = GateHelper(gate=gate)
        circuit = helper.circuit
        decomposed = cirq.Circuit(generator_decompose(circuit,max_decomposition_passes=2))
        resources = estimate_resources(gate,profile=True)
        resources_decomposed = estimate_resources(decomposed,profile=True)

        if L!=lz:
        # each data element is output repeatedly (corresponding to the atomic number) and so the variable spaced qrom optimization is used
            assert resources['T'] <= 4*(lz)
        else:
        # test the case when there are no repeats (no optimization) to verify paper costs
            assert resources['T'] == 4*(lz)
        assert resources['Rotations'] == 0

        for key in resources:
            if key != 'LogicalQubits':
                # unknown number of ancilla qubits allocated in decomposition
                assert resources[key] == resources_decomposed[key]

    @pytest.mark.parametrize("eta, L, nR, lz, is_adjoint",[(19,23,5,23,True),(33,3,4,50,True)])
    def test_PrepareZetaState_resources_adjoint(self,eta, L, nR, lz, is_adjoint):
        '''
        Tests resources are consistent with reference.
        '''
        gate = PrepareZetaState(num_atoms=L,num_bits_nuc_pos=nR,lambda_zeta=lz,eta=eta,is_adjoint=is_adjoint)
        helper = GateHelper(gate=gate)
        circuit = helper.circuit
        resources = estimate_resources(gate,profile=True)

        # assumes k=1 from reference but differs (lower) since not using clean ancilla implementation for fixup table.
        assert resources['T'] == 4*(np.ceil(lz/2)) 
        assert resources['Rotations'] == 0

    @pytest.mark.parametrize("L, nR, lz, is_adjoint",[(5,4,50,False),(5,4,50,True)])
    def test_PrepareZetaState_callgraph(self,L, nR, lz, is_adjoint):
        '''
        Tests ability to generate call_graph
        '''
        gate = PrepareZetaState(num_atoms=L,num_bits_nuc_pos=nR,lambda_zeta=lz,is_adjoint=is_adjoint)
        call_graph, _ = gate.call_graph()

    @pytest.mark.parametrize("eta, L, nR, lz, is_adjoint",[(19,23,5,23,False),(31,3,4,50,False)])
    def test_PrepareZetaState_iteration_length(self,eta, L, nR, lz, is_adjoint):
        gate = PrepareZetaState(num_atoms=L,num_bits_nuc_pos=nR,lambda_zeta=lz,eta=eta,is_adjoint=is_adjoint)
        iteration_length = gate.qrom_gate.selection_registers[0].dtype.iteration_length
        assert iteration_length == lz

    def test_PrepareZetaState_UserWarning(self):
        '''
        Tests the gate warns when number of precision bits is too high.
        '''
        max_width = int(np.log2((1 << (8 * ctypes.sizeof(ctypes.c_long) - 1)) - 1))
        nR = int(np.ceil(max_width/3))+1
        with pytest.warns(UserWarning, match='Desired precision'):
            gate = PrepareZetaState(num_atoms=23,num_bits_nuc_pos=nR,lambda_zeta=50,eta=19,is_adjoint=False)
            data = gate.R_l_data
