"""
Copyright (c) 2024 Massachusetts Institute of Technology 
SPDX-License-Identifier: BSD-2-Clause
"""
import pytest
import cirq
import numpy as np

from qualtran.cirq_interop.testing import assert_circuit_inp_out_cirqsim
from  pyLIQTR.utils.global_ancilla_manager import gam as gam
from  pyLIQTR.utils.printing import openqasm
from pyLIQTR.circuits.operators.PhaseGradientRotation import approx_angle_with_br_bits
from pyLIQTR.circuits.operators.RotationsQROM import RotationsQROM
from pyLIQTR.utils.resource_analysis import estimate_resources

class TestRotationsQROM:

    @pytest.mark.parametrize("n_select,n_data",[(4,4),(5,3),(4,10)])
    def test_RotationsQROM(self,n_select,n_data):
        '''
        Tests the gate carries out the expected operation by comparing simulation output with the expected output for different input values.
        '''
        # create registers
        num_coeffs = 2**n_select
        selection = cirq.NamedQubit.range(n_select,prefix='select')
        data_reg = cirq.NamedQubit.range(n_data,prefix='data')
        # create random data array
        np.random.seed(0)
        givens_angles = np.random.randint(low=0,high=2,size=(num_coeffs,n_data)) #each entry is randomly set to 0 or 1
        # initialilze oeprator
        qrom_rotations = RotationsQROM(
             [givens_angles],
            selection_bitsizes=(n_select,),
            target_bitsizes=(n_data,)
        )
        operation = qrom_rotations.on_registers(selection=selection,target0_=data_reg)
        # circuit
        circuit = cirq.Circuit(operation)
        # loop over selection indices
        for index in range(num_coeffs):
            binary_index = list(iter_bits(index,width=n_select))
            # input state
            input_state = binary_index + [0]*n_data
            # output state
            output_state = binary_index + list(givens_angles[index])
            # verify simulated results
            assert_circuit_inp_out_cirqsim(circuit,qubit_order=[*selection,*data_reg],inputs=input_state,outputs=output_state)

    @pytest.mark.parametrize("n_select,n_data",[(4,4),(5,3)])
    def test_RotationsQROM_inverse(self,n_select,n_data):
        '''
        Tests the gate inverse correctly uncomputes by checking all qubits return to initial state when gate*(gate)**(-1) is applied.
        '''
        # create registers
        num_coeffs = 2**n_select
        selection = cirq.NamedQubit.range(n_select,prefix='select')
        data_reg = cirq.NamedQubit.range(n_data,prefix='data')
        # create random data array
        np.random.seed(0)
        givens_angles = np.random.randint(low=0,high=2,size=(num_coeffs,n_data)) #each entry is randomly set to 0 or 1
        # initialilze oeprator
        qrom_rotations = RotationsQROM(
            [givens_angles],
            selection_bitsizes=(n_select,),
            target_bitsizes=(n_data,)
        )
        operation = qrom_rotations.on_registers(selection=selection,target0_=data_reg)
        # circuit
        circuit = cirq.Circuit(operation)
        circuit.append(operation**(-1))
        # loop over selection indices
        for index in range(num_coeffs):
            binary_index = list(iter_bits(index,width=n_select))
            # input state
            input_state = binary_index + [0]*n_data
            # output state equals input
            # verify simulated results
            assert_circuit_inp_out_cirqsim(circuit,qubit_order=[*selection,*data_reg],inputs=input_state,outputs=input_state)

    @pytest.mark.parametrize("n_select,n_data",[(4,4),(5,3)])
    def test_RotationsQROM_measurement_uncompute(self,n_select,n_data):
        '''
        Tests the method measurement_uncompute correctly uncomputes by checking all qubits return to initial state.
        '''
        # create registers
        num_coeffs = 2**n_select
        selection = cirq.NamedQubit.range(n_select,prefix='select')
        data_reg = cirq.NamedQubit.range(n_data,prefix='data')
        # create random data array
        np.random.seed(0)
        givens_angles = np.random.randint(low=0,high=2,size=(num_coeffs,n_data)) #each entry is randomly set to 0 or 1
        # initialilze oeprator
        qrom_rotations = RotationsQROM(
            [givens_angles],
            selection_bitsizes=(n_select,),
            target_bitsizes=(n_data,)
        )
        operation = qrom_rotations.on_registers(selection=selection,target0_=data_reg)
        # circuit
        circuit = cirq.Circuit(operation)
        ## measurement uncompute
        circuit.append([
            qrom_rotations.measurement_uncompute(selection=selection,data=data_reg)
        ])
        # loop over selection indices
        for index in range(num_coeffs):
            binary_index = list(iter_bits(index,width=n_select))
            # input state
            input_state = binary_index + [0]*n_data + [0]
            # output state equals input
            # verify simulated results
            gancilla_ind = [int(''.join(i for i in str(qb) if i.isdigit())) for qb in circuit.all_qubits() if 'gancilla' in str(qb)][0]
            assert_circuit_inp_out_cirqsim(circuit,qubit_order=[*selection,*data_reg,cirq.ops.CleanQubit(gancilla_ind, prefix='gancilla')],inputs=input_state,outputs=input_state)

    @pytest.mark.parametrize("n_select,n_data",[(4,4),(5,3)])
    def test_RotationsQROM_decomposes(self,n_select,n_data):
        '''
        Tests gate decomposition existence.
        '''
        # create registers
        num_coeffs = 2**n_select
        selection = cirq.NamedQubit.range(n_select,prefix='select')
        data_reg = cirq.NamedQubit.range(n_data,prefix='data')
        # create random data array
        np.random.seed(0)
        givens_angles = np.random.randint(low=0,high=2,size=(num_coeffs,n_data)) #each entry is randomly set to 0 or 1
        # initialilze oeprator
        qrom_rotations = RotationsQROM(
            [givens_angles],
            selection_bitsizes=(n_select,),
            target_bitsizes=(n_data,)
        )
        operation = qrom_rotations.on_registers(selection=selection,target0_=data_reg)
        # check decompose_once raises no error
        decomposed_once = cirq.decompose_once(operation)
        # check decompose returns decomposition not equal to operation itself
        decomposed = cirq.decompose(operation)
        assert([operation] != decomposed)

    def test_RotationsQROM_qasm(self):
        '''
        Tests qasm printing functionality.
        '''
        context = cirq.DecompositionContext(gam)
        n_select = 5
        n_data = 3
        # create registers
        num_coeffs = 2**n_select
        selection = cirq.NamedQubit.range(n_select,prefix='select')
        data_reg = cirq.NamedQubit.range(n_data,prefix='data')
        # create random data array
        np.random.seed(0)
        givens_angles = np.random.randint(low=0,high=2,size=(num_coeffs,n_data)) #each entry is randomly set to 0 or 1
        # initialilze oeprator
        qrom_rotations = RotationsQROM(
            [givens_angles],
            selection_bitsizes=(n_select,),
            target_bitsizes=(n_data,)
        )
        operation = qrom_rotations.on_registers(selection=selection,target0_=data_reg)

        qasm = openqasm(operation,rotation_allowed=True,context=context)
        assert qasm is not None

    def test_RotationsQROM_resources(self):
        '''
        Tests resources are counted correctly.
        '''
        n_select = 5
        n_data = 3
        # create registers
        num_coeffs = 2**n_select
        selection = cirq.NamedQubit.range(n_select,prefix='select')
        data_reg = cirq.NamedQubit.range(n_data,prefix='data')
        # create random data array
        np.random.seed(0)
        givens_angles = np.random.randint(low=0,high=2,size=(num_coeffs,n_data)) #each entry is randomly set to 0 or 1
        # initialilze oeprator
        qrom_rotations = RotationsQROM(
            [givens_angles],
            selection_bitsizes=(n_select,),
            target_bitsizes=(n_data,)
        )
        resources = estimate_resources(qrom_rotations)

        num_and = num_coeffs - 2 # minus 2 since uncontrolled
        cliffs_per_and = 9 + 2 # plus 2 for one zero control
        cliffs_per_and_adj = 4
        cliffs_for_unary_iteration = num_and*(cliffs_per_and+cliffs_per_and_adj) + num_coeffs
        cliffs_for_data = sum(sum(angle for angle in givens_angles))

        assert resources['T'] == 4*num_coeffs - 8 # minus 8 since uncontrolled
        assert resources['Clifford'] == cliffs_for_data + cliffs_for_unary_iteration
        assert resources['LogicalQubits'] == n_data+n_select