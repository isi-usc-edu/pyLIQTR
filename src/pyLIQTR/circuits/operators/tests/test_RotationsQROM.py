"""
DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.

This material is based upon work supported by the Under Secretary of Defense for
Research and Engineering under Air Force Contract No. FA8702-15-D-0001. Any opinions,
findings, conclusions or recommendations expressed in this material are those of the
author(s) and do not necessarily reflect the views of the Under Secretary of Defense
for Research and Engineering.

© 2023 Massachusetts Institute of Technology.

The software/firmware is provided to you on an As-Is basis

Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS Part
252.227-7013 or 7014 (Feb 2014). Notwithstanding any copyright notice, U.S. Government
rights in this work are defined by DFARS 252.227-7013 or DFARS 252.227-7014 as detailed
above. Use of this work other than as specifically authorized by the U.S. Government
may violate any copyrights that exist in this work.
"""
import pytest
import cirq
import numpy as np

from qualtran.cirq_interop.testing import assert_circuit_inp_out_cirqsim
from qualtran.cirq_interop.bit_tools import iter_bits
from  pyLIQTR.utils.global_ancilla_manager import gam as gam
from  pyLIQTR.utils.printing import openqasm
from pyLIQTR.circuits.operators.PhaseGradientRotation import approx_angle_with_br_bits
from pyLIQTR.circuits.operators.RotationsQROM import RotationsQROM

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
        givens_angles = np.random.randint(low=0,high=2,size=(num_coeffs,n_data)) #each entry is randomly set to 0 or 1
        # initialilze oeprator
        qrom_rotations = RotationsQROM(
            data = [givens_angles],
            selection_bitsizes=(n_select,),
            target_bitsizes=(n_data,)
        )
        operation = qrom_rotations.on_registers(selection0=selection,target0_=data_reg)
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

    @pytest.mark.parametrize("n_select,n_data",[(4,4),(5,3),(4,10)])
    def test_RotationsQROM_inverse(self,n_select,n_data):
        '''
        Tests the gate inverse correctly uncomputes by checking all qubits return to initial state when gate*(gate)**(-1) is applied.
        '''
        # create registers
        num_coeffs = 2**n_select
        selection = cirq.NamedQubit.range(n_select,prefix='select')
        data_reg = cirq.NamedQubit.range(n_data,prefix='data')
        # create random data array
        givens_angles = np.random.randint(low=0,high=2,size=(num_coeffs,n_data)) #each entry is randomly set to 0 or 1
        # initialilze oeprator
        qrom_rotations = RotationsQROM(
            data = [givens_angles],
            selection_bitsizes=(n_select,),
            target_bitsizes=(n_data,)
        )
        operation = qrom_rotations.on_registers(selection0=selection,target0_=data_reg)
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

    @pytest.mark.parametrize("n_select,n_data",[(4,4),(5,3),(4,10)])
    def test_RotationsQROM_measurement_uncompute(self,n_select,n_data):
        '''
        Tests the method measurement_uncompute correctly uncomputes by checking all qubits return to initial state.
        '''
        # create registers
        num_coeffs = 2**n_select
        selection = cirq.NamedQubit.range(n_select,prefix='select')
        data_reg = cirq.NamedQubit.range(n_data,prefix='data')
        # create random data array
        givens_angles = np.random.randint(low=0,high=2,size=(num_coeffs,n_data)) #each entry is randomly set to 0 or 1
        # initialilze oeprator
        qrom_rotations = RotationsQROM(
            data = [givens_angles],
            selection_bitsizes=(n_select,),
            target_bitsizes=(n_data,)
        )
        operation = qrom_rotations.on_registers(selection0=selection,target0_=data_reg)
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
            assert_circuit_inp_out_cirqsim(circuit,qubit_order=[*selection,*data_reg,cirq.ops.CleanQubit(0, prefix='gancilla')],inputs=input_state,outputs=input_state)

    @pytest.mark.parametrize("n_select,n_data",[(4,4),(5,3),(4,10)])
    def test_RotationsQROM_decomposes(self,n_select,n_data):
        '''
        Tests gate decomposition existence.
        '''
        # create registers
        num_coeffs = 2**n_select
        selection = cirq.NamedQubit.range(n_select,prefix='select')
        data_reg = cirq.NamedQubit.range(n_data,prefix='data')
        # create random data array
        givens_angles = np.random.randint(low=0,high=2,size=(num_coeffs,n_data)) #each entry is randomly set to 0 or 1
        # initialilze oeprator
        qrom_rotations = RotationsQROM(
            data = [givens_angles],
            selection_bitsizes=(n_select,),
            target_bitsizes=(n_data,)
        )
        operation = qrom_rotations.on_registers(selection0=selection,target0_=data_reg)
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
        n_select = 4
        n_data = 10
        # create registers
        num_coeffs = 2**n_select
        selection = cirq.NamedQubit.range(n_select,prefix='select')
        data_reg = cirq.NamedQubit.range(n_data,prefix='data')
        # create random data array
        givens_angles = np.random.randint(low=0,high=2,size=(num_coeffs,n_data)) #each entry is randomly set to 0 or 1
        # initialilze oeprator
        qrom_rotations = RotationsQROM(
            data = [givens_angles],
            selection_bitsizes=(n_select,),
            target_bitsizes=(n_data,)
        )
        operation = qrom_rotations.on_registers(selection0=selection,target0_=data_reg)

        qasm = openqasm(operation,rotation_allowed=True,context=context)
        assert qasm is not None