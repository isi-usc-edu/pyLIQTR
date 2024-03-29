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

from pyLIQTR.circuits.operators.BinaryToUnary import BinaryToUnaryBits
from qualtran.cirq_interop.testing import assert_circuit_inp_out_cirqsim
from qualtran.cirq_interop.bit_tools import iter_bits
from  pyLIQTR.utils.global_ancilla_manager import gam as gam
from  pyLIQTR.utils.printing import openqasm

class TestBinaryToUnaryBits:

    @pytest.mark.parametrize("selection_index",[0,1,2,4,5,8,11,12,13,15])
    def test_BinaryToUnaryBits(self,selection_index):
        '''
        Tests the gate carries out the expected operation by comparing simulation output with the expected output for different values of selection_index.
        '''
        n_bits = 4
        selection_register = cirq.NamedQubit.range(n_bits,prefix='select')
        target_reg = cirq.NamedQubit.range(n_bits,prefix='target')
        operation = BinaryToUnaryBits.make_on(selection=selection_register,target=target_reg)
        circuit = cirq.Circuit(operation)
        # input state is binary rep of selection index where msb is first reading left to right and target qubits all 0
        input_state = list(iter_bits(selection_index,width=n_bits))+[0]*n_bits
        # output state target qubits are binary rep of selection index rounded up to closest power of 2 where lsb is first reading left to right
        rounded_up_to_power_of_2 = 2**(selection_index).bit_length()
        num_ones = (rounded_up_to_power_of_2-1).bit_length()
        num_leading_zeros = n_bits - num_ones
        output_state = list(iter_bits(selection_index,width=n_bits))+[1]*num_ones +[0]*num_leading_zeros

        assert_circuit_inp_out_cirqsim(circuit,qubit_order=[*selection_register,*target_reg],inputs=input_state,outputs=output_state)

    @pytest.mark.parametrize("selection_index",[0,1,2,4,5,8,11,12,13,15])
    def test_BinaryToUnaryBits_inverse(self,selection_index):
        '''
        Tests the gate inverse correctly uncomputes by checking all target bits return to zero when gate*(gate)**(-1) is applied.
        '''
        n_bits = 4
        selection_register = cirq.NamedQubit.range(n_bits,prefix='select')
        target_reg = cirq.NamedQubit.range(n_bits,prefix='target')
        operation = BinaryToUnaryBits.make_on(selection=selection_register,target=target_reg)
        identity_circuit = cirq.Circuit(operation)
        identity_circuit.append(cirq.inverse(operation))
        # input state is binary rep of selection index where msb is first reading left to right and target qubits all 0
        input_state = list(iter_bits(selection_index,width=n_bits))+[0]*n_bits
        # output state is equal to input
        assert_circuit_inp_out_cirqsim(identity_circuit,qubit_order=[*selection_register,*target_reg],inputs=input_state,outputs=input_state)

    @pytest.mark.parametrize("n_bits",[1,2,5,8])
    def test_BinaryToUnaryBits_decomposes(self,n_bits):
        '''
        Tests gate decomposition existence.
        '''
        selection_register = cirq.NamedQubit.range(n_bits,prefix='select')
        target_reg = cirq.NamedQubit.range(n_bits,prefix='target')
        operation = BinaryToUnaryBits.make_on(selection=selection_register,target=target_reg)
        # check decompose_once raises no error
        decomposed_once = cirq.decompose_once(operation)
        # check decompose returns decomposition not equal to operation itself
        decomposed = cirq.decompose(operation)
        assert([operation] != decomposed)

    def test_BinaryToUnaryBits_qasm(self):
        '''
        Tests qasm printing functionality.
        '''
        n_bits = 4
        context = cirq.DecompositionContext(gam)
        selection_register = cirq.NamedQubit.range(n_bits,prefix='select')
        target_reg = cirq.NamedQubit.range(n_bits,prefix='target')
        operation = BinaryToUnaryBits.make_on(selection=selection_register,target=target_reg)

        qasm = openqasm(operation,rotation_allowed=True,context=context)
        assert qasm is not None