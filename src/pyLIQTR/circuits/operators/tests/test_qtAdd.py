"""
Copyright (c) 2024 Massachusetts Institute of Technology 
SPDX-License-Identifier: BSD-2-Clause
"""
import pytest
import cirq
import numpy as np

from qualtran.bloqs.arithmetic.addition import Add as qtAdd
from qualtran.cirq_interop.testing import assert_circuit_inp_out_cirqsim
from qualtran.cirq_interop.bit_tools import iter_bits_twos_complement
from qualtran._infra.data_types import QUInt
from  pyLIQTR.utils.global_ancilla_manager import gam as gam
from  pyLIQTR.utils.printing import openqasm
from pyLIQTR.circuits.operators.PhaseGradientRotation import approx_angle_with_br_bits
from qualtran.bloqs.rotations.phase_gradient import PhaseGradientState
from pyLIQTR.circuits.operators.tests.utils import get_partial_state

class TestqtAdd:

    @pytest.mark.parametrize("int_x,int_y",[(1,2),(6,9),(10,8),(8,3),(4,4)])
    def test_qtAdd_with_zero_padding(self,int_x,int_y):
        '''
        Tests the gate carries out the expected operation when adding/subtracting registers with different bitsizes by comparing simulation output with the expected output for different input values.
        '''
        # create registers
        nb_x = int_x.bit_length()
        nb_y = int_y.bit_length()
        nout = max(nb_x,nb_y)
        zero_pad = cirq.NamedQubit.range(nout-nb_x,prefix='zero')
        in_x = cirq.NamedQubit.range(nb_x,prefix='x')
        out_y = cirq.NamedQubit.range(nout,prefix='y')
        # addition circuit
        add_circuit = cirq.Circuit(qtAdd(a_dtype=QUInt(nout)).on_registers(a=zero_pad+in_x,b=out_y))
        # subtraction circuit
        subtract_circuit = cirq.Circuit()
        subtract_circuit.append(cirq.X.on_each(*out_y))
        subtract_circuit.append(qtAdd(a_dtype=QUInt(nout)).on_registers(a=zero_pad+in_x,b=out_y))
        subtract_circuit.append(cirq.X.on_each(*out_y))
        # input states
        int_out_add = int_x+int_y
        int_out_sub = int_y-int_x
        init_state = list(iter_bits_twos_complement(int_x,width=nb_x)) + list(iter_bits_twos_complement(int_y,width=nout)) + [0]*len(zero_pad)
        # output states
        out_state_add = list(iter_bits_twos_complement(int_x,width=nb_x)) + list(iter_bits_twos_complement(int_out_add,width=nout)) + [0]*len(zero_pad)
        out_state_sub = list(iter_bits_twos_complement(int_x,width=nb_x)) + list(iter_bits_twos_complement(int_out_sub,width=nout)) + [0]*len(zero_pad)
        # verify simulated results
        qubit_order = [*in_x,*out_y,*zero_pad]
        assert_circuit_inp_out_cirqsim(add_circuit,qubit_order=qubit_order,inputs=init_state,outputs=out_state_add)
        assert_circuit_inp_out_cirqsim(subtract_circuit,qubit_order=qubit_order,inputs=init_state,outputs=out_state_sub)

    @pytest.mark.parametrize("test_angle,br,bphi",[(6.24857,4,4),(-2.63212,4,4),(-1.73161,2,3),(0.79874,1,5)])
    def test_qtAdd_with_phase_gradient(self,test_angle:float,br:int,bphi:int):
        angle = cirq.NamedQubit.range(br,prefix='angle')
        phi = cirq.NamedQubit.range(bphi,prefix='phi')
        zero_pad = cirq.NamedQubit.range(bphi-br,prefix='zero')

        create_phase_grad_state = PhaseGradientState(bitsize=bphi)

        binary_angle = approx_angle_with_br_bits(test_angle,br)

        add_circuit = cirq.Circuit()
        sub_circuit = cirq.Circuit()
        # prepare angle register
        for i,bit in enumerate(angle):
            if binary_angle[i]:
                add_circuit.append(cirq.X.on(bit))
                sub_circuit.append(cirq.X.on(bit))

        # prepare phase gradient state
        add_circuit.append(create_phase_grad_state.on(*phi))
        sub_circuit.append(create_phase_grad_state.on(*phi))
        # flip phi for subtraction
        sub_circuit.append(cirq.X.on_each(*phi))

        add_gate = qtAdd(a_dtype=QUInt(bphi))
        add_circuit.append(add_gate.on_registers(a=zero_pad+angle,b=phi))
        sub_circuit.append(add_gate.on_registers(a=zero_pad+angle,b=phi))
        # unflip phi for subtraction
        sub_circuit.append(cirq.X.on_each(*phi))

        # expected outputs are phase gradient state coefficients with an additional phase based on the test angle (represented as integer ell)
        ell = int(''.join(str(b) for b in binary_angle), 2)
        add_expected_out=np.round(1/np.sqrt(2**bphi)*np.array([np.exp(-2j*np.pi*(n-ell)/(2**(bphi))) for n in range(2**bphi)]),8)
        sub_expected_out=np.round(1/np.sqrt(2**bphi)*np.array([np.exp(-2j*np.pi*(n+ell)/(2**(bphi))) for n in range(2**bphi)]),8)

        # isolate the state output on the phi register
        add_res = cirq.Simulator(dtype=np.complex128).simulate(add_circuit)
        sub_res = cirq.Simulator(dtype=np.complex128).simulate(sub_circuit)
        add_partial_state = get_partial_state(add_res.final_state_vector,np.arange(br,br+bphi))
        sub_partial_state = get_partial_state(sub_res.final_state_vector,np.arange(br,br+bphi))

        assert cirq.allclose_up_to_global_phase(add_partial_state, add_expected_out)
        assert cirq.allclose_up_to_global_phase(sub_partial_state, sub_expected_out)

    @pytest.mark.parametrize("int_x,int_y",[(1,2),(6,9),(10,8),(8,3),(4,4)])
    def test_qtAdd_with_zero_padding_inverse(self,int_x,int_y):
        '''
        Tests the gate inverse correctly uncomputes by checking all qubits return to initial state when gate*(gate)**(-1) is applied.
        '''
        # create registers
        nb_x = int_x.bit_length()
        nb_y = int_y.bit_length()
        nout = max(nb_x,nb_y)
        zero_pad = cirq.NamedQubit.range(nout-nb_x,prefix='zero')
        in_x = cirq.NamedQubit.range(nb_x,prefix='x')
        out_y = cirq.NamedQubit.range(nout,prefix='y')
        # addition identity circuit
        add_circuit = cirq.Circuit(qtAdd(a_dtype=QUInt(nout)).on_registers(a=zero_pad+in_x,b=out_y))
        add_circuit.append(cirq.inverse(qtAdd(a_dtype=QUInt(nout)).on_registers(a=zero_pad+in_x,b=out_y)))
        # subtraction identity circuit
        subtract_circuit = cirq.Circuit()
        subtract_circuit.append(cirq.X.on_each(*out_y))
        subtract_circuit.append(add_circuit)
        subtract_circuit.append(cirq.X.on_each(*out_y))
        # input states
        int_out_add = int_x+int_y
        int_out_sub = int_y-int_x
        init_state = list(iter_bits_twos_complement(int_x,width=nb_x)) + list(iter_bits_twos_complement(int_y,width=nout)) + [0]*len(zero_pad)
        # output states equal input
        # verify simulated results
        qubit_order = [*in_x,*out_y,*zero_pad]
        assert_circuit_inp_out_cirqsim(add_circuit,qubit_order=qubit_order,inputs=init_state,outputs=init_state)
        assert_circuit_inp_out_cirqsim(subtract_circuit,qubit_order=qubit_order,inputs=init_state,outputs=init_state)

    @pytest.mark.parametrize("int_x,int_y",[(1,2),(6,9),(10,8),(8,3),(4,4)])
    def test_qtAdd_decomposes(self,int_x,int_y):
        '''
        Tests gate decomposition existence.
        '''
        nb_x = int_x.bit_length()
        nb_y = int_y.bit_length()
        nout = max(nb_x,nb_y)
        zero_pad = cirq.NamedQubit.range(nout-nb_x,prefix='zero')
        in_x = cirq.NamedQubit.range(nb_x,prefix='x')
        out_y = cirq.NamedQubit.range(nout,prefix='y')
        operation = qtAdd(a_dtype=QUInt(nout)).on_registers(a=zero_pad+in_x,b=out_y)
        # check decompose_once raises no error
        decomposed_once = cirq.decompose_once(operation)
        # check decompose returns decomposition not equal to operation itself
        decomposed = cirq.decompose(operation)
        assert([operation] != decomposed)

    def test_qtAdd_qasm(self):
        '''
        Tests qasm printing functionality.
        '''
        context = cirq.DecompositionContext(gam)
        int_x = 6
        int_y = 9
        nb_x = int_x.bit_length()
        nb_y = int_y.bit_length()
        nout = max(nb_x,nb_y)
        zero_pad = cirq.NamedQubit.range(nout-nb_x,prefix='zero')
        in_x = cirq.NamedQubit.range(nb_x,prefix='x')
        out_y = cirq.NamedQubit.range(nout,prefix='y')
        operation = qtAdd(a_dtype=QUInt(nout)).on_registers(a=zero_pad+in_x,b=out_y)

        qasm = openqasm(operation,rotation_allowed=True,context=context)
        assert qasm is not None