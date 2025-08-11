"""
Copyright (c) 2024 Massachusetts Institute of Technology 
SPDX-License-Identifier: BSD-2-Clause
"""
import pytest
import cirq
import numpy as np
from collections import Counter
from pyLIQTR.ProblemInstances.getInstance import *
from pyLIQTR.BlockEncodings.getEncoding import *
from pyLIQTR.utils.printing import openqasm
from pyLIQTR.utils.resource_analysis import estimate_resources
from pyLIQTR.scheduler.scheduler import schedule_circuit

class TestFermiHubbardNNN:

    @pytest.fixture(scope="class")
    def fhnnn_instance(self):
        model = getInstance('FermiHubbardNNN',shape=(3,3),J1=-1,J2=-2,U=4.0,pbcs=(True,True))
        return model

    @pytest.fixture(scope="class")
    def lt_encoding(self, fhnnn_instance):
        return getEncoding(VALID_ENCODINGS.LinearT, instance=fhnnn_instance, energy_error=0.5)

    @pytest.fixture(scope="class")
    def lcu_encoding(self, fhnnn_instance):
        return getEncoding(VALID_ENCODINGS.PauliLCU, instance=fhnnn_instance)

    def test_FHNNN_ValueError(self):
        with pytest.raises(ValueError,match='2D lattices'):
            model = getInstance('FermiHubbardNNN',shape=(3,3,3),J1=-1,J2=-2,U=4.0,pbcs=(True,True))

        with pytest.raises(ValueError,match='square lattices'):
            model = getInstance('FermiHubbardNNN',shape=(3,2),J1=-1,J2=-2,U=4.0,pbcs=(True,True))

        with pytest.warns(UserWarning,match='periodic boundary'):
            model = getInstance('FermiHubbardNNN',shape=(3,3),J1=-1,J2=-2,U=4.0,pbcs=(False,True))
            [*model.yield_LinearT_Info('T')]

            model = getInstance('FermiHubbardNNN',shape=(3,3),J1=-1,J2=-2,U=4.0,pbcs=(True,False))
            [*model.yield_LinearT_Info('T')]

            model = getInstance('FermiHubbardNNN',shape=(3,3),J1=-1,J2=-2,U=4.0,pbcs=(False,False))
            [*model.yield_LinearT_Info('T')]

    @pytest.mark.parametrize('pbcs',[(True,True),(True,False),(False,True),(False,False)])
    def test_FHNNN_LCU_terms(self,pbcs):
        '''
        Tests 3x3 lattice Pauli LCU terms match expected.
        '''
        # set model parameters
        J1 = -1
        J2 = -2
        U = 4
        shape = (3,3)
        N = np.prod(shape)
        xzx_nn_coeff = J1/2
        xzx_nnn_coeff = J2/2
        zz_coeff = 2*U/8
        z_coeff = -U/4

        #set up analytically expected terms
        zz_inds = [(i,i+N) for i in range(N)]
        z_inds = [(i,) for i in range(2*N)]
        xzx_nn_inds = [(0,1),(0,1,2,3),(1,2),(1,2,3,4),(2,3,4,5),(3,4),(3,4,5,6),(4,5),(4,5,6,7),(5,6,7,8),(6,7),(7,8)]
        xzx_nn_inds += [tuple(i+N for i in inds) for inds in xzx_nn_inds]

        xzx_nn_pbc_x_inds = [(0,1,2),(3,4,5),(6,7,8)]
        xzx_nn_pbc_x_inds += [tuple(i+N for i in inds) for inds in xzx_nn_pbc_x_inds]
        xzx_nn_pbc_y_inds = [(0,1,2,3,4,5,6),(1,2,3,4,5,6,7),(2,3,4,5,6,7,8)]
        xzx_nn_pbc_y_inds += [tuple(i+N for i in inds) for inds in xzx_nn_pbc_y_inds]

        xzx_nnn_inds = [(0,1,2,3,4),(1,2,3),(1,2,3,4,5),(2,3,4),(3,4,5,6,7),(4,5,6),(4,5,6,7,8),(5,6,7)]
        xzx_nnn_inds += [tuple(i+N for i in inds) for inds in xzx_nnn_inds]

        xzx_nnn_pbc_x_inds = [(0,1,2,3,4,5),(2,3),(3,4,5,6,7,8),(5,6)]
        xzx_nnn_pbc_x_inds += [tuple(i+N for i in inds) for inds in xzx_nnn_pbc_x_inds]
        xzx_nnn_pbc_y_inds = [(0,1,2,3,4,5,6,7),(1,2,3,4,5,6),(1,2,3,4,5,6,7,8),(2,3,4,5,6,7)]
        xzx_nnn_pbc_y_inds += [tuple(i+N for i in inds) for inds in xzx_nnn_pbc_y_inds]
        xzx_nnn_pbc_corner_inds = [(0,1,2,3,4,5,6,7,8),(2,3,4,5,6)]
        xzx_nnn_pbc_corner_inds += [tuple(i+N for i in inds) for inds in xzx_nnn_pbc_corner_inds]
        assert len(xzx_nn_inds+xzx_nn_pbc_x_inds+xzx_nn_pbc_y_inds) == 2*N * 2
        assert len(xzx_nnn_inds+xzx_nnn_pbc_x_inds+xzx_nnn_pbc_y_inds+xzx_nnn_pbc_corner_inds) == 2*N * 2

        by_hand_zz_terms = [(inds,zz_coeff) for inds in zz_inds]
        by_hand_z_terms = [(inds,z_coeff) for inds in z_inds]
        by_hand_xzx_terms = [(inds,xzx_nn_coeff) for inds in xzx_nn_inds] + [(inds,xzx_nnn_coeff) for inds in xzx_nnn_inds]
        by_hand_yzy_terms = [(inds,xzx_nn_coeff) for inds in xzx_nn_inds] + [(inds,xzx_nnn_coeff) for inds in xzx_nnn_inds]
        if pbcs[0]:
            by_hand_xzx_terms += [(inds,xzx_nn_coeff) for inds in xzx_nn_pbc_x_inds]
            by_hand_xzx_terms += [(inds,xzx_nnn_coeff) for inds in xzx_nnn_pbc_x_inds]
            by_hand_yzy_terms += [(inds,xzx_nn_coeff) for inds in xzx_nn_pbc_x_inds]
            by_hand_yzy_terms += [(inds,xzx_nnn_coeff) for inds in xzx_nnn_pbc_x_inds]
        if pbcs[1]:
            by_hand_xzx_terms += [(inds,xzx_nn_coeff) for inds in xzx_nn_pbc_y_inds]
            by_hand_xzx_terms += [(inds,xzx_nnn_coeff) for inds in xzx_nnn_pbc_y_inds]
            by_hand_yzy_terms += [(inds,xzx_nn_coeff) for inds in xzx_nn_pbc_y_inds]
            by_hand_yzy_terms += [(inds,xzx_nnn_coeff) for inds in xzx_nnn_pbc_y_inds]
        if pbcs[0] or pbcs[1]:
            by_hand_xzx_terms += [(inds,xzx_nnn_coeff) for inds in xzx_nnn_pbc_corner_inds]
            by_hand_yzy_terms += [(inds,xzx_nnn_coeff) for inds in xzx_nnn_pbc_corner_inds]

        # generate model
        model = getInstance('FermiHubbardNNN',shape=shape,J1=J1,J2=J2,U=U,pbcs=pbcs)
        model_zz_terms = [(term[0],term[-1]) for term in model._ops.terms() if term[1] == 'ZZ']
        model_z_terms = [(term[0],term[-1]) for term in model._ops.terms() if term[1] == 'Z']
        model_xzx_terms = [(term[0],term[-1]) for term in model._ops.terms() if term[1][0] == 'X']
        model_yzy_terms = [(term[0],term[-1]) for term in model._ops.terms() if term[1][0] == 'Y']

        # compare
        assert Counter(model_zz_terms) == Counter(by_hand_zz_terms)
        assert Counter(model_z_terms) == Counter(by_hand_z_terms)
        assert Counter(model_xzx_terms) == Counter(by_hand_xzx_terms)
        assert Counter(model_yzy_terms) == Counter(by_hand_yzy_terms)
    
    def test_LinearT_decomposes(self, lt_encoding):
        '''
        Tests gate decomposition existence.
        '''
        num_qubits = cirq.num_qubits(lt_encoding.circuit)
        qubits = cirq.LineQubit.range(num_qubits)
        # initialize operator
        operation = lt_encoding.on(*qubits)
        # check decompose_once raises no error
        decomposed_once = cirq.decompose_once(operation)
        # check decompose returns decomposition not equal to operation itself
        decomposed = cirq.decompose(operation)
        assert([operation] != decomposed)

    def test_LinearT_qasm(self,lt_encoding):
        '''
        Tests qasm printing functionality.
        '''
        # create registers
        num_qubits = cirq.num_qubits(lt_encoding.circuit)
        qubits = cirq.LineQubit.range(num_qubits)
        # initialize operator
        operation = lt_encoding.on(*qubits)

        qasm = openqasm(operation,rotation_allowed=True)
        assert qasm is not None
        for line in qasm:
            pass

    def test_LinearT_resources(self, lt_encoding):
        '''
        Tests estimate_resources executes without error.
        '''
        resources = estimate_resources(lt_encoding.circuit)

    def test_LinearT_scheduling(self, lt_encoding):
        res = schedule_circuit(lt_encoding.circuit, full_profile=True, decomp_level=0)
        for r in res:
            pass

        res = schedule_circuit(lt_encoding.circuit, full_profile=True, decomp_level='Full')
        for r in res:
            pass

    def test_LCU_decomposes(self, lcu_encoding):
        '''
        Tests gate decomposition existence.
        '''
        num_qubits = cirq.num_qubits(lcu_encoding.circuit)
        qubits = cirq.LineQubit.range(num_qubits)
        # initialize operator
        operation = lcu_encoding.on(*qubits)
        # check decompose_once raises no error
        decomposed_once = cirq.decompose_once(operation)
        # check decompose returns decomposition not equal to operation itself
        decomposed = cirq.decompose(operation)
        assert([operation] != decomposed)

    def test_LCU_qasm(self,lcu_encoding):
        '''
        Tests qasm printing functionality.
        '''
        # create registers
        num_qubits = cirq.num_qubits(lcu_encoding.circuit)
        qubits = cirq.LineQubit.range(num_qubits)
        # initialize operator
        operation = lcu_encoding.on(*qubits)

        qasm = openqasm(operation,rotation_allowed=True)
        assert qasm is not None
        for line in qasm:
            pass

    def test_LCU_resources(self, lcu_encoding):
        '''
        Tests estimate_resources executes without error.
        '''
        resources = estimate_resources(lcu_encoding.circuit)

    def test_LCU_scheduling(self, lcu_encoding):
        res = schedule_circuit(lcu_encoding.circuit, full_profile=True, decomp_level=0)
        for r in res:
            pass

        res = schedule_circuit(lcu_encoding.circuit, full_profile=True, decomp_level='Full')
        for r in res:
            pass