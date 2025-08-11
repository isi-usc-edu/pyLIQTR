"""
Copyright (c) 2024 Massachusetts Institute of Technology 
SPDX-License-Identifier: BSD-2-Clause
"""
import pytest
import cirq

from qualtran.cirq_interop.testing import GateHelper
from pyLIQTR.utils.global_ancilla_manager import gam as gam
from pyLIQTR.utils.printing import openqasm
from pyLIQTR.circuits.operators.prepare_FirstQuantized import PrepareFirstQuantization
from pyLIQTR.scheduler.scheduler import schedule_circuit
from pyLIQTR.utils.circuit_decomposition import generator_decompose
from pyLIQTR.utils.utils import version_newer

class TestPrepareFirstQuantization:    

    @pytest.mark.parametrize("num_bits_p, eta, num_atoms, lz, nr",[(3,19,23,23,5),(3,31,3,50,4),(3,10,7,12,3)])
    def test_PrepareFirstQuantization_decomposes(self,num_bits_p, eta, num_atoms, lz, nr):
        '''
        Tests gate decomposition existence.
        '''
        m_param=2**(2*num_bits_p+2+2)
        # initialize operator
        gate = PrepareFirstQuantization(num_bits_p, eta, num_atoms, m_param=m_param, lambda_zeta=lz, num_bits_nuc_pos=nr)
        helper = GateHelper(gate=gate)
        operation = helper.operation
        # check decompose_once raises no error
        decomposed_once = cirq.decompose_once(operation)
        # check decompose returns decomposition not equal to operation itself
        decomposed = cirq.decompose(operation)
        decomposed = [*generator_decompose(operation)]
        assert([operation] != decomposed)

    @pytest.mark.parametrize("num_bits_p, eta, num_atoms, lz, nr",[(3,19,15,23,5),(3,10,7,12,3)])
    def test_PrepareFirstQuantization_qasm_grqm(self,num_bits_p, eta, num_atoms, lz, nr):
        '''
        Tests qasm printing functionality.
        '''
        context = cirq.DecompositionContext(cirq.GreedyQubitManager(prefix='greedy_ancilla'))
        m_param=2**(2*num_bits_p+2+2)
        # initialize operator
        gate = PrepareFirstQuantization(num_bits_p, eta, num_atoms, m_param=m_param, lambda_zeta=lz, num_bits_nuc_pos=nr)
        helper = GateHelper(gate=gate)
        operation = helper.operation

        qasm = openqasm(operation,rotation_allowed=True,context=context)
        assert qasm is not None
        for line in qasm:
            pass
    if version_newer('qualtran','0.4.0'):
        @pytest.mark.parametrize("num_bits_p, eta, num_atoms, lz, nr",[(3,19,15,23,5),(3,10,7,12,3)])
        def test_PrepareFirstQuantization_qasm_gam(self,num_bits_p, eta, num_atoms, lz, nr):
            '''
            Tests qasm printing functionality.
            '''
            context = cirq.DecompositionContext(gam)
            m_param=2**(2*num_bits_p+2+2)
            # initialize operator
            gate = PrepareFirstQuantization(num_bits_p, eta, num_atoms, m_param=m_param, lambda_zeta=lz, num_bits_nuc_pos=nr)
            helper = GateHelper(gate=gate)
            operation = helper.operation

            qasm = openqasm(operation,rotation_allowed=True,context=context)
            assert qasm is not None
            for line in qasm:
                pass

    @pytest.mark.parametrize("num_bits_p, eta, num_atoms, lz, nr",[(3,19,15,23,5),(3,10,7,12,3)])
    def test_PrepareFirstQuantization_qasm_sqm(self,num_bits_p, eta, num_atoms, lz, nr):
        '''
        Tests qasm printing functionality.
        '''
        context = cirq.DecompositionContext(cirq.SimpleQubitManager())
        m_param=2**(2*num_bits_p+2+2)
        # initialize operator
        gate = PrepareFirstQuantization(num_bits_p, eta, num_atoms, m_param=m_param, lambda_zeta=lz, num_bits_nuc_pos=nr)
        helper = GateHelper(gate=gate)
        operation = helper.operation

        qasm = openqasm(operation,rotation_allowed=True,context=context)
        assert qasm is not None
        for line in qasm:
            pass

    @pytest.mark.parametrize("num_bits_p, eta, num_atoms, lz, nr",[(3,19,23,23,5),(3,31,3,50,4)])
    def test_PrepareFirstQuantization_resources(self,num_bits_p, eta, num_atoms, lz, nr):
        '''
        Tests resources are consistent with decomposition.
        '''
        m_param=2**(2*num_bits_p+2+2)
        # initialize operator
        gate = PrepareFirstQuantization(num_bits_p, eta, num_atoms, m_param=m_param, lambda_zeta=lz, num_bits_nuc_pos=nr)

        assert gate.t_complexity() == gate.decompose_bloq().t_complexity()

    @pytest.mark.parametrize("num_bits_p, eta, num_atoms, lz, nr",[(3,19,15,23,5)])
    def test_PrepareFirstQuantization_callgraph(self,num_bits_p, eta, num_atoms, lz, nr):
        '''
        Tests ability to generate call_graph
        '''
        m_param=2**(2*num_bits_p+2+2)
        # initialize operator
        gate = PrepareFirstQuantization(num_bits_p, eta, num_atoms, m_param=m_param, lambda_zeta=lz, num_bits_nuc_pos=nr)
        call_graph, _ = gate.call_graph()
        
    @pytest.mark.filterwarnings("ignore::UserWarning")  
    @pytest.mark.parametrize("num_bits_p, eta, num_atoms, lz, nr",[(3,19,15,23,5)])
    def test_PrepareFirstQuantization_scheduler(self,num_bits_p, eta, num_atoms, lz, nr):
        m_param=2**(2*num_bits_p+2+2)
        # initialize operator
        gate = PrepareFirstQuantization(num_bits_p, eta, num_atoms, m_param=m_param, lambda_zeta=lz, num_bits_nuc_pos=nr)
        helper = GateHelper(gate=gate)
        res = schedule_circuit(helper.circuit, full_profile=True, decomp_level=0)
        assert res is not None
        for r in res:
            pass