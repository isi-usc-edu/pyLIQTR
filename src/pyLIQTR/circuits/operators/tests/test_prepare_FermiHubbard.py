"""
Copyright (c) 2025 Massachusetts Institute of Technology 
SPDX-License-Identifier: BSD-2-Clause
"""
import pytest
import cirq

from qualtran.cirq_interop.testing import GateHelper
from pyLIQTR.utils.global_ancilla_manager import gam as gam
from pyLIQTR.utils.printing import openqasm
from pyLIQTR.circuits.operators.prepare_FermiHubbard import PrepareHubbardPYL, PrepareHubbardPYL_invert_workaround
from pyLIQTR.scheduler.scheduler import schedule_circuit
from pyLIQTR.utils.circuit_decomposition import generator_decompose

class TestPrepareHubbardPYL:    

    @pytest.mark.parametrize("N",[2,5])
    def test_PrepareHubbardPYL_decomposes(self,N):
        '''
        Tests gate decomposition existence.
        '''
        # initialize operator
        gate = PrepareHubbardPYL( x_dim=N, y_dim=N, t=1, u=4 )
        helper = GateHelper(gate=gate)
        operation = helper.operation
        # check decompose_once raises no error
        decomposed_once = cirq.decompose_once(operation)
        # check decompose returns decomposition not equal to operation itself
        decomposed = cirq.decompose(operation)
        decomposed = [*generator_decompose(operation)]
        assert([operation] != decomposed)

    @pytest.mark.parametrize("N",[2,5])
    def test_PrepareHubbardPYL_qasm_gam(self,N):
        '''
        Tests qasm printing functionality.
        '''
        context = cirq.DecompositionContext(gam)
        
        # initialize operator
        gate = PrepareHubbardPYL( x_dim=N, y_dim=N, t=1, u=4 )
        helper = GateHelper(gate=gate)
        operation = helper.operation

        qasm = openqasm(operation,rotation_allowed=True,context=context)
        assert qasm is not None
        for line in qasm:
            pass

    @pytest.mark.parametrize("N",[2,5])
    def test_PrepareHubbardPYL_resources(self,N):
        '''
        Tests resources are consistent with decomposition.
        '''
        # initialize operator
        gate = PrepareHubbardPYL( x_dim=N, y_dim=N, t=1, u=4 )

        assert gate.t_complexity() == gate.decompose_bloq().t_complexity()

    @pytest.mark.parametrize("N",[2,5])
    def test_PrepareHubbardPYL_callgraph(self,N):
        '''
        Tests ability to generate call_graph
        '''
        # initialize operator
        gate = PrepareHubbardPYL( x_dim=N, y_dim=N, t=1, u=4 )
        call_graph, _ = gate.call_graph()

    @pytest.mark.parametrize("N",[2,5])
    def test_PrepareHubbardPYL_invert_callgraph(self,N):
        '''
        Tests ability to generate call_graph
        '''
        # initialize operator
        gate = PrepareHubbardPYL_invert_workaround( x_dim=N, y_dim=N, t=1, u=4 )
        call_graph, _ = gate.call_graph()
         
    @pytest.mark.parametrize("N",[2,5])
    def test_PrepareHubbardPYL_scheduler(self,N):
        # initialize operator
        gate = PrepareHubbardPYL( x_dim=N, y_dim=N, t=1, u=4 )
        helper = GateHelper(gate=gate)
        res = schedule_circuit(helper.circuit, full_profile=True, decomp_level=0)
        assert res is not None
        for r in res:
            pass