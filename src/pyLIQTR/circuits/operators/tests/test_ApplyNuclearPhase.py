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
from pyLIQTR.circuits.operators.ApplyNuclearPhase import ApplyNuclearPhase
from pyLIQTR.utils.resource_analysis import estimate_resources
from pyLIQTR.circuits.operators.tests.utils import ignore_bookkeeping

class TestApplyNuclearPhase:

    @pytest.mark.parametrize("nr,np",[(4,4),(4,2),(3,5)])
    def test_ApplyNuclearPhase_decomposes(self,nr,np):
        '''
        Tests gate decomposition existence.
        '''
        # initialilze operator
        gate = ApplyNuclearPhase(num_bits_nuc=nr,num_bits_p=np)
        helper = GateHelper(gate=gate)
        operation = helper.operation
        # check decompose_once raises no error
        decomposed_once = cirq.decompose_once(operation)
        # check decompose returns decomposition not equal to operation itself
        decomposed = cirq.decompose(operation)
        assert([operation] != decomposed)

    @pytest.mark.parametrize("nr,np",[(4,4),(4,2),(3,5)])
    def test_ApplyNuclearPhase_qasm(self,nr,np):
        '''
        Tests qasm printing functionality.
        '''
        gate = ApplyNuclearPhase(num_bits_nuc=nr,num_bits_p=np)
        helper = GateHelper(gate=gate)
        operation = helper.operation

        qasm = openqasm(operation,rotation_allowed=True)
        assert qasm is not None
        for line in qasm:
            pass

    @pytest.mark.parametrize("nr,np",[(4,4),(4,2),(3,5)])
    def test_ApplyNuclearPhase_resources(self,nr,np):
        '''
        Tests resources are counted correctly.
        '''
        gate = ApplyNuclearPhase(num_bits_nuc=nr,num_bits_p=np)
        resources = estimate_resources(gate)

        # match ref eq 97
        if nr>np:
            assert resources['T'] == 4*3*(2*np*nr-np*(np+1)-1) 
        else:
            assert resources['T'] == 4*3*nr*(nr-1)
        assert resources['LogicalQubits'] == 4*nr+3*(np+1)+2 # doesn't count ancilla
        assert gate.t_complexity() == gate.decompose_bloq().t_complexity()

    @pytest.mark.parametrize("nr,np",[(4,4),(4,2),(3,5)])
    def test_ApplyNuclearPhase_callgraph(self,nr,np):
        '''
        Tests ability to generate call_graph and that call_graph counts match decomposition counts.
        '''
        gate = ApplyNuclearPhase(num_bits_nuc=nr,num_bits_p=np)
        call_graph, _ = gate.call_graph()

        call_graph_counts = gate.bloq_counts(generalizer=ignore_bookkeeping)
        cbloq = gate.decompose_bloq()
        decomp_counts = cbloq.bloq_counts(generalizer=ignore_bookkeeping)

        assert call_graph_counts == decomp_counts