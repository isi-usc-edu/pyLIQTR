"""
Copyright (c) 2024 Massachusetts Institute of Technology 
SPDX-License-Identifier: BSD-2-Clause
"""
import pytest
import cirq
import numpy as np

from qualtran.cirq_interop.testing import GateHelper
from pyLIQTR.utils.printing import openqasm
from pyLIQTR.circuits.operators.PrepareTUVSuperpositions import PrepareTUVSuperpositions
from pyLIQTR.utils.resource_analysis import estimate_resources
from pyLIQTR.circuits.operators.tests.utils import ignore_bookkeeping

class TestPrepareTUVSuperpositions:

    @pytest.mark.parametrize("nt,eta,lz,theta",[(4,5,5,np.pi)])
    def test_PrepareTUVSuperpositions_decomposes(self,nt,eta,lz,theta):
        '''
        Tests gate decomposition existence.
        '''
        # initialilze operator
        gate = PrepareTUVSuperpositions(num_bits_t=nt,eta=eta,lambda_zeta=lz,theta=theta)
        helper = GateHelper(gate=gate)
        operation = helper.operation
        # check decompose_once raises no error
        decomposed_once = cirq.decompose_once(operation)
        # check decompose returns decomposition not equal to operation itself
        decomposed = cirq.decompose(operation)
        assert([operation] != decomposed)

    @pytest.mark.parametrize("nt,eta,lz,theta",[(4,5,5,np.pi)])
    def test_PrepareTUVSuperpositions_qasm(self,nt,eta,lz,theta):
        '''
        Tests qasm printing functionality.
        '''
        gate = PrepareTUVSuperpositions(num_bits_t=nt,eta=eta,lambda_zeta=lz,theta=theta)
        helper = GateHelper(gate=gate)
        operation = helper.operation

        context = cirq.DecompositionContext(cirq.SimpleQubitManager())
        qasm = openqasm(operation,rotation_allowed=True,context=context)
        assert qasm is not None
        for line in qasm:
            pass

    @pytest.mark.parametrize("nt,eta,lz,theta",[(4,5,5,np.pi)])
    def test_PrepareTUVSuperpositions_resources(self,nt,eta,lz,theta):
        '''
        Tests resources are counted correctly.
        '''
        gate = PrepareTUVSuperpositions(num_bits_t=nt,eta=eta,lambda_zeta=lz,theta=theta)
        resources = estimate_resources(gate,profile=True)

        # nt-1 for addition (rotation on first qubit) vs nt-3 (unoptimized addition)
        # n_eta_zeta for less than vs n_eta_zeta - 1 (not out of place adder)

        # flagged prep
        # note 1q costs assume n_eta_zeta = logL
        # 4 * inequality -> 4*logL (doesnt uncompute w/ Cliffords)
        # 2 rotations (vs 2*(br-3))
        # reflection -> logL-1
        d, a = gate.eta+2*gate.lambda_zeta, 0
        while d > 1 and d % 2 == 0:
            a += 1
            d = d // 2
        l, logL = int(d), gate.n_eta_zeta - a

        assert resources['T'] == 4*(nt-1 + 4*logL+logL-1+gate.n_eta_zeta+1) # more than reference due to above differences
        assert resources['LogicalQubits'] == nt+gate.n_eta_zeta+5
        assert resources['Rotations'] == 2
        assert gate.t_complexity() == gate.decompose_bloq().t_complexity()

    @pytest.mark.skip(reason='known but unresolved equality error')
    @pytest.mark.parametrize("nt,eta,lz,theta",[(4,5,5,np.pi)])
    def test_PrepareTUVSuperpositions_callgraph(self,nt,eta,lz,theta):
        '''
        Tests ability to generate call_graph and that call_graph counts match decomposition counts.
        '''
        gate = PrepareTUVSuperpositions(num_bits_t=nt,eta=eta,lambda_zeta=lz,theta=theta)
        call_graph, _ = gate.call_graph()

        call_graph_counts = gate.bloq_counts()
        cbloq = gate.decompose_bloq()
        decomp_counts = cbloq.bloq_counts(generalizer=ignore_bookkeeping)

        assert call_graph_counts == decomp_counts