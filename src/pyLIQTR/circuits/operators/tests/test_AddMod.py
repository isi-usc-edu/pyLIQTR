"""
Copyright (c) 2024 Massachusetts Institute of Technology 
SPDX-License-Identifier: BSD-2-Clause
"""
import pytest
import cirq

from qualtran.cirq_interop.testing import GateHelper
from pyLIQTR.utils.global_ancilla_manager import gam as gam
from pyLIQTR.utils.printing import openqasm
from pyLIQTR.circuits.operators.AddMod import AddMod
from pyLIQTR.utils.resource_analysis import estimate_resources
from pyLIQTR.utils.circuit_decomposition import decompose_once, generator_decompose
from pyLIQTR.scheduler.scheduler import schedule_circuit
from pyLIQTR.utils.global_ancilla_manager import gam

class TestAddMod:    

    @pytest.mark.parametrize("bitsize,mod",[(2,3),(2,4),(3,6),(4,16),(5,32),(8,143),(9,386)])
    def test_AddMod_decomposes_cirq_sqm(self,bitsize,mod):
        '''
        Tests gate decomposition existence.
        '''
        # initialilze operator
        gate = AddMod(bitsize=bitsize, mod=mod, add_val=1, cvs=[0, 0])
        helper = GateHelper(gate=gate)
        operation = helper.operation
        # check decompose_once raises no error
        decomposed_once = cirq.decompose_once(operation)
        # check decompose returns decomposition not equal to operation itself
        decomposed = cirq.decompose(helper.circuit,context=cirq.DecompositionContext(cirq.SimpleQubitManager()))
        assert([operation] != decomposed)
        for op in decomposed:
            pass

    @pytest.mark.parametrize("bitsize,mod",[(2,3),(2,4),(3,6),(4,16),(5,32),(8,143),(9,386)])
    def test_AddMod_decomposes_generator_sqm(self,bitsize,mod):
        '''
        Tests gate decomposition existence.
        '''
        # initialilze operator
        gate = AddMod(bitsize=bitsize, mod=mod, add_val=1, cvs=[0, 0])
        helper = GateHelper(gate=gate)
        decomposed =generator_decompose(helper.circuit,context=cirq.DecompositionContext(cirq.SimpleQubitManager()))
        for op in decomposed:
            pass

    @pytest.mark.parametrize("bitsize,mod",[(2,3),(2,4),(3,6),(4,16),(5,32),(8,143),(9,386)])
    def test_AddMod_decomposes_generator_gam(self,bitsize,mod):
        '''
        Tests gate decomposition existence.
        '''
        # initialilze operator
        gate = AddMod(bitsize=bitsize, mod=mod, add_val=1, cvs=[0, 0])
        helper = GateHelper(gate=gate)
        # check decompose returns decomposition not equal to operation itself
        decomposed =generator_decompose(helper.circuit,context=cirq.DecompositionContext(gam))
        for op in decomposed:
            pass

    @pytest.mark.parametrize("bitsize,mod",[(2,3),(2,4),(3,6),(4,16),(5,32),(8,143),(9,386)])
    def test_AddMod_qasm_sqm(self,bitsize,mod):
        '''
        Tests qasm printing functionality.
        '''
        gate = AddMod(bitsize=bitsize, mod=mod, add_val=1, cvs=[0, 0])
        helper = GateHelper(gate=gate)
        operation = helper.operation

        qasm = openqasm(operation,rotation_allowed=True,context=cirq.DecompositionContext(cirq.SimpleQubitManager()))
        assert qasm is not None
        for line in qasm:
            pass

    @pytest.mark.parametrize("bitsize,mod",[(2,3),(2,4),(3,6),(4,16),(5,32),(8,143),(9,386)])
    def test_AddMod_qasm_gam(self,bitsize,mod):
        '''
        Tests qasm printing functionality.
        '''
        gate = AddMod(bitsize=bitsize, mod=mod, add_val=1, cvs=[0, 0])
        helper = GateHelper(gate=gate)
        operation = helper.operation

        qasm = openqasm(operation,rotation_allowed=True,context=cirq.DecompositionContext(gam))
        assert qasm is not None
        for line in qasm:
            pass

    @pytest.mark.parametrize("bitsize,mod",[(2,3),(2,4),(3,6),(4,16),(5,32)])
    def test_AddMod_scheduler_sqm(self,bitsize,mod):
        gate = AddMod(bitsize=bitsize, mod=mod, add_val=1, cvs=[0, 0])
        helper = GateHelper(gate=gate)
        res = schedule_circuit(helper.circuit, full_profile=True, decomp_level=0, context=cirq.DecompositionContext(cirq.SimpleQubitManager()))

    @pytest.mark.parametrize("bitsize,mod",[(2,3),(2,4),(3,6),(4,16),(5,32)])
    def test_AddMod_scheduler_gam(self,bitsize,mod):
        gate = AddMod(bitsize=bitsize, mod=mod, add_val=1, cvs=[0, 0])
        helper = GateHelper(gate=gate)
        res = schedule_circuit(helper.circuit, full_profile=True, decomp_level=0, context=cirq.DecompositionContext(gam))