"""
Copyright (c) 2024 Massachusetts Institute of Technology 
SPDX-License-Identifier: BSD-2-Clause
"""
import pytest
import cirq

from qualtran.cirq_interop.testing import GateHelper
from pyLIQTR.utils.global_ancilla_manager import gam as gam
from pyLIQTR.utils.printing import openqasm
from pyLIQTR.circuits.operators.SelectUV_FirstQuantized import SelectUVFirstQuantization
from pyLIQTR.utils.resource_analysis import estimate_resources
from pyLIQTR.utils.circuit_decomposition import decompose_once, generator_decompose
from pyLIQTR.circuits.operators.tests.utils import ignore_bookkeeping
from pyLIQTR.scheduler.scheduler import schedule_circuit

class TestSelectUVFirstQuantization:    

    def test_SelectUVFirstQuantization_decomposes(self):
        '''
        Tests gate decomposition existence.
        '''
        # initialilze operator
        gate = SelectUVFirstQuantization(num_bits_p=3,eta=10,num_atoms=5,num_bits_nuc_pos=5)
        helper = GateHelper(gate=gate)
        operation = helper.operation
        # check decompose_once raises no error
        decomposed_once = cirq.decompose_once(operation)
        # check decompose returns decomposition not equal to operation itself
        decomposed = generator_decompose(operation)
        assert([operation] != decomposed)

    def test_SelectUVFirstQuantization_qasm(self):
        '''
        Tests qasm printing functionality.
        '''
        context = cirq.DecompositionContext(gam)
        bitsize = 3
        gate = SelectUVFirstQuantization(num_bits_p=3,eta=10,num_atoms=5,num_bits_nuc_pos=5)
        helper = GateHelper(gate=gate)
        operation = helper.operation

        qasm = openqasm(operation,rotation_allowed=True,context=context)
        assert qasm is not None
        for line in qasm:
            pass

    @pytest.mark.parametrize("bitsize, nr",[(3,7),(8,5)])
    def test_SelectUVFirstQuantization_resources(self,bitsize,nr):
        '''
        Tests resources are consistent with decomposition.
        '''
        gate = SelectUVFirstQuantization(num_bits_p=bitsize,eta=10,num_atoms=5,num_bits_nuc_pos=nr)
        helper = GateHelper(gate=gate)
        circuit = helper.circuit
        decomposed = decompose_once(circuit)
        resources = estimate_resources(gate,profile=True)
        resources_decomposed = estimate_resources(decomposed,profile=True)
        decomp_ancilla_allocated = 3*(bitsize+1)
        resources['LogicalQubits'] += decomp_ancilla_allocated

        integer_conversion = 6*(bitsize-2) + 6*(bitsize-1) # differs from ref by 6
        controlled_arithmetic = 6*bitsize+6*(bitsize+1) # differs from ref by 6
        if nr>bitsize:
            apply_phase = 3*( 2*bitsize*nr - bitsize*(bitsize+1) -1)
        else:
            apply_phase = 3*nr*(nr-1)

        # differs from ref by 12 Toffoli due to np vs np+1 bits in addition/integer conversion
        assert resources['T'] == 4*(integer_conversion+controlled_arithmetic+apply_phase) 

        assert resources == resources_decomposed

    def test_SelectUVFirstQuantization_callgraph(self):
        '''
        Tests ability to generate call_graph.
        '''
        gate = SelectUVFirstQuantization(num_bits_p=3,eta=10,num_atoms=5,num_bits_nuc_pos=5)
        call_graph, _ = gate.call_graph()
        call_graph_counts = gate.bloq_counts()
        cbloq = gate.decompose_bloq()
        decomp_counts = cbloq.bloq_counts(generalizer=ignore_bookkeeping)

        # cant compare directly bc it sees each instance of SignedIntegerToTwosComplement and its adjoint as a different object in the decomposition. So, assert that the difference between the counts is equal to the number of occurences of these gates and then filter them out to compare the rest of the counts.
        assert len(set(decomp_counts).difference(set(call_graph_counts))) == 12

        filtered_call_graph_counts = {item for item in call_graph_counts if 'SignedInteger' not in str(item)}
        filtered_decomp_counts = {item for item in decomp_counts if 'SignedInteger' not in str(item)}

        assert filtered_call_graph_counts == filtered_decomp_counts, f"{filtered_call_graph_counts}, \n{filtered_decomp_counts}"

    def test_SelectUVFirstQuantization_scheduler(self):
        # initialize operator
        gate = SelectUVFirstQuantization(num_bits_p=3,eta=10,num_atoms=5,num_bits_nuc_pos=5)
        helper = GateHelper(gate=gate)
        res = schedule_circuit(helper.circuit, full_profile=True, decomp_level=0)
        assert res is not None
        for r in res:
            pass