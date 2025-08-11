"""
Copyright (c) 2024 Massachusetts Institute of Technology 
SPDX-License-Identifier: BSD-2-Clause
"""
import cirq
import pytest
import numpy as np
from pyLIQTR.ProblemInstances.getInstance import *
from pyLIQTR.BlockEncodings.getEncoding import *
from pyLIQTR.utils.printing import openqasm
from pyLIQTR.utils.resource_analysis import estimate_resources
from pyLIQTR.scheduler.scheduler import schedule_circuit
from pyLIQTR.utils.global_ancilla_manager import gam
from pyLIQTR.utils.circuit_decomposition import generator_decompose
from pyLIQTR.utils.utils import version_newer


class TestFirstQuantizedEncoding:

    @pytest.fixture(scope="class")
    def example_encoding(self):
        lengths=[12.7,12.7,19.9]
        example_instance = getInstance('MoleculeParameters',num_electrons=19,num_atoms=50,cell_volume=np.prod(lengths),sum_atomic_num=23,num_plane_waves=300)
        first_quantized_encoding = getEncoding(VALID_ENCODINGS.FirstQuantized, instance=example_instance, eps_M=1e-5,n_R=5)
        return first_quantized_encoding

    def test_FirstQuantized_decomposes_cirq_sqm(self,example_encoding):
        '''
        Tests gate decomposition existence.
        '''
        num_qubits = cirq.num_qubits(example_encoding.circuit)
        qubits = cirq.LineQubit.range(num_qubits)
        # initialize operator
        operation = example_encoding.on(*qubits)
        # check decompose_once raises no error
        decomposed_once = cirq.decompose_once(operation)
        # check decompose returns decomposition not equal to operation itself
        decomposed = cirq.decompose(example_encoding.circuit,context=cirq.DecompositionContext(cirq.SimpleQubitManager()))
        assert([operation] != decomposed)
        for op in decomposed:
            pass

    def test_FirstQuantized_decomposes_gen_sqm(self,example_encoding):
        '''
        Tests gate decomposition existence.
        '''
        decomposed = generator_decompose(example_encoding.circuit,context=cirq.DecompositionContext(cirq.SimpleQubitManager()))
        for op in decomposed:
            pass

    def test_FirstQuantized_qasm_sqm(self,example_encoding):
        '''
        Tests qasm printing functionality.
        '''
        qasm = openqasm(example_encoding.circuit,rotation_allowed=True,context=cirq.DecompositionContext(cirq.SimpleQubitManager()))
        assert qasm is not None
        for line in qasm:
            pass

    def test_FirstQuantized_callgraph(self,example_encoding):
        '''
        Tests ability to generate call_graph.
        '''
        call_graph, _ = example_encoding.call_graph()

    def test_FirstQuantized_resources(self, example_encoding):
        '''
        Tests estimate_resources executes without error.
        '''
        resources = estimate_resources(example_encoding.circuit)

    def test_FirstQuantized_scheduler_sqm(self,example_encoding):
        # initialize operator
        res = schedule_circuit(example_encoding.circuit, full_profile=True, decomp_level=0, context=cirq.DecompositionContext(cirq.SimpleQubitManager()))
        assert res is not None
        for r in res:
            pass

    def test_FirstQuantized_scheduler_full_sqm(self,example_encoding):
        # initialize operator
        res = schedule_circuit(example_encoding.circuit, full_profile=True, decomp_level='Full', context=cirq.DecompositionContext(cirq.SimpleQubitManager()))
        assert res is not None
        for r in res:
            pass
        
    if version_newer('qualtran','0.4.0'):
        def test_FirstQuantized_scheduler_full_gqm(self,example_encoding):
            # initialize operator
            res = schedule_circuit(example_encoding.circuit, full_profile=True, decomp_level='Full',context=cirq.DecompositionContext(gam))
            assert res is not None
            for r in res:
                pass

        def test_FirstQuantized_decomposes_gen_gam(self,example_encoding):
            '''
            Tests gate decomposition existence.
            '''
            decomposed = generator_decompose(example_encoding.circuit,context=cirq.DecompositionContext(gam))
            for op in decomposed:
                pass

        def test_FirstQuantized_scheduler_gam(self,example_encoding):
            # initialize operator
            res = schedule_circuit(example_encoding.circuit, full_profile=True, decomp_level=0)
            assert res is not None
            for r in res:
                pass

        def test_FirstQuantized_decomposes_gen_grqm(self,example_encoding):
            '''
            Tests gate decomposition existence.
            '''
            decomposed = generator_decompose(example_encoding.circuit,context=cirq.DecompositionContext(cirq.GreedyQubitManager(prefix='greedy_ancilla')))
            for op in decomposed:
                pass

        def test_FirstQuantized_qasm_gam(self,example_encoding):
            '''
            Tests qasm printing functionality.
            '''
            qasm = openqasm(example_encoding.circuit,rotation_allowed=True,context=cirq.DecompositionContext(gam))
            assert qasm is not None
            for line in qasm:
                pass

        def test_FirstQuantized_qasm_grqm(self,example_encoding):
            '''
            Tests qasm printing functionality.
            '''
            qasm = openqasm(example_encoding.circuit,rotation_allowed=True,context=cirq.DecompositionContext(cirq.GreedyQubitManager(prefix='greedy_ancilla')))
            assert qasm is not None
            for line in qasm:
                pass