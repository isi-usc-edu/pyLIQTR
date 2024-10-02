"""
Copyright (c) 2024 Massachusetts Institute of Technology 
SPDX-License-Identifier: BSD-2-Clause
"""
import pytest
import numpy as np
from pyLIQTR.ProblemInstances.getInstance import *
from pyLIQTR.BlockEncodings.getEncoding import *
from pyLIQTR.utils.resource_analysis import estimate_resources

class TestFirstQuantizedEncoding:

    def test_FirstQuantized_callgraph(self):
        '''
        Tests ability to generate call_graph.
        '''
        lengths=[12.7,12.7,19.9]
        example_instance = getInstance('MoleculeParameters',num_electrons=100,num_atoms=50,cell_volume=np.prod(lengths),sum_atomic_num=100,num_plane_waves=4000)
        first_quantized_encoding = getEncoding(VALID_ENCODINGS.FirstQuantized, instance=example_instance, eps_M=1e-5,n_R=20)
        call_graph, _ = first_quantized_encoding.call_graph()

    def test_FirstQuantized_resources(self):
        '''
        Tests estimate_resources executes without error.
        '''
        lengths=[12.7,12.7,19.9]
        example_instance = getInstance('MoleculeParameters',num_electrons=100,num_atoms=50,cell_volume=np.prod(lengths),sum_atomic_num=100,num_plane_waves=4000)
        first_quantized_encoding = getEncoding(VALID_ENCODINGS.FirstQuantized, instance=example_instance, eps_M=1e-5,n_R=20)
        resources = estimate_resources(first_quantized_encoding.circuit)
        