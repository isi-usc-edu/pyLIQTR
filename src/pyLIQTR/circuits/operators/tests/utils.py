"""
Copyright (c) 2024 Massachusetts Institute of Technology 
SPDX-License-Identifier: BSD-2-Clause
"""
import cirq

def get_partial_state(state_vector,keep_indices):
    partial_state = cirq.partial_trace_of_state_vector_as_mixture(state_vector,keep_indices=keep_indices)
    if partial_state[0][0] == 1.0:
            return partial_state[0][1]
    else:
        for state in partial_state:
            print(state[0],cirq.dirac_notation(state[1]))
        raise RuntimeError('state is entangled')
        