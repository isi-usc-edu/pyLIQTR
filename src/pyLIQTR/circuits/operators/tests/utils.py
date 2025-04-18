"""
Copyright (c) 2024 Massachusetts Institute of Technology 
SPDX-License-Identifier: BSD-2-Clause
"""
import cirq
from typing import Optional
from qualtran import Bloq

def strip_bloqs(item,bloqs_to_strip=('Allocate','Free')):
    new_item = []
    for subitem in item:
        if all(x not in subitem.__str__() for x in bloqs_to_strip):
            new_item.append(subitem)
    return cirq.Moment.from_ops(*new_item)

def ignore_bookkeeping(b: Bloq) -> Optional[Bloq]:
    """A generalizer that ignores split and join operations."""
    from qualtran.bloqs.bookkeeping import Cast, Join, Partition, Split, Allocate, Free, Cast

    if isinstance(b, (Split, Join, Partition, Cast, Allocate, Free, Cast)):
        return None
    return b

def get_partial_state(state_vector,keep_indices):
    partial_state = cirq.partial_trace_of_state_vector_as_mixture(state_vector,keep_indices=keep_indices)
    if partial_state[0][0] == 1.0:
            return partial_state[0][1]
    else:
        for state in partial_state:
            print(state[0],cirq.dirac_notation(state[1]))
        raise RuntimeError('state is entangled')
        