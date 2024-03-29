"""
DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.

This material is based upon work supported by the Under Secretary of Defense for
Research and Engineering under Air Force Contract No. FA8702-15-D-0001. Any opinions,
findings, conclusions or recommendations expressed in this material are those of the
author(s) and do not necessarily reflect the views of the Under Secretary of Defense
for Research and Engineering.

© 2023 Massachusetts Institute of Technology.

The software/firmware is provided to you on an As-Is basis

Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS Part
252.227-7013 or 7014 (Feb 2014). Notwithstanding any copyright notice, U.S. Government
rights in this work are defined by DFARS 252.227-7013 or DFARS 252.227-7014 as detailed
above. Use of this work other than as specifically authorized by the U.S. Government
may violate any copyrights that exist in this work.
"""
"""This set of unit tests will verify that we can still correctly decompose and analyze the resources,
specificallty T-count, Rotations and Cliffords of the various cirq-ft gates.

This is the current list of the gates and likely will need to be expanded as Google releases more
cirq-ft gates.

Test Process:
1. Create an instance of the given gate
2. Create an operator from the gate using the registers and associated qubits
3. Run cirq_ft.t_complexity() against the operator
4. Capture this result, it will give cirq's evaluation of T-count, Rotations and Cliffords
5. Use cirq.decompose_once to evaluate the topmost level only
6. Create an instance of our pyLCircuit
7. Assign the operator to this pyLCircuit instance
8. Run pyLCircuit.resourceAnalyze() against the pyLCircuit instance
9. Capture the results of resourceAnalyze()
10. Compare the T-count, Rotations and Cliffords counts against each other

As long as these number match we are still correctly decomposing the cirq-ft gates.
"""

import numpy as np
from typing import *

import cirq

from qualtran import _infra
from qualtran.bloqs.and_bloq import And
from qualtran.cirq_interop.t_complexity_protocol import t_complexity
from qualtran.bloqs.hubbard_model import *

from pyLIQTR.circuits.pyLCircuit import pyLCircuit as pylc

def _z_to_odd(n: int):
    if n % 2 == 1:
        return cirq.Z
    return cirq.I

def get_resource_state(m: int):
    """Returns a state vector representing the resource state on m qubits from Eq.17 of Ref-1.
    Returns a numpy array of size 2^{m} representing the state vector corresponding to the state
    $$
        \sqrt{\frac{2}{2^m + 1}} \sum_{n=0}^{2^{m}-1} \sin{\frac{\pi(n + 1)}{2^{m}+1}}\ket{n}
    $$
    
    Params:
        m: Number of qubits to prepare the resource state on.
    
    Ref:
        1) [Encoding Electronic Spectra in Quantum Circuits with Linear T Complexity]
            (https://arxiv.org/abs/1805.03662)
            Eq. 17
        2) [Google]
            (https://github.com/quantumlib/Cirq/blob/master/cirq-ft/cirq_ft/algos/phase_estimation_of_quantum_walk.ipynb)
    """
    den = 1 + 2 ** m
    norm = np.sqrt(2 / den)
    return norm * np.sin(np.pi * (1 + np.arange(2**m)) / den)


class TestCirqFTSIngleGateDecompose:
    """This Class will test the Cirq-FT gates, but only one at a time. The goal is to make sure we are able
    to correctly analyse the resources necessary to build the circuts as defined now and in the future.

    WARNING: we set the decompose_rotations=False flag in pyLCircuit.resourceAnalyze() so that we can keep the comparison 
    with the cirq-ft resource analysis compatible. If we let it default to True then pyLIQTR will convert the rotation gates into
    Clifford+T's and cause all of the counts to be drastically different.
    """
    def test_cirqft_and(self):
        """This tests the cirq-ft And gate and compares our resource analysis against that."""
        gate = And() 
        cft_op = gate.on_registers(**_infra.gate_with_registers.get_named_qubits(gate.signature))
        cft = t_complexity(cft_op)

        cirq.decompose_once(cft_op)

        pylc_and = pylc([cft_op])
        pylcra = pylc_and.resourceAnalyze(decompose_rotations=False)

        assert pylcra['CliffT']['T'] == cft.t
        assert pylcra['CliffT']['Rotations'] == cft.rotations
        assert pylcra['CliffT']['Clifford'] == cft.clifford

    def test_cirqft_adjoint_and(self):
        """This tests our ability to analyze the resources of cirq-ft's Adjoint And gate."""
        gate = And() ** -1
        cft_op = gate.on_registers(**_infra.gate_with_registers.get_named_qubits(gate.signature))
        cft = t_complexity(cft_op)

        cirq.decompose_once(cft_op)

        pylc_and = pylc([cft_op])
        pylcra = pylc_and.resourceAnalyze(decompose_rotations=False)

        assert pylcra['CliffT']['T'] == cft.t
        assert pylcra['CliffT']['Rotations'] == cft.rotations
        assert pylcra['CliffT']['Clifford'] == cft.clifford

    # Attempting to figure out why cirq can't decompose it's own Quantum Fourier Transform operator
    # def test_cirq_ft(self):
    #     import cirq
    #     from cirq_ft.infra.t_complexity_protocol import t_complexity

    #     circuit = cirq.Circuit()
    #     qubits = [cirq.q(f'm_{i}') for i in range(3)]
    #     qft = cirq.qft(*qubits, inverse=True)
    #     circuit.append(qft)
    #     qfttype = type(qft)
    #     # comp = t_complexity(circuit)
    #     # print(comp)

    #     pylc_qft = pylc([qft])
    #     pylcra = pylc_qft.resourceAnalyze(decompose_rotations=False)

    #     assert pylcra['CliffT']['T'] == 1
    #     assert pylcra['CliffT']['Rotations'] == 1
    #     assert pylcra['CliffT']['Clifford'] == 1
