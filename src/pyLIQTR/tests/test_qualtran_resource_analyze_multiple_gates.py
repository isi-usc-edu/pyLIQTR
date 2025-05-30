"""
Copyright (c) 2024 Massachusetts Institute of Technology 
SPDX-License-Identifier: BSD-2-Clause
"""
from qualtran.bloqs.basic_gates import ZGate

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

import qualtran as qt
import qualtran.cirq_interop as qt_ci
from qualtran import BQUInt as BoundedQUInt
import qualtran.bloqs.chemistry.hubbard_model.qubitization as qt_hm
from qualtran.bloqs.multiplexers.select_pauli_lcu import SelectPauliLCU
from qualtran.bloqs.mcmt import MultiTargetCNOT, MultiControlPauli
import pytest
from qualtran.bloqs.chemistry.ising.walk_operator import get_walk_operator_for_1d_ising_model

from pyLIQTR.utils.resource_analysis import pylqt_t_complexity as t_complexity
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


class TestCirqFTMultipleGateDecompose:
    """This class handles testing multiple gate decomposition and resource analysis. The goal is to make sure we are able
    to correctly analyse the resources necessary to build the circuts as defined now and in the future.
    
    Unit tests enforce code that doesn't introduce breaking changes to the base.

    WARNING: we set the decompose_rotations=False flag in pyLCircuit.resourceAnalyze() so that we can keep the comparison 
    with the cirq-ft resource analysis compatible. If we let it default to True then pyLIQTR will convert the rotation gates into
    Clifford+T's and cause all of the counts to be drastically different.
    """
    def test_cirqft_generic_select(self):
        target_bitsize = 4
        us = ['XIXI', 'YIYI', 'ZZZZ', 'ZXYZ']
        us = [cirq.DensePauliString(u) for u in us]
        selection_bitsize = int(np.ceil(np.log2(len(us))))

        gate = SelectPauliLCU(selection_bitsize, target_bitsize, select_unitaries=us)
        cft_op = gate.on_registers(**qt_ci.get_named_qubits(gate.signature))
        cft = t_complexity(cft_op)

        cirq.decompose_once(cft_op)

        pylc_and = pylc([cft_op])
        pylcra = pylc_and.resourceAnalyze(decompose_rotations=False)

        assert pylcra['CliffT']['T'] == cft.t
        assert pylcra['CliffT']['Rotations'] == cft.rotations
        assert pylcra['CliffT']['Clifford'] == cft.clifford

    def test_cirqft_apply_gate_to_lth_qubit(self):
        apply_z_to_odd = qt.bloqs.multiplexers.apply_gate_to_lth_target.ApplyGateToLthQubit(
            qt.Register('selection', dtype=BoundedQUInt(3, 4)),
            nth_gate=_z_to_odd,
            # control_regs=cirq_ft.Signature.build(control=2),
        )
        cft_op = apply_z_to_odd.on_registers(**qt_ci.get_named_qubits(apply_z_to_odd.signature))
        cft = t_complexity(cft_op)

        cirq.decompose_once(cft_op)

        pylc_and = pylc([cft_op])
        pylcra = pylc_and.resourceAnalyze(decompose_rotations=False)

        assert pylcra['CliffT']['T'] == cft.t
        assert pylcra['CliffT']['Rotations'] == cft.rotations
        assert pylcra['CliffT']['Clifford'] == cft.clifford

    def test_cirqft_hubbard(self):
        x_dim, y_dim, t = 2, 2, 5
        u = 4 * t
        gate = qt_hm.PrepareHubbard(x_dim=x_dim, y_dim=x_dim, t=t, u=u)
        cft_op = gate.on_registers(**qt_ci.get_named_qubits(gate.signature))
        cft = t_complexity(cft_op)

        cirq.decompose_once(cft_op)

        pylc_and = pylc([cft_op])
        pylcra = pylc_and.resourceAnalyze(decompose_rotations=False)

        assert pylcra['CliffT']['T'] == cft.t
        assert pylcra['CliffT']['Rotations'] == cft.rotations
        assert pylcra['CliffT']['Clifford'] == cft.clifford

    def test_cirqft_multi_target_cnot(self):
        gate = MultiTargetCNOT(bitsize=2) 
        cft_op = gate.on_registers(**qt_ci.get_named_qubits(gate.signature))
        cft = t_complexity(cft_op)

        cirq.decompose_once(cft_op)

        pylc_and = pylc([cft_op])
        pylcra = pylc_and.resourceAnalyze(decompose_rotations=False)

        assert pylcra['CliffT']['T'] == cft.t
        assert pylcra['CliffT']['Rotations'] == cft.rotations
        assert pylcra['CliffT']['Clifford'] == cft.clifford

    def test_cirqft_multicontrol_pauli(self):
        cvs = (0, 1)
        gate = MultiControlPauli(cvs=cvs, target_bloq=ZGate())
        cft_op = gate.on_registers(**qt_ci.get_named_qubits(gate.signature))
        cft = t_complexity(cft_op)

        cirq.decompose_once(cft_op)

        pylc_and = pylc([cft_op])
        pylcra = pylc_and.resourceAnalyze(decompose_rotations=False)

        assert pylcra['CliffT']['T'] == cft.t
        assert pylcra['CliffT']['Rotations'] == cft.rotations
        assert pylcra['CliffT']['Clifford'] == cft.clifford

    def test_cirqft_walk_operator(self):
        num_sites: int = 200
        eps: float = 1e-5
        m_bits: int = 14
        walk, _ = get_walk_operator_for_1d_ising_model(num_sites, eps)
        cft_op = walk.on_registers(**qt_ci.get_named_qubits(walk.signature))
        cft = t_complexity(cft_op)

        cirq.decompose_once(cft_op)

        pylc_and = pylc([cft_op])
        pylcra = pylc_and.resourceAnalyze(decompose_rotations=False)

        assert pylcra['CliffT']['T'] == cft.t
        assert pylcra['CliffT']['Rotations'] == cft.rotations
        assert pylcra['CliffT']['Clifford'] == cft.clifford

    def test_cirqft_prepare_uniform_superposition(self):
        # For now I'll tie the number if T-Basis states to 10 to test this
        # there is no particular reason for using 10 just to keep the number of iterations low enough to not impact performance.
        gate = qt.bloqs.state_preparation.PrepareUniformSuperposition(n=10, cvs=(1,))
        cft_op = gate.on_registers(**qt_ci.get_named_qubits(gate.signature))
        cft = t_complexity(gate)

        cirq.decompose_once(cft_op)

        pylc_and = pylc([cft_op])
        pylcra = pylc_and.resourceAnalyze(decompose_rotations=False)

        assert pylcra['CliffT']['T'] == cft.t
        assert pylcra['CliffT']['Rotations'] == cft.rotations
        assert pylcra['CliffT']['Clifford'] == cft.clifford
