"""
Copyright (c) 2024 Massachusetts Institute of Technology 
SPDX-License-Identifier: BSD-2-Clause
"""
import pytest
import cirq
import numpy as np

from pyLIQTR.PhaseEstimation.pe_gates import Trotter_Unitary
from openfermion import FermionOperator, QubitOperator

class TestPEGates:

    @pytest.mark.parametrize("ham_op",[FermionOperator("2^ 15"),QubitOperator(((1, "X"), (2, "Y"), (3, "Z")))])
    def test_Trotter_Unitary_instantiates(self,ham_op):
        """
        Tests instantiation works for both openfermion.FermionOperator and openfermion.QubitOperator
        """
        prec_qb = cirq.NamedQubit('prec')
        reg_qbs = cirq.NamedQubit.range(3,prefix='reg')
        args = { 'trotterize':True, 'mol_ham':ham_op, 'ev_time':2, 'trot_ord':1, 'tort_num':1}
        gate = Trotter_Unitary(tgt_qubit=prec_qb,reg_qubits=reg_qbs,prec_order=1,operator_power=1,kwargs=args)

