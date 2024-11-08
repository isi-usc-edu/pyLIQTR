"""
Copyright (c) 2024 Massachusetts Institute of Technology 
SPDX-License-Identifier: BSD-2-Clause
"""
import pytest
import platform
import cirq
from pyLIQTR.ProblemInstances.getInstance import *
from pyLIQTR.BlockEncodings.getEncoding import *
from pyLIQTR.utils.printing import openqasm
from pyLIQTR.utils.resource_analysis import estimate_resources
from openfermion.chem import MolecularData

@pytest.mark.skipif(platform.system() == 'Windows', reason = "cannot run this test on Windows")
class TestDoubleFactorizedEncoding:

    @pytest.fixture(scope="class")
    def h2_instance(self):
        from openfermionpyscf import run_pyscf
        mol_data = MolecularData([('H', (0.0, 0.0, 0.63164)), ('H', (0.0, 0.0, 1.76836))],\
                                 'sto-3g', 1.0, 0, 'H2')
        mol = run_pyscf(mol_data, run_scf=1, run_mp2=0, run_cisd=0, run_ccsd=0, run_fci=0, verbose=0)
        mol_instance = getInstance('ChemicalHamiltonian',mol_ham=mol.get_molecular_hamiltonian(),mol_name='H2')
        return mol_instance

    @pytest.fixture(scope="class")
    def df_encoding(self, h2_instance):
        return getEncoding(instance=h2_instance, encoding=VALID_ENCODINGS.DoubleFactorized,df_error_threshold=1e-2,br=4,step_error=1e-1)

    def test_DoubleFactorized_decomposes(self, df_encoding):
        '''
        Tests gate decomposition existence.
        '''
        num_qubits = cirq.num_qubits(df_encoding.circuit)
        qubits = cirq.LineQubit.range(num_qubits)
        # initialize operator
        operation = df_encoding.on(*qubits)
        # check decompose_once raises no error
        decomposed_once = cirq.decompose_once(operation)
        assert([operation] != decomposed_once)
        # check decompose returns decomposition not equal to operation itself
        decomposed = cirq.decompose(operation)
        assert([operation] != decomposed)

    def test_DoubleFactorized_qasm(self,df_encoding):
        '''
        Tests qasm printing functionality.
        '''
        context = cirq.DecompositionContext(cirq.SimpleQubitManager())
        num_qubits = cirq.num_qubits(df_encoding.circuit)
        qubits = cirq.LineQubit.range(num_qubits)
        # initialize operator
        operation = df_encoding.on(*qubits)

        qasm = openqasm(operation,rotation_allowed=True,context=context)
        assert qasm is not None
        for line in qasm:
            pass

    def test_DoubleFactorized_sf_thresh_zero(self, h2_instance):
        '''
        Tests gate instantiates with sf_error_threshold=0.0.
        '''
        df_encoding = getEncoding(instance=h2_instance, encoding=VALID_ENCODINGS.DoubleFactorized,df_error_threshold=1e-2,sf_error_threshold=0.0,br=4,step_error=1e-1)

    def test_DoubleFactorized_callgraph(self,df_encoding):
        '''
        Tests ability to generate call_graph.
        '''
        call_graph, _ = df_encoding.call_graph()

    def test_DoubleFactorized_resources(self, df_encoding):
        '''
        Tests estimate_resources executes without error.
        '''
        resources = estimate_resources(df_encoding.circuit)
        