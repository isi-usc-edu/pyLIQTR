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
import pytest
import cirq
import numpy as np
from pyLIQTR.ProblemInstances.getInstance import *
from pyLIQTR.utils.get_hdf5 import  get_hdf5 
from pyLIQTR.BlockEncodings.getEncoding import *
from  pyLIQTR.utils.global_ancilla_manager import gam as gam
from  pyLIQTR.utils.printing import openqasm
# from openfermionpyscf import run_pyscf
from openfermion.chem import MolecularData

class TestDoubleFactorizedEncoding:

    @pytest.mark.skip
    @pytest.fixture(scope="class")
    def h2_instance(self):
        mol_data = MolecularData([('H', (0.0, 0.0, 0.63164)), ('H', (0.0, 0.0, 1.76836))],\
                                 'sto-3g', 1.0, 0, 'H2')
        mol = run_pyscf(mol_data, run_scf=1, run_mp2=0, run_cisd=0, run_ccsd=0, run_fci=0, verbose=0)
        mol_instance = getInstance('ChemicalHamiltonian',mol_ham=mol.get_molecular_hamiltonian(),mol_name='H2')
        return mol_instance

    @pytest.mark.skip
    def test_DoubleFactorized_decomposes(self, h2_instance):
        '''
        Tests gate decomposition existence.
        '''
        df_encoding = getEncoding(instance=h2_instance, encoding=VALID_ENCODINGS.DoubleFactorized,df_error_threshold=1e-2,br=4,step_error=1e-1)
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

    @pytest.mark.skip
    def test_DoubleFactorized_qasm(self,h2_instance):
        '''
        Tests qasm printing functionality.
        '''
        context = cirq.DecompositionContext(gam)
        # create registers
        df_encoding = getEncoding(instance=h2_instance, encoding=VALID_ENCODINGS.DoubleFactorized,df_error_threshold=1e-2,br=4,step_error=1e-1)
        num_qubits = cirq.num_qubits(df_encoding.circuit)
        qubits = cirq.LineQubit.range(num_qubits)
        # initialize operator
        operation = df_encoding.on(*qubits)

        qasm = openqasm(operation,rotation_allowed=True,context=context)
        assert qasm is not None