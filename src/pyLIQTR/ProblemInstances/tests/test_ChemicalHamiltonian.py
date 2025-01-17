"""
Copyright (c) 2024 Massachusetts Institute of Technology 
SPDX-License-Identifier: BSD-2-Clause
"""
import pytest
import platform
import numpy as np
from openfermion import InteractionOperator
from pyLIQTR.ProblemInstances.getInstance import *

@pytest.mark.skipif(platform.system() == 'Windows', reason = "pyscf not supported on Windows")
class TestChemicalHamiltonian:

    @pytest.fixture(scope="class")
    def h2_instance(self):
        obt = np.array([[-1.04689337,  0.0,  0.0,  0.0],
                        [ 0.0, -1.04689337,  0.0,  0.0],
                        [ 0.0,  0.0, -0.62248428,  0.0],
                        [ 0.0,  0.0,  0.0, -0.62248428]])
        tbt = np.array([[[[0.30156089, 0.0, 0.0, 0.0],
         [0.0, 0.0, 0.0, 0.0],
         [0.0, 0.0, 0.10281136, 0.0],
         [0.0, 0.0, 0.0, 0.0]],
        [[0.0, 0.0, 0.0, 0.0],
         [0.30156089, 0.0, 0.0, 0.0],
         [0.0, 0.0, 0.0, 0.0],
         [0.0, 0.0, 0.10281136, 0.0]],
        [[0.0, 0.0, 0.10281136, 0.0],
         [0.0, 0.0, 0.0, 0.0],
         [0.3011551 , 0.0, 0.0, 0.0],
         [0.0, 0.0, 0.0, 0.0]],
        [[0.0, 0.0, 0.0, 0.0],
         [0.0, 0.0, 0.10281136, 0.0],
         [0.0, 0.0, 0.0, 0.0],
         [0.3011551 , 0.0, 0.0, 0.0]]],
       [[[0.0, 0.30156089, 0.0, 0.0],
         [0.0, 0.0, 0.0, 0.0],
         [0.0, 0.0, 0.0, 0.10281136],
         [0.0, 0.0, 0.0, 0.0]],
        [[0.0, 0.0, 0.0, 0.0],
         [0.0, 0.30156089, 0.0, 0.0],
         [0.0, 0.0, 0.0, 0.0],
         [0.0, 0.0, 0.0, 0.10281136]],
        [[0.0, 0.0, 0.0, 0.10281136],
         [0.0, 0.0, 0.0, 0.0],
         [0.0, 0.3011551 , 0.0, 0.0],
         [0.0, 0.0, 0.0, 0.0]],
        [[0.0, 0.0, 0.0, 0.0],
         [0.0, 0.0, 0.0, 0.10281136],
         [0.0, 0.0, 0.0, 0.0],
         [0.0, 0.3011551 , 0.0, 0.0]]],
       [[[0.0, 0.0, 0.3011551 , 0.0],
         [0.0, 0.0, 0.0, 0.0],
         [0.10281136, 0.0, 0.0, 0.0],
         [0.0, 0.0, 0.0, 0.0]],
        [[0.0, 0.0, 0.0, 0.0],
         [0.0, 0.0, 0.3011551 , 0.0],
         [0.0, 0.0, 0.0, 0.0],
         [0.10281136, 0.0, 0.0, 0.0]],
        [[0.10281136, 0.0, 0.0, 0.0],
         [0.0, 0.0, 0.0, 0.0],
         [0.0, 0.0, 0.31597877, 0.0],
         [0.0, 0.0, 0.0, 0.0]],
        [[0.0, 0.0, 0.0, 0.0],
         [0.10281136, 0.0, 0.0, 0.0],
         [0.0, 0.0, 0.0, 0.0],
         [0.0, 0.0, 0.31597877, 0.0]]],
       [[[0.0, 0.0, 0.0, 0.3011551 ],
         [0.0, 0.0, 0.0, 0.0],
         [0.0, 0.10281136, 0.0, 0.0],
         [0.0, 0.0, 0.0, 0.0]],
        [[0.0, 0.0, 0.0, 0.0],
         [0.0, 0.0, 0.0, 0.3011551 ],
         [0.0, 0.0, 0.0, 0.0],
         [0.0, 0.10281136, 0.0, 0.0]],
        [[0.0, 0.10281136, 0.0, 0.0],
         [0.0, 0.0, 0.0, 0.0],
         [0.0, 0.0, 0.0, 0.31597877],
         [0.0, 0.0, 0.0, 0.0]],
        [[0.0, 0.0, 0.0, 0.0],
         [0.0, 0.10281136, 0.0, 0.0],
         [0.0, 0.0, 0.0, 0.0],
         [0.0, 0.0, 0.0, 0.31597877]]]])
        h2_op = InteractionOperator(constant=0.4655299554155818,one_body_tensor=obt,two_body_tensor=tbt)
        mol_instance = getInstance('ChemicalHamiltonian',mol_ham=h2_op,mol_name='H2')
        return mol_instance

    def test_ChemicalHamiltonian_alpha(self, h2_instance):
        lcu_alpha1 = h2_instance.get_alpha()
        assert lcu_alpha1 == 1.4527183600000002

        lcu_alpha2 = h2_instance.get_alpha(encoding='PauliLCU')
        assert lcu_alpha2 == 1.4527183600000002

        df_alpha = h2_instance.get_alpha(encoding='DF')
        assert df_alpha == 1.2543735419921236

        df_alpha_pass_err = h2_instance.get_alpha(encoding='DF',df_error_threshold=0.99)
        assert df_alpha_pass_err == 0.8006020005033776

        df_alpha_pass_xi = h2_instance.get_alpha(encoding='DF',df_cutoffs=[1,1])
        assert df_alpha_pass_xi == 0.6463849605033776

        with pytest.raises(ValueError,match='provide only df_error_threshold or df_cutoffs'):
            alpha = h2_instance.get_alpha(encoding='DF',df_cutoffs=[1,1],df_error_threshold=1e-2)