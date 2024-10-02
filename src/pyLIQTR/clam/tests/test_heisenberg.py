"""
Copyright (c) 2024 Massachusetts Institute of Technology 
SPDX-License-Identifier: BSD-2-Clause
"""
import pytest
from pyLIQTR.ProblemInstances.spin_models import Heisenberg
from pyLIQTR.clam.lattice_definitions import SquareLattice


class TestClamHeisenberg:
    """These tests are tests of the clam utilities as they are used within the Heisenberg Notebooks"""
    def test_heisenberg_zero(self):
        instance = Heisenberg((2,2),J=(1,2,3),h=(0,0,0),cell=SquareLattice)
        assert instance._model == "Heisenberg Model - SquareLattice(regular)"

    def test_heisenberg_non_zero(self):
        instance = Heisenberg((2,2),J=(1,2,3),h=(0.,0.,0.1),cell=SquareLattice)
        assert instance._model == "Heisenberg Model - SquareLattice(regular)"
