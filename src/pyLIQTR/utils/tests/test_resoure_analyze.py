# """
# DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.

# This material is based upon work supported by the Under Secretary of Defense for
# Research and Engineering under Air Force Contract No. FA8702-15-D-0001. Any opinions,
# findings, conclusions or recommendations expressed in this material are those of the
# author(s) and do not necessarily reflect the views of the Under Secretary of Defense
# for Research and Engineering.

# © 2022 Massachusetts Institute of Technology.

# The software/firmware is provided to you on an As-Is basis

# Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS Part
# 252.227-7013 or 7014 (Feb 2014). Notwithstanding any copyright notice, U.S. Government
# rights in this work are defined by DFARS 252.227-7013 or DFARS 252.227-7014 as detailed
# above. Use of this work other than as specifically authorized by the U.S. Government
# may violate any copyrights that exist in this work.
# """
import cirq
import pytest
import pyLIQTR.utils.resource_analysis as ra
import numpy as np
# Below are specific for testing using notebooks
import os
import pyLIQTR
from pathlib import Path
from testbook import testbook

angles = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])*np.pi
# Get the `src` directory, because we need the route to the Notebooks relative to it
SRC_DIR = Path(pyLIQTR.__file__).parent.parent
FH_DIR = Path(SRC_DIR).parent / "Examples/ApplicationInstances/FermiHubbard"
FH_DYNAMICS_QUBITIZED = Path(FH_DIR) / "fermi_hubbard-dynamics-qubitized.ipynb"
RE_DIR = Path(SRC_DIR).parent / "Examples/Algorithms_and_Infrastructure"
HARDWARE_RE = Path(RE_DIR) / "hardware_resource_estimation.ipynb"

class TestLogicalResourceAnalysis:
    @testbook(FH_DYNAMICS_QUBITIZED, execute=['imports', 'getInstance', 'blockEncoding'])
    def test_FH_dynamics_qubitized_be(self, tb):
        pass

    @testbook(FH_DYNAMICS_QUBITIZED, execute=['imports', 'getInstance', 'blockEncoding'])
    def test_FH_logical_resource_estimate(self, tb):
        """tests that our logical estimates are working properly."""

