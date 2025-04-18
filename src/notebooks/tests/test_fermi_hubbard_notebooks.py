import os
import pyLIQTR
from pathlib import Path
from testbook import testbook


# Get the `src` directory, because we need the route to the Notebooks relative to it
SRC_DIR = Path(pyLIQTR.__file__).parent.parent
FH_DIR = Path(SRC_DIR).parent / "Examples/ApplicationInstances/FermiHubbard"
FH_DYNAMICS_QUBITIZED = Path(FH_DIR) / "fermi_hubbard-dynamics-qubitized.ipynb"


@testbook(FH_DYNAMICS_QUBITIZED, execute=['imports', 'getInstance'])
def test_dynamics_qubitized(tb):
    tb.inject("assert model.alpha == 18.0")

@testbook(FH_DYNAMICS_QUBITIZED, execute=['imports', 'getInstance', 'blockEncoding'])
def test_dynamics_qubitized_be(tb):
    tb.inject("assert block_encoding.alpha > 0")

# Commented this one out because we fail due to nb client CellExecutionTimeout (default is 60 seconds)
# @testbook(FH_DYNAMICS_QUBITIZED, execute=True)
# def test_dynamics_qubitized_all(tb):
#     tb.inject("assert block_encoding.alpha > 0")

# NOTES
# 1. for some reason I can't wrap @testbook into a pytest class....
# 2. most of the APIs testbook provides use the tb.ref() to grab a reference to a function or a basic data type, then 
#      pass the one to the other. Grabbing a reference to an object seems to be out of their functionality. So we'll
#      need to get clever with how we write our notebooks to allow us to tag different cells and do things like reset the 
#      shape(N, N) being passed to getInstance('FermiHubbard',shape=shape, J=J, U=U, cell=SquareLattice)
# 3. NBclient has a default timeout of 60 seconds to cell execution time. Have yet to figure out a way to extend this, currently running
#      the complete notebook is not possible because we have several cells that take minutes to execute