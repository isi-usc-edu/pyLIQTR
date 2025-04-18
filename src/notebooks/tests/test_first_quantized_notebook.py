import os
import pyLIQTR
from pathlib import Path
from testbook import testbook


# Get the `src` directory, because we need the route to the Notebooks relative to it
SRC_DIR = Path(pyLIQTR.__file__).parent.parent
FQ_DIR = Path(SRC_DIR).parent / "Examples/ApplicationInstances/PeriodicChemistry"
NB = Path(FQ_DIR) / "encoding_in_first_quantization.ipynb"


@testbook(NB, execute=['imports', 'getInstance', 'getEncoding'])
def test_fq_init(tb):
    tb.inject("assert dimer_encoding.lam == 801761.4958623915")
    tb.inject("assert supercell_encoding.lam == 185127454.27157325")

@testbook(NB, execute=['imports', 'getInstance', 'getEncoding'])
def test_fq_resources(tb):
    tb.inject("assert estimate_resources(dimer_encoding._prepare_gate)['T'] == 6961")
    tb.inject("assert estimate_resources(dimer_encoding._prepare_gate)['LogicalQubits'] == 422")