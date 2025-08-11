import sys
import pyLIQTR
from pathlib import Path
from testbook import testbook

SRC_DIR = Path(pyLIQTR.__file__).parent.parent
ANI_DIR = Path(SRC_DIR).parent / "Examples/Algorithms_and_Infrastructure"
NB = Path(ANI_DIR) / "scheduling_example.ipynb"

@testbook(NB, execute=['imports', 'circuit_gen', 'estimate_resources'])
def test_instance_creation(tb):
    tb.inject("assert resources['T'] == 3320")
    tb.inject("assert resources['LogicalQubits'] == 9")

@testbook(NB, execute=['imports', 'circuit_gen', 'schedule_circuit'])
def test_standard_scheduler_results(tb):
    tb.inject("assert res['Total time for execution']>596000 and res['Total time for execution']<597500")
    tb.inject("assert res['Circuit T-depth'] > 2730 and res['Circuit T-depth'] < 2735")
    tb.inject("assert res['Number of qubits used'] == 13")
    tb.inject("assert res['Gate profile']['T'] == 2760")

'''
@testbook(NB, execute=['imports', 'circuit_gen', 'build_factory', 'build_arch', 'factory_schedule'])
def test_factory_scheduler_results(tb):
    tb.inject("assert res['Total time for execution'] == 18710")
    tb.inject("assert res['Circuit T-depth'] == 2760")
    tb.inject("assert res['Number of qubits used'] == 13")
    tb.inject("assert res['Gate profile']['T'] == 2760")
'''
