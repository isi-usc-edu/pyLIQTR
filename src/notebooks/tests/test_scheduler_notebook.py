import sys
import pyLIQTR
from pathlib import Path
from testbook import testbook

SRC_DIR = Path(pyLIQTR.__file__).parent.parent
ANI_DIR = Path(SRC_DIR).parent / "Examples/Algorithms_and_Infrastructure"
NB = Path(ANI_DIR) / "scheduling_example.ipynb"

@testbook(NB, execute=['imports', 'circuit_gen', 'estimate_resources'])
def test_instance_creation(tb):
    tb.inject("assert estimate_resources(circuit, profile=True)['T'] == 248")
    tb.inject("assert estimate_resources(circuit, profile=True)['LogicalQubits'] == 15")

@testbook(NB, execute=['imports', 'circuit_gen', 'schedule_circuit'])
def test_standard_scheduler_results(tb):
    tb.inject("assert res['Total time for execution'] >= 1350 and res['Total time for execution'] <= 1385")
    tb.inject("assert res['Circuit T-depth'] == 124")
    tb.inject("assert res['Number of qubits used'] == 20")
    tb.inject("assert res['Gate profile']['T'] == 248")

@testbook(NB, execute=['imports', 'circuit_gen', 'execution_timings'])
def test_custom_timeset(tb):
    tb.inject("assert res['Total time for execution'] >= 1745 and res['Total time for execution'] <= 1758")
    tb.inject("assert res['Circuit T-depth'] == 180")
    tb.inject("assert res['Gate profile']['T'] == 248")

@testbook(NB, execute=['imports', 'circuit_gen', 't_cliff_and_gateset'])
def test_custom_gateset_1(tb):
    tb.inject("assert res['Gate profile']['T'] == 12978")
    tb.inject("assert res['Gate profile']['And'] == 124")

@testbook(NB, execute=['imports', 'circuit_gen', 't_cliff_gateset'])
def test_custom_gateset_2(tb):
    tb.inject("assert res['Gate profile']['T'] == 13226")
    tb.inject("assert res['Gate profile']['Miscellaneous'] == 124")

@testbook(NB, execute=['imports', 'circuit_gen', 'cx_t_cliff_gateset'])
def test_custom_gateset_3(tb):
    tb.inject("assert res['Gate profile']['T'] == 13226")
    tb.inject("assert res['Gate profile']['CX'] == 666")
