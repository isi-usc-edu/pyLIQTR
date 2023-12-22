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
from numpy import pi

from pyLIQTR.circuits.operators.reflect import Reflect
from pyLIQTR.utils.printing import _build_qasm_qubit_map

class TestReflect:
    @pytest.fixture(scope="class")
    def getAngleAndQubits(self):
        all_qubits = cirq.LineQubit.range(7)
        ctl = all_qubits[0:4]
        phs = all_qubits[4]
        anc = all_qubits[5:]
        angle = 30
        
        yield(angle, ctl, phs, anc)
        del ( angle, ctl, phs, anc)
        
    @pytest.fixture(scope="class")
    def getAngleAndQubits2(self):
        all_qubits = cirq.LineQubit.range(1,8)
        ctl = all_qubits[0:4]
        phs = all_qubits[4]
        anc = all_qubits[5:]
        angle = 40
        
        yield(angle, ctl, phs, anc)
        del ( angle, ctl, phs, anc)
        
    def test_reflect_basic(self,getAngleAndQubits):
        angle = getAngleAndQubits[0]
        ctlQ  = getAngleAndQubits[1]
        phsQ  = getAngleAndQubits[2]
        ancQ  = getAngleAndQubits[3]
        allQ  = [*ctlQ, *[phsQ], *ancQ]
        
        # Should be able to be created:
        operator = Reflect(phi=angle,
                           control_qubits=ctlQ,
                           phase_qubit=phsQ,
                           ancilla_qubits=ancQ)
        
        # Should be able to be appended onto a circuit:
        circuit = cirq.Circuit()
        circuit.append(operator.on(*allQ))
        
        # Let's check that the string version is correct:
        opStr = str(operator)
        assert(opStr == "Reflect (q(5),q(6),q(0),q(1),q(2),q(3),q(4))")
        
        # Check the number of qubits:
        assert(operator.num_qubits() == 7)
        
    def test_reflect_toQasm(self,getAngleAndQubits):
        angle = getAngleAndQubits[0]
        ctlQ  = getAngleAndQubits[1]
        phsQ  = getAngleAndQubits[2]
        ancQ  = getAngleAndQubits[3]
        allQ  = [*ctlQ, *[phsQ], *ancQ]
        
        # Should be able to be created:
        operator = Reflect(phi=angle,
                           control_qubits=ctlQ,
                           phase_qubit=phsQ,
                           ancilla_qubits=ancQ)
        
        circuit = cirq.Circuit()
        circuit.append(operator.on(*allQ))
        
        # Check the openQASM decomposition:
        trueQasmStr ="""
OPENQASM 2.0;
include "qelib1.inc";


// Qubits: [q(0), q(1), q(2), q(3), q(4), q(5), q(6)]
qreg q[7];


Reflect (q(5),q(6),q(0),q(1),q(2),q(3),q(4))"""
        qasm_args,__,__ = _build_qasm_qubit_map(circuit)
        # for op in circuit[0]:
        opQasmStr = cirq.qasm(circuit)  

        # opQasmStr = cirq.qasm(operator, args=qasm_args, qubits=allQ)
        assert(opQasmStr.endswith(trueQasmStr))
        
    def test_reflect_equality(self, getAngleAndQubits, getAngleAndQubits2):
        
        angle = getAngleAndQubits[0]
        ctlQ  = getAngleAndQubits[1]
        phsQ  = getAngleAndQubits[2]
        ancQ  = getAngleAndQubits[3]
        allQ  = [*ctlQ, *[phsQ], *ancQ]
        
        # Should be able to be created:
        operator = Reflect(phi=angle,
                           control_qubits=ctlQ,
                           phase_qubit=phsQ,
                           ancilla_qubits=ancQ)
        
        angle2 = getAngleAndQubits2[0]
        ctlQ2  = getAngleAndQubits2[1]
        phsQ2  = getAngleAndQubits2[2]
        ancQ2  = getAngleAndQubits2[3]
        allQ2  = [*ctlQ2, *[phsQ2], *ancQ2]
        
        # Should be able to be created:
        operator2 = Reflect(phi=angle2,
                           control_qubits=ctlQ2,
                           phase_qubit=phsQ2,
                           ancilla_qubits=ancQ2)
        
        assert(operator!=operator2)
        assert(operator==operator)
        assert(operator2==operator2)
        
    @pytest.mark.parametrize('execution_number', range(10))
    def test_reflect_count(self, getAngleAndQubits, execution_number):
        angle = getAngleAndQubits[0]
        ctlQ  = getAngleAndQubits[1]
        phsQ  = getAngleAndQubits[2]
        ancQ  = getAngleAndQubits[3]
        allQ  = [*ctlQ, *[phsQ], *ancQ]
        
        # Should be able to be created:
        operator = Reflect(phi=angle,
                           control_qubits=ctlQ,
                           phase_qubit=phsQ,
                           ancilla_qubits=ancQ)
        
        count_exact_dict = {
            'H': 71, 
            'CNOT': 36, 
            'T**-1': 23,
            'T': 79, 
            # 'S': 41, 
            'S**-1': 1,
            'Z': 1}
        auto_count = operator.count_exact()

        # something changed in Cirqso that the reflect count of the 'S' component is verying 
        # between 25 and 41, we haven't been able to figure out why yet so broke this value out
        # into it's own test
        assert(auto_count['S'] == 41 or auto_count['S'] == 25)
        for key in count_exact_dict:
            assert(auto_count[key] == count_exact_dict[key])