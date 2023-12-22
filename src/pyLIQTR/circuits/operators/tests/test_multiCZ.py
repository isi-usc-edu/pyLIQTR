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

from pyLIQTR.circuits.operators.multiCZ import MultiCZ
from pyLIQTR.utils.printing import _build_qasm_qubit_map

class TestMultiCZ:
    @pytest.fixture(scope="class")
    def getQubits(self):
        all_qubits = cirq.LineQubit.range(7)
        ctl = all_qubits[0:4]
        phs = [all_qubits[4]]
        anc = all_qubits[5:]
        
        yield(ctl, phs, anc)
        del ( ctl, phs, anc)
        
    @pytest.fixture(scope="class")
    def getQubits2(self):
        all_qubits = cirq.LineQubit.range(1,8)
        ctl = all_qubits[0:4]
        phs = [all_qubits[4]]
        anc = all_qubits[5:]
        
        yield(ctl, phs, anc)
        del ( ctl, phs, anc)
        
    def test_multi_cz_basic(self, getQubits):
        ctlQ = getQubits[0]
        phsQ = getQubits[1]
        ancQ = getQubits[2]
        allQ = [*ctlQ, *phsQ, *ancQ]
        
        # Should be able to be created:
        operator = MultiCZ(control_qubits=ctlQ,
                           target_qubit=phsQ,
                           ancilla_qubits=ancQ)
        
        # Should be able to be appeneded onto a circuit
        circuit = cirq.Circuit()
        circuit.append(operator.on(*allQ))
        
        #let's check the string version of this.
        opStr = str(operator)
        assert(opStr ==  'MultiCZ (q(5),q(6),q(0),q(1),q(2),q(3),q(4))')
            
        # Check the number of qubits:
        assert(operator.num_qubits() == 7)
        
    def test_multi_cz_toQasm(self, getQubits):
        ctlQ = getQubits[0]
        phsQ = getQubits[1]
        ancQ = getQubits[2]
        allQ = [*ctlQ, *phsQ, *ancQ]
        
        # Should be able to be created:
        operator = MultiCZ(control_qubits=ctlQ,
                           target_qubit=phsQ,
                           ancilla_qubits=ancQ)
        
        # Should be able to be appeneded onto a circuit
        circuit = cirq.Circuit()
        circuit.append(operator.on(*allQ))
        
         # Check the openQASM decomposition:
        trueQasmStr = """
OPENQASM 2.0;
include "qelib1.inc";


// Qubits: [q(0), q(1), q(2), q(3), q(4), q(5), q(6)]
qreg q[7];


MultiCZ(q(5),q(6),q(0),q(1),q(2),q(3),q(4))
"""
        qasm_args,__,__ = _build_qasm_qubit_map(circuit)
        # for op in circuit[0]:
        opQasmStr = cirq.qasm(circuit)  


        # opQasmStr = cirq.qasm(operator, args=qasm_args, qubits=allQ)
        assert(opQasmStr.endswith(trueQasmStr))
        
    def test_multi_cz_equality(self, getQubits, getQubits2):
        ctlQ1 = getQubits[0]
        phsQ1 = getQubits[1]
        ancQ1 = getQubits[2]
        allQ1 = [*ctlQ1, *phsQ1, *ancQ1]
        
        ctlQ2 = getQubits2[0]
        phsQ2 = getQubits2[1]
        ancQ2 = getQubits2[2]
        allQ2 = [*ctlQ2, *phsQ2, *ancQ2]
        
        # Should be able to be created:
        operator1 = MultiCZ(control_qubits=ctlQ1,
                           target_qubit=phsQ1,
                           ancilla_qubits=ancQ1)
        operator2 = MultiCZ(control_qubits=ctlQ2,
                           target_qubit=phsQ2,
                           ancilla_qubits=ancQ2)
        
        assert((operator1!=operator2))
        assert(operator1==operator1)
        assert(operator2==operator2)
        
    def test_multi_cz_count(self, getQubits):
        ctlQ1 = getQubits[0]
        phsQ1 = getQubits[1]
        ancQ1 = getQubits[2]
        allQ1 = [*ctlQ1, *phsQ1, *ancQ1]
              
        operator = MultiCZ(control_qubits=ctlQ1,
                           target_qubit=phsQ1,
                           ancilla_qubits=ancQ1)
        
        count_exact_dict = {
            "H": 4,
            "CNOT": 18,
            "T**-1": 12,
            "T": 9,
            "S": 3
        }

        auto_count = operator.count_exact()
        print(auto_count)
        for key in count_exact_dict:
            assert(auto_count[key] == count_exact_dict[key])
        
        