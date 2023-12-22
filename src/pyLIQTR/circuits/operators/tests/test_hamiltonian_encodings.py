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

from pyLIQTR.utils.printing import _build_qasm_qubit_map
from pyLIQTR.utils.circuit_decomposition import decompose_once

from pyLIQTR.utils.Hamiltonian import Hamiltonian
from pyLIQTR.circuits.operators.selectV \
    import ChildsSelectV,QROMSelectV
from pyLIQTR.circuits.operators.prepare import Prepare
from pyLIQTR.circuits.operators.hamiltonian_encodings \
    import UnitaryBlockEncode, SzegedyWalkOperator

class TestEncodings:
    @pytest.fixture(scope="class")
    def getHamiltonianAndQubits(self):
        terms = [("XXXX",1),("YYYY",1),("ZZZZ",1),
                 ("XIIY",-1),("IXYI",-2),("IZIZ",-3)]
        ham = Hamiltonian(terms)
        ham.adjust_hamiltonian()
        allqubits = cirq.LineQubit.range(4+ham.loglen+1+3)
        trgt = allqubits[0:4]
        ctl = allqubits[4:7]
        phs = [allqubits[7]]
        anc = allqubits[8:]

        yield (ham,trgt,ctl,phs,anc)
        del (ham,trgt,ctl,phs,anc)
        
    @pytest.fixture(scope="class")
    def getHamiltonianAndQubits2(self):
        terms = [("XXXX",-1),("YYYY",-1),("ZZZZ",-1),
                 ("XIIY",+1),("IXYI",+2),("IZIZ",+3)]
        ham = Hamiltonian(terms)
        ham.adjust_hamiltonian()
        allqubits = cirq.LineQubit.range(4+ham.loglen+1+3)
        trgt = allqubits[0:4]
        ctl = allqubits[4:7]
        phs = [allqubits[7]]
        anc = allqubits[8:]

        yield (ham,trgt,ctl,phs,anc)
        del (ham,trgt,ctl,phs,anc)

    @pytest.fixture(scope="class")
    def getOperator(self, getHamiltonianAndQubits):
        ham = getHamiltonianAndQubits[0]
        trgtQ = getHamiltonianAndQubits[1]
        ctlQ = getHamiltonianAndQubits[2]
        phsQ = getHamiltonianAndQubits[3]
        ancillaQ = getHamiltonianAndQubits[4]
        allQ = [*trgtQ,*ctlQ,*phsQ,*ancillaQ]
        
        #A hamiltonian encoding only generates the action of a 
        #prepare and select, no other guarantees
        selVO = ChildsSelectV(hamiltonian = ham,
                            phase_qubit = phsQ,
                            target_qubits = trgtQ,
                            control_qubits = ctlQ,
                            ancilla_qubits = ancillaQ)
        prepO = Prepare(qubit_reg = ctlQ, alphas = ham.alphas)
        operator = UnitaryBlockEncode(hamiltonian=ham,
                            selectOracle=selVO, prepareOracle=prepO,
                            phase_qubit = phsQ,
                            target_qubits = trgtQ,
                            control_qubits = ctlQ,
                            ancilla_qubits = ancillaQ)

        yield operator, selVO, prepO, allQ
    
    def test_unitary_basic(self, getOperator):
        operator = getOperator[0]
        selVO = getOperator[1]
        prepO = getOperator[2]
        allQ = getOperator[3]
        
        circuit = cirq.Circuit()
        circuit.append(operator.on(*allQ))
        opStr = str(operator)
        #Check printing
        assert("UnitaryBE q(7),q(8),q(9),q(10),q(4),q(5),q(6),q(0),q(1),q(2),q(3)" == \
                opStr)

        #check qasm print
        qasm_args,__,__ = _build_qasm_qubit_map(circuit)
        trueQasmStr = "UnitaryBE(q(7),q(8),q(9),q(10),q(4),q(5),q(6),q(0),q(1),q(2),q(3))\n"
        for op in circuit[0]:
            opQasmStr = cirq.qasm(op, args = qasm_args)
            assert(opQasmStr == trueQasmStr)
        #Check number of qubits
        assert(operator.num_qubits() == 11)

        #We wrap these two operators, therefore, we should add one level of decomposition
        assert(operator.total_decomp == selVO.total_decomp+1)

    def test_unitary_decomp(self, getOperator):
        operator = getOperator[0]
        selVO = getOperator[1]
        prepO = getOperator[2]
        allQ = getOperator[3]

        circuit = cirq.Circuit()
        circuit.append(operator.on(*allQ))
        decomposed_circuit = cirq.align_left(decompose_once(circuit))
        true_qasm_str = """
OPENQASM 2.0;
include "qelib1.inc";


// Qubits: [q(0), q(1), q(2), q(3), q(4), q(5), q(6), q(7), q(8), q(9), q(10)]
qreg q[11];


Prepare(q(4),q(5),q(6))
ChildsSelectV(q(7),q(8),q(9),q(10),q(4),q(5),q(6),q(0),q(1),q(2),q(3))
"""
        cq_qasm = cirq.qasm(decomposed_circuit)
        assert(cq_qasm.endswith(true_qasm_str))