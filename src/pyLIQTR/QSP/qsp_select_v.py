"""
DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.

This material is based upon work supported by the Under Secretary of Defense for
Research and Engineering under Air Force Contract No. FA8702-15-D-0001. Any opinions,
findings, conclusions or recommendations expressed in this material are those of the
author(s) and do not necessarily reflect the views of the Under Secretary of Defense
for Research and Engineering.

© 2022 Massachusetts Institute of Technology.

The software/firmware is provided to you on an As-Is basis

Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS Part
252.227-7013 or 7014 (Feb 2014). Notwithstanding any copyright notice, U.S. Government
rights in this work are defined by DFARS 252.227-7013 or DFARS 252.227-7014 as detailed
above. Use of this work other than as specifically authorized by the U.S. Government
may violate any copyrights that exist in this work.
"""

import cirq
import numpy as np

#toffoli with appropriate basis change gates
def toffoli(b0,b1,ctl0,ctl1,trgt):
        #3 qubits.
        basis_change = []
        if not b0:
            basis_change.append(cirq.X.on(ctl0))
        if not b1:
            basis_change.append(cirq.X.on(ctl1))
        
        cnX = cirq.X(trgt).controlled_by(ctl0,ctl1)

        return (basis_change+[cnX]+basis_change[::-1])

#implemented & checked
def walkDown(bTree, qubits, ancilla):
    def walkDown_helper(myc,bt,qbt,anc):
        if len(bt)==0:
            return myc
        else:
            myc.append(toffoli(bt[0],True, qbt[0], anc[0], anc[1]))
            myc = walkDown_helper(myc, bt[1::], qbt[1::], anc[1::])
            return myc
        
    circuit = cirq.Circuit()
    circuit.append(toffoli(bTree[0],bTree[1], qubits[0], qubits[1], ancilla[0]))
    circuit = walkDown_helper(circuit, bTree[2::], qubits[2::], ancilla)
    
    return circuit

#implemented
#given a binary, insert the gates required to move from one node to a node on the right
def stepRight(circuit, bTree, qubits, ancilla):
    def myTof(c1,c2,trgt):
        return toffoli(True,False,c1,c2,trgt)

    def nth2last(n):
        return -(n+1)
    def h_distance(val1,val2):
        return sum([0 if v1==v2 else 1 for v1,v2 in zip(val1,val2)])
    #particular tree traversal rules
    distance = h_distance(bTree,incrementBooleanTree(bTree))
    n_anc = len(ancilla)
    if distance == 0:
        pass #no need to do anything.
    elif distance == 1:
        if n_anc == 1:
            q0 = ancilla[nth2last(0)]
            circuit.append(cirq.X.on(q0))
        else:
            q0 = ancilla[nth2last(0)]
            q1 = ancilla[nth2last(1)]
            circuit.append(cirq.CX.on(q1,q0))
    elif distance == 2:
        if n_anc == 2:
            q0 = ancilla[nth2last(0)]
            q1 = ancilla[nth2last(1)]
            x0 = qubits[nth2last(0)]
            circuit.append(cirq.CX.on(q1,q0))
            circuit.append(cirq.CX.on(x0,q0))
            circuit.append(cirq.X.on(q1))
            circuit.append(cirq.X.on(q0))
        else:
            q0 = ancilla[nth2last(0)]
            q1 = ancilla[nth2last(1)]
            q2 = ancilla[nth2last(2)]
            x0 = qubits[nth2last(0)]
            circuit.append(cirq.CX.on(q1,q0))
            circuit.append(myTof(q2,x0,q0))
            circuit.append(cirq.CX.on(q2,q1))
    else:
        #recurse
        x0 = qubits[nth2last(0)]
        q0 = ancilla[nth2last(0)]
        q1 = ancilla[nth2last(1)]

        qubits2 = qubits[0:-1]
        ancilla2 = ancilla[0:-1]
        m_bTree = bTree[0:-1]
        circuit.append(toffoli(True, True, q1, x0, q0))
        circuit = stepRight(circuit, m_bTree, qubits2, ancilla2)
        circuit.append(toffoli(True, False, q1, x0, q0))
    
    return circuit

#implemented
def applyAndStep(operators, start_bTree, end_bTree, controls, ancilla, target):
    def applyAndStep_helper(myc, ops, sbT, ebT, qbs, anc, tgt):
        if sbT == ebT:
            myc.append(ops[0](anc[-1],tgt))
            return myc
        else:
            myc.append(ops[0](anc[-1],tgt))
            myc = stepRight(myc, sbT.copy(), qbs, anc)
            return applyAndStep_helper(myc, ops[1::], incrementBooleanTree(sbT),\
                            ebT, qbs, anc, tgt)

    circuit = cirq.Circuit()
    return applyAndStep_helper(circuit, operators, start_bTree, end_bTree,\
                     controls, ancilla, target)



#these two methods should just get wrapped into a class.
def constructBooleanTree(n,m):
    n = bin(n)[2:]
    value = ''.join([''.join(['0' for __ in range(m-len(n))]),n])
    value = [True if v=='1' else False for v in value ]
    return value

def incrementBooleanTree(n):
    def incBT(m):
        h = m[0]
        m = m[1::]
        if h == False:
            return [True] + m
        else:
            return [False] + incBT(m)
    n = n.copy()
    n.reverse()
    n = incBT(n)
    n.reverse()
    return n



class SelVBase(cirq.Gate):
    #do we need knowledge of the hamiltonian here?
    def __init__(self,invert, hamiltonian,phase_qubit, target_qubits, control_qubits, ancilla_qubits,\
                    n_controls=2):
        self.__phs_q = phase_qubit
        self.__ctl_q = control_qubits
        self.__tgt_q = target_qubits
        self.__anc_q = ancilla_qubits
        self.__hamiltonian = hamiltonian

        super(SelVBase, self)

    def _num_qubits_(self):
        return len(self.__tgt_q)+len(self.__ctl_q)+1+len(self.__anc_q)

    def convert_hamiltonian_terms_to_operators(self):
        def controlled_pauli_gate(pauliString, ctrl, targets):
            gate_sequence = []
            for p,tq in zip(pauliString,targets):
                if p == "I":
                    continue
                elif p == "X":
                    gate_sequence.append(cirq.CX.on(ctrl,tq))
                elif p == "Y":
                    gate_sequence.append(cirq.inverse(cirq.S.on(tq)))
                    gate_sequence.append(cirq.CX.on(ctrl,tq))
                    gate_sequence.append(cirq.S.on(tq))
                elif p == "Z":
                    gate_sequence.append(cirq.CZ.on(ctrl,tq))

            return gate_sequence

        operator_generators = []
        for term in self.__hamiltonian.terms:
            if (term[1] < 0):
                add_z = lambda ctrl, qubits : [cirq.Z.on(ctrl)]
                gate_seq = lambda ctrl, qubits, pauli=term[0]: add_z(ctrl, qubits)+controlled_pauli_gate(pauli,ctrl,qubits)
                operator_generators.append(gate_seq)
            else:
                gate_seq = lambda ctrl, qubits, pauli=term[0]: controlled_pauli_gate(pauli,ctrl,qubits)
                operator_generators.append(gate_seq)
        return operator_generators

    def _decompose_(self, qubits):
        #this is selectV!
        #helper functions~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        #identify pos1 and pos2
        #need to map hamiltonian terms to operators
        ham_ps_is_identity = [(np.all([p=="I" for p in term[0]]) or term[1]==0) for term in self.__hamiltonian.terms]
        def find_first_non_identity_term(check):
            for idx,k in enumerate(check):
                if not k:
                    return idx    
        pos1 = find_first_non_identity_term(ham_ps_is_identity)
        pos2 = len(self.__hamiltonian) - find_first_non_identity_term(ham_ps_is_identity[::-1])
        #this travels a tree and implements the selectV as appropriate
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        ham_as_ops = self.convert_hamiltonian_terms_to_operators()[pos1::]
        start_node = [True]+constructBooleanTree(pos1,len(self.__ctl_q))
        end_node = [True]+constructBooleanTree(pos2-1,len(self.__ctl_q))

        downCircuit = walkDown(start_node, [self.__phs_q]+self.__ctl_q, self.__anc_q)
        applyCircuit = applyAndStep(ham_as_ops, start_node, end_node, \
                    self.__ctl_q, [self.__phs_q]+self.__anc_q, self.__tgt_q)

        upCircuit =  cirq.inverse(walkDown(end_node, [self.__phs_q]+self.__ctl_q, self.__anc_q))
        
        yield downCircuit
        yield applyCircuit
        yield upCircuit

    def _circuit_diagram_info_(self, args):
        return ["SelVBase"] * self.num_qubits()
