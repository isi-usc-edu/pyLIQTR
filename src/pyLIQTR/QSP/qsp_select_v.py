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

def convert_hamiltonian_terms_to_operators(ham_terms):
    def apply_gate(p, ctrl, tq):
        gate_list = []
        if p == "I":
            pass
        elif p == "X":
            gate_list.append(cirq.CX.on(ctrl,tq))
        elif p == "Y":
            gate_list.append(cirq.inverse(cirq.S.on(tq)))
            gate_list.append(cirq.CX.on(ctrl,tq))
            gate_list.append(cirq.S.on(tq))
        elif p == "Z":
            gate_list.append(cirq.CZ.on(ctrl,tq))

        return gate_list

    # pauliString can be a string or a list of tuples of the form: [(P,0), ...]
    #
    def controlled_pauli_gate(pauliString, ctrl, targets):
        gate_sequence = []
        if type(pauliString) is list:
            for pv in pauliString:
                p = pv[0]
                tq = targets[pv[1]]
                gate_sequence.append(apply_gate(p, ctrl, tq))
        else:
            for p,tq in zip(pauliString,targets):
                gate_sequence.append(apply_gate(p, ctrl, tq))

        return gate_sequence

    operator_generators = []
    for term in ham_terms:
        if (term[1] < 0):
            add_z = lambda ctrl, qubits : [cirq.Z.on(ctrl)]
            gate_seq = lambda ctrl, qubits, pauli=term[0]: add_z(ctrl, qubits)+controlled_pauli_gate(pauli,ctrl,qubits)
            operator_generators.append(gate_seq)
        else:
            gate_seq = lambda ctrl, qubits, pauli=term[0]: controlled_pauli_gate(pauli,ctrl,qubits)
            operator_generators.append(gate_seq)
    return operator_generators

# apply operators from Hamiltonian controlled on address bits using a QROM
#
def applyAndWalk(ham_terms, pos1, pos2, ctl_q, sel_q, anc_q, tgt_q, opType='select'):

    # Generate a circuit that implements a QROM (based on construction in Fig. 2 of arXiv:1905.07682)
    # This circuit accepts a dictionary with (address,data) pairs for each uniquue entry
    # The bit-width of any address must be less than the specified bit-width, however,
    # all addresses do not need to be specified
    # The QROM works by iterating over the addresses in numeric order
    # A set of ancilla bits (b) are used to build up the current selected address from the address bits (a)
    # Each subsequent bit adds a bit of the selected address to the ancilla register
    # For example, if we have a 3-bit address then: (where a_i corresponds to the current value of the address)
    # b_2 = a_2
    # b_1 = a_2 ^ a_1
    # b_0 = a_2 ^ a_1 ^ a_0
    #
    # To set the current value, from a cleared ancilla register, we build it up one bit at a time
    # with a CCX gate per bit (the control is either zero or one controlled based on the value of the address bit)
    #
    # To move from one current address value to another there are three main operations:
    #      1) flipping the bit value of the current address (this is done as CX: b_i+1 -> b_i)
    #      2) clearing the current address (this is the inverse process of setting the value)
    #      3) setting a new value (this is the same as above)
    # This process of moving from one address to the next consists of three steps:
    #      1) Clearing the current address bits differ in only one position
    #      2) Flipping the bit that differs
    #      3) Setting the bits that were cleared to the new value
    #
    # Parameters:
    # qubit_dict - contains starting location of qubits
    # width - number of address bit in QROM
    # key_values - list of address values
    # operators - operators for each key
    #
    def qrom(qubit_dict, width, key_vals, operators, opType):

        circuit = cirq.Circuit()
        circuit2 = cirq.Circuit()
        
        prev_key_val = -1
        addrQubits = qubit_dict['a'][::-1]   # address bits
        ancQubits = qubit_dict['b'][::-1]   # ancilla that holds current address
        selQubit = qubit_dict['s']   # global select
        trgtQubits = qubit_dict['t']   # target for memory

        #print("num keys = {}, len(addr) = {}, len(anc) = {}, len(trgt) = {}".format(len(key_vals), len(addrQubits), len(ancQubits), len(trgtQubits)))
    
        for kidx,key_val in enumerate(key_vals):
            if prev_key_val == -1:
                # special case for the first value, build up the initial set of ancilla
                for bi in range(width)[::-1]:
                    ctlSense = (key_val >> bi) % 2 == 1

                    # Extend the address starting from the MSB, the first gate is controlled on the global select
                    if bi+1 == width:
                        if selQubit==None:
                            if not ctlSense:
                                circuit.append(cirq.X.on(addrQubits[bi]))
                            circuit.append(cirq.CX.on(addrQubits[bi], ancQubits[bi]))
                            if not ctlSense:
                                circuit.append(cirq.X.on(addrQubits[bi]))
                        else:
                            circuit.append(toffoli(True, ctlSense, selQubit , addrQubits[bi], ancQubits[bi]))
                    else:
                        circuit.append(toffoli(True, ctlSense, ancQubits[bi + 1] , addrQubits[bi], ancQubits[bi]))
                    
            else:
                # move from previous key value to the new key value
                ci = 0

                # clear the current address until we get to a state where the bit values are equal
                while not (prev_key_val >> (ci+1)) == (key_val >> (ci+1)):
                    ctlSense = (prev_key_val >> ci) % 2 == 1
                    circuit.append(toffoli(True, ctlSense, ancQubits[ci + 1], addrQubits[ci], ancQubits[ci]))

                    ci += 1

                # flip the address of the least signfificant bit that is different
                if (ci + 1) == width:
                    if selQubit==None:
                        circuit.append(cirq.X.on(ancQubits[ci]))
                    else:
                        circuit.append(cirq.CX.on(selQubit, ancQubits[ci]))
                else:
                    circuit.append(cirq.CX.on(ancQubits[ci + 1], ancQubits[ci]))

                # work our way down to the new value
                while ci > 0:
                    ctlSense = (key_val >> (ci-1)) % 2 == 1
                    circuit.append(toffoli(True, ctlSense, ancQubits[ci], addrQubits[ci - 1], ancQubits[ci - 1]))
                    
                    ci -= 1

            # apply the operator
            if opType=='select':
                circuit.append(operators[key_val](ancQubits[0], trgtQubits))
            elif opType=='fermion':
                circuit.append(operators[key_val](ancQubits[0], trgtQubits))
                if (kidx+1) != len(key_vals):
                    circuit.append(cirq.CX.on(ancQubits[0], selQubit))
                    assert(key_val < len(trgtQubits))
                    circuit.append(cirq.CZ.on(selQubit, trgtQubits[key_val]))
                        
            prev_key_val = key_val

        # clear the ancilla register
        if not prev_key_val == -1:
            for ci in range(width):
                ctlSense = (prev_key_val >> ci) % 2 == 1

                if (ci + 1) == width:
                    if selQubit==None:
                        if not ctlSense:
                            circuit.append(cirq.X.on(addrQubits[ci]))
                        circuit.append(cirq.CX.on(addrQubits[ci], ancQubits[ci]))
                        if not ctlSense:
                            circuit.append(cirq.X.on(addrQubits[ci]))
                    else:
                        circuit.append(toffoli(True, ctlSense, selQubit, addrQubits[ci], ancQubits[ci]))
                else:
                    circuit.append(toffoli(True, ctlSense, ancQubits[ci + 1], addrQubits[ci], ancQubits[ci]))

        return circuit

    qubit_dict = dict()
    qubit_dict['a'] = ctl_q
    qubit_dict['b'] = anc_q
    qubit_dict['s'] = sel_q
    qubit_dict['t'] = tgt_q
    
    key_values = [ii+pos1 for ii in range(pos2-pos1)]
    operators = convert_hamiltonian_terms_to_operators(ham_terms)
    
    return qrom(qubit_dict, len(ctl_q), key_values, operators, opType)

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
        if self.__phs_q == None:
            return len(self.__tgt_q)+len(self.__ctl_q)+len(self.__anc_q)
        else:
            return len(self.__tgt_q)+len(self.__ctl_q)+1+len(self.__anc_q)

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

        # this is the new way to do SelectV
        #ham_as_ops = convert_hamiltonian_terms_to_operators(self.__hamiltonian.terms)
        circuit = applyAndWalk(self.__hamiltonian.terms, pos1, pos2, self.__ctl_q, self.__phs_q, self.__anc_q, self.__tgt_q)
        yield circuit

    def _circuit_diagram_info_(self, args):
        return ["SelVBase"] * self.num_qubits()

class UnaryIterate(cirq.Gate):
    def __init__(self, ham_terms, phase_qubit, target_qubits, control_qubits, ancilla_qubits, op_type):
        self.__ham = ham_terms
        self.__phs_q = phase_qubit
        self.__ctl_q = control_qubits
        self.__tgt_q = target_qubits
        self.__anc_q = ancilla_qubits
        self.__op_type = op_type

        super(UnaryIterate)

    def _num_qubits_(self):
        return len(self.__tgt_q)+len(self.__ctl_q)+1+len(self.__anc_q)

    def _decompose_(self, qubits):
        yield applyAndWalk(self.__ham, 0, len(self.__ham), self.__ctl_q, self.__phs_q, self.__anc_q, self.__tgt_q, opType=self.__op_type)
        
    def _circuit_diagram_info_(self, args):
        return ["UnaryIterate"] * self.num_qubits()

class FermionSelect(cirq.Gate):

    def __init__(self, hamiltonian, phase_qubit, target_qubits, control_qubits, ancilla_qubits):
        #from pyLIQTR.QSP.Hamiltonian import Hamiltonian as pyH
        
        self.__phs_q = phase_qubit
        self.__ctl_q = control_qubits
        self.__tgt_q = target_qubits
        self.__anc_q = ancilla_qubits
        self.__hamiltonian = hamiltonian

        print("len(ctl) = {}, len(tgt) = {}, len(anc) = {}".format(len(self.__ctl_q), len(self.__tgt_q), len(self.__anc_q)))
        
        # the Hamiltonian contains three components:
        # U(p), T(p-q), V(p-q)
        # Each component contains a number of terms: [(p,alpha,q,beta), amp]
        # The number of orbitals + spin is specified as N
        N = self.__hamiltonian.problem_size
        Nu = N  # number of U terms in the Hamiltonian (p,sigma,p,sigma)
        Nt = 2*((N//2)**2 - N//2)   # number of Utterms in the Hamiltonian (p,sigma,q,sigma) (p != q)
        #Nt = 2*Nt    # for now the prep does all combination of alpha,beta
        Nv = (N**2) - N  # number of V terms in the Hamiltonian (p,alpha) != (q,beta)
        #print("N = {}, Nu = {}, Nt = {}, Nv = {}".format(N, Nu, Nt, Nv))
        
        # separate the coefficients for each term
        coefU = [self.__hamiltonian.terms[ii][1] for ii in range(0,N)]
        coefT = [self.__hamiltonian.terms[ii][1] for ii in range(N,2*N)]
        coefV = [self.__hamiltonian.terms[ii][1] for ii in range(2*N,4*N)]

        # Create the three select iterators used
        fermY = [([("Y",ni)], 1.0) for ni in range(N)]   # Fermionic YpZp-1...Z0
        self.__selFermY = fermY
        fermX = [([("X",ni)], 1.0) for ni in range(N)]   # Fermionic XpZp-1...Z0
        self.__selFermX = fermX
        fermZ = [([("Z",ni)], 1.0) for ni in range(N)]   # Zp
        self.__selAplyZ = fermZ

        self.__N = N
        self.__np = int(np.ceil(np.log2(N)))
        
        super(FermionSelect, self)
        
    def _num_qubits_(self):
        return len(self.__tgt_q)+len(self.__ctl_q)+1+len(self.__anc_q)

    def _decompose_(self, qubits):
        from pyLIQTR.QSP.qsp_cirq_gates import MultiFredkin
        
        np = self.__np
        
        # Iterate on p
        #ham_as_ops = convert_hamiltonian_terms_to_operators(self.__selFermY)
        yield UnaryIterate(self.__selFermY, self.__phs_q, self.__tgt_q, self.__ctl_q[:np], self.__anc_q[:np], 'fermion').\
            on(*([self.__phs_q]+self.__tgt_q+self.__ctl_q[:np]+self.__anc_q[:np]))

        # Controlled swap p,alpha with q,beta
        yield MultiFredkin(self.__ctl_q[2*np+1], self.__ctl_q[:np], self.__ctl_q[np:2*np]).\
          on(*([self.__ctl_q[2*np+1]] + self.__ctl_q[:np] + self.__ctl_q[np:2*np]))
          
        # Iterate on q
        ham_as_ops = convert_hamiltonian_terms_to_operators(self.__selFermX)
        yield UnaryIterate(self.__selFermX, self.__phs_q, self.__tgt_q, self.__ctl_q[np:2*np], self.__anc_q[:np], 'fermion').\
            on(*([self.__phs_q]+self.__tgt_q+self.__ctl_q[np:2*np]+self.__anc_q[:np]))
            
        # Controlled swap p,alpha with q,beta
        yield MultiFredkin(self.__ctl_q[2*np+1], self.__ctl_q[:np], self.__ctl_q[np:2*np]).\
          on(*([self.__ctl_q[2*np+1]] + self.__ctl_q[:np] + self.__ctl_q[np:2*np]))

        yield cirq.CCX.on(self.__phs_q, self.__ctl_q[2*np+1], self.__anc_q[np])
        
        # Iterate on q
        ham_as_ops = convert_hamiltonian_terms_to_operators(self.__selAplyZ)
        yield UnaryIterate(self.__selAplyZ, self.__anc_q[np], self.__tgt_q, self.__ctl_q[np:2*np], self.__anc_q[:np], 'select').\
            on(*([self.__anc_q[np]]+self.__tgt_q+self.__ctl_q[np:2*np]+self.__anc_q[:np]))

        yield cirq.CCX.on(self.__phs_q, self.__ctl_q[2*np+1], self.__anc_q[np])

    def _circuit_diagram_info_(self, args):
        return ["FermionSelect"] * self.num_qubits()

def SelectOracle(hamiltonian, phase_qubit, target_qubits, control_qubits, ancilla_qubits):
    if hamiltonian.is_fermionic:
        # this is a fermionic Hamiltonian
        yield FermionSelect(hamiltonian, phase_qubit, target_qubits, control_qubits, ancilla_qubits).\
              on(*([phase_qubit] + target_qubits + control_qubits + ancilla_qubits))
    else:
        yield SelVBase(False, hamiltonian, phase_qubit, target_qubits, control_qubits, ancilla_qubits).\
              on(*([phase_qubit] + target_qubits + control_qubits + ancilla_qubits))
    
