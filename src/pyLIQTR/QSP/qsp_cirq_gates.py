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
from typing import ( Optional, Tuple )
from pyLIQTR.QSP.qsp_helpers  import getLineQubitIndexMap, getQubitFromMap
from pyLIQTR.QSP.qsp_select_v import SelVBase, SelectOracle
from pyLIQTR.QSP.qsp_prepare  import QSP_Prepare, PrepareOracle

        
class SelectV(cirq.Gate):
    def __init__(self,hamiltonian,phi,phase_qubit,\
            target_qubits, control_qubits, ancilla_qubits):
        self.__hamiltonian = hamiltonian
        self.__angle = phi #this is in radians already
        self.__phs_q = phase_qubit
        self.__tgt_q = target_qubits
        self.__ctl_q = control_qubits
        self.__anc_q = ancilla_qubits
        super(SelectV, self)

    def _num_qubits_(self):
        return len(self.__tgt_q)+len(self.__ctl_q)+1+len(self.__anc_q)

    def _decompose_(self, qubits):
        rotation = self.__angle
        
        yield QSP_Prepare(qubit_reg = self.__ctl_q, \
                            alphas = self.__hamiltonian.alphas, 
                            pp_exp=cirq.Y, pp_ctl = cirq.X, pp_com = cirq.I).\
                                on(*self.__ctl_q)
        #RESET ANCILLA?
        for q in self.__anc_q:
            yield cirq.ResetChannel().on(q)
        yield SelVBase(False, self.__hamiltonian, self.__phs_q, self.__tgt_q, self.__ctl_q, self.__anc_q).\
                    on(*([self.__phs_q] + self.__tgt_q + self.__ctl_q + self.__anc_q))
        yield cirq.Ry(rads = rotation).on(self.__phs_q)
        yield cirq.inverse(SelVBase(False, self.__hamiltonian, self.__phs_q, self.__tgt_q, self.__ctl_q, self.__anc_q).\
                    on(*([self.__phs_q] + self.__tgt_q + self.__ctl_q + self.__anc_q)))
        for q in self.__anc_q:
            yield cirq.ResetChannel().on(q)
        #RESET ANCILLA?            
        yield cirq.inverse(QSP_Prepare(qubit_reg = self.__ctl_q, \
                            alphas = self.__hamiltonian.alphas, 
                            pp_exp=cirq.Y, pp_ctl = cirq.X, pp_com = cirq.I).\
                                on(*self.__ctl_q))

    def _circuit_diagram_info_(self, args):
        return ["SelectV".format(self.__angle)] * self.num_qubits()

"""
Generate a Unitary Block encoding of the LCU Hamiltonian
This only includes a single Prepare+Select Oracle
"""
class UnitaryBlockEncode(cirq.Gate):
    def __init__(self, hamiltonian, phase_qubit, target_qubits, control_qubits, ancilla_qubits):
        self.__hamiltonian = hamiltonian
        self.__phs_q = phase_qubit    # this is the global control (None if there is none)
        self.__tgt_q = target_qubits
        self.__ctl_q = control_qubits
        self.__anc_q = ancilla_qubits
        super(UnitaryBlockEncode, self)

    def _num_qubits_(self):
        if self.__phs_q == None:
            return len(self.__tgt_q)+len(self.__ctl_q)+len(self.__anc_q)
        else:
            return len(self.__tgt_q)+len(self.__ctl_q)+1+len(self.__anc_q)

    def _decompose_(self, qubits):

        # prepare expects qubits in |1> state
        circuitX = cirq.Circuit()
        for qb in (self.__ctl_q):
            circuitX.append(cirq.X.on(qb))
        yield circuitX
        
        yield QSP_Prepare(qubit_reg = self.__ctl_q, \
                            alphas = self.__hamiltonian.alphas, 
                            pp_exp=cirq.Y, pp_ctl = cirq.X, pp_com = cirq.I).\
                                on(*self.__ctl_q)
        #RESET ANCILLA?
        for q in self.__anc_q:
            yield cirq.ResetChannel().on(q)
        if self.__phs_q == None:
            yield SelVBase(False, self.__hamiltonian, self.__phs_q, self.__tgt_q, self.__ctl_q, self.__anc_q).\
                           on(*(self.__tgt_q + self.__ctl_q + self.__anc_q))
        else:
            yield SelVBase(False, self.__hamiltonian, self.__phs_q, self.__tgt_q, self.__ctl_q, self.__anc_q).\
                           on(*([self.__phs_q] + self.__tgt_q + self.__ctl_q + self.__anc_q))
                           
            
        for q in self.__anc_q:
            yield cirq.ResetChannel().on(q)

        yield cirq.inverse(QSP_Prepare(qubit_reg = self.__ctl_q, \
                            alphas = self.__hamiltonian.alphas, 
                            pp_exp=cirq.Y, pp_ctl = cirq.X, pp_com = cirq.I).\
                                on(*self.__ctl_q))
                                
    def _circuit_diagram_info_(self, args):
        return ["UnitaryBE"] * self.num_qubits()

class SzegedyWalkOperator(cirq.Gate):
    def __init__(self, hamiltonian, phase_qubit, target_qubits, control_qubits, ancilla_qubits):
        self.__hamiltonian = hamiltonian
        self.__phs_q = phase_qubit    # this is the global control (None if there is none)
        self.__tgt_q = target_qubits
        self.__ctl_q = control_qubits
        self.__anc_q = ancilla_qubits
        super(SzegedyWalkOperator, self)

    def _num_qubits_(self):
        return len(self.__tgt_q)+len(self.__ctl_q)+1+len(self.__anc_q)

    def _decompose_(self, qubits):
        print("generating Walk operator, total_qubits = {}".format(self._num_qubits_()))
        
        yield SelectOracle(self.__hamiltonian, self.__phs_q, self.__tgt_q, self.__ctl_q, self.__anc_q)

        yield PrepareOracle(self.__hamiltonian, self.__ctl_q, self.__anc_q)

        # produce a zero-controlled multi-CZ gate
        circuitX = cirq.Circuit()
        for qb in (self.__ctl_q):
            circuitX.append(cirq.X.on(qb))
        yield circuitX
        
        n_anc = len(self.__ctl_q) - 2
        yield MultiCZ(self.__ctl_q, [self.__phs_q], self.__anc_q[:n_anc], clear_ancilla=True).on(*(self.__ctl_q + [self.__phs_q] + self.__anc_q[:n_anc]))

        yield circuitX
        # this should probably be inverted
        yield PrepareOracle(self.__hamiltonian, self.__ctl_q, self.__anc_q)
        
    def _circuit_diagram_info_(self, args):
        return ["WalkOP"] * self.num_qubits()
    

class MultiCZ(cirq.Gate):
    def __init__(self, control_qubits, target_qubits, ancilla_qubits, clear_ancilla=False):
        self.__ctl_q   = control_qubits
        self.__trgt_q  = target_qubits
        self.__anc_q   = ancilla_qubits
        self.__clear_ancilla = clear_ancilla
        super(MultiCZ,self)

    def _num_qubits_(self):
        return len(self.__ctl_q)+len(self.__trgt_q) +len(self.__anc_q)
        

    #TODO : make this work so that executes correctly even if the number of ancilla qubits
    #provided are more than this actually requires.
    def _decompose_(self, qubits): 
        # Note: there may be moment issues when constructing the circuit
        # this way, I fixed them downstream but maybe we want it different.

        # Lets start by getting these tuple-maps:
        ctl_list = getLineQubitIndexMap(self.__ctl_q,  'ctl')
        tgt_list = getLineQubitIndexMap(self.__trgt_q, 'tgt')
        anc_list = getLineQubitIndexMap(self.__anc_q,  'anc')
        # start this while loop to do the thing...
        if len(ctl_list) == 2:
            yield(cirq.CCZ(self.__ctl_q[0],self.__ctl_q[1],self.__trgt_q[0]))
        else:
            reverse_circ = cirq.Circuit()
            
            while len(ctl_list) > 2:
                # get two ctl qubits:
                ctl_1 = ctl_list[0]
                ctl_2 = ctl_list[1]
                ctl_list.remove(ctl_1)
                ctl_list.remove(ctl_2)

                # Get the target qubit:
                tgt_1 = anc_list[0]
                anc_list.remove(tgt_1)
                
                # add the most recent target to the end of the control list
                ctl_list.append(tgt_1)

                # Grab the appropriate qubits
                ctl_q_1 = getQubitFromMap(ctl_1, self.__ctl_q, self.__trgt_q, self.__anc_q)
                ctl_q_2 = getQubitFromMap(ctl_2, self.__ctl_q, self.__trgt_q, self.__anc_q)
                tgt_q_1 = getQubitFromMap(tgt_1, self.__ctl_q, self.__trgt_q, self.__anc_q)

                # trying to use my new class
                yield(cirq.CCX(ctl_q_1, ctl_q_2, tgt_q_1))
                if self.__clear_ancilla:
                    reverse_circ.append(cirq.CCX.on(ctl_q_1, ctl_q_2, tgt_q_1))
                    
                if len(ctl_list) == 2 and not anc_list:
                    ctl_q_1 = getQubitFromMap(ctl_list[0], self.__ctl_q, self.__trgt_q, self.__anc_q)
                    ctl_q_2 = getQubitFromMap(ctl_list[1], self.__ctl_q, self.__trgt_q, self.__anc_q)
                    tgt_q_1 = getQubitFromMap(tgt_list[0], self.__ctl_q, self.__trgt_q, self.__anc_q)
                    yield(cirq.CCZ(ctl_q_1, ctl_q_2, tgt_q_1))
                    
            yield cirq.inverse(reverse_circ)
            
    def _circuit_diagram_info_(self, args):
        return ["MultiCZ"] * self.num_qubits()

# This is almost a CCX gate, up to an additional CS applied to the coontrol qubits
# This can be used in cases where the CCX gate is paired with an inverse and the control
# qubits have no intervening gates applied to them
#
class CCXi(cirq.Gate):
    def __init__(self, c0, c1, t):
        self.__ctl_q0 = c0
        self.__ctl_q1 = c1
        self.__trg_q = t
        
        super(CCXi, self)

    def __str_(self) -> str:
        if self._exponent == 1:
            return 'ccxi {}, {}, {}'.format(self.__ctl_q0, self.__ctl_q1, self.__trg_q)
        else:
            return 'ccxi**{} {}, {}, {}'.format(self._exponent, self.__ctl_q0, self.__ctl_q1, self.__trg_q)
            
    def __repr_(self) -> str:
        if self._exponent == 1:
            return 'ccxi {}, {}, {}'.format(self.__ctl_q0, self.__ctl_q1, self.__trg_q)
        else:
            return 'ccxi**{} {}, {}, {}'.format(self._exponent, self.__ctl_q0, self.__ctl_q1, self.__trg_q)
    
    def _num_qubits_(self):
        return 3

    def _decompose_(self, qubits):
        c0, c1, t = qubits
        yield(cirq.CCX(self.__ctl_q0, self.__ctl_q1, self.__trg_q))

    def _circuit_diagram_info_(self, args):
        return ["CCXi"] * self.num_qubits()        

    def _qasm_(self, args: 'cirq.QasmArgs', qubits: Tuple['cirq.Qid', ...]) -> Optional[str]:
        args.validate_version('2.0')
        lines = [
            args.format('ccxi {0},{1},{2};\n', qubits[0], qubits[1], qubits[2]),
        ]
        return ''.join(lines)

class Reflect(cirq.Gate):
    def __init__(self, phi, phase_qubit, control_qubits, ancilla_qubits):
        self.__angle = phi #IN DEGREES
        self.__phs_q = phase_qubit
        self.__ctl_q = control_qubits
        self.__anc_q = ancilla_qubits
        super(Reflect, self)

    # What is this for?
    def _num_qubits_(self):
        return (len(self.__ctl_q) + len(self.__anc_q) + 1) 
        # return len(self.__ctl_q)+1


    def _decompose_(self, qubits):
        rotation = self.__angle
        #This is the easy guy.
        #we do use some ancilla...
        yield MultiCZ(control_qubits=self.__ctl_q, 
                      target_qubits=[self.__phs_q],
                      ancilla_qubits=self.__anc_q).\
                      on(*([self.__phs_q] + self.__ctl_q + self.__anc_q))

        yield cirq.Rx(rads = rotation).on(self.__phs_q)

        yield cirq.inverse(MultiCZ(control_qubits=self.__ctl_q, 
                      target_qubits=[self.__phs_q],
                      ancilla_qubits=self.__anc_q).\
                      on(*([self.__phs_q] + self.__ctl_q + self.__anc_q)))

    def _circuit_diagram_info_(self, args):
        return ["Reflect".format(self.__angle)] * self.num_qubits()

class MultiFredkin(cirq.Gate):
    def __init__(self, select_q, qubit_reg1, qubit_reg2):
        self.__sel_q = select_q
        self.__reg1_q = qubit_reg1
        self.__reg2_q = qubit_reg2

        super(MultiFredkin, self)
        
    def _num_qubits_(self):
        return (len(self.__reg1_q) + len(self.__reg2_q) + 1)

    def _decompose_(self, qubits):

        circuit = cirq.Circuit()
        for q1,q2 in zip(self.__reg1_q, self.__reg2_q):
            circuit.append(cirq.CX.on(q1,q2))
            circuit.append(cirq.CCX.on(self.__sel_q,q2,q1))
            circuit.append(cirq.CX.on(q1,q2))

        yield circuit
        
    def _circuit_diagram_info_(self, args):
        return ["Fredkin"] * self.num_qubits()
