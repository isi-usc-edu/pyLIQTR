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
import cirq
import numpy as np
import pyLIQTR.circuits.pyLOperator as pyLO
from typing import List,Optional,Tuple
from pyLIQTR.circuits.operators.selectVutils import *




class ChildsSelectV(pyLO.pyLOperator):
    def __init__(self,hamiltonian,
                    phase_qubit, 
                    target_qubits, 
                    control_qubits, 
                    ancilla_qubits):
        """
        Implements the SelectV operator from https://arxiv.org/pdf/1711.10980.pdf

        """
        if not isinstance(phase_qubit,list):
            phase_qubit = [phase_qubit]
        self.__phs_q = phase_qubit
        self.__ctl_q = control_qubits
        self.__tgt_q = target_qubits
        self.__anc_q = ancilla_qubits
        self.__hamiltonian = hamiltonian
        super(ChildsSelectV, self).__init__()

        self.total_decomp = 2
    
    def __str__(self) -> str:
        allQ = [*self.__phs_q, *self.__anc_q, *self.__ctl_q, *self.__tgt_q]
        qStr = ",".join([str(x) for x in allQ])
        return "ChildsSelectV {}".format(qStr)

    def _qasm_(self, args: 'cirq.QasmArgs', qubits: Tuple['cirq.Qid', ...]) -> Optional[str]:
        args.validate_version('2.0')
        allQ = [*self.__phs_q, *self.__anc_q, *self.__ctl_q, *self.__tgt_q]
        allQStr = ",".join([args.format(str(x)) for x in allQ])
        return f"ChildsSelectV({allQStr})\n"

    def _num_qubits_(self):
        """
        The ChildsSelectV acts on the target qubits, control qubits, the phase qubit, and any ancilla qubits
        """
        return len(self.__tgt_q)+len(self.__ctl_q)+1+len(self.__anc_q)
    
    def _circuit_diagram_info_(self, args) -> List[str]:
        return ["ChildsSelectV"] * self.num_qubits()

    def _decompose_(self, qubits):
        """
        Decomposes the ChildsSelectV (https://arxiv.org/pdf/1711.10980.pdf) into CX & Pauli gates
        """
        ham_ps_is_identity = [(np.all([p=="I" for p in term[0]]) or term[1]==0) for term in self.__hamiltonian.terms]
        def find_first_non_identity_term(check):
            for idx,k in enumerate(check):
                if not k:
                    return idx    
        pos1 = find_first_non_identity_term(ham_ps_is_identity)
        pos2 = len(self.__hamiltonian) - find_first_non_identity_term(ham_ps_is_identity[::-1])

        # this is the new way to do selectV
        ham_as_ops = convert_hamiltonian_terms_to_operators(self.__hamiltonian.terms)[pos1::]
        start_node = [True]+constructBooleanTree(pos1,len(self.__ctl_q))
        end_node = [True]+constructBooleanTree(pos2-1,len(self.__ctl_q))

        downCircuit = cirq.Circuit(walkDown(start_node, self.__phs_q+self.__ctl_q, self.__anc_q))
        applyCircuit = cirq.Circuit(applyAndStep(ham_as_ops, start_node, end_node, \
                                    self.__ctl_q, self.__phs_q+self.__anc_q, self.__tgt_q))

        upCircuit =  cirq.inverse(cirq.Circuit(walkDown(end_node, self.__phs_q+self.__ctl_q, self.__anc_q)))

        yield downCircuit
        yield applyCircuit
        yield upCircuit

    def __eq__(self, other):
        if self.__class__ != other.__class__:
            return False
        else:
            return ((self.__hamiltonian == other.__hamiltonian) and \
                (self.__phs_q == other.__phs_q) and \
                (self.__ctl_q == other.__ctl_q)and \
                (self.__tgt_q == other.__tgt_q) and \
                (self.__anc_q == other.__anc_q) )

    def _get_as_circuit(self):
        allQ = [*self.__phs_q, *self.__anc_q, *self.__ctl_q, *self.__tgt_q]
        return cirq.Circuit(self.on(*allQ))


class QROMSelectV(pyLO.pyLOperator):
    def __init__(self,hamiltonian,
                    phase_qubit, 
                    target_qubits, 
                    control_qubits, 
                    ancilla_qubits):
        """
        Implements the SelectV operator from ????

        """
        self.__phs_q = phase_qubit
        self.__ctl_q = control_qubits
        self.__tgt_q = target_qubits
        self.__anc_q = ancilla_qubits
        self.__hamiltonian = hamiltonian
        super(QROMSelectV, self).__init__()
        self.total_decomp = 2
        
    
    def _num_qubits_(self):
        """
        The QROMSelectV acts on the target qubits, control qubits, the phase qubit, and any ancilla qubits
        """
        return len(self.__tgt_q)+len(self.__ctl_q)+1+len(self.__anc_q)
    
    def _circuit_diagram_info_(self, args) -> List[str]:
        return ["SelectV"] * self.num_qubits()

    def _decompose_(self, qubits):
        """
        Decomposes the QROM from ???? into gates
        """
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


        ham_as_ops = convert_hamiltonian_terms_to_operators(self.__hamiltonian.terms)
        circuit = applyAndWalk(ham_as_ops, pos1, pos2, self.__ctl_q, *self.__phs_q, self.__anc_q, self.__tgt_q)
        yield circuit
    
    def __eq__(self, other):
        if self.__class__ != other.__class__:
            return False
        else:
            return ((self.__hamiltonian == other.__hamiltonian) and \
                (self.__phs_q == other.__phs_q) and \
                (self.__ctl_q == other.__ctl_q)and \
                (self.__tgt_q == other.__tgt_q) and \
                (self.__anc_q == other.__anc_q) )
    
    def __str__(self) -> str:
        allQ = [*self.__phs_q, *self.__anc_q, *self.__ctl_q, *self.__tgt_q]
        qStr = ",".join([str(x) for x in allQ])
        return "SelectV {}".format(qStr)

    def _qasm_(self, args: 'cirq.QasmArgs', qubits: Tuple['cirq.Qid', ...]) -> Optional[str]:
        args.validate_version('2.0')
        allQ = [*self.__phs_q, *self.__anc_q, *self.__ctl_q, *self.__tgt_q]
        allQStr = ",".join([args.format(str(x)) for x in allQ])
        return f"SelectV({allQStr})"

    def _get_as_circuit(self):
        allQ = [*self.__phs_q, *self.__anc_q, *self.__ctl_q, *self.__tgt_q]
        return cirq.Circuit(self.on(*allQ))
