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
### This file contains functions to help generate classical shadow circuits in cirq ###

### General Imports ###
import cirq
import random
import numpy as np
import numpy.linalg      as la
import pyLIQTR.sim_methods.quantum_ops  as  qops 

from pyLIQTR.QSP          import gen_qsp   as   qspFuncs
from pyLIQTR.QSP          import QSP       as   pQSP

from pyLIQTR.QSP.Hamiltonian import Hamiltonian as pyH

# Useful gates
S_dagger = cirq.inverse(cirq.S)
had = cirq.H


################################################################################
###
###   FUNCTION     random_unitary_circ(qubits, e)
###
################################################################################
###
###   DESCRIPTION
###
###      Generates a random unitary circuit. Over many trials, these random 
###      unitaries approximate a unitary 2-design. See Dankert et al.
###
###   ARGUMENTS
###
###      qubits      =  cirq qubit objects which are used in the circuit
###      e           =  error factor
###
###   RETURNS
###
###      C           =  A unitary cirq circuit
###                 
###
###
###   REQUIRES
###
###      numpy as np
###      qantum_ops as qops
###
### HELPERS:
def random_XOR(qubits):
    gates = []
    cnot_qubits = [qubits[0]]

    for q in qubits[1:]:
        if not int(random.randint(1,4)/4):
            cnot_qubits.append(q)

    gates_rev = []
    
    for d in range(1, np.ceil(np.log2(len(cnot_qubits))).astype(int)+1):
        for i in range(0, len(cnot_qubits), 2**d):
            s = 2**(d-1)
            if i+s < len(cnot_qubits):
                gate = cirq.CNOT(cnot_qubits[i+s], cnot_qubits[i])
                gates.append(gate)
                if i != 0:
                    gates_rev = [gate] + gates_rev
    
    gates = gates + gates_rev
    return gates

def c1p1(qubits):
    gates = []
    for q in qubits:
        r = random.randint(0,2)
        if r == 1:
            gates.append([cirq.H.on(q), cirq.S.on(q)])
        elif r == 2:
            gates.append([cirq.H.on(q), cirq.S.on(q), cirq.H.on(q), cirq.S.on(q)])
    return gates

def random_unitary_circ(e, qubits):
    C = cirq.Circuit()
    for i in range(int(np.log10(1/e))):
        C.append(c1p1(qubits))

        C.append(random_XOR(qubits))

        C.append(cirq.H.on(qubits[0]))
        C.append(c1p1(qubits[1:]))

        C.append(random_XOR(qubits))

        C.append(cirq.H.on(qubits[0]))
        C.append(c1p1(qubits[1:]))

        if random.randint(0,1):
            C.append(cirq.S.on(qubits[0]))

        C.append(random_XOR(qubits))

        C.append(c1p1([qubits[0]]))
    return C

################################################################################
###
###   FUNCTION     random_pBasis(n)
###
################################################################################
###
###   DESCRIPTION
###
###      Returns an circuit which projects a state into a random Pauli basis,  
###      i.e. each qubit is randomly projected into the X, Y, or Z basis
###
###   ARGUMENTS
###
###      qubits      =  cirq qubit objects which are used in the circuit
###
###   RETURNS
###
###      C           =  A cirq circuit 
###
###   REQUIRES
###
###      random
###

def random_pauli_circ(qubits):
    C = cirq.Circuit()
    for q in qubits:
        r = random.randint(0,2)
        if r == 0:
            C.append(had.on(q))
        elif r == 1:
            C.append(S_dagger.on(q))
            C.append(had.on(q))
    return C

################################################################################
###
###   FUNCTION     nisq_decomp(circuit)
###
################################################################################
###
###   DESCRIPTION
###
###      Returns an circuit which decomposes all toffolis/three-qubit gates into
###      single and two-qubit gates via cirq.decompose()
###
###   ARGUMENTS
###
###     circuit      =  cirq circuit to be decomposed
###
###   RETURNS
###
###     C            =  A cirq circuit which has been decomposed
###
###   REQUIRES
###
###      random
###

def nisq_decomp(circuit):
    ops = []
    for moment in circuit:
        for op in moment:
            gate = str(op.gate).strip()
            ccx = False
            ccz = False
            cx = False
            for g in ["CCX", "ccx", "TOFFOLI", "toffoli"]:
                if gate.startswith(g):
                    ccx = True
            for g in ["CCZ", "ccz"]:
                if gate.startswith(g):
                    ccz = True
            for g in ["CX", "cx", "CNOT", "cnot"]:
                cx = True
            if ccx or ccz:
                ops.append(cirq.decompose(op))
            elif cx:
                ops.append(cirq.decompose(op))
            else:
                ops.append(op)
    return cirq.Circuit(ops)

def prep_qsp(type, n, beta):
    N = n
    if type == 'XXZ':
        J_z  =    1.0;

        J_x  =    1.1*J_z;     J_y  =   J_x;               
        h_x  =   -1.0*J_z;     h_y  =   0.0*J_z;     h_z  =   0.0*J_z;

        # Generate input parameter sets
        tuples,types,coeffs = qops.params_heisenberg_1d(N,
                                                        J_x=J_x,J_y=J_y,J_z=J_z,
                                                        h_x=h_x,h_y=h_y,h_z=h_z,
                                                        periodic=False)

        # Generate Hamiltonian Matrix
        H = qops.ham_from_pauli_tuples(tuples,types,coeffs,N)

        # Pre processing for QSP requires Min-Max Rescaling -- HOW do we incorporate this with our tuples and such?
        # Is it better to use other rescaling?

        evals,evecs = la.eig(H)


        e_min = np.min(evals)
        e_max = np.max(evals)

        e_diff = e_max - e_min

        ham_strings = qops.ps_text_full_set(tuples,types,N,Coeffs=coeffs / e_diff)
        ham_strings

        tau = beta*(e_diff).real/2.0
        qsp_H  = pyH(ham_strings)

        required_precision = 1e-6

        #Old, deprecated call:
        #angles, tolerances = qspFuncs.get_phis(qsp_H, simtime=dt*sclf, 
        #                                       req_prec=required_precision);
        angles, tolerances = qspFuncs.compute_hamiltonian_angles(qsp_H,simtime=tau, req_prec=required_precision,\
                                                mode="random")
        
        # Initialize circuit to the correct size
        qsp_generator = pQSP.QSP(phis=angles, hamiltonian=qsp_H, 
                         target_size=qsp_H.problem_size)
        
        return qsp_generator.circuit()

