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
import sys
from urllib.parse import quote_from_bytes

import cirq
from pyLIQTR.circuits.pyLCircuit import pyLCircuit
from pyLIQTR.circuits.operators.prepare import *
from pyLIQTR.circuits.operators.selectV import *
from pyLIQTR.circuits.operators.hamiltonian_encodings import *
from pyLIQTR.circuits.operators.reflect import *


#hamiltonian, angles, qubits
def generate_qubits_from_size(hamiltonian, target_size):
    trgt_q = cirq.LineQubit.range(target_size)

    if hamiltonian.is_lcu:
        num_ctl = hamiltonian.loglen
    elif hamiltonian.is_fermionic:
        # N is the total system size (functions + spins)
        # the control contains (p,alpha), (q,beta), and (U,V,theta)
        N = hamiltonian.problem_size
        nval = int(np.ceil(np.log2(N)))
        num_ctl = 2*nval + 3
    ctl_q = cirq.NamedQubit.range(len(trgt_q)+1,\
                    len(trgt_q)+1+num_ctl,
                    prefix="ctl_q")
    phase_q =  cirq.NamedQubit.range(len(trgt_q),\
                    len(trgt_q)+1, prefix="phs_q")
    return trgt_q, ctl_q,phase_q

def generate_QSP_circuit(hamiltonian,
                        angles,
                        target_size):
    """
    Returns a pyLIQTR Circuit object
    """
    sys.setrecursionlimit(2**16)
    if hamiltonian.is_lcu:
        hamiltonian.adjust_hamiltonian()

    trgt_q, ctl_q, phase_q = generate_qubits_from_size(hamiltonian, target_size)
    total_qubits=len(trgt_q)+len(ctl_q)+len(phase_q)
    ancillaQ = cirq.NamedQubit.range(total_qubits,\
        total_qubits+len(ctl_q),\
            prefix="z_anc_q")

    circuit = []#pyLCircuit()
    #prepare circuit only acts on |1> state
    prepare_prepare_gates = [cirq.X.on(q) for q in ctl_q]
    n_ = len(ctl_q)+1
    """
    selVO = ChildsSelectV(hamiltonian = hamiltonian,
                            phase_qubit = phase_q,
                            target_qubits = trgt_q,
                            control_qubits = ctl_q,
                            ancilla_qubits = ancillaQ)
    """
    selVO = QROMSelectV(hamiltonian = hamiltonian,
                            phase_qubit = phase_q,
                            target_qubits = trgt_q,
                            control_qubits = ctl_q,
                            ancilla_qubits = ancillaQ)
    prepO = Prepare(qubit_reg = ctl_q, alphas = hamiltonian.alphas)
    reflO = lambda angle : \
        Reflect(angle, *phase_q, ctl_q, ancillaQ[:(n_-3)]).\
            on(*([*phase_q] + ctl_q + ancillaQ[:(n_-3)]))
    
    operator = UnitaryBlockEncode(hamiltonian=hamiltonian,
                        selectOracle=selVO, prepareOracle=prepO,
                        phase_qubit = phase_q,
                        target_qubits = trgt_q,
                        control_qubits = ctl_q,
                        ancilla_qubits = ancillaQ).\
                            on(*trgt_q,*ctl_q,*phase_q,*ancillaQ)
    inv_operator = cirq.inverse(operator)
    
    phi0 = angles[0]
    angles = angles[1:]
    phiLen = int(len(angles)/2)
    phiLo = angles[0:phiLen-1]
    phiMid = angles[phiLen-1]
    phiHi = angles[phiLen:-1]
    phiN = angles[-1]

    circuit.append(prepare_prepare_gates)
    circuit.append(cirq.Rx(rads=-phi0/2).on(*phase_q))
    for idx in range(0,len(phiLo)-1,2):
        circuit.append(operator)
        circuit.append(cirq.Ry(rads=2*phiLo[idx]).on(*phase_q))
        circuit.append(inv_operator)
        circuit.append(reflO(phiLo[idx+1]))

    circuit.append(operator)
    circuit.append(cirq.Ry(rads=2*phiLo[-1]).on(*phase_q))
    circuit.append(inv_operator)

    circuit.append(reflO(phiMid))

    circuit.append(operator)
    circuit.append(cirq.Ry(rads=2*phiHi[0]).on(*phase_q))
    circuit.append(inv_operator)

    for idx in range(1,len(phiHi),2):
        circuit.append(reflO(phiHi[idx]))
        circuit.append(operator)
        circuit.append(cirq.Ry(rads=2*phiHi[idx+1]).on(*phase_q))
        circuit.append(inv_operator)

    circuit.append(cirq.Rx(rads=-phiN/2).on(*phase_q))
    circuit.append(prepare_prepare_gates)
    return pyLCircuit(circuit)
