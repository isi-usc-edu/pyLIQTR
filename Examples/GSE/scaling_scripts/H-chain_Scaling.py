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
import time
import os
import numpy as np

from openfermion.chem import MolecularData
from openfermionpyscf import run_pyscf

from pyLIQTR.GSE.GSE                     import GSE 
from pyLIQTR.QSP.qsp_helpers             import qsp_decompose_once
from pyLIQTR.gate_decomp.cirq_transforms import clifford_plus_t_direct_transform
from pyLIQTR.utils.utils                 import count_T_gates

###
### Define parameters
###
prec_     = 3
trot_ord_ = 1
trot_num_ = 4
E_max     = 2.0
E_min     = -2.0
chain_lens = [2,4,8,16,32,64,128,256]

def gen_H_chain(num_H, bond_len=0.7414):
    geometry = [
        ('H', (0., 0., 0.)), 
        ('H', (0., 0., bond_len))]
    
    if num_H > 2:
        for ii in range(num_H-2):
            geometry.append(
                ('H', (0., 0., (ii+2)*bond_len)))

    print(geometry)

    basis = 'sto-3g'
    mult  = 1
    chrg  = 0
    description = f'{num_H}_H_chain_{bond_len}'
    molecule = MolecularData(geometry, basis, mult, chrg, description)
    molecule = run_pyscf(molecule,
                         run_scf=1,
                         run_mp2=0,
                         run_cisd=0,
                         run_ccsd=0,
                         run_fci=0,
                         verbose=True)
    # molecule.load() 

    mol_ham = molecule.get_molecular_hamiltonian()
    return mol_ham

###
### The function that does the calc and writes to csv:
###
def generate_gse_circuit_stats(mol_ham, n_electrons, prec_ord, trot_ord, trot_num, csv_out='gse_HChain_scaling.csv'):

    t_start_time = time.time()

    # Generate the circuit:
    HChain_kwargs = {
    'trotterize' : True,
    'mol_ham'    : mol_ham,
    'ev_time'    : 1,
    'trot_ord'   : trot_ord,
    'trot_num'   : trot_num}

    # Set the initial state so electrons are in lowest orbitals
    init_state = np.zeros(n_electrons*2)
    for ii in range(n_electrons):
        init_state[ii] = 1

    gse_inst = GSE(
        precision_order=prec_ord,
        init_state=list(init_state),
        E_max = E_max,
        E_min = E_min,
        kwargs=HChain_kwargs)
    
    curr_circ = gse_inst.pe_inst.pe_circuit
    t_init_circuit_gen = time.time()

    # decompose circuit:
    decomp_circuit        = cirq.align_left(qsp_decompose_once(qsp_decompose_once(curr_circ)))
    cliffT_decomp_circuit = cirq.align_left(clifford_plus_t_direct_transform(decomp_circuit))

    t_circ_decomp = time.time()

    if os.path.exists(csv_out):
        need_header = False
    else:
        need_header = True
    outfile = open(csv_out,'a')
    if need_header:
        outfile.write('Num_Qubits,Prec_Bits,Trot_Order,Trot_Steps,Circ_Gen_Time,CliffT_Decomp_Time,CliffT_Depth,CliffT_T_Gate_Count\n')
    
    # Calculate times:
    circ_gen_time      = t_init_circuit_gen - t_start_time
    cliffT_decomp_time = t_circ_decomp      - t_init_circuit_gen

    # Calculate depth
    depth       = len(cliffT_decomp_circuit.moments)
    num_qubits  = len(cliffT_decomp_circuit.all_qubits())
    num_T_gates = count_T_gates(cliffT_decomp_circuit)

    # Format the print string:
    print_string = f'{num_qubits},{prec_ord},{trot_ord},{trot_num},{circ_gen_time},{cliffT_decomp_time},{depth},{num_T_gates}\n'

    outfile.write(print_string)


for chain_len in chain_lens:
    print(f'> On H-chain of length {chain_len}')
    mol_ham = gen_H_chain(chain_len)
    generate_gse_circuit_stats(
        mol_ham=mol_ham,
        n_electrons=chain_len,
        prec_ord=prec_,
        trot_ord=trot_ord_,
        trot_num=trot_num_)