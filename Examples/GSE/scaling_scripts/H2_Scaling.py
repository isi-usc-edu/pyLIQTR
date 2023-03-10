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

from openfermion.chem import MolecularData

from pyLIQTR.GSE.GSE                     import GSE 
from pyLIQTR.QSP.qsp_helpers             import qsp_decompose_once
from pyLIQTR.gate_decomp.cirq_transforms import clifford_plus_t_direct_transform
from pyLIQTR.utils.utils                 import count_T_gates

###
### Define parameters to loop over:
prec_     = [3,4,6]
trot_ord_ = [1,2]
trot_num_ = [1,2,4,8]
E_max     = 2.0
E_min     = -2.0



###
### Define the molecule
### and get the hamiltonian
###
diatomic_bond_length = 0.7414
geometry             = [
    ('H', (0., 0., 0.)), 
    ('H', (0., 0., diatomic_bond_length))]
basis = 'sto-3g'
multiplicity = 1
charge = 0
description = str(diatomic_bond_length)

molecule = MolecularData(geometry, basis, multiplicity, charge, description)
molecule.load()

mol_ham = molecule.get_molecular_hamiltonian()


###
### The function that does the calc and writes to csv:
###
def generate_gse_circuit_stats(prec_ord, trot_ord, trot_num, csv_out='gse_H2_scaling.csv'):

    t_start_time = time.time()

    # Generate the circuit:
    H2_kwargs = {
    'trotterize' : True,
    'mol_ham'    : mol_ham,
    'ev_time'    : 1,
    'trot_ord'   : trot_ord,
    'trot_num'   : trot_num
    }


    gse_inst = GSE(
        precision_order=prec_ord,
        init_state=[1,1,0,0],
        E_max = E_max,
        E_min = E_min,
        kwargs=H2_kwargs)
    
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

###
### Do the loops
###
for prec in prec_:
    for trot_ord in trot_ord_:
        for trot_num in trot_num_:
            t_start = time.time()
            print(f'>---------------------------------------')
            print(f'> Doing run for:')
            print(f'>  prec:        {prec}')
            print(f'>  trot order:  {trot_ord}')
            print(f'>  trot_number: {trot_num}')
            
            generate_gse_circuit_stats(
                prec_ord=prec,
                trot_ord=trot_ord,
                trot_num=trot_num
            )
            t_end = time.time()
            print(f'>    Time: {t_end-t_start:0.2f} sec')
