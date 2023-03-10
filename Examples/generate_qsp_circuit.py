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

#!/usr/bin/env python3
from __future__ import print_function

###
### Imports to support the pyQSP Gate-Based Hamiltonian simulation
###
import os
import sys
import math
import argparse
import cirq
from cirq.contrib.svg import SVGCircuit
import time

import pyLIQTR.QSP.gen_qsp                 as qspFuncs
import pyLIQTR.QSP.QSP                     as pQSP
import pyLIQTR.model_simulators.vlasovsim  as vs        


from pyLIQTR.QSP.Hamiltonian             import Hamiltonian as pyH
from pyLIQTR.QSP.qsp_helpers             import qsp_decompose_once, print_to_openqasm, prettyprint_qsp_to_qasm # these should move to a utils.
from pyLIQTR.gate_decomp.cirq_transforms import clifford_plus_t_direct_transform, count_rotation_gates

###
### User Input: Vlasov Equation Parameterization 
###

N      = 100                                    # Number of terms in the Vlasov-Hermite 
                                               # expansion (equivalent to qubits)

k      =  2.0                                  # Fourier wavenumber

nu     =  0.0                                  # Collisional damping parameter : 
                                               # (for use only with statevector propagation)
                                               # (SET TO 0 for now, since imag otherwise)

alpha  =  0.6                                  # Electric field parameter


required_precision   = 1e-2
timestep_of_interest = 0.05 # sim_time

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("-n", "--numb", help="Width of operations", type=int, default=3)
parser.add_argument("-p", "--precision", help="Precision of circuit in negative powers of 10, e.g. 3 => 10-3", type=int, default=3)
parser.add_argument("-t", "--timestep", help="Time step in 100's, e.g. 5 = 0.05", type=int, default=5)
parser.add_argument("-o", "--staq_opt", help="Run Staq optimization", action="store_true")
parser.add_argument("-ls", "--lattice_surgery", help="Run Lattice surgery transfoorm", action="store_true")
args = parser.parse_args()

N = args.numb
required_precision = 10**-args.precision
timestep_of_interest = float(args.timestep) * 0.01

print("Running simulation, N = {}, precision = {}, time = {}".format(N, required_precision, timestep_of_interest))

ham_string = vs.hamiltonian_wfn_vlasov_hermite_linear_sym_string(k, alpha, nu, N)

qsp_H = pyH(ham_string)

print('\n: --- Visual Check Hamiltonian ---')
print(f': qsp_H = {qsp_H}\n')

# Generate angles
angles, tolerances = qspFuncs.get_phis(qsp_H, simtime=timestep_of_interest, req_prec=required_precision)

print(qspFuncs.get_phis(qsp_H, simtime=timestep_of_interest, req_prec=required_precision, steps_only=True))
print(len(angles))

print("Generating QSP circuit")
# Generate the QSP circuit
time1 = time.perf_counter()
_singleElement = True
qsp_generator = pQSP.QSP(phis=angles, hamiltonian=qsp_H, target_size=qsp_H.problem_size, singleElement=_singleElement)
qsp_circ      = qsp_generator.circuit()

# decompose the circuit
print("Decompose circuit(Step 1)")
decomposed_once_circuit = (qsp_decompose_once(qsp_circ))
print("Decompose circuit(Step 2)")
decomposed_circuit      = (qsp_decompose_once(decomposed_once_circuit))
time2 = time.perf_counter()
generate_time = time2 - time1
#print(decomposed_circuit)

# Calculate the required precision of rotation gates, we use the specified precision and assume
# that the precision of all rotations is 0.5*required_precision
#
time1 = time.perf_counter()
num_rots = count_rotation_gates(decomposed_circuit)
if _singleElement:
    num_rots = num_rots * len(angles)/2
    print("singleElement, updating num_rots = {}, num unique rotations = {}".format(num_rots, num_rots/(len(angles)/2)))
rot_res = required_precision / (num_rots*2)
rot_res10 = math.ceil(-1.0*math.log10(rot_res))
if True:
    print("Decomposing into Clifford+T circuit, total_rotations = {}, precision of each = 10e-{}".format(num_rots, rot_res10))
    cliff_plus_T_circuit    = clifford_plus_t_direct_transform(decomposed_circuit, precision=rot_res10)
else:
    cliff_plus_T_circuit = None
time2 = time.perf_counter()
decompose_time = time2 - time1

"""
print("decomposed circuit")
for moment in decomposed_circuit:
    for op in moment:
        print(op)
"""

print("Writing OpenQASM circuit")
time1 = time.perf_counter()
# write to the output file
with open('qsp_circ_N{}.qasm'.format(N),'w') as f:
    print_to_openqasm(f, decomposed_circuit)
if not cliff_plus_T_circuit==None:
    with open('qsp_circ_N{}_CliffordT.qasm'.format(N),'w') as f:
        print_to_openqasm(f, cliff_plus_T_circuit)
time2 = time.perf_counter()
qasm_time = time2 - time1

current_circuit = 'qsp_circ_N{}_CliffordT.qasm'.format(N)

staq_time = None
if args.staq_opt:
    time1 = time.perf_counter()
    staq_circuit = 'qsp_circ_N{}_CliffordT.staq.qasm'.format(N)
    cmd = "staq -S -i -r --simplify {} > {}".format(current_circuit, staq_circuit)
    print("Performing Staq optimization => {}".format(staq_circuit))
    os.system(cmd)
    current_circuit = staq_circuit
    time2 = time.perf_counter()
    staq_time = time2 - time1

ls_time = None
if args.lattice_surgery:
    time1 = time.perf_counter()
    # perform lattice surgery transform
    staq_ls_circuit = '.'.join(current_circuit.split('.')[:-1]+['json'])
    cmd = "staq_lattice_surgery < {} > {}".format(current_circuit, staq_ls_circuit)
    print("Performing lattice surgery transformation => {}".format(staq_ls_circuit))
    os.system(cmd)
    time2 = time.perf_counter()
    ls_time = time2 - time1

    # process JSON to get stats
    staq_ls_stats = '.'.join(staq_ls_circuit.split('.')[:-1]+['dat'])
    cmd = "./calc_lattice_surgery_resources.py {} > {}".format(staq_ls_circuit, staq_ls_stats)
    print("Calculating lattice surgery resources => {}".format(staq_ls_stats))
    os.system(cmd)

print("Generate time = {}, Decompose time = {}, QASM time = {}".format(generate_time, decompose_time, qasm_time))
if not staq_time==None:
    print("Staq time = {}".format(staq_time))
if not ls_time==None:
    print("LS time = {}".format(ls_time))


    
    
