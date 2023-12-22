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
from fable_functions import *
import numpy as np
import scipy as sp
import random
import cirq
import time


#build full LS-FABLE circuit from sparse matrix description
def LS_FALBE_full(i1,i2,v,n):
	oracle = LS_orace_from_sparse(i2,i1,v,n)
	fable_circ = oracle_surround(oracle,n)
	final_circ = sparse_surround(fable_circ,n)
	return(final_circ)
	
	


#builds oracle from structure and rotations
def oracle_from_struc(struc, v, n):
	circuit = cirq.Circuit()
	q = cirq.LineQubit.range(2*n + 1)
	circuit.append(cirq.Ry(rads=np.pi)(q[0]))
	L = len(struc)
	i = 0
	for x in range(L):
		if struc[x] == 0:
			a = -2*v[i]/(2**n)
			circuit.append(cirq.Ry(rads=a)(q[0]))
			i += 1
		else:
			j = 2*n + 1 - struc[x]
			circuit.append(cirq.CNOT(q[j],q[0]))
	return(circuit)


#build LS-FABLE oracle from sparse matrix description
def LS_orace_from_sparse(i1,i2,v,n):
	N = 2**n
	I = two_to_one_index(i1,i2,N)
	[I1,v1] = convert_mat_to_gray(I,v,n)
	C = cnot_base_structure(I1,n)
	circuit = oracle_from_struc(C, v1, n)
	return(circuit)


#surround oracle with FABLE gates
def oracle_surround(circuit,n):
	big_H = cirq.Circuit()
	big_SWAP = cirq.Circuit()
	q = cirq.LineQubit.range(2*n + 1)
	for x in range(n):
		big_H.append(cirq.H(q[x+1]))
		big_SWAP.append(cirq.SWAP(q[x+1],q[x+n+1]))
	new_circuit = big_H[:]
	new_circuit.append(circuit)
	new_circuit.append(big_SWAP)
	new_circuit.append(big_H)
	return(new_circuit)


#surround FABLE circuit with 0-controlled H gate for sparse
def sparse_surround(c,n):
	circuit = cirq.Circuit()
	circuit.append(big_CH(n))
	circuit.append(c)
	circuit.append(big_CH(n))
	return(circuit)


#build multi-controlled single H gates
def single_CH(b,n):
	big_H = cirq.Circuit()
	q = cirq.LineQubit.range(2*n + 1)
	H_op = cirq.H(q[b])
	for x in range(n+1):
		H_op = H_op.controlled_by(q[x])
	big_H.append(H_op)
	return(big_H)


#build large 0-controlled H gate
def big_CH(n):
	big_H = cirq.Circuit()
	big_X = cirq.Circuit()
	q = cirq.LineQubit.range(2*n + 1)
	for x in range(n+1):
		big_X.append(cirq.X(q[x]))
	big_H.append(big_X)
	for x in range(n):
		big_H.append(single_CH(x+n+1,n))
	big_H.append(big_X)
	return(big_H)



























