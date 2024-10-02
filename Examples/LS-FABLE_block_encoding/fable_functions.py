"""
Copyright (c) 2024 Massachusetts Institute of Technology 
SPDX-License-Identifier: BSD-2-Clause
"""
##LS-FABLE Implementation
##Parker Kuklinski

#Given a matrix array, this code generates an LS-FABLE block encoding circuit and diagnostics on the encoding


import numpy as np
import scipy as sp
import random




#Define Hadamard tensor
def hadamard(n):
	H=[[1,1],[1,-1]]
	H=H/np.sqrt(2)
	HH=H
	for k in range(0,n-1):
		HH=np.kron(HH,H)
	return(HH)


#Generate block approximation using LS-FABLE
def ls_block(A):
	size_matrix = A.shape
	num_qubits = int(np.log2(size_matrix[0]))
	H = hadamard(num_qubits)
	A1 = np.dot(np.dot(H,A),H)
	A2 = np.sin(A1)
	A3 = np.dot(np.dot(H,A2),H)
	return(A3)


#Generate sparse array as sparse
def gen_sparse(n,s):
	N = 2**n
	num_nonzero = int(np.floor(N*s))
	I = random.sample(range(N**2), num_nonzero)
	i1,i2 = one_to_two_index(I,N)
	v = 2 * np.random.rand(num_nonzero) - 1
	return([i1,i2,v])


#convert sparse description to matrix
def sparse_to_mat(i1,i2,v,N):
	Z = np.zeros((N,N))
	L = len(i1)
	for x in range(L):
		Z[i1[x],i2[x]] = v[x]	
	return(Z)


#l2norm
def l2norm(A):
	B = A.transpose()
	C = B - 2j*B.imag
	
	M = np.dot(A,C)
	eigenvalues, eigenvectors = np.linalg.eig(M)
	L = np.sqrt(max(eigenvalues))
	return(L.real)
		

#generate pattern of CNOT controls
def gray_digit(n):
	v=[]
	for k in range(1,2*n+1):
		v1=v[:]
		v.append(k)
		v.extend(v1)
	v.append(2*n)
	return(v)
	 

#convert vector of binary to decimal number
def binatodeci(binary):
    return sum(val*(2**idx) for idx, val in enumerate(binary))


#convert decimal number to binary vector
def decitobina(n):
	v = [int(x) for x in bin(n)[2:]]
	v.reverse()
	return(v)
	

#generate gray code permutation list
def gray_perm(n):
	b = np.zeros(2*n).astype(int)
	v = [0]
	D = gray_digit(n)
	for k in range(2**(2*n)-1):
		d = D[k]-1
		b[d] = (b[d] + 1) % 2
		m = binatodeci(b)
		v.append(m)
	return(v)


#takes CNOT structure as input and eliminates duplicates
def cnot_eliminate(v):
	L = len(v)
	m = max(v)
	v1 = []
	c = np.zeros(m).astype(int)
	for n in range(L):
		x = v[n]
		if x == 0:
			for k in range(m):
				if c[k] == 1:
					v1.append(k+1)
			v1.append(0)
			c = np.zeros(m).astype(int)
		else:
			c[x-1] = (c[x-1] + 1) % 2
		if n == L-1:
			for k in range(m):
				if c[k] == 1:
					v1.append(k+1)
	return(v1)


#build CNOT structure from gray
def cnot_base_structure(I,n):
	N = 2**n
	list1 = np.zeros(N**2).astype(int) - 1
	list2 = gray_digit(n)
	result = [None]*(len(list1)+len(list2))
	result[::2] = list1
	result[1::2] = list2
	L = len(I)
	for x in range(L):
		i = 2*I[x]
		result[i] = result[i] + 1
	elim = [ele for ele in result if ele != -1]
	elim1 = cnot_eliminate(elim)
	return(elim1)


#convert sparse to gray
def convert_mat_to_gray(I,v,n):
	L = len(I)
	G = gray_perm(n)
	I1 = []
	for x in range(L):
		#I1.append(G[I[x]])
		I1.append(G.index(I[x]))
	i1 = sorted(range(L), key=lambda k: I1[k])
	I1 = sorted(I1)
	v1 = []
	for x in range(L):
		v1.append(v[i1[x]])
	return([I1,v1])	

				
#convert N^2 index to NxN
def one_to_two_index(I,N):
	Ia = np.array(I)
	i1 = np.floor(Ia/N).astype(int)
	i2 = (Ia % N).astype(int)
	return([i1,i2])


#convert NxN index to N^2 index
def two_to_one_index(i1,i2,N):
	I1 = [x * N for x in i2]
	I2 = np.add(I1,i1)
	return(I2)
	
		
		
	
















	
	
