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
import juliapkg
juliapkg.require_julia("~1.8,~1.9")
juliapkg.resolve()

import juliacall
from juliacall import Main as jl

jl.seval('import Pkg')
jl.seval('Pkg.add("QuantumMAMBO")')
jl.seval("using QuantumMAMBO")
jl.seval("using LinearAlgebra")
mambo = jl.QuantumMAMBO

import numpy as np
import scipy as sp
import itertools
from openfermion import FermionOperator, hermitian_conjugated, normal_ordered, QubitOperator
import openfermion as of
from warnings import warn

"""!!! ALL FUNCTIONS BELOW ARE FROM QUANTUMMAMBO UNTIL INDICATED !!!"""
# Copyright (c) 2023 iloaiza
def get_spin_orbitals(H : FermionOperator):
    '''
    Obtain the number of spin orbitals of H
    '''
    n = -1 
    for term, val in H.terms.items():
        if len(term) == 4:
            n = max([
                n, term[0][0], term[1][0],
                term[2][0], term[3][0]
            ])
        elif len(term) == 2:
            n = max([
                n, term[0][0], term[1][0]])
    n += 1 
    return n

def get_ferm_op_two(tbt, spin_orb):
    n = tbt.shape[0]
    op = FermionOperator.zero()
    for i in range(n):
        for j in range(n):
            for k in range(n):
                for l in range(n): 
                    if not spin_orb:
                        for a in range(2):
                            for b in range(2):
                                op += FermionOperator(
                                    term = (
                                        (2*i+a, 1), (2*j+a, 0),
                                        (2*k+b, 1), (2*l+b, 0)
                                    ), coefficient=tbt[i, j, k, l]
                                )
                    else:
                        op += FermionOperator(
                            term=(
                                (i, 1), (j, 0),
                                (k, 1), (l, 0)
                            ), coefficient=tbt[i, j, k, l]
                        )
    return op

def get_obt(H : FermionOperator, n = None, spin_orb=False, tiny=1e-12):
    '''
    Obtain the 2-rank tensor that represents one body interaction in H. 
    In addition, simplify tensor assuming symmetry between alpha/beta coefficients
    '''
    # getting N^2 phy_tbt and then (N/2)^2 chem_tbt 
    if n is None:
        n = get_spin_orbitals(H)
    
    obt = np.zeros((n,n))
    for term, val in H.terms.items():
        if len(term) == 2:
            if term[0][1] == 1 and term[1][1] == 0:
                obt[term[0][0], term[1][0]] = val.real
            elif term[1][1] == 1 and term[0][1] == 0:
                obt[term[1][0], term[0][0]] = -val.real
            else:
                warn("Warning, one-body operator has double creation/annihilation operators!",stacklevel=2)
                quit()

    if spin_orb:
        return obt

    # Spin-orbital to orbital 
    n_orb = obt.shape[0]
    n_orb = n_orb // 2

    obt_red_uu = np.zeros((n_orb, n_orb))
    obt_red_dd = np.zeros((n_orb, n_orb))
    obt_red_ud = np.zeros((n_orb, n_orb))
    obt_red_du = np.zeros((n_orb, n_orb))
    for i in range(n_orb):
        for j in range(n_orb):
            obt_red_uu[i,j] = obt[2*i, 2*j]
            obt_red_dd[i,j] = obt[2*i+1, 2*j+1]
            obt_red_ud = obt[2*i, 2*j+1]
            obt_red_du = obt[2*i+1, 2*j]

    if np.sum(np.abs(obt_red_du)) + np.sum(np.abs(obt_red_ud)) != 0:
        warn("one-body tensor spin-orbit couplings are not 0. This should not affect the DoubleFactorized resource estimates.",stacklevel=2)
    if np.sum(np.abs(obt_red_uu - obt_red_dd)) > tiny:
        warn("one-body tensor isn't symmetric to spin-flips. This should not affect the DoubleFactorized resource estimates.",stacklevel=2)
        # print("obt_uu - obt_dd = {}".format(obt_red_uu - obt_red_dd))

    obt = (obt_red_uu + obt_red_dd) / 2

    return obt

def get_tbt(H : FermionOperator, n=None):
    
    if n is None:
        n = get_spin_orbitals(H)

    phy_tbt = np.zeros((n, n, n, n))
    for term, val in H.terms.items():
        if len(term) == 4:
            phy_tbt[
                term[0][0], term[1][0],
                term[2][0], term[3][0]
            ] = np.real_if_close(val)

    chem_tbt = np.transpose(phy_tbt, [0, 3, 1, 2])
    chem_tbt_sym = (chem_tbt - np.transpose(chem_tbt, [0,3,2,1]) + np.transpose(chem_tbt, [2,3,0,1]) - np.transpose(chem_tbt, [2,1,0,3]) ) / 4.0

    # Spin-orbital to orbital
    n_orb = phy_tbt.shape[0]
    n_orb = n_orb // 2
    alpha_indices = list(range(0, n_orb * 2, 2))
    beta_indices = list(range(1, n_orb * 2, 2))

    chem_tbt_orb = (chem_tbt_sym[np.ix_(alpha_indices, alpha_indices, beta_indices, beta_indices)]
                    - np.transpose(chem_tbt_sym[np.ix_(alpha_indices, beta_indices, beta_indices, alpha_indices)], [0,3,2,1]))
   
    return chem_tbt_orb


def to_tensors(H : FermionOperator, n=None, spin_orb=False):
    #returns constant term, one-body tensor, and two-body tensor from FermionOperator.
    no_h_ferm = normal_ordered(H)
    tbt = get_tbt(no_h_ferm, )
    h1b = no_h_ferm - get_ferm_op_two(tbt, spin_orb)
    h1b = normal_ordered(h1b)
    obt = get_obt(h1b, spin_orb=spin_orb)

    return H.constant, obt, tbt

"""!!! ALL ABOVE FUNCTIONS ARE FROM QUANTUMMAMBO, BELOW ARE ORIGINAL !!!"""


def givens_func(U):
    #returns list of givens rotations from unitary U??
    N = U[1].size

    num_G = int(N*(N-1)/2)
    G = np.zeros(num_G)

    u_curr = mambo.givens_real_orbital_rotation(N, G)
    return u_curr

def coeffs_to_givens(coeffs):
    #returns list of givens rotation angles from coefficients
    N = len(coeffs)
    thetas = 0.5 * np.pi * np.ones(N-1)
    thetas[0] = (mambo.r_acos(coeffs[0])) / 2

    for i in range(1,N-1):
        cum_coef = np.sum((np.absolute(coeffs[0:i-1]))**2)

        if abs(coeffs[i]) > 1e-10:
            ccur = coeffs[i] / np.prod(np.sin(2 * thetas[0:i-1]))
            if abs(ccur) > 1:
                ccur = np.sign(ccur)
        else:
            ccur = 0
            
        thetas[i] = (mambo.r_acos(ccur)) / 2
        
    if abs(coeffs[-1]) > 1e-10:
        thetas[-1] = thetas[-1] * np.sign(coeffs[-1])
    
    return thetas


def DF_decomposition(h0,obt,tbt, tol=1e-8, tiny=1e-12):
    #performs DF routine on FermionOperator, returns fragments with data to be 
    #parsed by yield_DF_Info().
    
    # h0, obt, tbt = to_tensors(H)
    orbs = np.size(obt, 0)
    n = int(orbs)
    N = n**2
    
    #symmetry test
    tbt_full = np.reshape(tbt, (N,N))
    tbt_res = jl.LinearAlgebra.Symmetric(tbt_full)

    if np.sum(abs(np.subtract(tbt_full, tbt_res))) > tiny:
        warn("Non-symmetric two-body tensor as input for DF routine, calculations might have errors...", stacklevel=2)
        tbt_res = tbt_full

    #first eigendecomposition (outer sum)
    all_vals, all_vecs = np.linalg.eigh(tbt_res) # Note, all_vals ordered from min absolute value to max, let's reverse
    all_vals = all_vals[::-1]
    all_vecs = all_vecs[:,::-1]

    #threshold test
    vals = []
    for i in range(len(all_vals)):
        test_val = all_vals[i]
        if abs(test_val) < tol:
            # all eigenvalues less than or equal to tol are thrown out
            num_ops = i-1
            break
    
    vals = all_vals[:num_ops]
    vecs = all_vecs[:,0:num_ops]
    
    frags = []

    #second eigendecomposition (inner sum)
    for i in range(0,num_ops):
        full_l = np.reshape(vecs[:,i], (n,n))
        cur_l = jl.LinearAlgebra.Symmetric(full_l)
        sym_dif = np.sum((abs(np.subtract(cur_l, full_l)))**2)
        if sym_dif > tiny:
            #Hermitian test
            if np.sum(abs(np.add(full_l, full_l.transpose()))) > tiny:
                warn(f"Fragment {i} is neither Hermitian or anti-Hermitian!",stacklevel=2) 
                cur_l = jl.LinearAlgebra.Hermitian(full_l)
                vals[i] = -1 * vals[i]
        wl, Ul = np.linalg.eigh(cur_l)


        #build fragment
        Rl = mambo.f_matrix_rotation(n, Ul)
        C = mambo.cartan_1b(False, wl, n)
        frags.append(mambo.F_FRAG(1, jl.tuple(Rl), mambo.DF(), C, n, False, all_vals[i], True))

    return(frags)


def to_OBF(obt):
    #tranform one-body tensor into QuantumMAMBO F_FRAG.
    N = np.size(obt,0)
    D, U = np.linalg.eig(obt)
    C = mambo.cartan_1b(False, D, N)
    fU = jl.tuple(mambo.f_matrix_rotation(N, U))
    return mambo.F_FRAG(1, fU, mambo.OBF(), C, N, False, 1, False)
    

def U_to_Givens(U: mambo.F_UNITARY, k: int):
    #transform unitary into one-body coefficients and wrap coeffs_to_givens function to return givens angles.
    one_bod = mambo.one_body_unitary(U)
    size = int(np.sqrt(len(one_bod)))
    one_bod_list = []
    for i in one_bod:
        one_bod_list.append(i)
    one_bod_array = np.reshape(one_bod_list, (size,size))

    coeffs = []
    for i in range(len(one_bod_array[k])):
        coeffs.append(one_bod_array[i][k])

    givens = coeffs_to_givens(coeffs)

    return givens
    

def calc_xi(f_p_abs, th):
    #calculate the number of elements of DF results which meet or surpass truncation threshold. 
    xi_vals = []
    for l in f_p_abs:
        norm_sc = sum(l)
        truncation_limit = [i * norm_sc for i in l]
        keep_elements = []
        for i in truncation_limit:
            if i > th:
                keep_elements.append(1)
            else:
                keep_elements.append(0)
        xi = sum(keep_elements)
        xi_vals.append(xi)

    return xi_vals  



            
