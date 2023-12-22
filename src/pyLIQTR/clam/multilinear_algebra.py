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

################################################################################
###                                                                          ###
###    MULTILINEAR_ALGEBRA - Methods for dense/sparse multilinear algebra    ###
###                                                                          ###
################################################################################



import  numpy              as  np
import  scipy.sparse       as  sps



## mk(ops), mk_s(ops)
## -------------------------------------------------------------
##
## Kronecker product of operators in list <ops>.  Sparse matrix
## version is mk_s().
##

def mk(ops):

    A = 1

    for op in ops:
        A = np.kron(A,op)

    return(A)


def mk_s(ops):

    A = 1

    for op in ops:
        A = sps.kron(A,op)

    return(A)





## mk_n(op,N)
## -------------------------------------------------------------
##
## N-fold Kronecker product of operator /
## matrix op
##

def mk_n(op,N):
    return mk([op]*N)

def mk_n_s(op,N):
    return mk_s([op]*N)





## mk_id(op,N)
## -------------------------------------------------------------
##
## N-fold Kronecker product of identity element with dimension 
## dim (default=2).  Sparse version is mk_id_s().
##

def mk_id(N,dim=2):
    id_op = np.eye(dim**N)
    return(id_op)

def mk_id_s(N,dim=2):
    id_op = sps.eye(dim**N)
    return(id_op)





## mk_ipad(op,N)
## -------------------------------------------------------------
##
## N-fold Kronecker product of operator, padded with n identity 
## elements of dimension dim on the left and m of the same on the 
## right:
##
## I*I*I*I*I*op*I*I*I*I*I
##     |            |
##  n-fold       m-fold
##
## Sparse matrix version is mk_ipad_s().
##

def mk_idpad(n,op,m,dim=2):
    
    lid = mk_id(n,dim=dim)
    rid = mk_id(m,dim=dim)
    
    pop = mk([lid,op,rid])

    return pop

def mk_idpad_s(n,op,m,dim=2):
    
    lid = mk_id_s(n,dim=dim)
    rid = mk_id_s(m,dim=dim)
    
    pop = mk_s([lid,op,rid])

    return pop



## mk_ksum(op,N)
## -------------------------------------------------------------
##
## Sum of identity padded Kronecker product "strings" from
## Carleman linearization.  Sparse version is mk_ksum_s().
##
##

def mk_ksum(op,N,dim=2):
    
    ks = mk([mk_id(0,dim=dim),op,mk_id(N-1,dim=dim)])
    
    for n in range(1,N):
      
        ks += mk([mk_id(n,dim=dim),op,mk_id(N-n-1,dim=dim)])
    
    return ks


def mk_ksum_s(op,N,dim=2):
    
    ks = mk_s([mk_id_s(0,dim=dim),op,mk_id_s(N-1,dim=dim)])
    
    for n in range(1,N):
      
        ks += mk_s([mk_id_s(n,dim=dim),op,mk_id_s(N-n-1,dim=dim)])
    
    return ks