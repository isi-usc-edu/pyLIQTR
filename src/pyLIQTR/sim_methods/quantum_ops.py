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
import numpy as np
import numpy.linalg as la
import scipy.linalg as sla

from math import log2



# Define Pauli matrices + identity

px = np.array([[0.0, 1.0], [1.0, 0.0]])
py = np.array([[0.0, -1j], [1j,  0.0]])
pz = np.array([[1.0, 0.0], [0.0, -1.0]])

id = np.array([[1.0, 0.0], [0.0, 1.0]])

# Spin ladder operators

sigp = (px + 1j*py)/2
sigm = (px - 1j*py)/2

# Pauli z projectors

przp = (id+pz)/2
przm = (id-pz)/2

# Define up and down spin states

sz_u = np.array([1,0])
sz_d = np.array([0,1])









def ham_from_pauli_tuples(tuples,types,coeffs,N):

    H = 0.0j

    for j in range(len(tuples)):

        p_string = ps_text_pad(tuples[j],types[j],N)
        H       += coeffs[j]*ps_text_to_matrix(p_string)

    return(H)





################################################################################
###
###   FUNCTION     ps_text_to_ops(string)
###
################################################################################
###
###   DESCRIPTION
###
###      Returns an array containing the Pauli operators represented by the
###      Pauli string <string>, starting from the least significant element.
###      This method is case insensitive.
###
###   ARGUMENTS
###
###      string  =  Pauli string consisting of X,Y,Z and I (projectors and the
###                 like are not yet implemented).
###
###   RETURNS
###
###      op_set  =  Array of Pauli operators in the canonical matrix
###                 representation for su(2)
###
###
###   REQUIRES
###
###      numpy as np
###

def ps_text_to_ops(string):

    plabels = {'x','y','z','i'}

    N = len(string)

    op_set = []

    for j in range(N):
        op = label_to_pauli(string[j])
        op_set.append(op)

    return(op_set)





################################################################################
###
###   FUNCTION     ps_label_to_pauli(label)
###
################################################################################
###
###   DESCRIPTION
###
###      Returns an a matrix representation of the Pauli operator corresponding
###      to the label <label>.  This method is case insensitive.
###
###   ARGUMENTS
###
###      string  =  Text label for the Pauli; may be 'X', 'Y', 'Z' or 'I'
###
###   RETURNS
###
###      op      =  Matrix representation of the operator
###      -1      =  If label is not recognized.
###
###   REQUIRES
###
###      numpy as np
###

def label_to_pauli(label):

    if  (label.lower() == 'x'):
        return(px)
    elif(label.lower() == 'y'):
        return(py)
    elif(label.lower() == 'z'):
        return(pz)
    elif(label.lower() == 'i'):
        return(id)
    else:
        print("\nNo matching Pauli operator: ", label,"\n")
        return(-1)





################################################################################
###
###   FUNCTION     ps_text_to_matrix(string)
###
################################################################################
###
###   DESCRIPTION
###
###      Generates a matrix representation of the operator corresponding to
###      the text-based Pauli string <string> (example: 'IXXIYII').
###
###
###   ARGUMENTS
###
###      string  =  Pauli string consisting of X,Y,Z and I (projectors and the
###                 like are not yet implemented).
###
###   RETURNS
###
###      op      =  Matrix representation of the operator
###
###   REQUIRES
###
###      numpy as np
###

def ps_text_to_matrix(string):

    ops = ps_text_to_ops(string)
    M   = multikron(ops)

    return(M)





################################################################################
###
###   FUNCTION     ps_text_pad(string)
###
################################################################################
###
###   DESCRIPTION
###
###      Pads the shorthand text string a mulit-qubit Pauli operator with
###      identity elements to give an N-qubit Pauli string. That is,
###      an operator described by <string> acting on spins/qubits
###      in the tuple <sites> is padded by identity operators to give a string
###      acting on <N> qubits. If <sites> = (0,2,3) and <string> = 'XYZ' with
###      N = 6, this would return XIYZII. Note that the indices in sites
###      follow pythonic array indexing, while <N> is the absolute number of
###      qubits (NOT the largest array index).
###
###
###
###   ARGUMENTS
###
###      sites  =   Tuple containing site/qubit indices.
###      string =   Text string describing the operator
###      N      =
###
###   RETURNS
###
###      op      =  Matrix representation of the operator
###
###   REQUIRES
###
###      numpy as np
###

def ps_text_pad(sites,string,N):

    padded = ''

    ## Check if we have been given a scalar or a tuple.
    ##
    is_tuple = hasattr(sites, "__len__")

    if (is_tuple):
        for k in range(N):
            if (k in sites):
                sidx    = sites.index(k)
                padded += string[sidx]
            else:
                padded += 'I'
    else:
        padded = 'I'*(sites) + string + 'I'*(N-sites-1)

    return(padded)




def ps_text_full_set(Sites,Strings,N,Coeffs=None):

    Padded = []

    if (Coeffs is not None):
        for pair in zip(Sites,Strings,Coeffs):
            Padded.append((ps_text_pad(pair[0],pair[1],N),pair[2]))
    else:
        for pair in zip(Sites,Strings):
            Padded.append(ps_text_pad(pair[0],pair[1],N))

    return(Padded)




def ps_print_set(tuples,types,coeffs):

    print("\nSites     \t Type \t\t Coefficient\n",end="")
    print("-"*80,"\n")
    for j in range(len(tuples)):
        print(tuples[j],"\t\t", types[j], "\t\t", coeffs[j])
    print("")



def ps_print_set_merged(tuples):

    print("\nStrings     \t Coefficients\n",end="")
    print("-"*50,"\n")
    for j in range(len(tuples)):
        print(tuples[0],"\t\t", tuples[1])
    print("")




def params_tfim_1d( N, J_x=0.0, J_z=0.0, h_x=0.0, h_z=0.0, periodic=False ):

    # Returns the model in either a ZZ or XX coupling representation
    # (i.e., via Kramers-Wannier duality).  If the J_z arguments are
    # nonzero this convention will take precedence.

    if   (J_z != 0.0):
        params = params_heisenberg_1d( N, J_z=J_z, h_x=h_x, periodic=periodic )
    elif (J_x != 0.0):
        params = params_heisenberg_1d( N, J_x=J_x, h_z=h_z, periodic=periodic )
    else:
        params = -1

    return(params)




def params_heisenberg_xxz_1d( N, J=0.0, Delta=0.0, h_x=0.0, h_y=0.0,
                              h_z=0.0, periodic=False ):

    params = params_heisenberg_1d( N, J_x=J, J_y=J, J_z=(Delta*J), h_x=h_x,
                                   h_y=h_y, h_z=h_z, periodic=periodic )

    return(params)




def params_heisenberg_xxx_1d( N, J=0.0, h_x=0.0, h_y=0.0,h_z=0.0,
                              periodic=False ):

    params = params_heisenberg_1d( N, J_x=J, J_y=J, J_z=J, h_x=h_x, h_y=h_y,
                                   h_z=h_z, periodic=periodic )

    return(params)






def params_heisenberg_1d(N,J_x=0.0,J_y=0.0,J_z=0.0,h_x=0.0,h_y=0.0,h_z=0.0,
                          periodic=False):

    tuples  = []
    types   = []
    coeffs  = []

    J_vals  =  [J_x, J_y, J_z];     J_labels  =  ['XX','YY','ZZ']
    H_vals  =  [h_x, h_y, h_z];     H_labels  =  ['X','Y','Z']
    NJs     =  len(J_vals);         NHs       =  len(J_vals)


    for idx in range(NJs):
        if (J_vals[idx] != 0):
            for k in range(N-1):
                tuples.append((k,k+1))
                types.append(J_labels[idx])
                coeffs.append(J_vals[idx])

    if (periodic == True):
        for idx in range(NJs):
            if (J_vals[idx] != 0):
                tuples.append((0,N-1))
                types.append(J_labels[idx])
                coeffs.append(J_vals[idx])

    for idx in range(NHs):
         if (H_vals[idx] != 0):
            for k in range(N):
                tuples.append((k))
                types.append(H_labels[idx])
                coeffs.append(H_vals[idx])


    return(tuples,types,coeffs)






################################################################################
###
###   FUNCTION     multikron(ops)
###
################################################################################
###
###   DESCRIPTION
###
###      Returns the kronecker product of a set of operators ops passed in a
###      tuple or array.  The product is formed starting with the least
###      significant (zeroth) operator first and taking the right product with
###      subsequent operators, consistent with the tensor product:
###
###             op_0 \otimes op_1 \otimes op_2 \otimes ... \otimes \op_N
###
###   ARGUMENTS
###
###      ops = array containing the matrix representation of operators from
###            to form the kronecker product.
###
###   RETURNS
###
###      A   = Kronecker product of the operator set
###
###   REQUIRES
###
###      numpy as np
###

def multikron(ops):

    A = 1

    for op in ops:
        A = np.kron(A,op)

    return(A)





################################################################################
###
###   FUNCTION     vectorize_cvec(M)
###
################################################################################
###
###   DESCRIPTION
###
###      "Re-vectorizes" a one-dimensional numpy array (vector) with N complex
###      entries, forming a new vector with the N real entries first and their
###      N imaginary parts second (all real valued):
###
###        [a_0 + i*b_0, a_1 + i*b_1, ... a_N + i*b_N] ------>
###             [a_0, a_1, ... a_N, b_0, b_1, ... b_N]
###
###      This routine facilitates argument passing for ODEPACK solvers.
###
###   ARGUMENTS
###
###      M = complex-valued vector to be re-vectorized
###
###   RETURNS
###
###      A   = Re-vectorized real-valued array
###
###   REQUIRES
###
###      numpy as np
###

def vectorize_cvec(M):

    M_vec = np.append(M.real,M.imag)

    return(M_vec)





################################################################################
###
###   FUNCTION     devectorize_cvec(M)
###
################################################################################
###
###   DESCRIPTION
###
###      Converts a real-valued "re-vectorized" representation of a 1d-array of
###      complex numbers back into the a standard complex-valued array.
###      [see devectorize_cvec() for details].  The precise transformation is:
###
###         [a_0, a_1, ... a_N, b_0, b_1, ... b_N]  ------>
###               [a_0 + i*b_0, a_1 + i*b_1, ... a_N + i*b_N]
###
###      This routine facilitates argument passing for ODEPACK solvers.
###
###   ARGUMENTS
###
###      M = Real valued array representing an array of complex numbers
###
###   RETURNS
###
###      A = Complex-valued array representing N complex numbers
###
###   REQUIRES
###
###      numpy as np
###

def devectorize_cvec(M_vec):

    N = int(M_vec.shape[0]/2)

    M = M_vec[0:N] + 1j*M_vec[N:(2*N)]

    return(M)




################################################################################
###
###   FUNCTION     measure_obs_site(wfn,obs_op)
###
################################################################################
###
###   DESCRIPTION
###
###      Calculates the expectation value of the oservable operator obs_op
###      for each qubit / spin site subject as described by the wavefunction
###      wfn.
###
###   ARGUMENTS
###
###      wnf      = Full wavefunction
###      obs_op   = Matrix representation of observable
###
###   RETURNS
###
###      obs_val  = Vector containing expectation value of operator for
###                 each qubit / spin site.
###
###   REQUIRES
###
###      numpy as np
###

def measure_obs_site(wfn,obs_op):

    # really inefficient. should save measuremet operators.

    N       = int(log2(wfn.shape[0]))
    obs_val = 1j*np.zeros(N)

    for j in range(N):
        mop         = multi_qb_op([obs_op],[j],N)
        obs_val[j]  = ip_sv_expect(wfn,mop,wfn)

    return(np.abs(obs_val))




def measure_obs_pair(wfn,obs_op):

    # really inefficient. should save measuremet operators.

    N       = int(log2(wfn.shape[0]))
    obs_val = 1j*np.zeros(N-1)

    for j in range(N):
        mop         = multi_qb_op([obs_op],[j],N-1)
        obs_val[j]  = ip_sv_expect(wfn,mop,wfn)

    return(obs_val)




################################################################################
###
###   FUNCTION     multi_qb_op(ops,idxs,N)
###
################################################################################
###
###   DESCRIPTION
###
###      Returns a multi-site operator, formed from the tensor product of
###      operators in the array <ops> acting on the (respective) qubits
###      with indices <idxs>.  The product is formed following the order
###      that the qubits are specified in <idxs>.  For an N=6 qubit example,
###
###            ops  = [opA,opB,opC]
###            idxs = [0, 3, 4]
###
###      we form the tensor product
###
###         opA \otimes Id \otimes Id \otimes opB \otimes opC \otimes Id
###
###      where Id is the identity operator (eg. the least significant qubit
###      is the leftmost qubit).
###
###   ARGUMENTS
###
###      ops    = set of operators
###      idxs   = indexes of sites acted on by these operators
###      N      = total number of qubits / spins
###
###   RETURNS
###
###      OP = the multi-site operator
###
###   REQUIRES
###
###      numpy as np
###

def multi_qb_op(ops,idxs,N):

    OP = 1

    for j in range(N):
        if (j in idxs):
            OP = np.kron(OP,ops[idxs.index(j)])
        else:
            OP = np.kron(OP,id)

    return(OP)






################################################################################
###
###   FUNCTION     ip_sv_expect(wfnl,op,wfnr)
###
################################################################################
###
###   DESCRIPTION
###
###      Calculates the inner product <wfnl | op | wfnr> = <wfnl | op*wfnr>,
###      which corresponds to the expectation value of op when wfnl = wfnr.
###
###   ARGUMENTS
###
###      wnfl     = Wavefunction 1
###      wfnr     = Wavefunction 2
###      op       = Matrix representation of operator
###
###   RETURNS
###
###      eev      = Value associated with inner product.
###
###   REQUIRES
###
###      numpy as np
###

def ip_sv_expect(wfnl,op,wfnr):

    ev = np.matmul(wfnl.conj().T,np.matmul(op,wfnr))

    return(ev)


################################################################################
###
###   FUNCTION     ip_sv_(wfnl,wfnr)
###
################################################################################
###
###   DESCRIPTION
###
###      Calculates the inner product <wfnl | wfnr> = <wfnl, wfnr> between
###      wavefunctions wfnl and wfnr (the so-called "overlap")
###
###   ARGUMENTS
###
###      wnfl     = Wavefunction 1
###      wfnr     = Wavefunction 2
###
###   RETURNS
###
###      eev      = Value associated with inner product.
###
###   REQUIRES
###
###      numpy as np
###
def ip_sv(wfnl,wfnr):

    ev = np.matmul(wfnl.conj().T,wfnr)

    return(ev)



################################################################################
###
###   FUNCTION     parse_wfn_trajz_pz(ts_vec,traj_raw)
###
################################################################################
###
###   DESCRIPTION
###
###      Extracts the observable (1+pz)/2 for each qubit / spin described by a
###      time series of wavefunctions that were evolved directly with the
###      Schrodinger propagator.
###
###   ARGUMENTS
###
###      ts_vec   = Vector containing timesteps from the trajectory
###      traj_raw = Raw wavefunction trajectory from integrator
###
###   RETURNS
###
###      traj_proc = Processed trajectory.  The first element of each row is
###                  the timestep, while subsequent entries contain the
###                  expectation value (amplitude) of (1+pz)/2 for each qubit /
###                  spin site.
###
###   REQUIRES
###
###      numpy as np
###


### Add vectorization flag

def parse_wfn_traj(ts_vec,traj_raw,op,indices=None,vectorized=True):

    #   * D_op = N_qb  -->  this is a global observable
    #   * D_op = 1     -->  this is a single qubit observable
    #   * D_op = 2     -->  this is a two-qubit observable


    # Get number of qubits in trajectory
    if ( vectorized==True ):
        N_qb = int(np.log2(traj_raw[0].shape[0]/2))
    else:
        N_qb = int(np.log2(traj_raw[0].shape[0]))

    D_op = int(np.log2(op.shape[0]))              # Number of qubits associated
                                                  # with operator


    # N_ops = len(ops)
    N_ts  = len(ts_vec)

    # traj_proc = 1j*np.zeros((N_ops,N_ts,N+1))

    # add code to handle evalation on select site indices


    if  (D_op == 1):
        traj_proc = 1j*np.zeros((N_ts,N_qb+1))
    elif (D_op == 2):
        traj_proc = 1j*np.zeros((N_ts,N_qb))
    elif (D_op == N_qb):
        traj_proc = 1j*np.zeros(N_ts)
    else:
        print(" ERROR: parse_wfn_traj only handles 1, 2, and N qubit operators.\n")
        print("        Other operators shoudl be passed as an N-qubit operator after \n")
        print("        padding with the identity using multi_qb_op.\n")
        return()


    for j, t in enumerate(ts_vec):
        traj_proc[j,0] = t

        if (vectorized == True):
            TFr = traj_raw[j,0:(2**N_qb)] + 1j*traj_raw[j,2**N_qb:(2*(2**N_qb))]
        else:
            TFr = traj_raw[j]

        if (D_op == 1):
            traj_proc[j,1:(N_qb+1)] = measure_obs_site(TFr,op)
        elif (D_op == 2):
            traj_proc[j,1:(N_qb)]   = measure_obs_site(TFr,op)
        elif (D_op == N_qb):
            traj_proc[j,1]          = ip_sv_expect(TFr,op,TFr)

    return(traj_proc)







################################################################################
###
###   FUNCTION     wfn_prop (WFv,t,H,N,dt)
###
################################################################################
###
###   DESCRIPTION
###
###      Returns the Hamiltonian applied to a (multipled by -i) wavefunction,
###      for use in solving the time-dependent Schrodinger equation (hbar=1).
###      Time parameters are included as arguments for ease of modification
###      when adding time-depedent perturbations.  Arguments follow that for
###      generic numpy wrappers to ODEPACK.
###
###   ARGUMENTS
###
###      WFv = Vectorized wavefunction, with real components first and imaginary
###            components second.
###      t   = Current / absolute time
###      H   = Hamiltonian (as a real or complex matrix)
###      N   = Number of spins / qubits in Hamiltonian
###      dt  = Integrator timestep
###
###   RETURNS
###
###      WFp_dt = Infintesimal time evolution generated by the Hamiltonian
###               opeartor
###
###   REQUIRES
###
###      numpy as np
###

def prop_wfn(WFv,t,H):

    WF = devectorize_cvec(WFv)

    WF_dt = -1j*np.matmul(H,WF)

    WFv_dt = vectorize_cvec(WF_dt)

    return(WFv_dt)



################################################################################
###
###   FUNCTION     trace_distance (A,B)
###
################################################################################
###
###   DESCRIPTION
###
###      Returns the trace distance between matrix operators A and B, normalized
###      by a factor of 1/2 (i.e., the "density matrix" convention)
###
###   ARGUMENTS
###
###      A = Matrix representation of an operator
###      B = Matrix representation of an operator
###
###   RETURNS
###
###      tr_dist = trace distance between oeprators A and B
###
###   REQUIRES
###
###      numpy          as np
###      scipy.linalg   as sla
###

def trace_distance(A,B):

    C = A-B
    tr_dist = 0.5 * trace_norm(C)

    return(tr_dist)

################################################################################
###
###   FUNCTION     trace_norm (A)
###
################################################################################
###
###   DESCRIPTION
###
###      Calculates the trace norm of the matrix operator A
###
###   ARGUMENTS
###
###      A = Matrix representation of an operator
###
###   RETURNS
###
###      tr_norm = trace norm of oeprator A
###
###   REQUIRES
###
###      numpy          as np
###      scipy.linalg   as sla
###

def trace_norm(A):

    tr_norm = np.trace(sla.sqrtm(np.matmul((A).conj().T,(A))))

    return(tr_norm)

################################################################################
###
###   FUNCTION     hamiltonian_cm_vlasov_hermite_linear_symm(k,alpha,nu,N)
###
################################################################################
###
###   DESCRIPTION
###
###      Vlasov-Hermite linear system implemeted as a quadratic Hamitonian
###      for noninteracting fermions.  Corresponds to SYMMETERIZED amplitudes.
###
###      Returns the Hamiltonian matrix corresponding to the <N> term
###      Hermite expansion of the velocity component in a fully spectral
###      decomposition of the linearized Vlasov equation.  This expansion
###      is associated with the spatial Fourier mode of wavevector <k>
###      with an electric field parameter <alpha> and damping constant
###      <nu>.
###
###      This operator is provided in the SYMMETRIC form (e.g., WITH A CHANGE
###      OF VARIABLES to give a symmetric Hamiltonian).
###
###      This Hamiltonian is non-Hermitian whenever (nu != 0).
###
###      This Hamiltonian is suitable for use with an evolution given by the
###      Liouville-von Neumann equation or as a matrix for use in a conventional
###      solver for the Vlasov equation (it is simply the matrix associated with
###      the classical problem).
###
###   ARGUMENTS
###
###      k      = Fourier mode wavevector
###      alpha  = Electric field parameter
###      nu     = Damping parameter
###      N      = Maximal mode in the Hermite expansion.  This corresponds to
###               the number of Fermionic modes / qubits in the quantum problem.
###
###   RETURNS
###
###      H      = Matrix representation of the linearized Vlasov Hamiltonian.
###
###   REQUIRES
###
###      numpy as np
###

def hamiltonian_cm_vlasov_hermite_linear_sym(k,alpha,nu,N):

    H = 1j*np.zeros((N,N))

    ###  With a change of variable, the electric field has been symmetrically
    ###  wrapped into the coupling between the first and second modes.
    ###
    H[0,1] = k*np.sqrt((1.0+alpha)/2.0)
    H[1,0] = k*np.sqrt((1.0+alpha)/2.0)
    H[1,2] = k

    for j in range(2,N-1):

        # Hopping terms
        H[j,j-1] = k*np.sqrt(j / 2.0)
        H[j,j+1] = k*np.sqrt((j+1) / 2.0)

        # Collisional damping
        H[j,j]   = -1j*nu*j

    H[N-1,N-2] = k*np.sqrt((N-1) / 2.0)
#    H[N-1,N-1] = -1j*nu*(N-1)

    H[N-1,N-2] = k*np.sqrt((N-1) / 2.0)
    H[N-1,N-1] = -1j*nu*(N-1)


    return(H)





################################################################################
###
###   FUNCTION     hamiltonian_cm_vlasov_hermite_linear_asym(k,alpha,nu,N)
###
################################################################################
###
###   DESCRIPTION
###
###      Returns the Hamiltonian matrix corresponding to the <N> term
###      Hermite expansion of the velocity component in a fully spectral
###      decomposition of the linearized Vlasov equation.  This expansion
###      is associated with the spatial Fourier mode of wavevector <k>
###      with an electric field parameter <alpha> and damping constant
###      <nu>.
###
###      This operator is provided in the ASYMMETRIC form (e.g., WITHOUT
###      A SYMMETERIZING CHANGE OF VARIABLES for the electric field term).
###
###      This Hamiltonian is non-Hermitian whenever (alpha !=0) or (nu != 0).
###
###      This Hamiltonian is suitable for use with an evolution given by a
###      direct statevector evolution (e.g. with exponential scaling in N).
###
###   ARGUMENTS
###
###      k      = Fourier mode wavevector
###      alpha  = Electric field parameter
###      nu     = Damping parameter
###      N      = Maximal mode in the Hermite expansion.  This corresponds to
###               the number of Fermionic modes / qubits in the quantum problem.
###
###   RETURNS
###
###      H      = Matrix representation of the linearized Vlasov Hamiltonian.
###
###   REQUIRES
###
###      numpy as np
###

def hamiltonian_wfn_vlasov_hermite_linear_asym(k,alpha,nu,N):

    pjzp = (id+pz)/2.0

    H = 0.0

    for m in range(0,N-1):

        ###  "Coupling" term in spectral expansion
        ###
        cf = (k/2.0)*pf*np.sqrt((m+1)/2)
        H += (multi_qb_op((cf*px,px),(m,m+1),N) +
                                      multi_qb_op((cf*py,py),(m,m+1),N))

        ###  Relaxation due to collisional damping
        ###
        if (m >= 2):
            ci = -1.0j*nu*m
            H += multi_qb_op((ci*pjzp,),(m,),N)

    ###  Electric field term alpha via raising and lowering operators
    ###
    H += (k*alpha/4.0)*multi_qb_op((sigp,sigm),(1,0),N)

    return(H)



def parse_wfn_trajz_pzpj(ts_vec,traj_raw):

    N = int(np.log2(traj_raw[0].shape[0]/2))

    traj_proc = 1j*np.zeros((len(ts_vec),N+1))

    for j, t in enumerate(ts_vec):
        traj_proc[j,0] = t
        TFr = traj_raw[j,0:(2**N)] + 1j*traj_raw[j,2**N:(2*(2**N))]
        traj_proc[j,1:(N+1)] = measure_obs_site(TFr,(id+pz)/2)

    return(traj_proc)



def parse_wfn_traj_measure(ts_vec,traj_raw,op):

    N = int(np.log2(traj_raw[0].shape[0]/2))

    traj_proc = 1j*np.zeros((len(ts_vec),N+1))

    for j, t in enumerate(ts_vec):
        traj_proc[j,0] = t
        TFr = traj_raw[j,0:(2**N)] + 1j*traj_raw[j,2**N:(2*(2**N))]
        traj_proc[j,1:(N+1)] = measure_obs_site(TFr,op)

    return(traj_proc)


################################################################################
###
###   FUNCTION     hamiltonian_cm_vlasov_hermite_linear_symm(k,alpha,nu,N)
###
################################################################################
###
###   DESCRIPTION
###
###      Vlasov-Hermite linear system implemeted as a quadratic Hamitonian
###      for noninteracting fermions.  Corresponds to SYMMETERIZED amplitudes.
###
###      Returns the Hamiltonian matrix corresponding to the <N> term
###      Hermite expansion of the velocity component in a fully spectral
###      decomposition of the linearized Vlasov equation.  This expansion
###      is associated with the spatial Fourier mode of wavevector <k>
###      with an electric field parameter <alpha> and damping constant
###      <nu>.
###
###      This operator is provided in the SYMMETRIC form (e.g., WITH A CHANGE
###      OF VARIABLES to give a symmetric Hamiltonian).
###
###      This Hamiltonian is non-Hermitian whenever (nu != 0).
###
###      This Hamiltonian is suitable for use with an evolution given by the
###      Liouville-von Neumann equation or as a matrix for use in a conventional
###      solver for the Vlasov equation (it is simply the matrix associated with
###      the classical problem).
###
###   ARGUMENTS
###
###      k      = Fourier mode wavevector
###      alpha  = Electric field parameter
###      nu     = Damping parameter
###      N      = Maximal mode in the Hermite expansion.  This corresponds to
###               the number of Fermionic modes / qubits in the quantum problem.
###
###   RETURNS
###
###      H      = Matrix representation of the linearized Vlasov Hamiltonian.
###
###   REQUIRES
###
###      numpy as np
###

def hamiltonian_cm_vlasov_hermite_linear_sym(k,alpha,nu,N):

    H = 1j*np.zeros((N,N))

    ###  With a change of variable, the electric field has been symmetrically
    ###  wrapped into the coupling between the first and second modes.
    ###
    H[0,1] = k*np.sqrt((1.0+alpha)/2.0)
    H[1,0] = k*np.sqrt((1.0+alpha)/2.0)
    H[1,2] = k

    for j in range(2,N-1):

        # Hopping terms
        H[j,j-1] = k*np.sqrt(j / 2.0)
        H[j,j+1] = k*np.sqrt((j+1) / 2.0)

        # Collisional damping
        H[j,j]   = -1j*nu*j

    H[N-1,N-2] = k*np.sqrt((N-1) / 2.0)
#    H[N-1,N-1] = -1j*nu*(N-1)

    H[N-1,N-2] = k*np.sqrt((N-1) / 2.0)
    H[N-1,N-1] = -1j*nu*(N-1)


    return(H)





################################################################################
###
###   FUNCTION     hamiltonian_cm_vlasov_hermite_linear_asym(k,alpha,nu,N)
###
################################################################################
###
###   DESCRIPTION
###
###      Returns the Hamiltonian matrix corresponding to the <N> term
###      Hermite expansion of the velocity component in a fully spectral
###      decomposition of the linearized Vlasov equation.  This expansion
###      is associated with the spatial Fourier mode of wavevector <k>
###      with an electric field parameter <alpha> and damping constant
###      <nu>.
###
###      This operator is provided in the ASYMMETRIC form (e.g., WITHOUT
###      A SYMMETERIZING CHANGE OF VARIABLES for the electric field term).
###
###      This Hamiltonian is non-Hermitian whenever (alpha !=0) or (nu != 0).
###
###      This Hamiltonian is suitable for use with an evolution given by a
###      direct statevector evolution (e.g. with exponential scaling in N).
###
###   ARGUMENTS
###
###      k      = Fourier mode wavevector
###      alpha  = Electric field parameter
###      nu     = Damping parameter
###      N      = Maximal mode in the Hermite expansion.  This corresponds to
###               the number of Fermionic modes / qubits in the quantum problem.
###
###   RETURNS
###
###      H      = Matrix representation of the linearized Vlasov Hamiltonian.
###
###   REQUIRES
###
###      numpy as np
###

def hamiltonian_wfn_vlasov_hermite_linear_asym(k,alpha,nu,N):

    pjzp = (id+pz)/2.0

    H = 0.0

    for m in range(0,N-1):

        ###  "Coupling" term in spectral expansion
        ###
        cf = (k/2.0)*pf*np.sqrt((m+1)/2)
        H += (multi_qb_op((cf*px,px),(m,m+1),N) +
                                      multi_qb_op((cf*py,py),(m,m+1),N))

        ###  Relaxation due to collisional damping
        ###
        if (m >= 2):
            ci = -1.0j*nu*m
            H += multi_qb_op((ci*pjzp,),(m,),N)

    ###  Electric field term alpha via raising and lowering operators
    ###
    H += (k*alpha/4.0)*multi_qb_op((sigp,sigm),(1,0),N)

    return(H)





################################################################################
###
###   FUNCTION     hamiltonian_wfn_vlasov_hermite_linear_sym(k,alpha,nu,N)
###
################################################################################
###
###   DESCRIPTION
###
###      Returns the Hamiltonian matrix corresponding to the <N> term
###      Hermite expansion of the velocity component in a fully spectral
###      decomposition of the linearized Vlasov equation.  This expansion
###      is associated with the spatial Fourier mode of wavevector <k>
###      with an electric field parameter <alpha> and damping constant
###      <nu>.
###
###      This operator is provided in the SYMMETRIC form (e.g., WITH A CHANGE
###      OF VARIABLES to give a symmetric Hamiltonian).
###
###      This Hamiltonian is non-Hermitian whenever (alpha !=0) or (nu != 0).
###
###      This Hamiltonian is suitable for use with an evolution given by a
###      direct statevector evolution (e.g. with exponential scaling in N).
###
###   ARGUMENTS
###
###      k      = Fourier mode wavevector
###      alpha  = Electric field parameter
###      nu     = Damping parameter
###      N      = Maximal mode in the Hermite expansion.  This corresponds to
###               the number of Fermionic modes / qubits in the quantum problem.
###
###   RETURNS
###
###      H      = Matrix representation of the linearized Vlasov Hamiltonian.
###
###   REQUIRES
###
###      numpy as np
###

def hamiltonian_wfn_vlasov_hermite_linear_sym(k,alpha,nu,N):

    H = 0.0

    for m in range(0,N-1):

        ###  "Coupling" term in spectral expansion.  With a change of variable,
        ###  the electric field has been symmetrically wrapped into the zeroth
        ###  term in the expanion.
        ###
        if (m == 0):
            pf = np.sqrt((m+1.0+alpha)/2.0)
        else:
            pf = np.sqrt((m+1.0)/2.0)

        cf = (k/2.0)*pf
        H += (multi_qb_op((cf*px,px),(m,m+1),N) + \
                      multi_qb_op((cf*py,py),(m,m+1),N))

        ###  Relaxation due to collisional damping
        ###
        if (m >= 2):
            ci = -1.0j*nu*m
            H += multi_qb_op((ci*przp,),(m,),N)

    ci = -1.0j*nu*(N-1)
    H += multi_qb_op((ci*przp,),(N-1,),N)


    return(H)





################################################################################
###
###   FUNCTION    hamiltonian_wfn_vlasov_hermite_linear_sym_string(k,alpha,nu,N)
###
################################################################################
###
###   DESCRIPTION
###
###      Returns pauli strings represnting the Hamiltonian for the <N> term
###      Hermite expansion of the velocity component in a fully spectral
###      decomposition of the linearized Vlasov equation.  This expansion
###      is associated with the spatial Fourier mode of wavevector <k>
###      with an electric field parameter <alpha> and damping constant
###      <nu>.
###
###      This operator is provided in the SYMMETRIC form (e.g., WITH A CHANGE
###      OF VARIABLES to give a symmetric Hamiltonian).
###
###      This Hamiltonian is non-Hermitian whenever (alpha !=0) or (nu != 0).
###
###      The notation within the Pauli string reflects the following
###      convention:
###
###      "X"    = Pauli X gate
###      "Y"    = Pauli Y gate
###      "Z"    = Pauli Z gate
###      "X"    = Identity gate
###      "+"    = Pauli Z+ projector (I+Z)/2
###      "-"    = Pauli Z- projector (I-Z)/2
###
###   ARGUMENTS
###
###      k      = Fourier mode wavevector
###      alpha  = Electric field parameter
###      nu     = Damping parameter
###      N      = Maximal mode in the Hermite expansion.  This corresponds to
###               the number of Fermionic modes / qubits in the quantum problem.
###
###   RETURNS
###
###      str_set = A list of tuples corresponding to terms in the Hamiltonian.
###                The first element of the tuple is the Pauli string, while
###                the second is the coefficient for terms in the sring.  The
###                coefficient is a double precision real or complex number,
###                an imaginary term appearing only when the damping is nonzero
###                (i.e., nu != 0).
###
###   REQUIRES
###
###      numpy as np
###

def hamiltonian_wfn_vlasov_hermite_linear_sym_string(k,alpha,nu,N):

    str_set = []

    for m in range(0,N-1):

        ###  "Coupling" term in spectral expansion.  With a change of variable,
        ###  the electric field has been symmetrically wrapped into the zeroth
        ###  term in the expanion.
        ###
        if (m == 0):
            pf = np.sqrt((m+1.0+alpha)/2.0)
        else:
            pf = np.sqrt((m+1.0)/2.0)

        cf = (k/2.0)*pf

        strxx = multi_qb_op_string(("X","X"),(m,m+1),N)[::-1]
        stryy = multi_qb_op_string(("Y","Y"),(m,m+1),N)[::-1]
        str_set.append((strxx,cf))
        str_set.append((stryy,cf))

        ###  Relaxation due to collisional damping
        ###
        if ( (m >= 2) and (nu != 0.0)):
            ci   = -1.0j*nu*m
            strz = multi_qb_op_string(("+",),(m,),N)[::-1]
            str_set.append((strz,ci))

    if (nu != 0.0):
        ci = -1.0j*nu*(N-1)
        strz = multi_qb_op_string(("+",),(N-1,),N)[::-1]
        str_set.append((strz,ci))


    return(str_set)


################################################################################
###
###   FUNCTION     multi_qb_op_string(ops,idxs,N)
###
################################################################################
###
###   DESCRIPTION
###
###      Returns a Pauli string corresponding to a muti-site operator, with
###      names (strings) for these operators passed as a tuple/array in ops and
###      intergers corresponding to the qubits on which they act in the tuple
###      <idxs>.  The product is formed following the order
###      that the qubits are specified in <idxs>.  Here, the least significant
###      qubit is the first qubit in the string (i.e., the leftmost qubit).
###
###   ARGUMENTS
###
###      ops    = tuple/array containing string for operator names
###               (e.g., "X", "Y", "Z", "I", ...)
###      idxs   = indexes of sites acted on by these operators
###      N      = total number of qubits / spins
###
###   RETURNS
###
###      OP = sring corresponding multi-qubit operator
###
###   REQUIRES
###
###      numpy as np
###

def multi_qb_op_string(ops,idxs,N):

    op_str = ""

    for j in range(N):
        if (j in idxs):
            op_str += str(ops[idxs.index(j)])
        else:
            op_str += "I"

    return(op_str)





################################################################################
###
###   FUNCTION     wfn_product_state_single (state,N)
###
################################################################################
###
###   DESCRIPTION
###
###      Returns the product state formed by taking the tensor product of the
###      state <state> with times <N> times.
###
###   ARGUMENTS
###
###      state = matrix representing the state used to construct product state
###      N     = number of copies of <state> in the product.
###
###   RETURNS
###
###      wnf   = statevector corresponding to the product state
###
###   REQUIRES
###
###      numpy as np
###

def wfn_product_state_single(state,N):

    # wfn = 1.0 + 1j*0.0
    #
    # for j in range(N):
    #     wfn = np.kron(wfn,state)

    wfn = multikron([state for j in range(N)])

    return wfn





################################################################################
###
###   FUNCTION     wfn_product_state(states)
###
################################################################################
###
###   DESCRIPTION
###
###      Returns the product state formed by taking the tensor product of the
###      states in the vector <states>.  The state is formed as:
###
###        |s_0> \otimes |s_1> \otimes |s_2> \otimes ... \otimes |s_N>
###
###      where |s_0> is the least signficant state in <states>.
###
###   ARGUMENTS
###
###      states = vector with entries corresponding to the matrix representation
###               of states used to form the product state.
###
###   RETURNS
###
###      wnf   = statevector corresponding to the product state
###
###   REQUIRES
###
###      numpy as np
###

def wfn_product_state(states):

    wfn = multikron(states)

    return wfn





################################################################################
###
###   FUNCTION     lvn_prop (CMv,t,H,N,dt)
###
################################################################################
###
###   DESCRIPTION
###
###      Returns the infintesimal time evolution of a density / correlation
###      matrix <rho> that is generated by a Hamiltonian <H> for use in solving
###      the Liouville-von Neumann equation.
###
###   ARGUMENTS
###
###      WFv = Vectorized wavefunction, with real components first and imaginary
###            components second.
###      t   = Current / absolute time
###      H   = Hamiltonian (as a real or complex matrix)
###      N   = Number of spins / qubits in Hamiltonian
###      dt  = Integrator timestep
###
###   RETURNS
###
###      WFp_dt = Infintesimal time evolution generated by the Hamiltonian
###               opeartor
###
###   REQUIRES
###
###      numpy as np
###

def prop_lvn(Rv,t,H):

    # assumes that the only non-Hermitian part comes from the diagonal entries
    # which are imaginary.

    R = devectorize_cmat(Rv)

#    R_dt = -1j*(np.matmul(R,H) - np.matmul(H,R))

    # R_dt = (-1j*(np.matmul(R,H.real) - np.matmul(H.real,R)) +
    #                (np.matmul(R,H.imag) + np.matmul(H.imag,R)))

    R_dt = (-1j*(np.matmul(H.real,R) - np.matmul(R,H.real)) +
                   (np.matmul(H.imag,R) + np.matmul(R,H.imag)))

    Rv_dt = vectorize_cmat(R_dt)

    return(Rv_dt)





################################################################################
###
###   FUNCTION     vectorize_cmat(M)
###
################################################################################
###
###   DESCRIPTION
###
###      Vectorizes an N x N dimensional array of complex numbers.  The real
###      and imaginary parts are individually reshaped into 1-d real arrays of
###      N^2 elements, with elements placed as consecutive rows (numpy stores
###      arrays in row-major order).  The final 2*N^2 element, real-valued array
###      array is the concatenation of the real and imaginary parts, with the
###      real parts first.  More precisely:
###
###       [[a_{1,1} + i*b_{1,1}, a_{1,2} + i*b_{1,2}, ..., a_{1,N} + i*b_{1,N}],
###        [a_{2,1} + i*b_{2,1}, a_{2,2} + i*b_{2,2}, ..., a_{1,N} + i*b_{1,N}],
###          . . .
###        [a_{N,1} + i*b_{N,1}, a_{N,2} + i*b_{N,2}, ..., a_{N,N} + i*b_{N,N}]]
###
###          is vectorized as:
###
###        [a_{1,1}, a_{1,2}, ..., a_{1,N}, a_{2,1}, a_{2,2}, a_{2,N}, ...
###          a_{N,N}, b_{1,1}, b_{1,2}, ..., b_{1,N}, b_{2,1}, b_{2,2},
###           b_{2,N}, ..., a_{N,N}]
###
###   ARGUMENTS
###
###      M = complex-valued, NxN array to be vectorized
###
###   RETURNS
###
###      A   = Real-valued vectorization of M
###
###   REQUIRES
###
###      numpy as np
###

def vectorize_cmat(M):

    N = M.shape[0]

    M_vec = np.append(M.reshape(N**2).real,M.reshape(N**2).imag)

    return(M_vec)





################################################################################
###
###   FUNCTION     devectorize_cmat(M)
###
################################################################################
###
###   DESCRIPTION
###
###      Reverses the vectorization performed by vectorize_cmat
###
###   ARGUMENTS
###
###      M = Real-valued vectorization of an NxN complex matrix
###
###   RETURNS
###
###      A = NxN complex matrix
###
###   REQUIRES
###
###      numpy as np
###

def devectorize_cmat(M_vec):

    N = int(np.sqrt(M_vec.shape[0]/2))

    # The interpreter seems to handle this efficiently... though I doubt that
    # it scales well unless is is really smart.
    M = M_vec.reshape(2,N,N)[0,:,:] + 1j*M_vec.reshape(2,N,N)[1,:,:]

    return(M)





################################################################################
###
###   FUNCTION     parse_cm_traj(ts_vec,traj_raw)
###
################################################################################
###
###   DESCRIPTION
###
###      Extracts diagonal ("occpancy") terms of the correlation matrix in a
###      time series from a Liouville-von Neumann simulation.
###
###   ARGUMENTS
###
###      ts_vec   = Vector containing timesteps from the trajectory
###      traj_raw = Raw CM trajectory from Integrator
###
###   RETURNS
###
###      traj_proc = Processed trajectory.  The first element of each row is
###                  the timestep, while subsequent entries are the diagonal
###                  terms.
###
###   REQUIRES
###
###      numpy as np
###

def parse_cm_traj(ts_vec,traj_raw):

    N = int(np.sqrt(traj_raw[0].shape[0]/2))

    traj_proc = 1j*np.zeros((len(ts_vec),N+1))

    for j, t in enumerate(ts_vec):

        traj_proc[j,0] = t
        TFr = traj_raw[j].reshape(2,N,N)
        for i in range(N):
            traj_proc[j,(i+1)] = TFr[0,i,i] + 1j*TFr[1,i,i]


    return(traj_proc)






################################################################################
###
###   FUNCTION     pprint(A)
###
################################################################################
###
###   DESCRIPTION
###
###      "Pretty print" the matrix A for debugging.
###
###   ARGUMENTS
###
###      A        = Matrix
###
###   RETURNS
###
###      NONE
###
###   REQUIRES
###
###      numpy as np
###

def pprint(A):

    print("\n")
    print(np.array2string(A, formatter={'complex_kind': '{0:.2f}'.format}))



################################################################################
###
###   FUNCTION     fcr(m,N)
###
################################################################################
###
###   DESCRIPTION
###
###      Fermionic creation operator acting on site m in a one dimensional array
###      of N qubits.  The operator is defined using the Jordan-Winger
###      transformation.
###
###   ARGUMENTS
###
###      m     =  site on which creation operator is acting
###      N     =  total number of qubits
###
###   RETURNS
###
###      OP = numpy array corresponding to the creation operator
###
###   REQUIRES
###
###      numpy as np
###

def fcr(m,N):

    opr = 1

    for j in range(m):
        opr = np.kron(opr,pz)

    opr = np.kron(opr,sigp)

    for j in range(m+1,N):
        opr = np.kron(opr,id)

    return(opr)





################################################################################
###
###   FUNCTION     fan(m,N)
###
################################################################################
###
###   DESCRIPTION
###
###      Fermionic annihilation operator acting on site m in a one dimensional
###      array of N qubits.  The operator is defined using the Jordan-Winger
###      transformation.
###
###   ARGUMENTS
###
###      m     =  site on which annihilation operator is acting
###      N     =  total number of qubits
###
###   RETURNS
###
###      OP = numpy array corresponding to the annihilation operator
###
###   REQUIRES
###
###      numpy as np
###

def fan(m,N):

    opr = (fcr(m,N)).conj().T

    return(opr)





################################################################################
###
###   FUNCTION     wfn_to_cm(wfn)
###
################################################################################
###
###   DESCRIPTION
###
###      Calculates the Fermionic correlation matrix <c^dagger_i c_j>
###      corresponding to the spin wavefunction wfn via the Jordan-Wigner
###      transformation.
###
###   ARGUMENTS
###
###      wfn     =  target wavefunction expressed in a Pauli basis
###
###   RETURNS
###
###      CM      =  correlation matrix with C_{ij} = <c^dagger_i c_j>
###
###   REQUIRES
###
###      numpy as np
###

def wfn_to_cm(wfn):

    N = int(np.log2(len(wfn)))

    CM = 1.0j*np.zeros((N,N))

    for j in range(N):
        for k in range(N):
            cm_op   = np.matmul(fcr(j,N),fan(k,N))
            CM[j,k] = np.matmul(np.conjugate(wfn.T),np.matmul(cm_op,wfn))

    return(CM)


################################################################################
###
###   FUNCTION     fcr(m,N)
###
################################################################################
###
###   DESCRIPTION
###
###      Defines the N-fermion vacuum state via the Joran-Wigner transformation.
###      By convention, we adopt the spin-down product state as the vacuum.
###
###   ARGUMENTS
###
###      N     =  total number of qubits
###
###   RETURNS
###
###      OP = numpy array corresponding to the N-fermion vacuum state
###
###   REQUIRES
###
###      numpy as np
###

def state_vacuum(N):

    vac_state = wfn_product_state([sz_d for j in range(N)])

    return(vac_state)





################################################################################
###
###   FUNCTION     state_multifermi_idx(occ_idxs,N)
###
################################################################################
###
###   DESCRIPTION
###
###      Returns the state containg m fermions in the sites specified by
###      the indices in occ_idxs.  This is defined by the action of creation
###      operators on the N-fermion vacuum state:
###
###         $c^dag_{i_1} c^dag_{i_2} \dots c^dag_{i_m} |0^N\rangle$
###
###      The states are defined in their spin/qubit analogue via the Jordan-
###      Winger transformation.
###
###   ARGUMENTS
###
###      occ_idxs     =  tuple containing indices of fermions
###      N            =  total number of fermionic sites
###
###   RETURNS
###
###      OP = numpy array corresponding to the state with m fermions in N sites.
###
###   REQUIRES
###
###      numpy as np
###

def state_multifermi_idx(occ_idxs,N):

    wfn = state_vacuum(N)

    for idx in occ_idxs:
        wfn = np.matmul(fcr(idx,N),wfn)

    return(wfn)




################################################################################
###
###   FUNCTION     prop_vlasov_classical(g_v,t,A)
###
################################################################################
###
###   DESCRIPTION
###
###      Classical time derivative of the Vlasov-Hermite expansion, defined
###      for use with ODEPACK solvers.
###
###   ARGUMENTS
###
###      g_v   =  vectorized set of of amplitudes
###      t     =  Integration time
###      A     =  Evolution operator for the Vlasov-Hermite system
###
###   RETURNS
###
###      g_dt_v = vectorized numpy array containing the time derivative
###
###   REQUIRES
###
###      numpy as np
###

def prop_vlasov_classical(g_v,t,A):

    N = int(g_v.shape[0]/2)

    g = g_v[0:N] + 1j*g_v[N:(2*N)]

    g_dt = np.matmul(A,g)

    g_dt_v = np.append(g_dt.real,g_dt.imag)

    return(g_dt_v)


################################################################################
###
###   FUNCTION     prop_vlasov_exponential(A,g0,dt,Ns)
###
################################################################################
###
###   DESCRIPTION
###
###      Direct time propagation of the Vlasov-Hermite system.
###
###      Applies the exponential propagtor exp[A*dt] to the state g0 for Ns
###      repetitions.  This corresponds to a time evolution up to t = Ns*dt,
###      where dt is the timestep.
###
###   ARGUMENTS
###
###      A     =  Evolution operator for the Vlasov-Hermite system
###      g0    =  Initial set of amplutides
###      dt    =  Timestep
###      Ns    =  Number of timesteps
###
###   RETURNS
###
###      gt_arr = numpy array containing the amplitudes at each timestep.
###
###   REQUIRES
###
###      numpy as np
###      scipy.linalg as sla
###

def prop_vlasov_exponential(A,g0,dt,Ns):

    N = A.shape[0]

    gt_arr = 0j*np.zeros((Ns+2,N))

    gt = g0

    for j in range(Ns):
        gt = np.matmul(sla.expm(A*dt),gt)
        gt_arr[j+1,0:N] = gt

    return(gt_arr)
