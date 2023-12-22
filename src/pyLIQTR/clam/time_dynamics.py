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
###    TIME DYNAMICS  - Simulating the time-evolution of quantum states      ###
###                                                                          ###
################################################################################



import numpy as np



######                            #############################################
######  Time-derivatives          #############################################
######                            #############################################


def cm_prop(CM_p,t,H,dummy):
    
    N_tot = H.shape[0]
    
    CM =    CM_p.reshape(2,N_tot,N_tot)[0,:,:] + \
         1j*CM_p.reshape(2,N_tot,N_tot)[1,:,:]
    
    CM_dt = 1j*(H @ CM - CM @ H)
    
    CM_dt = np.append(CM_dt.reshape(N_tot**2).real,CM_dt.reshape(N_tot**2).imag)
    
    return(CM_dt)



def cm_simulate(H,CM_0,t_max,dt,atol=1.0e-16):
    
    N_tot = H.shape[0]
    
    tvec  = np.arange(0,t_max+dt,dt)
        
    CM_p  = vectorize_cmat(CM_0)
    
    traj = spi.odeint(prop_lvn,CM_p,tvec,args=(H,),atol=atol)
    
    return(traj)
    



##
##  Modify for non-Hermitian density matrix
##


def prop_lvn(Rv,t,H):

    # assumes that the only non-Hermitian part comes from the diagonal entries
    # which are imaginary.

    R = devectorize_cmat(Rv)

    R_dt = (-1j*(np.matmul(H.real,R) - np.matmul(R,H.real)) +
                   (np.matmul(H.imag,R) + np.matmul(R,H.imag)))

    Rv_dt = vectorize_cmat(R_dt)

    return(Rv_dt)    




















################################################################################
###
###   FUNCTION   parse_wfn_traj(ts_vec,traj_raw,op,indices=None,vectorized=True)
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
def parse_wfn_traj_op(traj_raw,op,indices=None,vectorized=False):

    traj_proc = calc_traj_op_wfn(traj_raw,op,indices=indices,vectorized=vectorized)
    return(traj_proc)

### Add vectorization flag

def calc_traj_op_wfn(traj_raw,op,indices=None,vectorized=False):

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

    N_ts = len(traj_raw)

    # traj_proc = 1j*np.zeros((N_ops,N_ts,N+1))

    # add code to handle evalation on select site indices


    if  (D_op == 1):
        traj_proc = 1j*np.zeros((N_ts,N_qb))
    elif (D_op == 2):
        traj_proc = 1j*np.zeros((N_ts,N_qb-1))
    elif (D_op == N_qb):
        traj_proc = 1j*np.zeros(N_ts)
    else:
        print(" ERROR: parse_wfn_traj only handles 1, 2, and N qubit operators.\n")
        print("        Other operators shoudl be passed as an N-qubit operator after \n")
        print("        padding with the identity using multi_qb_op.\n")
        return()


    for j in range(N_ts):

        if (vectorized == True):
            TFr = traj_raw[j,0:(2**N_qb)] + 1j*traj_raw[j,2**N_qb:(2*(2**N_qb))]
        else:
            TFr = traj_raw[j][:]

        if (D_op == 1):
            traj_proc[j,0:N_qb]       = measure_obs_site(TFr,op)
        elif (D_op == 2):
            traj_proc[j,0:(N_qb-1)]   = measure_obs_site(TFr,op)
        elif (D_op == N_qb):
            traj_proc[j]              = TFr.conj().T @ op @ TFr

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

    WF = vs.devectorize_cvec(WFv)

    WF_dt = -1j*np.matmul(H,WF)

    WFv_dt = vectorize_cvec(WF_dt)

    return(WFv_dt)






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

    wfn = mla.mk(states)

    return wfn





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

    # really inefficient. should save measurement operators.

    N       = int(np.log2(wfn.shape[0]))
    obs_val = 1j*np.zeros(N)

    for j in range(N):
        mop         = qo.multi_qb_op([obs_op],[j],N)
        obs_val[j]  = wfn.conj().T @ mop @ wfn

    return(obs_val)















