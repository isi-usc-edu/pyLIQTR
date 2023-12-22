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
###  VLASOV_SPECTRAL - Spectral representation of the Vlasov-Poisson system  ###
###                                                                          ###
################################################################################



import  numpy            as  np
import  scipy.sparse     as  sps



def vlasov_spectral_system(N_x,N_v,L_x,epsilon,gamma,sparse=False):

    ## there is some issue with the sparse part hanging, so
    ## we just convert at the end

    # if sparse:
    #     F_1     =  vlasov_coeffs_linear_1d_sparse(N_x,N_v,L_x,epsilon,gamma)
    #     F_2     =  vlasov_coeffs_nonlinear_1d_sparse(N_x,N_v,L_x,epsilon,gamma)
    # else:
    #     F_1     =  vlasov_coeffs_linear_1d(N_x,N_v,L_x,epsilon,gamma)
    #     F_2     =  vlasov_coeffs_nonlinear_1d(N_x,N_v,L_x,epsilon,gamma)

    F_1     =  vlasov_coeffs_linear_1d(N_x,N_v,L_x,epsilon,gamma)
    F_2     =  vlasov_coeffs_nonlinear_1d(N_x,N_v,L_x,epsilon,gamma)

    if sparse:
        coeff_set = [sps.coo_matrix(F_1), sps.coo_matrix(F_2)]
    else:
        coeff_set = [F_1, F_2]

    return(coeff_set)



################################################################################
###
###   FUNCTION     vlasov_spectral_linear(k,alpha,nu,N)
###
################################################################################
###
###   DESCRIPTION
###
###      Returns a matrix encoding velocity components for a dual-spectral 
###      (Fourier-Hermite) representation of the 1d Vlasov-Poisson system. 
###
###   ARGUMENTS
###
###      k      =  Fourier mode wavevector
###      alpha  =  Electric field parameter
###      nu     =  Collisional damping parameter
###      N      =  Maximal mode in the Hermite expansion.  
###
###   RETURNS
###
###      H       =  Matrix for dual spectral representation
###
###
###   REQUIRES
###
###      numpy as np
###

def vlasov_spectral_linear(k,alpha,nu,N):

    H = 1j*np.zeros((N,N))

    ###  With a change of variable, the electric field has been symmetrically
    ###  wrapped into the coupling between the first and second modes.
    ###
    H[0,1] = k*np.sqrt((1.0+alpha)/2.0)
    H[1,0] = k*np.sqrt((1.0+alpha)/2.0)
    H[1,2] = k

    for j in range(2,N-1):

        # "Hopping" terms
        H[j,j-1] = k*np.sqrt(j / 2.0)
        H[j,j+1] = k*np.sqrt((j+1) / 2.0)

        # Collisional damping
        H[j,j]   = -1j*nu*j

    H[N-1,N-2] = k*np.sqrt((N-1) / 2.0)
    H[N-1,N-1] = -1j*nu*(N-1)


    return(H)





################################################################################
###
###   FUNCTION     vlasov_spectral_linear_sparse(k,alpha,nu,N)
###
################################################################################
###
###   DESCRIPTION
###
###      Returns a matrix encoding velocity components for a dual-spectral 
###      (Fourier-Hermite) representation of the 1d Vlasov-Poisson system. 
###      Sparse matrix arithmetic variant.
###
###   ARGUMENTS
###
###      k      =  Fourier mode wavevector
###      alpha  =  Electric field parameter
###      nu     =  Collisional damping parameter
###      N      =  Maximal mode in the Hermite expansion.  
###
###   RETURNS
###
###      H       =  Matrix for dual spectral representation
###
###
###   REQUIRES
###
###      numpy as np
### 

def vlasov_spectral_linear_sparse(k,alpha,nu,N):

    H = 1j*sps.csr_matrix((N,N))

    ###  With a change of variable, the electric field has been symmetrically
    ###  wrapped into the coupling between the first and second modes.
    ###
    H[0,1] = k*np.sqrt((1.0+alpha)/2.0)
    H[1,0] = k*np.sqrt((1.0+alpha)/2.0)
    H[1,2] = k

    for j in range(2,N-1):

        # "Hopping" terms
        H[j,j-1] = k*np.sqrt(j / 2.0)
        H[j,j+1] = k*np.sqrt((j+1) / 2.0)

        # Collisional damping
        H[j,j]   = -1j*nu*j

    H[N-1,N-2] = k*np.sqrt((N-1) / 2.0)
    H[N-1,N-1] = -1j*nu*(N-1)


    return(H)







def get_vector_I(lmax,gamma,eps):

    N_l = lmax+1

    ## Use iterative approach in lieu of exact 
    ## expresson to avoid numerical issues

    I_0   =  np.sqrt(np.pi)/eps
    sclf  =  lambda l : np.sqrt((l+1)/(l+2))*(gamma**2 - 1.0)

    I_vec = np.zeros(N_l)

    I_vec[0]  =  I_0

    for ell in range(1,N_l):
        if ( (ell % 2) == 0 ):
            I_vec[ell] = sclf(ell)*I_vec[ell-2]
        else:
            I_vec[ell] = 0.0

    return(I_vec)



def vlasov_coeffs_linear_1d(N_x,N_v,L_x,eps,gamma):

    N_x_tot  =  2*N_x + 1
    N_tot    =  N_x_tot * N_v

    coeffs   =  np.zeros((N_tot,N_tot))

    C        =  np.sqrt(2.0)*np.pi/(gamma*eps*L_x)

    for k in range(N_x_tot):
        for l in range(N_v):

            gidx = k*N_v + l

            if (l < N_v-1):    
                coeffs[gidx,gidx+1] = C*(k-N_x)*np.sqrt(l+1)
            
            if (l >= 1):
                coeffs[gidx,gidx-1] = C*(k-N_x)*np.sqrt(l)

    return(coeffs)



def vlasov_coeffs_linear_1d_sparse(N_x,N_v,L_x,eps,gamma):

    N_x_tot  =  2*N_x + 1
    N_tot    =  N_x_tot * N_v

    coeffs   =  sps.csr_matrix((N_tot,N_tot))

    C        =  np.sqrt(2.0)*np.pi/(gamma*eps*L_x)

    for k in range(N_x_tot):
        for l in range(N_v):

            gidx = k*N_v + l

            if (l < N_v-1):    
                coeffs[gidx,gidx+1] = C*(k-N_x)*np.sqrt(l+1)
            
            if (l >= 1):
                coeffs[gidx,gidx-1] = C*(k-N_x)*np.sqrt(l)

    return(coeffs)




def vlasov_coeffs_nonlinear_1d(N_x,N_v,L_x,eps,gamma):

    N_x_tot  =  2*N_x + 1
    N_tot    =  N_x_tot * N_v

    coeffs   =  np.zeros((N_tot,N_tot**2))

    I_vec    =  get_vector_I(N_v-1,gamma,eps)

    C_1      =  np.sqrt(2)*eps/gamma
    C_2      =  C_1*(gamma**2 - 1.0)

    for k in range(N_x_tot):
        for l in range(N_v):

            gidx_1 = k*N_v + l

            for j in range(N_x_tot):
                for m in range(N_v):

                    if ( (l+1) <= (N_v-1) ):

                        gidx_2 = N_tot*(j*N_v + (l+1)) + ((k-j)*N_v + m)

                        if ( (k-j) != 0 ):
                            coeffs[gidx_1,gidx_2] += C_2*I_vec[m]*L_x*np.sqrt(l+1)*(2*np.pi)/(k-j)

                    if ( (l-1) >= 0 ):

                        gidx_2 = N_tot*(j*N_v + (l-1)) + ((k-j)*N_v + m)

                        if ( (k-j) != 0):
                            coeffs[gidx_1,gidx_2] += C_1*I_vec[m]*L_x*np.sqrt(l)*(2*np.pi)/(k-j)

    return(-1.0j*coeffs)





def vlasov_coeffs_nonlinear_1d_sparse(N_x,N_v,L_x,eps,gamma):

    N_x_tot  =  2*N_x + 1
    N_tot    =  N_x_tot * N_v

    coeffs   =  sps.csr_matrix((N_tot,N_tot**2))

    I_vec    =  get_vector_I(N_v-1,gamma,eps)

    C_1      =  np.sqrt(2)*eps/gamma
    C_2      =  C_1*(gamma**2 - 1.0)

    for k in range(N_x_tot):
        for l in range(N_v):

            gidx_1 = k*N_v + l

            for j in range(N_x_tot):
                for m in range(N_v):

                    if ( (l+1) <= (N_v-1) ):

                        gidx_2 = N_tot*(j*N_v + (l+1)) + ((k-j)*N_v + m)

                        if ( (k-j) != 0 ):
                            coeffs[gidx_1,gidx_2] += C_2*I_vec[m]*L_x*np.sqrt(l+1)*(2*np.pi)/(k-j)

                    if ( (l-1) >= 0 ):

                        gidx_2 = N_tot*(j*N_v + (l-1)) + ((k-j)*N_v + m)

                        if ( (k-j) != 0):
                            coeffs[gidx_1,gidx_2] += C_1*I_vec[m]*L_x*np.sqrt(l)*(2*np.pi)/(k-j)

    return(-1.0j*coeffs)