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
###       HAMILTONIANS - Generate Hamiltonians as an LCU via Pauli Strings   ###
###                                                                          ###
################################################################################


import numpy             as  np
import operator_strings  as  os



## hamiltonian_fh_2d_strings
## ---------------------------------------
##
##  Return Pauli strings for the 2d 
##  Fermi-Hubbard Hamiltonian as an
##  lcu_strings object
##
##     N_x  =  x dimension of lattice
##     N_y  =  y dimension of lattice
##     J    =  Hopping parameter
##     U    =  Density-density interaction
##             strength between spin states
##     mu   =  Chemical potential
##     hz   =  External z-field strength
##

def hamiltonian_fh_2d_strings(N_x=4,N_y=4,J=1.0,U=0.0,mu=0.0,hz=0.0):

    ### TO-DO: Set this up to accept arrays of
    ###        site-dependent parameters

    N_tot   = 2*N_x * N_y
    N_th    = int(N_tot/2)

    Jcr = -J/2
    Jcc =  J/2 * ((-1)**N_x)

    lcu_str = os.lcu_strings(N_tot)


    ###########################################
    ###
    ###   In-row hopping 
    ###
    ###########################################

    ###  Spin-up sublattice hopping
    ### 

    for n in range(0,N_y):
        nx = 2*N_x*n

        for m in range(0,N_x-1):
            nxm = nx + 2*m

            lcu_str.append((nxm, nxm+1, nxm+2),'XZX',Jcr)
            lcu_str.append((nxm, nxm+1, nxm+2),'YZY',Jcr)


    ###  Spin-down sublattice hopping
    ###

    for n in range(0,N_y):
        nx = 2*N_x*n

        for m in range(0,N_x-1):
            nxm = nx + 2*m

            lcu_str.append((nxm+1, nxm+2, nxm+3),'XZX',Jcr)
            lcu_str.append((nxm+1, nxm+2, nxm+3),'YZY',Jcr)



    ###########################################
    ###
    ###   In-column hopping 
    ###
    ###########################################


    ###  Spin-up sublattice hopping
    ### 

    xx_str = 'X'+('Z'*(2*N_x-1))+'X'
    yy_str = 'Y'+('Z'*(2*N_x-1))+'Y'


    for m in range(0,N_th-N_x):

        idx_set = np.arange(2*m,2*m+2*N_x+1,1).tolist()
        
        lcu_str.append(idx_set,xx_str,Jcc)
        lcu_str.append(idx_set,yy_str,Jcc)


    ###  Spin-down sublattice hopping
    ### 
    for m in range(0,N_th-N_x):

        idx_set = np.arange(2*m+1,2*m+2*N_x+2,1).tolist()
        
        lcu_str.append(idx_set,xx_str,Jcc)
        lcu_str.append(idx_set,yy_str,Jcc)


    ###########################################
    ###
    ###   Many-Body Terms
    ###
    ###########################################

    ###  Spin-spin coupling
    ###
    for m in range(0,N_th):
        lcu_str.append((2*m,2*m+1),'++',U)


    ###  Chemical potential
    ###

    if (mu != 0.0):
        for m in range(0,N_tot):
            lcu_str.append((m,),'+',-mu)


    ###  External magnetic field
    ###

    if (hz != 0.0):
        for m in range(N_th):
            lcu_str.append((2*m,),   '+',  hz/2)
            lcu_str.append((2*m+1,), '+', -hz/2)

    return(lcu_str)



## hamiltonian_heisenberg_2d_strings
## ---------------------------------------
##
##  Return Pauli strings for the 2d 
##  Heisenberg Hamiltonian as an
##  lcu_strings object
##
##     N_x  =  x dimension of lattice
##     N_y  =  y dimension of lattice
##     J_x  =  Nearest-neighbor XX coupling
##             strength 
##     J_y  =  Nearest-neighbor YY coupling
##             strength 
##     J_z  =  Nearest-neighbor ZZ coupling
##             strength 
##     h_x  =  External X field strength
##     h_y  =  External Y field strength
##     h_z  =  External Z field strength
##

def hamiltonian_heisenberg_2d_strings(N_x,N_y,J_x=1.0,J_y=1.0,J_z=1.0,h_x=1.0,h_y=1.0,h_z=1.0):

    ### Set this up to handle input arrays

    N_tot   = N_x * N_y

    lcu_str = os.lcu_strings(N_tot)

    ###  In-chain coupling (row)
    ### 
    for n in range(0,N_y):
        nx = N_x*n

        for m in range(0,N_x-1):
            nxm = nx + m

            if (J_x != 0.0):
                lcu_str.append((nxm, nxm+1),'XX',J_x)
            if (J_y != 0.0):
                lcu_str.append((nxm, nxm+1),'YY',J_y)
            if (J_z != 0.0):
                lcu_str.append((nxm, nxm+1),'ZZ',J_z)


    ###  Cross-chain coupling (column)
    ### 
    for m in range(0,N_tot-N_x):
        
        if (J_x != 0.0):
            lcu_str.append((m,m+N_x),'XX',J_x)
        if (J_y != 0.0):
            lcu_str.append((m,m+N_x),'YY',J_y)
        if (J_z != 0.0):
            lcu_str.append((m,m+N_x),'ZZ',J_z)


    ###  Local fields
    ###
    for m in range(0,N_tot):
        
        if (h_x != 0.0):
            lcu_str.append((m,),'X',h_x)
        if (h_y != 0.0):
            lcu_str.append((m,),'Y',h_y)
        if (h_z != 0.0):
            lcu_str.append((m,),'Z',h_z)
        

    return(lcu_str)