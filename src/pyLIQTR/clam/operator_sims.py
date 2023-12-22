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
###    OPERATOR_SIMS - Operator-level simulations for quantum algorithms     ###
###                                                                          ###
################################################################################


##
##  TODO: UPDATE TO INCLUDE NON-SCALAR ARGUMENTSn
##


import  numpy  as  np
import  copy   as  copy



######                    #####################################################
######  QSP / QET/ QSVT   #####################################################
######                    #####################################################



## phi_Wx_to_O(phiset)
## -------------------------------------------------------------
##
## Convert Wx convention phase angles <phiset> to O convention 
## for QSP.
##

def phi_Wx_to_O(phiset):

    N           =  len(phiset)
    phiset_new  =  copy.deepcopy(phiset)

    phiset_new[0]    -=  np.pi/4
    phiset_new[N-1]  +=  np.pi/4

    return(phiset_new)





## phi_Wx_to_qsvt(phiset), phi_Wx_to_R(phiset)
## -------------------------------------------------------------
##
## Convert Wx convention phase angles <phiset> to R convention 
## used by quantum eigenvalue transform (QET) and the 
## quantum singular value transform (QSVT)
##

def phi_Wx_to_qsvt(phiset):

    N           =  len(phiset)
    phiset_new  =  copy.deepcopy(phiset)

    phiset_new[0]    +=  np.pi/4
    phiset_new[N-1]  +=  np.pi/4

    for n in range(1,N-1):
        phiset_new[n] += np.pi/2

    return(phiset_new)



def phi_Wx_to_R(phiset):
    return(phi_Wx_to_qsvt(phiset))





## get_qsp_signal_op(signal_convention)
## -------------------------------------------------------------
##
## Get a function generating the signal operator for QSP/QET/QSVT
## following the phase angle convention <signal_convention>
## (valid arguments: {"Wx","Wz","R"})
##
## The returned function sig_op(a) computes the signal operator 
## for a scalar argument <a>.
##

def get_qsp_signal_op(signal_convention):

    if signal_convention == "Wx":
        def sig_op(a): return np.array(
            [[a, 1j * np.sqrt(1 - a**2)],
             [1j * np.sqrt(1 - a**2), a]])
        
    elif signal_convention == "R":
        def sig_op(a): return np.array(
            [[a, -np.sqrt(1 - a**2)],
             [np.sqrt(1 - a**2), a]] )

        # def sig_op(a): return np.array(
        #     [[a, np.sqrt(1 - a**2)],
        #      [np.sqrt(1 - a**2), -a]])

    elif signal_convention == "Wz":
        H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)

        def sig_op(a): return H @ np.array(
            [[a, 1j * np.sqrt(1 - a**2)],
             [1j * np.sqrt(1 - a**2), a]]) @ H

    return sig_op





## get_qsp_signal_op(signal_convention)
## -------------------------------------------------------------
##
## Get a function generating the phase angle rotation operator
## for QSP/QET/QSVT following the phase angle convention 
## specified by <signal_convention> ( arguments: {"Wx","Wz","R"})
##
## The returned function sig_op(a) computes the signal operator 
## for a scalar argument <a>.
##

def get_qsp_phase_op(signal_convention):

    if signal_convention == "Wx":
        def qsp_op(phi): return np.array(
            [[np.exp(1j * phi), 0.],
             [0., np.exp(-1j * phi)]])
        
    elif signal_convention == "R":
        def qsp_op(phi): return np.array(
            [[np.exp(1j * phi), 0.],
             [0., np.exp(-1j * phi)]])

    elif signal_convention == "Wz":
        H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)

        def qsp_op(phi): return H @ np.array(
            [[np.exp(1j * phi), 0.],
             [0., np.exp(-1j * phi)]]) @ H

    return qsp_op





## get_qsp_signal_op(signal_convention)
## -------------------------------------------------------------
##
## Return the signal state used to extract the target subspace
## from a QSP/QET/QSVT sequence. The phase angle convention is
## specified by <signal_convention> (arguments: {"Wx","Wz","R"})
##

def get_qsp_signal_state(signal_convention):
    
    if signal_convention == "Wx":
        p_state = np.array([[1.], [1.]]) / np.sqrt(2)
    elif signal_convention == "R":
        p_state = np.array([[1.], [1.]]) / np.sqrt(2)
    elif signal_convention == "Wz":
        p_state = np.array([[1.], [0.]])
    
    return p_state





## get_qsp_signal_op(signal_convention)
## -------------------------------------------------------------
##
## Return the signal state used to extract the target subspace
## from a QSP/QET/QSVT sequence. The phase angle convention is
## specified by <signal_convention> (arguments: {"Wx","Wz","R"})
##

def qsp_response(x_set,phiset,signal_convention):

    pmats      =  []
    pdat       =  []

    if ( (not isinstance(x_set,np.ndarray)) and
         (not isinstance(x_set,list))    and
         (not isinstance(x_set,tuplelist)) ):
        
        x_set = [x_set,]

    signal_op  =  get_qsp_signal_op(signal_convention)
    phase_op   =  get_qsp_phase_op(signal_convention)

    for phi in phiset:
        pmats.append(phase_op(phi))

    for a in x_set:
        W = signal_op(a)
        U = pmats[0]
        for pm in pmats[1:]:
            U = U @ W @ pm
        pdat.append( U )

    pdat = np.array(pdat, dtype=np.complex128)

    return(pdat)



## qsvt_response(signal_convention)
## -------------------------------------------------------------
##
## Return the QSP/QET/QSVT response function (sequence) for a 
## series of signal operators parameterized by the scalar <a> 
## and the phase angle sequence <phiset> with signal convention
## <signal_convention>.  A reflection is applied within the 
## sequence by default <reflect=True>. 
##

def qsvt_response(x_set,phiset,signal_convention,reflect=True):

    pmats      =  []
    pdat       =  []

    Zpi        =  np.array([[1.0,0.0],[0.0,-1.0]])

    signal_op  =  get_qsp_signal_op(signal_convention)
    phase_op   =  get_qsp_phase_op(signal_convention)

    for phi in phiset:
        pmats.append(phase_op(phi))

    N_pm = len(pmats)

    for a in x_set:

        if reflect:
            W = signal_op(a) @ Zpi          
        else:
            W = signal_op(a)          

        if (N_pm % 2 != 0):
            U = pmats[0]
            for n in range(1,int((N_pm-1) / 2)+1):
                U = U @ W.conj().T @ pmats[2*n-1] @ W @ pmats[2*n]
            pdat.append( U )
        else:
            U = pmats[0] @ W @ pmats[1]
            for n in range(1,int(N_pm / 2)+1):
                U = U @ W.conj().T @ pmats[2*n] @ W @ pmats[2*n+1]
            pdat.append( U )

    pdat = np.array(pdat, dtype=np.complex128)

    return(pdat)





##
##  From angle_test-formal-6.ipynb  Tidy this up.
## 

#def simulate_qsvt(convention="R"):
    


    xpts       =  np.linspace(-1.0,1.0,100)


    ##
    ## need to change signal state for "R" to "Wx"

    sig_state  =  get_qsp_signal_state(convention)

    #sig_state  =  get_qsp_signal_state("Wx")



    #qsp_unitaries = qsp_response(xpts,phiset_cos,signal_convention)

    if (convention == 'R'):
        qsp_unitaries = qsp_response(xpts,phi_Wx_to_O(phiset_cos),signal_convention)
    else:
        qsp_unitaries = qsp_response(xpts,phiset_cos,signal_convention)


    # if (signal_convention == 'R'):
    #     qsp_unitaries = qsvt_response(xpts,phi_Wx_to_qsvt(phiset_cos),signal_convention)
    # else:
    #     qsp_unitaries = qsvt_response(xpts,phiset_cos,signal_convention, reflect=False)



    qsp_F = 1j*np.zeros(len(qsp_unitaries))

    idx = 0
    for U in qsp_unitaries:
        qsp_F[idx] = (sig_state.T @ U @ sig_state)
        print(U)
        print(qsp_F[idx])

        idx += 1
