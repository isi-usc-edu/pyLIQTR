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
###     SIMQSP   -  Operator-level simulation of QSP and QSVT sequences      ###
###                                                                          ###
################################################################################
###
###   REVISION: 11/08/2022
###
###   Justin E. Elenewski  -  MIT Lincoln Laboratory
###   justin dot elenewski at ll dot mit dot edu
###
###

import numpy as np
import numpy.linalg as nla
import scipy.linalg as sla

import copy as copy
from pyLIQTR.sim_methods.__internal__ import phaseCos1x,phaseSin1x, phaseCos10x,phaseSin10x

def phase_w_to_r(phasesW):

    N = len(phasesW)

    phasesR      = [ (phasesW[j] - (np.pi/2.0)) for j in range(N)]

    phasesR[0]   = phasesW[0] + ((2.0*(N-1)) - 1.0)*np.pi/4.0
    phasesR[N-1] = phasesW[N-1] - np.pi/4.0

    return(phasesR)




class SimQSP:

    def __init__(self,phases=None,M_op=None,angler=None,parity=None):

        if (phases is not None):
            self.phases = phases
            if (parity is not None):
                self.parity = parity
            else:
                print ("\nERROR: Parity not specified for phase sequence.\n")
                return
        elif (angler is not None):
            self.phases = copy.deepcopy(angler.phases)
            self.parity = copy.deepcopy(angler.parity)


        if (M_op is not None):
            self.M_op = M_op

        self.qsp_seq = None

        self.IdM     = None
        self.Ua      = None
        self.ZPi     = None
        self.Rot     = None




    ################################################################################
    ###
    ###   FUNCTION     get_qsp_operators (M_op)
    ###
    ################################################################################
    ###
    ###   DESCRIPTION
    ###
    ###      Returns the rotation and reflection constitutuents of a QSP sequence
    ###      (in the Z-representation) when given the matrix representation of a
    ###      normal operator Mop.  This method performs the qubitization using
    ###      a single ancilary qubit (i.e., without the use of explicit select
    ###      and prepare oracles).
    ###
    ###   ARGUMENTS
    ###
    ###      M_op = Normal operator to be qubitized
    ###
    ###   RETURNS
    ###
    ###      Ua  = Block encoding unitary for the operator Mop
    ###      ZPi = Reflection operator in the qubitized space
    ###      IdM = Identity operator in the qubitized space
    ###
    ###   REQUIRES
    ###
    ###      numpy          as np
    ###      scipy.linalg   as sla
    ###

    def get_qsp_operators(self, M_op=None, conv="r"):


        if ((M_op is None) and (self.M_op is not None)):
            M_op = self.M_op
        elif ((M_op is None) and (self.M_op is None)):
            print("\n\nERROR:   No operator specified for block encoding.\n\n")
            return
        else:
            self.M_op = M_op

        # Rank of the operator Mop
        N = self.M_op.shape[0]

        # Number of qubits corresponding to the operator
        Nqb = int(np.log2(N))

        # Identity and projection operators for a two-dimensional Hilbert space
        Id = np.array([[1.0, 0.0], [0.0, 1.0]])
        PO = np.array([[1.0, 0.0], [0.0, 0.0]])


        # Generate the identity operator on the space formed between the eigenspace
        # of Mop and a single ancilary qubit
        IdM = Id

        for j in range(1,Nqb):
            IdM = np.kron(IdM,Id)

        ZPi = 2.0*np.kron(PO,IdM) - np.kron(Id,IdM)


        # Calculate defect terms for off-diagonal using an explicit matrix root.
        # Note that this can become costly for large matrices and numerically
        # unstable depending on the condition number of Mop.
        MopDef = sla.sqrtm(IdM - np.matmul(self.M_op.conj().T,self.M_op))

        # Build the minimal block encoding of Mop
        Ua =  1j*np.zeros((2*N,2*N))

        if (conv.lower() == "r"):
            Ua[0:N,   0:N  ] =  self.M_op
            Ua[N:2*N, 0:N  ] =  MopDef
            Ua[0:N,   N:2*N] =  MopDef
            Ua[N:2*N, N:2*N] = -self.M_op
            Rot              = Ua @ ZPi

        elif (conv.lower() == "w"):
            Ua[0:N,   0:N  ] =  Mop
            Ua[N:2*N, 0:N  ] =  1.0j*MopDef
            Ua[0:N,   N:2*N] =  1.0j*MopDef
            Ua[N:2*N, N:2*N] = Mop
            Rot              = Ua

        else:
            print("\n\nERROR:   Unsupported QSP covention <",conv,">\n")

        # Build the reflection operator on the qubitized space

        self.Ua  = Ua
        self.ZPi = ZPi
        self.IdM = IdM
        self.Rot = Rot


    def get_qsp_operators_lcu_r(self, M_ops=None):

        return




################################################################################
###
###   FUNCTION     qsp_sequence(Rot,Refl,phaseSeq)
###
################################################################################
###
###   DESCRIPTION
###
###      Return one iteration of the QSP sequence defined by the SU(2) rotation
###      operator Rot, the reflection operator Refl, and the phases specified
###      in the array phaseSeq.
###
###   ARGUMENTS
###
###      Rot      = qubitized rotation operator
###      Relf     = reflection operator acting on the qubitized Hilbert space
###      phaseSeq = array containing QSP phases (i.e., for z-rotations), with
###                 the lowest-degree phase component (\phi_0) first and the
###                 highest degree component last (\phi_d)
###
###   RETURNS
###
###      theSeq   = matrix representation of the QSP sequence
###
###   REQUIRES
###
###      numpy          as np
###      scipy.linalg   as sla
###

#    def qsp_sequence(self,Rot,Refl,phaseSeq):
    def qsvt_sequence(self,phases=None,parity=None):

        if (phases is not None):
            self.phases = phases
            if (parity is not None):
                self.parity = parity
            else:
                print ("\nERROR: Parity not specified for phase sequence.\n")
                return

        # Number of phases in QSP sequence / degree of approximated polynomial
        # transformation.
        N = len(self.phases)

        phase_hold = copy.deepcopy(self.phases)

        # Symmetric convention - decided not to go this route, since it
        # requires change elsewhere.  Recorded for posterity nonetheless.
        #
        # self.phases[0]     = phase_hold[0]   + np.pi/4
        # self.phases[N-1]   = phase_hold[N-1] + np.pi/4
        #
        # for j in range(1,N-1):
        #     self.phases[j] = phase_hold[j]   + np.pi/2


        self.phases[0]     = phase_hold[0]

        for j in range(1,N):
            self.phases[j] = phase_hold[j]  + np.pi/2

        # Define 0-th z-reflection
        Zphi0 = sla.expm(1j*self.phases[0]*self.ZPi)

        if (self.parity == 1):

            Zphi1p = sla.expm(1j*self.phases[ 1 ]*self.ZPi)


            Nmx    = int((N-1)/2)
            theSeq = Zphi0 @ (self.Ua @ Zphi1p)

            for m in range(1,Nmx+1):

                Zphi1  = sla.expm(1j*self.phases[ (2*m)   ]*self.ZPi)
                Zphi2  = sla.expm(1j*self.phases[ (2*m)+1 ]*self.ZPi)

                theSeq = theSeq @ self.Ua.conj().T @ Zphi1 @ self.Ua @ Zphi2


        elif (self.parity == 0):


            # here we use symmetric qsp with alternating phase signs
            # to eliminate need for conjugate (need separate routine with
            # conjugate as well

            Nmx    = int(N/2)
            theSeq = Zphi0

            for m in range(1,Nmx+1):

                Zphi1  = sla.expm(1j*self.phases[ (2*m)-1   ]*self.ZPi)
                Zphi2  = sla.expm(1j*self.phases[ (2*m)     ]*self.ZPi)

                theSeq = theSeq @ self.Ua.conj().T @ Zphi1 @ self.Ua @ Zphi2


        self.qsp_seq = ((-1.0j)**(N-1)) * theSeq

        self.phases = phase_hold

        return(self.qsp_seq)



################################################################################
###
###   FUNCTION     qsp_sequence(Rot,Refl,phaseSeq)
###
################################################################################
###
###   DESCRIPTION
###
###      Return one iteration of the QSP sequence defined by the SU(2) rotation
###      operator Rot, the reflection operator Refl, and the phases specified
###      in the array phaseSeq.
###
###   ARGUMENTS
###
###      Rot      = qubitized rotation operator
###      Relf     = reflection operator acting on the qubitized Hilbert space
###      phaseSeq = array containing QSP phases (i.e., for z-rotations), with
###                 the lowest-degree phase component (\phi_0) first and the
###                 highest degree component last (\phi_d)
###
###   RETURNS
###
###      theSeq   = matrix representation of the QSP sequence
###
###   REQUIRES
###
###      numpy          as np
###      scipy.linalg   as sla
###

#    def qsp_sequence(self,Rot,Refl,phaseSeq):
    def qsp_sequence(self,phases=None):

        if (phases is not None):
            self.phases = phases

        # Number of phases in QSP sequence / degree of approximated polynomial
        # transformation.
        N = len(self.phases)

        # Define 0-th z-reflection
        Zphi = sla.expm(1j*self.phases[0]*self.ZPi)

        # Starting point for the QSP operator product
        theSeq = Zphi

        for m in range(1,N):

            # Define m-th z-reflection operator
            Zphi = sla.expm(1j*self.phases[m]*self.ZPi)

            # Multiply into operator product defining QSP sequence
            # theSeq = np.matmul(theSeq, (self.Rot @ Zphi) )
            theSeq = np.matmul(theSeq, np.matmul(self.Rot, Zphi))

        self.qsp_seq = theSeq


        return(self.qsp_seq )




    # assumes we already loaded in some phases
    def eval_sequence(self,xpts,conv="r"):

        N = len(xpts)

        qvals = 1j*np.zeros(N)

        F = lambda x : np.array([[x,0.0],[0.0,-x]])

        for j in range(N):

            M_op = F(xpts[j])

            self.get_qsp_operators(M_op,conv=conv)
            op_ite = self.qsp_sequence()

            qvals[j] = self.qsp_seq[0,0]

        return(np.real(qvals))



    def eval_sequence_qsvt(self,xpts,conv="r"):

        N = len(xpts)

        qvals = 1j*np.zeros(N)

        F = lambda x : np.array([[x,0.0],[0.0,-x]])

        for j in range(N):

            M_op = F(xpts[j])

            self.get_qsp_operators(M_op,conv=conv)
            op_ite = self.qsvt_sequence()

            qvals[j] = self.qsp_seq[0,0]

        return(np.real(qvals))




################################################################################
###
###   FUNCTION     qsp_exp_sim (H)
###
################################################################################
###
###   DESCRIPTION
###
###      Calculates the real-time unitary propagator exp(-i * H * dt) using
###      a canned QSP sequence and a timestep $dt = 1.0 \omega_0^{-1}$,
###      where $\omega_0$ is the characteristic frequency associated with the
###      the Hamiltonian H.  Other timesteps may be used by rescaling the
###      Hamiltonian or providing an alternate phase angle sequence (as sines
###      and cosines via the Jacobi-Anger approximation)
###
###      As structured, this function can also give an imaginary time evolution
###      if the H is rescaled by -i (or a nonunitary evolution if imaginary
###      terms appear in H.  Note that this behavior *cannot* be emulated Using
###      a straightforward gate-based representation of the QSP sequence.
###
###   ARGUMENTS
###
###      H = Matrix representation of a Hamiltonian operator
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
#
def qsp_exp_sim_r(Ham,phaseCos=phaseCos10x, phaseSin=phaseSin10x):
    cos = SimQSP(M_op=Ham)
    sin = SimQSP(M_op=Ham)
    cos.get_qsp_operators()
    sin.get_qsp_operators()

    # Get QSP operators representing the half-sine and half-cosine.
    op_cos = cos.qsp_sequence(phaseCos)
    op_sin = sin.qsp_sequence(phaseSin)

    # We calculate the sine and cosine terms directly as the real part of the
    # sequence given by the respective QSP sequences (hence why the phases
    # encode 1/2 sin(x) and 1/2 cos(x)).  This could be practically implemented,
    # for instance, using controlled gates for the encoding QSP sequence and
    # its conjugate as conditioned on the |+> Hadamard state.
    op_cos = op_cos + np.conj(op_cos)
    op_sin = op_sin + np.conj(op_sin)

    # Generate the qubitized exponential operator
    op_exp = op_cos - 1j*op_sin

    return(op_exp)


#
#
# def qsp_exp_sim_w(Ham,phaseCos=phaseCos10x, phaseSin=phaseSin10x):
#
#
#     # Get qubitized block encoding unitary Ua, reflection operator ZPi,
#     # and identity IdM
#     Rot,ZPi,IdM = get_qsp_operators_w(Ham)
#
#     # Get rotation operator as product of block encoding unitary Ua and
#     # reflecton ZPi
#     # Rot = np.matmul(Ua,ZPi)
#
#     # Get QSP operators representing the half-sine and half-cosine.
#     op_cos = qsp_sequence(Rot,ZPi,phaseCos)
#     op_sin = qsp_sequence(Rot,ZPi,phaseSin)
#
#     # We calculate the sine and cosine terms directly as the real part of the
#     # sequence given by the respective QSP sequences (hence why the phases
#     # encode 1/2 sin(x) and 1/2 cos(x)).  This could be practically implemented,
#     # for instance, using controlled gates for the encoding QSP sequence and
#     # its conjugate as conditioned on the |+> Hadamard state.
#     op_cos = op_cos + np.conj(op_cos)
#     op_sin = op_sin + np.conj(op_sin)
#
#     # Generate the qubitized exponential operator
#     op_exp = op_cos - 1j*op_sin
#
#     return(op_exp)


    # sig = (1.0 / np.sqrt(2.0))*np.array([1,1])
    # wfn = np.kron(sig,IdM)
    #
    # qsp_op = np.matmul(wfn,np.matmul(op_exp,np.transpose(wfn)))
    # ref_op = sla.expm(-1j*10.0*Mop)
    #
    # trd = trace_distance(qsp_op,ref_op)
