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
###       ANGLER   -    Phase angles for quantum signal processing.          ###
###                                                                          ###
################################################################################
###
###   REVISION: 11/21/2022
###
###   Justin E. Elenewski  -  MIT Lincoln Laboratory
###   justin dot elenewski at ll dot mit dot edu
###

import numpy                as np
import scipy.special        as sfn
import copy                 as copy

from   pyLIQTR.sim_methods.ChebyshevPoly        import ChebyshevPoly




################################################################################
###
###   FUNCTION     handle_scalar(x)
###
################################################################################
###
###   DESCRIPTION
###
###      Converts a scalar value into a numpy array, while leaving a numpy
###      array unchanged
###
###   ARGUMENTS
###
###      x  =   Scalar value
###
###   RETURNS
###
###      x  =   The value x as a numpy array
###
###   REQUIRES
###
###      numpy          as np
###

def handle_scalar(x):

    x = np.asarray(x)

    if x.ndim == 0:
        x = x[np.newaxis]

    return(x)







################################################################################
###
###   DATACLASS     eval_chebyshev(x, coeffs, parity)
###
################################################################################
###
###   DESCRIPTION
###
###      Evaluate the approximation to a function f(x) of parity <parity>
###      as defined by a Chebyshev expansion with coefficients <coeffs>.  The
###      array <coeffs> is assumed to contain terms for both even and odd
###      parity.
###
###   ARGUMENTS
###
###      x       =   value or array of values for function evaluation
###      coeffs  =   array of coefficients defining the Chebyshev expansion
###      parity  =   parity of the polynomial to evaluate (i.e., nonzero
###                  coefficents to include in the sum):
###
###                           0   ->   even terms
###                           1   ->   odd terms
###                           2   ->   all terms
###
###   RETURNS
###
###      N/A
###
###   REQUIRES
###
###      fx  =   value of function evaluated at point(s) x
###

def eval_chebyshev(x, coeffs, parity):


    x     = handle_scalar(x)      # Transforms scalar input to a 0-d numpy array

    N_cfs = len(coeffs)           # Number of coefficients defining f(x)
    fx    = np.zeros(len(x))      # Store evaluation of of f(x)
    acx   = np.arccos(x)          # Arccos appearing in Chebyshev terms

    if   (parity == 0):
        for k in range(0,N_cfs,2):
            fx += coeffs[k] * np.cos(k*acx)
    elif (parity == 1):
        for k in range(1,N_cfs,2):
            fx += coeffs[k] * np.cos(k*acx)
    else:
        for k in range(0,N_cfs):
            fx += coeffs[k] * np.cos(k*acx)

    return(fx)





################################################################################
###
###   DATACLASS     eval_chebyshev_strict(x, coeffs, parity)
###
################################################################################
###
###   DESCRIPTION
###
###      Evaluate the approximation to a function f(x) of parity <parity>
###      as defined by a Chebyshev expansion with coefficients <coeffs>.  The
###      array <coeffs> is assumed to only contain coefficients corresponding
###      to the parity <parity>.
###
###   ARGUMENTS
###
###      x       =   value or array of values for function evaluation
###      coeffs  =   array of coefficients defining the Chebyshev expansion
###      parity  =   parity of the polynomial terms to evaluate:
###
###                           0   ->   even terms
###                           1   ->   odd terms
###
###   RETURNS
###
###      N/A
###
###   REQUIRES
###
###      fx  =   value of function evaluated at point(s) x
###

def eval_chebyshev_strict(x, coeffs, parity):


    x     = handle_scalar(x)      # Transforms scalar input to a 0-d numpy array

    N_cfs = len(coeffs)           # Number of coefficients defining f(x)
    fx    = np.zeros(len(x))      # Array to store evaluation of of f(x)
    acx   = np.arccos(x)          # Arccos appearing in Chebyshev terms


    if (parity == 0):
        for k in range(0,N_cfs):
            fx += coeffs[k] * np.cos((2*k)*acx)
    else:
        for k in range(0,N_cfs):
            fx += coeffs[k] * np.cos((2*k+1)*acx)

    return(fx)



class Angler:

    ## QSP unitary in the "W" convention
    ##
    # _Wx    = lambda x : np.array( [[          x,             1.0j*np.sqrt(1-x**2) ],
    #                         [ 1.0j*np.sqrt(1-x**2),            x           ]] )

    ## Z-rotation by pi/4
    ##
    _Z_pi4  = np.array( [[ np.exp(1j*np.pi/4.0),              0.0              ],
                         [          0.0,         np.conj(np.exp(1j*np.pi/4.0)) ]])

    def _Wx(self,x):
        M =  np.array( [[          x,             1.0j*np.sqrt(1-x**2) ],
                        [ 1.0j*np.sqrt(1-x**2),            x           ]] )
        return(M)


    def __init__(self,coeffs=[1.0],parity=0,cheb_poly=None,bfgs_conv=1.0e-12,max_iter=5e4):


        ## Make a copy of coefficients from an existing ChebyshevPoly object.
        ##
        if (cheb_poly is not None):
            self.poly    = copy.deepcopy(cheb_poly)
        else:
            self.poly    = ChebyshevPoly(coeffs=coeffs,parity=parity)


        ## Set parity equal to polynomial parity and make the target function
        ## simply evaluation of the Chebyshev polynomial (makes for easier
        ## access by internal and external routines).
        ##
        self.parity      = self.poly.parity
        self.f_tgt       = lambda x : self.poly.eval(np.array([x]))

        ## Set optimizer properties that will frequently be changed between
        ## different fucntions.
        ##
        self.bfgs_conv   = bfgs_conv        # Convergence threshold for L-BFGS
        self.max_iter    = max_iter         # Max. number of L-BFGS iterations


        ## Defaults for L-BFGS parameters that are much less likely to be
        ## changed.
        ##
        self.gamma       = 0.5
        self.accept_rate = 1.0e-3
        self.min_step    = 1.0e-5
        self.lmem        = 200
        self.print_conv  = True

        ## Placeholders for the generated phase angles and final value of the
        ## L-BFGS objective function.
        ##
        self.phases      = None
        self.obj_value   = None




    def set_poly(self,coeffs=None,parity=None,cheb_poly=None):

        if (cheb_poly is not None):
            self.poly    = copy.deepcopy(cheb_poly)
        else:
            self.poly    = ChebyshevPoly(coeffs=coeffs,parity=parity)

        # self.f_tgt = lambda x : eval_chebyshev_strict(np.array([x]),
        #                                              self.coeffs, self.parity)

        self.parity      = self.poly.parity

        self.f_tgt = lambda x : self.poly.eval(np.array([x]))

        return

################################################################################
###
###   FUNCTION     _qsp_unitary_symmetric(phi,x, parity)
###
################################################################################
###
###   DESCRIPTION
###
###      Calculate the QSP unitary that that is defined by the phase factors
###      <phi> and evaluate it with an argument of <x>.  The parity of the
###      encoded polynomials are given by <parity>.  Phase factors are assumed
###      to have been generated with a SYMMETRIC convention.
###
###   ARGUMENTS
###
###      phi     =   array containing QSP phase factors
###      x       =   value or array of values for function evaluation
###      parity  =   parity of the polynomial terms to evaluate:
###
###                           0   ->   even terms
###                           1   ->   odd terms
###
###   RETURNS
###
###      fx  =   value of function evaluated at point(s) x
###
###   REQUIRES
###
###      numpy   as   np
###

    def _qsp_unitary_symmetric(self,phi,x):

        if (phi.ndim == 1):
            Np = phi.shape[0]
        else:
            Np = phi.shape[1]

        phi    = np.reshape(phi,phi.size,order='F')     # currently uses "fortran"
                                                        # column-major ordering

        Z_phi = lambda n : np.array( [[ np.exp(1.0j*phi[n]),           0.0       ],
                                      [        0.0,          np.exp(-1.0j*phi[n])]])
        ## EVEN PARITY
        ##
        if (self.parity == 0):

            seq = np.eye(2)

            for k in range(1,Np):
                seq  =  (seq @ self._Wx(x)) @ Z_phi(k)

            seq  =  seq @ self._Z_pi4
            Ua   =  seq.T @ (Z_phi(0) @ seq)

        ## ODD PARITY
        ##
        else:

            seq = Z_phi(0)

            for k in range(1,Np):
                seq  =  seq @ (self._Wx(x) @ Z_phi(k))

            seq  =  seq   @ self._Z_pi4
            Ua   =  seq.T @ (self._Wx(x) @ seq)


        return(Ua)





################################################################################
###
###   FUNCTION     _qsp_obj_func(phi,x_vec,opts)
###
################################################################################
###
###   DESCRIPTION
###
###      Calculate the objective function associated with the optimization-based
###      approach to QSP phase angle generation.
###
###   ARGUMENTS
###
###      phi     =   array containing QSP phase factors
###      x_vec   =   value or array of values for function evaluation
###      opts    =   params_gradopt dataclass
###
###   RETURNS
###
###      obj  =   value of function evaluated at point(s) x_vec
###
###   REQUIRES
###
###      numpy   as   np
###

    def _qsp_objective(self,phi,x_vec):

        M   = len(x_vec)
        Np  = phi.shape[0]

        obj = np.zeros((M))

        for j in range(M):

            Ua = self._qsp_unitary_symmetric(phi,x_vec[j])
            obj[j] = 0.5*(np.real(Ua[0,0]) - self.f_tgt(x_vec[j]))**2

        return(obj)






################################################################################
###
###   FUNCTION     _qsp_obj_func_grad(phi,x_vec,opts)
###
################################################################################
###
###   DESCRIPTION
###
###      Calculate the analytical gradient of the objective function, as needed
###      for the optimization-based approach to QSP phase angle generation.
###
###   ARGUMENTS
###
###      phi     =   array containing QSP phase factors
###      x_vec   =   value or array of values for function evaluation
###      opts    =   params_gradopt dataclass
###
###   RETURNS
###
###      obj  =   value of the objective function evaluated at point(s) x_vec
###               with phases phi
###      grad =   gradient of the objective function with respect to the angles
###               phi, when evaluated at point(s) x_vec
###
###   REQUIRES
###
###      numpy   as   np
###

    def _qsp_objective_grad(self,phi,x_vec):

        if (phi.ndim == 1):
            D = phi.shape[0]
        else:
            D = phi.shape[1]

        phi     = np.reshape(phi,phi.size,order='F')

        M       = len(x_vec)

        obj     = np.zeros(M)                # Array to hold values of objective function
        grad    = np.zeros((M,D))            # Array to hold values of gradient for each degree d

        IV      = np.array([[1.0j, 0.0], [0.0, -1.0j]])

        ##
        Z_phi = lambda n : np.array( [[ np.exp(1.0j*phi[n]),           0.0       ],
                                      [        0.0,          np.exp(-1.0j*phi[n])]])


        for m in range(M):

            x        = x_vec[m]

            ## Temporary arrays which will to hold components of the objective
            ## gradient with respect to phase set.
            ##
            tmp1         =  1.0j*np.zeros((2,2,D))
            tmp2         =  1.0j*np.zeros((2,2,D))

            tmp1[:,:,0]  =  np.eye(2)

            tmp2[:,:,0]  =  np.matmul(Z_phi(D-1),self._Z_pi4)


            for d1 in range(1,D):

                # "Right" side terms
                tmp1[:,:,d1]  = (tmp1[:,:,d1-1] @ Z_phi(d1-1)) @ self._Wx(x)

                # "Left" side terms
                tmp2[:,:,d1]  = Z_phi(D-d1-1) @ (self._Wx(x) @ tmp2[:,:,d1-1])

            if (self.parity == 0):

                # Compute unitary from "left" and "right" parts, then estimate the
                # deficit in the objective function
                #
                Ua         = (tmp2[:,:,D-2].T) @ (self._Wx(x) @ tmp2[:,:,D-1])
                leftseq    = (tmp2[:,:,D-2].T) @ self._Wx(x)

                d_obj      = np.real(Ua[0,0]) - self.f_tgt(x)


                for d2 in range(D):

                    grad_tmp   = leftseq @ ((tmp1[:,:,d2] @ IV) @ tmp2[:,:,D-1-d2])
                    grad[m,d2] = 2*np.real(grad_tmp[0,0])*d_obj

                grad[m,0]    = grad[m,0]/2.0
                obj[m]       = 0.5*( np.real(Ua[0,0]) - self.f_tgt(x) )**2

            else:

                Ua          = (tmp2[:,:,D-1].T  @  self._Wx(x))  @  tmp2[:,:,D-1]
                leftseq     =  tmp2[:,:,D-1].T  @  self._Wx(x)

                d_obj       = np.real(Ua[0,0]) - self.f_tgt(x)

                for d2 in range(D):

                    grad_tmp   = leftseq @ ((tmp1[:,:,d2] @ IV) @ tmp2[:,:,D-1-d2])
                    grad[m,d2] = 2*np.real(grad_tmp[0,0])*d_obj

                obj[m] = 0.5*(np.real(Ua[0,0])-self.f_tgt(x))**2

        return(grad,obj)





################################################################################
###
###   FUNCTION     _phases_lbfgs(obj_func,grad_func,x_vec,phi,opts)
###
################################################################################
###
###   DESCRIPTION
###
###      Runs an L-BFGS optimizer for QSP phase angle generation.  Pulled
###      very directly from the methods in QSPPACK:
###
###                 Xiang Meng, K. Birgitta Whaley, Lin Lin.
###                 "Efficient Phase Factor Evaluation in Quantum
###                     Signal Processing"   [arXiv:2002.11649].
###
###   ARGUMENTS
###
###      obj_func     =   objective function
###      grad_func    =   gradient of objective function
###      x_vec        =   array of points for evaluation
###      phi          =   initial phase angle set
###      opts         =   params_gradopt dataclass
###
###   RETURNS
###
###      phi          =   optimized set of phase angles
###      obj_max      =   maximal value of the objective function over x_vec
###                       in the final optimization step
###      itr          =   total number of iterations for the optimizer
###
###   REQUIRES
###
###      numpy   as   np
###

    def _phases_lbfgs(self,obj_func,grad_func,x_vec,phi):

        print("#"*80)
        print("#             GRADIENT OPTIMIZER for QSP Phase Angle Determination             #")
        print("#"*80,"\n\n")
        print("Iteration\t\tMax. Obj.")
        print("="*50,"\n")

        accuracy_target = (self.bfgs_conv)**2

        d         = len(phi)     # number of phases

        ## Define memory terms for previous iterations of BFGS optimizer
        ##
        mem_size  = 0;                          mem_now   = 0;
        mem_grad  = np.zeros((self.lmem,d));    mem_obj   = np.zeros((self.lmem,d));
        mem_dot   = np.zeros(self.lmem);

        ## Calculate value of BFGS objecive function (e.g., the mean gradient)
        ##
        grad_s,obj_s = grad_func(phi,x_vec)
        obj_value    = np.mean(obj_s)

        grad         = np.mean(grad_s,axis=0)

        run_flag = True
        itr      = 0

        while(run_flag):

            itr      += 1
            theta_d   = grad

            alpha     = np.zeros(mem_size)

            for k in range(mem_size):
                subsc    = np.mod(mem_now-k,self.lmem)
                alpha[k] = mem_dot[subsc]    *  (mem_obj[subsc,:] @ theta_d)
                theta_d  = theta_d-alpha[k]  *   mem_grad[subsc,:]


            theta_d   = 0.5*theta_d

            if(self.parity == 0):
                theta_d[0] = 2*theta_d[0]

            for k in range(mem_size):
                subsc    = np.mod(mem_now-(mem_size-k)+1,self.lmem)
                beta     = mem_dot[subsc]    *   np.dot(mem_grad[subsc,:],theta_d)
                theta_d  = theta_d + np.dot(alpha[mem_size-k-1]-beta,
                                            mem_obj[subsc,:])


            step      = 1
            exp_des   = (grad @ theta_d)


            ts_flag   = True

            while(ts_flag):
                theta_new    = phi - (step*theta_d)[...,None]



                obj_snew     = obj_func(theta_new,x_vec)

                obj_valuenew = np.mean(obj_snew)
                ad           = (obj_value - obj_valuenew)

                # this is awful style for flow control
                if ((ad > exp_des*step*self.accept_rate) or (step < self.min_step)):
                    break

                step = step*self.gamma

            phi = theta_new
            obj_value = obj_valuenew
            obj_max = np.max(obj_snew)
            [grad_s,obj_s_tmp] = grad_func(phi,x_vec)
            grad_new = np.mean(grad_s,axis=0)
            mem_size = np.min((self.lmem,mem_size+1))
            mem_now  = np.mod(mem_now,self.lmem)+1
            mem_grad[mem_now,:] = grad_new - grad
            mem_obj[mem_now,:]  = -step*theta_d

            mem_dot[mem_now]    = 1.0/(mem_grad[mem_now,:] @ mem_obj[mem_now,:])
            grad = grad_new


            if (self.print_conv == True):
                #print(itr, obj_max, ad/(exp_des*step))
                print(itr,"\t\t\t{a:.6e}".format(a=obj_max))


            if (itr >= self.max_iter):
                print("\nGRADOPT-LBFGS: Stopping. Max iterations (",self.max_iter,") reached.  ")
                print("GRADOPT-LBFGS: Increase iterations or change starting guess...  ")
                run_flag = False

            if (obj_max < accuracy_target):
                print("\nGRADOPT-LBFGS: Optimization converged to target accuracy (max_obj <", accuracy_target,").")
                run_flag = False

        return(phi,obj_max,itr)





################################################################################
###
###   FUNCTION     phases_gradopt(poly_coeffs,opts)
###
################################################################################
###
###   DESCRIPTION
###
###      Wrapper function for the optimizer-based refinement of phase angles
###      for a given QSP sequence.  The target function is approximated by
###      Chebyshev polynomial coeffients <poly_coeffs>, and the parity of
###      this function is specified through opts.parity
###
###   ARGUMENTS
###
###      poly_coeffs  =   array of Chebyshev expansion coefficients that
###                       approximate a target function
###      opts         =   params_gradopt dataclass
###
###   RETURNS
###
###      phi_proc     =   array of optimized phases
###      obj_value    =   final (maximal) value of the optimizer's objective
###                       function (over all roots of the Chebyshev polynomial)
###
###   REQUIRES
###
###      numpy   as   np
###

    def gradient_optimization(self):

        # Number of coefficients in the approximating Chebyshev polynomial
        #
        # Nc            = len(self.coeffs)

        Nc            = self.poly.N

        # Define target points for fitting in x_vec, corresponding to the
        # zeros of a Chebyshev polynomial with degree 2*Nc
        #
        apts          = np.arange(1,(2*Nc),2)
        x_vec         = np.cos(apts*np.pi/(4*Nc))

        # Array to hold resulting phase factors
        #
        phi0          = np.zeros(Nc)

        # Target function is the Chebyshev polynomial defined by only the even (or odd) terms
        # specified by opts.parity and contained in poly_coefs.
        #
        # self.f_tgt = lambda x : eval_chebyshev_strict(np.array([x]),
        #                                                  self.coeffs, self.parity)


        phi,obj_value,itr = self._phases_lbfgs(self._qsp_objective, self._qsp_objective_grad,
                                               x_vec,phi0)

        # Set initial phase vector to [pi/4, 0.0, 0.0, ... 0.0], which gives robust convergence
        # for exponential functions.  This is specific to the symmetrized approach, while
        # [pi/4, 0.0, 0.0, ... pi/4] would be needed otherwise.
        #
        phif          = phi[:,1]
        phif[-1]     += np.pi/4.0

        Nphi          = len(phif)

        ## Optimization has computed the target set of phase vectors, but these
        ## only corespond to unique values.  Here we apply symmetry to get a
        ## phase angle vector that containts the actual QSP sequence.
        ##
        if (self.parity == 0):

            Nproc = 2*Nphi-1

            phi_proc                  = np.zeros(Nproc)
            phi_proc[0:(Nphi-1)]      = phif[-1:0:-1]
            phi_proc[(Nphi-1):Nproc]  = phif[:]

        else:

            Nproc = 2*Nphi

            phi_proc                  = np.zeros(Nproc)
            phi_proc[0:Nphi]          = phif[::-1]
            phi_proc[Nphi:Nproc]      = phif[:]

        self.phases    = phi_proc
        self.obj_value = obj_value

        return(phi_proc,obj_value)





################################################################################
###
###   FUNCTION     optimize_phases_ja_exp(tau,acc_tgt)
###
################################################################################
###
###   DESCRIPTION
###
###      Generate refined phase angles for representing f(x) = exp[-i*tau*x]
###      using QSP, specifically via an optimization-based strategy.  In the
###      case of Hamiltonian simulation, <tau> corresponds to the timestep.
###      Coefficients for constituent sine and cosine terms are generated
###      using the Jacobi-Anger expansion.
###
###   ARGUMENTS
###
###      tau          =   timestep / argument of exponential function
###      acc_target   =   target accuracy for QSP sequence
###
###   RETURNS
###
###      phi_cos      =   QSP phase set for cosine terms
###      phi_sin      =   QSP phase set for sine terms
###      obj_cos      =   maximal objective function value during cos angle
###                       determination
###      obj_sin      =   maximal objective function value during sin angle
###                       determination
###
###   REQUIRES
###
###      numpy                as   np
###      scipy.special        as   sfn
###
#
# def optimize_phases_ja_exp(tau,acc_tgt,opts=None):
#
#     # SET UP OPTS FIRST
#
#     # Maximum order of polynomial needed to hit accuracy target.
#     maxorder = int(np.ceil(1.4*np.abs(tau)+np.log(1.0/acc_tgt)))
#
#     if (np.mod(maxorder,2) == 1):
#         maxorder = maxorder - 1
#
#     print("Maximum order of phase set = ", maxorder)
#
#     Ncf = int(maxorder/2 + 1)
#
#     if (opts == None):
#         opts = params_gradopt(1)
#
#     ###
#     ### CALCULATE COSINE PHASES (EVEN PARITY)
#     ###
#
#     coeffs_cos = np.zeros(Ncf)
#
#     # Coefficients for the Jacobi-Anger expansion of cos
#     # in terms of Bessel functions
#     #
#     for j in range(1,(Ncf+1)):
#         coeffs_cos[j-1] = ((-1)**(j-1))*sfn.jv(2*(j-1),tau)
#
#     coeffs_cos[0] = coeffs_cos[0] / 2.0
#
#     opts.parity = 0
#
#     print("\nComputing cosine terms...\n")
#     phi_cos,obj_cos = phases_gradopt(coeffs_cos,opts)
#
#     ###
#     ### CALCULATE SINE PHASES (EVEN PARITY)
#     ###
#
#     # Coefficients for the Jacobi-Anger expansion of cos
#     # in terms of Bessel functions
#     #
#     coeffs_sin = np.zeros(Ncf)
#
#     for j in range(1,Ncf+1):
#         coeffs_sin[j-1] = ((-1)**(j-1))*sfn.jv(2*j - 1,tau)
#
#     opts.parity = 1
#
#     print("\nComputing sine terms...\n")
#     phi_sin,obj_sin = phases_gradopt(coeffs_sin,opts)
#
#
#     return(phi_cos,phi_sin,obj_cos,obj_sin)
