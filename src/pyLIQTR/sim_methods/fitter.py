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
###       FITTER   -    Approximation in a Chebyshev polynomial basis        ###
###                                                                          ###
################################################################################
###
###   REVISION: 11/21/2022
###
###   Justin E. Elenewski  -  MIT Lincoln Laboratory
###   justin dot elenewski at ll dot mit dot edu
###
###


import numpy                as np
import numpy.linalg         as nla
import scipy.linalg         as sla
import scipy.special        as sfn
import scipy.optimize       as sopt

import mpmath               as mpm
import copy                 as copy

from   pyLIQTR.sim_methods.ChebyshevPoly   import ChebyshevPoly

mpm.mp.prec = 512



def chebyshev_func_m(x,sol,parity,offset=0.0):

    y = offset

    for i in range(1,len(sol)+1):

        if  (parity == 0):
            pdeg = 2*(i-1)
        elif (parity == 1):
            pdeg = (2*i)-1
        else:
            pdeg = (i-1)

        y += sol[i-1]*chebyshev_m(x,pdeg)

    return(y)


def chebyshev_m(x,n):
    if (mpm.fabs(x)<=1):
        rv = mpm.cos(mpm.fmul(n,mpm.acos(x)))
    elif (x>1):
        rv = mpm.cosh(mpm.fmul(n,mpm.acosh(x)))
    else:
        rv = ((-1)**n)*mpm.cosh(mpm.fmul(n,mpm.acosh(-x)))

    return(rv)


def m_to_np(mat):
    npa = np.array(mat.tolist(), dtype=float)
    return(npa)





class Fitter(ChebyshevPoly):

    _SOLVER_SET = ['anderson', 'ridder', 'pegasus']


    # changed default to kappa = None

    def __init__(self, f_tgt, deg, parity, kappa=0.0, i_lower=None, i_upper=None,
                 pts_init=None, mpm_prec=512):

        super().__init__(parity=parity)

        self.mpm_prec            =  mpm_prec     # Numerical precision for mpmath
        self.root_solver         = 'pegasus'     # Root solver for Remez method

        self.r_max_iter          = 100          # Maximum number of Remez iterations
        self.r_num_samp          = 50           # Number of points to sample in
                                                # each subinterval
        self.r_eps_tot           = 1.0e-6
        self.r_eps_rel           = 1.0e-4

        self.r_fail_thr          = 1.0e-12
        self.r_absc_thr          = 1.0e-4

       # Options for root solvers
        self.root_eps_fixed      = 1.0e-18
        self.root_eps_scaling    = 1.0e-4
        self.root_max_iter       = 1000

        # Options specific to internal Brent root solver
        self.brent_dpm_init      = 1.0e-10

        self.f_tgt               = f_tgt
        self.deg                 = deg
        self.parity              = parity


        self.kappa               = kappa

        if i_lower is None:
            if ( (self.kappa is None) or (self.kappa == 0.0 )):
                self.i_lower = 0.0
            else:
                self.i_lower = 1.0/kappa
        else:
            self.i_lower = i_lower


        if i_upper is None:
            self.i_upper = 1.0
        else:
            self.i_upper = i_upper

        # set mpmath precision to default for class

        mpm.mp.prec = self.mpm_prec

        # self.chebyshev_coeffs   = None
        # self.chebyshev_coeffs_m = None

        self.coeffs_m       = None


        self.fit_err_linf   = -np.Inf
        self.fit_err_rel    = -np.Inf
        self.conv_err_linf  = []

        if pts_init is None:
            self.pts = self._get_chebyshev_roots_m()
        else:
            self.pts = copy.deepcopy(pts_init)


    ## These routines can easily be made more general, but currently apply
    ## to a Chebyshev expansion that is relevant to QSP.
    ##

    def eval_func(self,xpts):

        fx = np.zeros(len(xpts))

        for j in range(N):
            fx[j]     = self.f_tgt(xpts[j])

        return(fx)


    def eval_approx(self,xpts):

        N = len(xpts)

        fx  = np.zeros(N);   BAPx  = np.zeros(N);    resi = np.zeros(N)

        if (self.coeffs is None):
            print("\n\nERROR:   Cannot evaluate Chebyshev expansion since a fit has not ",
                  "been performed.  Rerun after calling remez().\n\n")

        for j in range(N):

            x = xpts[j]

            fx[j]     = self.f_tgt(x)
            BAPx[j]   = self.eval(x)
            resi[j]   = BAPx[j] - fx[j]

        return(BAPx,fx,resi)


    def get_pts(self):
        return(m_to_np(self.pts).flatten())

    def get_pts_m(self):
        return(self.pts.flatten())


    def set_precision(self, prec):

        self.mpm_prec = prec
        mpm.mp.prec   = self.mpm_prec

        return


    def set_root_solver(self, solver):

        if  ((solver in self._SOLVER_SET) or solver == 'brent_mpm'):

            self.root_solver         = solver

        else:

            print("\n\n The root solver <", self.root_solver,
                  "> is not supported. Current solver remains ",
                  self.root_solver, ".\n")

        return




    def get_chebyshev_domain(self,kappa):

        self.kappa = kappa

        self.pts = self._get_chebyshev_roots_m()

        return


    def remez(self):

        print("#"*80)
        print("#    FITTER:REMEZ   ---   Approximation in a Chebyshev polynomial basis.       #")
        print("#"*80,"\n")

        mpm.mp.prec = self.mpm_prec

        ## Do iterations for Remez approximation.
        ##
        for itr in range(self.r_max_iter):


            ##  Solve linear system for the best approximating polynomial; residual
            ##  is stored in bap_sol[deg]
            ##
            bap_sol    =  self._get_bap_m()


            ##  Set eps for root solver based to minimum of a fixed value
            ##  or a scaling of the approximation residual.
            ##
            eps_root   =  np.min( [mpm.mpf(self.root_eps_fixed),
                                    self.root_eps_scaling*mpm.fabs(bap_sol[self.deg])] )


            roots      =  mpm.zeros(self.deg,1)

            ##  Evaluate Chebyshev polynomial fit and find roots of the
            ##  residual function (we assume a Chebyshev basis from the outset)
            ##
            for j in range(self.deg):

                ## Define residual function for root finding routines
                ##
                f_resi = lambda x : chebyshev_func_m(x,bap_sol[0:self.deg],self.parity)-self.f_tgt(x)
                # xtol should be about 10^4 * rtol

                ##  Run root finding routines on each subinterval
                ##

                pt_pair = (self.pts[j],self.pts[j+1])

                if (self.root_solver == 'brent_mpm'):
                    roots[j]  = self._findroot_brent_m(f_resi,pt_pair,
                                                  maxsteps=self.root_max_iter,
                                                  tol=eps_root,
                                                  dpm_init=self.brent_dpm_init)

                elif (self.root_solver in self._SOLVER_SET):
                    roots[j]  = mpm.findroot(f_resi,pt_pair,
                                             solver=self.root_solver,
                                             maxsteps=self.root_max_iter,
                                             tol=eps_root)

                else:
                    print("\n\n The root solver <", self.root_solver,
                          "> is not supported. Exiting...")
                    return

                if (roots[j] == np.NaN):
                    return


            ## Run procedure to find points that maximize residuals on each
            ## subinterval of the target interval (exchange steps).
            ##
            pts_new           =  copy.deepcopy(self.pts)
            max_resi_tot      =  mpm.fabs(bap_sol[self.deg])

            rabs_max   = -np.Inf;       rrel_max   = -np.Inf


            ##  Find points that maximize the fitting residual
            ##  between pairs of adjacent roots.
            ##
            for j in range(self.deg+1):

                max_resi = mpm.fabs(bap_sol[self.deg])

                pm = ((-1)**(j+1)) * np.sign(bap_sol[self.deg])

                ##  Set interval between consecutive roots, being mindful of
                ##  interval endpoints.
                ##
                if (j == 0):
                    lb = self.i_lower
                else:
                    lb = roots[j-1]

                if (j == self.deg):
                    ub = self.i_upper
                else:
                    ub = roots[j]


                subsamps = mpm.linspace(lb,ub,self.r_num_samp)


                ##  Evaluate approximating polynomial on subinterval and adjust
                ##  domain accordingly - that is, find a point between adjacent
                ##  roots that maximizes fitting residuals.
                ##
                ##     Tracking parmeters:
                ##
                ##          rabs_max       -  running maximal absolute residual
                ##          rrel_max       -  running maximal relative residual
                ##          maxi_resi_tot  -  running maximal residual (overall)
                ##
                for k in range(len(subsamps)):

                    resi = (chebyshev_func_m(subsamps[k], bap_sol[0:self.deg], self.parity)
                              - self.f_tgt(subsamps[k]))

                    if (np.abs(resi) > rabs_max):
                        rabs_max = np.abs(resi)

                    if (resi*pm > rrel_max):
                        rrel_max = resi*pm

                    if (resi*pm > max_resi):
                        max_resi     = resi*pm
                        max_resi_tot = np.max(np.array([max_resi_tot,max_resi]))
                        pts_new[j] = subsamps[k]


            ## Store new approximation points for next iteration.
            ##
            self.pts = pts_new


            ## Give an update on our status.
            ##
            print("REMEZ\t\t Iter: ", itr, "\t  Linf_Err:  ", float(max_resi_tot))


            ## Test failure criteria between relative and absolute error measures.
            ##
            if (np.abs(rabs_max-rrel_max) > self.r_fail_thr):
                print("REMEZ:\t\tREMEZ may have failed.  Check results.\n")


            ## Test convergence criteria to see if we have generated the target
            ## approximation on this iteration.  Passing either test will conunt as
            ## a success.
            ##
            crel_err = np.abs((bap_sol[self.deg] - max_resi_tot) / bap_sol[self.deg])

            abs_pass = (max_resi_tot < self.r_eps_tot)
            rel_pass = (crel_err < self.r_eps_rel)

            self.conv_err_linf.append(float(max_resi_tot))

            if ( abs_pass or rel_pass ):

                print("\nREMEZ:\t\tFITTING COMPLETE!\n")

                self.fit_err_linf = float(max_resi_tot)
                self.fit_err_rel  = float(crel_err)
                self.coeffs_m = bap_sol[0:(self.deg)]
                self.coeffs   = m_to_np(self.coeffs_m)
                self.N        = len(self.coeffs)
                return ()



    def _get_bap_m(self):

        A, b   =  self._get_bap_system_m()
        sol    =  mpm.lu_solve(A,b)

        return(sol)




    def _get_bap_system_m(self):

        A = mpm.matrix(self.deg+1,self.deg+1);     b = mpm.matrix(self.deg+1,1)

        comp = [(-1)**(n) for n in range(self.deg+1)]

        for j in range(1,self.deg+1):

            if (self.parity == 0):
                pdeg = 2*(j-1)
            elif (self.parity == 1):
                pdeg = (2*j)-1
            else:
                pdeg = j-1

            for k in range(1,self.deg+2):
                A[k-1,j-1] = chebyshev_m(self.pts[k-1],pdeg)

        for j in range(self.deg+1):
            A[j,self.deg]  =  comp[j]
            b[j]           =  self.f_tgt(self.pts[j])

        return(A,b)





    def _get_chebyshev_roots_m(self):

        mpm.mp.prec = self.mpm_prec

        kp = self.kappa

        xpts   =  mpm.matrix(self.deg+1,1)
        cpts   =  mpm.linspace(mpm.pi/2,0.0,self.deg+1)

        if (kp != 0.0):
            for j in range(self.deg+1):
               xpts[j] = mpm.cos(mpm.mpf(cpts[j])) * ((kp-1)/kp) + (1/kp)
        else:
            for j in range(self.deg+1):
               xpts[j] = mpm.cos(mpm.mpf(cpts[j]))

        return(xpts)






##
## Change to more natural flow control.  This is beasically the numerical
## methods implementation.
##
def _findroot_brent_m(f, ibounds, maxsteps=1000, tol=1.0e-18, tol_diff=None,
                      dpm_init=None):

    mpm.mp.prec = self.mpm_prec

    a = ibounds[0];    b = ibounds[1]

    fa = f(a);     fb = f(b);

    if ( f(a)*f(b) > 0 ):
        print("REMEZ-BRENT:\tError in endpoint condition f(a)f(b) > 0\n")
        return(np.NaN)

    tol_func = tol

    if tol_diff is None:
        tol_diff = tol_func

    if dpm_init is None:
        dpm_init = 1e-10


    if( mpm.fabs(f(a)) < mpm.fabs(f(b)) ):

        tmp1  =  a;       tmp2  =  fa;
        a     =  b;       fa    =  fb;
        b     =  tmp1;    fb    =  tmp2;

    c = a;    fc = fa;   s = b;

    d      =  self.b_dpm_init;      itr    =  0;
    mflag  =  True

    while(True):

        itr += 1

        if( itr > self.root_max_iter ):
            print("REMEZ-BRENT:\tMax number ( N = ", self.root_max_iter, ") of iterations reached!\n")
            return(np.NaN)

        if( ( mpm.fabs(b-a) < tol_diff ) or ( mpm.fabs(f(s)) < tol_func ) ):
            return(s)

        if( (fa != fc) and (fb != fc) ):
            s = (a*fb*fc/((fa-fb)*(fa-fc)) +
                 b*fa*fc/((fb-fa)*(fb-fc)) +
                 c*fa*fb/((fc-fa)*(fc-fb)))
        else:
            s = b - fb*(b-a)/(fb-fa)

        # changed tolerance here to tol_diff
        if(   ( s>=b )                                                  or
              ( s <= (3*a+b)/4 )                                        or
              ( (2*mpm.fabs(s-b)*2 > mpm.fabs(b-c))  and       mflag)   or
              ( (2*mpm.fabs(s-b) > mpm.fabs(d-c))    and  (not mflag))  or
              ( (tol_diff > mpm.fabs(b-c))           and       mflag )  or
              ( (tol_diff > mpm.fabs(d-c))           and  (not mflag)) ):

            s = (a+b)/2
            mflag = True

        else:
            mflag = False

        fs = f(s);     d  = c;
        c  = b;        fc = fb;

        if( (fa*fs) < 0):
            b = s;      fb = fs;
        else:
            a = s;      fa = fs;

        if( mpm.fabs(f(a)) < mpm.fabs(f(b)) ):

            tmp   =  a;      tmp2  =  fa;
            a     =  b;      fa    =  fb;
            b     =  tmp;    fb    =  tmp2;
