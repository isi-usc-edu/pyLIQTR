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
###       EXPANDER   -    Analytical expansions in a Chebyshev basis.        ###
###                                                                          ###
################################################################################
###
###   REVISION: 11/08/2022
###
###   Justin E. Elenewski  -  MIT Lincoln Laboratory
###   justin dot elenewski at ll dot mit dot edu
###
###

import numpy                as np
import scipy.special        as sfn

from   pyLIQTR.sim_methods.ChebyshevPoly        import ChebyshevPoly




def handle_scalar(x):

    x = np.asarray(x)

    if x.ndim == 0:
        x = x[np.newaxis]

    return(x)



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



################################################################################
###
###   CLASS     Expander
###
################################################################################
###
###   DESCRIPTION
###
###      Provides analytically-defined expansions of functions in a Chebyshev
###      polynomial basis, for input to QSP angle generation routines.
###
###   ARGUMENTS
###
###      NONE
###
###   RETURNS
###
###      Expander object
###
###   REQUIRES
###
###      numpy          as  np
###      numpy.special  as  sfn
###

class Expander(ChebyshevPoly):



    def __init__(self):

        super().__init__()

        self.type    = None
        self.err_tgt = None


    ############################################################################
    ###
    ###   FUNCTION     _ja_get_trig_order
    ###
    ############################################################################
    ###
    ###   DESCRIPTION
    ###
    ###      Calculates the target order for the Jacobi-Anger expansion of
    ###      trigonometric functions (sine, cosine), as needed to surpass
    ###      an accuracy threshold of acc_tgt.
    ###
    ###   ARGUMENTS
    ###
    ###      acc_tgt    -    Accuracy target for the polynomial expansion
    ###      tau        -    Constant argument of the sine or cosine function
    ###                      (e.g. f(x) = cos(tau*x) ).
    ###
    ###   RETURNS
    ###
    ###      max_order  -    Maximal order of the expansion
    ###      N_cf       -    Number of coefficients appearing in expansion
    ###
    ###   REQUIRES
    ###
    ###      numpy          as  np
    ###

    def _ja_get_trig_order(self, tau, err_tgt):

        # Maximum order of polynomial needed to hit accuracy target.
        max_order = int(np.ceil(1.4*np.abs(tau)+np.log(1.0/err_tgt)))

        if (np.mod(max_order,2) == 1):
            max_order = max_order - 1

        N_cf = int(max_order/2 + 1)

        return(max_order, N_cf)





    ############################################################################
    ###
    ###   FUNCTION     ja_cos
    ###
    ############################################################################
    ###
    ###   DESCRIPTION
    ###
    ###      Returns coefficients for a Chebyshev polynomial representation
    ###      of the cosine function via the Jacobi-Anger expansion.
    ###
    ###   ARGUMENTS
    ###
    ###      tau        -    Constant argument of the cosine function
    ###                      (e.g. f(x) = cos(tau*x) ).
    ###      acc_tgt    -    Accuracy target for the polynomial expansion
    ###
    ###   RETURNS
    ###
    ###      self.chebyshev_coeffs   -   Chebyshev expansion oefficients
    ###      self.parity             -   Parity of the target function
    ###      self.type               -   Descriptive string for the expansion.
    ###
    ###   REQUIRES
    ###
    ###      numpy          as  np
    ###      numpy.special  as  sfn
    ###

    ### NOTE NOTE NOTE - Edit text to note that this gives cos/2

    ### need to make sure this is compatible with EVAL

    def ja_cos(self, tau, err_tgt):

        max_order, N_cf = self._ja_get_trig_order(tau, err_tgt)

        #print("Maximum order of phase set = ", max_order)

        coeffs = np.zeros(N_cf)

        # Coefficients for the Jacobi-Anger expansion of cos
        # in terms of Bessel functions
        #
        for j in range(1,(N_cf+1)):
            coeffs[j-1] = ((-1)**(j-1))*sfn.jv(2*(j-1),tau)

        coeffs[0] = coeffs[0] / 2.0

        self.set_coeffs(coeffs,parity=0)


        return





    ############################################################################
    ###
    ###   FUNCTION     ja_sin
    ###
    ############################################################################
    ###
    ###   DESCRIPTION
    ###
    ###      Returns coefficients for a Chebyshev polynomial representation
    ###      of the sine function via the Jacobi-Anger expansion.
    ###
    ###   ARGUMENTS
    ###
    ###      tau        -    Constant argument of the cosine function
    ###                      (e.g. f(x) = sin(tau*x) ).
    ###      acc_tgt    -    Accuracy target for the polynomial expansion
    ###
    ###   RETURNS
    ###
    ###      self.chebyshev_coeffs   -   Chebyshev expansion oefficients
    ###      self.parity             -   Parity of the target function
    ###      self.type               -   Descriptive string for the expansion.
    ###
    ###   REQUIRES
    ###
    ###      numpy          as  np
    ###      numpy.special  as  sfn
    ###

    def ja_sin(self, tau, acc_tgt):

        max_order, N_cf = self._ja_get_trig_order(tau, acc_tgt)

        #print("Maximum order of phase set = ", max_order)

        coeffs = np.zeros(N_cf)

        # Coefficients for the Jacobi-Anger expansion of cos
        # in terms of Bessel functions
        #
        for j in range(1,N_cf+1):
            coeffs[j-1] = ((-1)**(j-1)) * sfn.jv(2*j - 1,tau)

        self.set_coeffs(coeffs,parity=1)


        return



    ############################################################################
    ###
    ###   FUNCTION     exponential_decay
    ###
    ############################################################################
    ###
    ###   DESCRIPTION
    ###
    ###      Returns coefficients for the exponential function with a negative
    ###      real argument, e.g., f(x) = exp(-beta*x)
    ###
    ###   ARGUMENTS
    ###
    ###      beta        -   Constant argument of the exponential function
    ###                      (e.g. f(x) = exp(-beta*x) ).
    ###      epsilon     -    Accuracy target for the polynomial expansion
    ###
    ###   RETURNS
    ###
    ###      self.chebyshev_coeffs   -   Chebyshev expansion coefficients
    ###                                  of even parity
    ###      self.parity = 0         -   Parity of the target function (even)
    ###      self.type               -   Descriptive string for the expansion.
    ###
    ###   REQUIRES
    ###
    ###      numpy          as  np
    ###      numpy.special  as  sfn
    ###
    ###   REFERENCES
    ###
    ###      Discussed contextually in:
    ###
    ###         G. H. Low and I. L. Chuang
    ###         "Hamiltonian Simulation by Uniform Spectral Ampliification"
    ###         arXiv:1707.05391
    ###
    ###      Expansion originally from:
    ###
    ###         S. Sachdeva and N. K. Vishnoi
    ###         "Faster algorithms via approximation theory"
    ###         Found. Trends Theoo. Comp. Sci. 9, 125 (2014)
    ###


    #
    # From:
    #
    #       A. M. Childs, R. Kothari, and R. D. Somma
    #       "Quantuum algorithm for systems of linear equations with
    #        exponentially improved dependence on precision.
    #       SIAM J. Comput. 46, 1920 (2017)
    #
    def exponential_decay_even(self,beta,epsilon):

        oterm1    = np.sqrt(2.0*np.ceil(np.max([beta*np.e**2,np.log(2.0/epsilon)])))
        oterm2    = np.sqrt(np.log(4.0/epsilon))

        max_order = np.ceil(oterm1 * oterm2)

        N_cf = int(max_order/2)

        coeffs = np.zeros(N_cf)

        coeffs[0] = sfn.iv(0,beta)

        # note that the coefficents are of T_j(-x) - should we change to make
        # this consistent with trigonometric routines?

        for j in range(1,N_cf):
            coeffs[j] = 2*sfn.iv(2*j,beta)

        self.set_coeffs(coeffs,parity=0)


        return




    def exponential_decay_odd(self,beta,epsilon):

        oterm1    = np.sqrt(2.0*np.ceil(np.max([beta*np.e**2,np.log(2.0/epsilon)])))
        oterm2    = np.sqrt(np.log(4.0/epsilon))

        max_order = np.ceil(oterm1 * oterm2)

        N_cf = np.int(max_order/2)

        coeffs = np.zeros(N_cf)

        # note that the coefficents are of T_j(-x) - should we change to make
        # this consistent with trigonometric routines?

        for j in range(0,N_cf):
            coeffs[j] = 2*sfn.iv(2*j+1,beta)

        self.set_coeffs(coeffs,parity=1)


        return





    ############################################################################
    ###
    ###   FUNCTION     jacobi_anger_gaussian
    ###
    ############################################################################
    ###
    ###   DESCRIPTION
    ###
    ###      Returns coefficients for the gaussan function with a
    ###      real argument, e.g., f(x) = exp(-(alpha^2)*x)
    ###
    ###   ARGUMENTS
    ###
    ###      alpha       -    Constant argument of the exponential function
    ###                       (e.g. f(x) = exp(-(alpha^2)*x) ).
    ###      epsilon     -    Accuracy target for the polynomial expansion
    ###
    ###   RETURNS
    ###
    ###      self.chebyshev_coeffs   -   Chebyshev expansion oefficients
    ###      self.parity             -   Parity of the target function
    ###      self.type               -   Descriptive string for the expansion.
    ###
    ###   REQUIRES
    ###
    ###      numpy          as  np
    ###      numpy.special  as  sfn
    ###


    def gaussian(self,alpha,epsilon):

        max_order = np.ceil( np.sqrt((alpha + np.log(1.0/epsilon)) *
                              np.log(1.0/epsilon)))

#        max_order = max_order + 10

        if (np.mod(max_order,2) == 1):
            max_order = max_order - 1

        N_cf      = int(max_order/2 + 1)

        coeffs    = np.zeros(N_cf)

        asqh      =  (alpha / 2.0)

        coeffs[0] = np.exp(-asqh) * sfn.iv(0,asqh)

        for j in range(1,N_cf):
            coeffs[j] = 2*((-1)**j) * np.exp(-asqh) * sfn.iv(j,asqh)

        self.set_coeffs(coeffs,parity=0)

        return



    def erf(self, kappa):

        return


    def erf_shifted(self, kappa):

        return


    #
    # From:
    #
    #       A. M. Childs, R. Kothari, and R. D. Somma
    #       "Quantuum algorithm for systems of linear equations with
    #        exponentially improved dependence on precision.""
    #       SIAM J. Comput. 46, 1920 (2017)
    #
    def inverse(self, kappa, acc_tgt):


        b = int(np.ceil((kappa**2) * np.log10(kappa / acc_tgt)))
        d = int(np.ceil(np.sqrt(b*np.log10(4*b/acc_tgt))))

#        d = d+10

        coeffs = np.zeros(d+1)

        for j in range(d+1):
            coeffs[j] = 4 * ((-1)**j) * (binomsum(j+1,b) / (2**(2*b)))

        ### need to modify code to handle analytical expansion of binom
        ### so that denominator isn't problematic

        # coeffs are of T_{2j+1}(x)

        self.set_coeffs(coeffs,parity=1)

        return



def binomsum(min,b):

    val = 0.0

    for i in range(min,b+1):
        val += sfn.binom(2*b,b+i)

    return(val)
