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
import  numpy      as      np
import  mpmath     as      mpm
from    functools  import  cache 

mpm.mp.prec = 512       # precision for fitting methods

## Import different angle generation algorithms
##
from  pyLIQTR.phase_factors.fourier_response.fourier_response    import  Angler_fourier_response
from  pyLIQTR.phase_factors.optimization.angler_optimization     import  Angler_opt
from  pyLIQTR.phase_factors.angler                               import  Angler_random

from  pyLIQTR.phase_factors.conversions                          import  phases_Wx_to_R

## Import function approximation algorithms
##
from  pyLIQTR.phase_factors.optimization.expander                import  Expander
from  pyLIQTR.phase_factors.optimization.fitter                  import  Fitter

## Routines for plotting error
##
from  pyLIQTR.utils.plot_helpers            import  plot_func_error






class PhaseFactors:

    def __init__( self,  
                #   eps=1e-3,
                  precompute=False, 
                  angler_max_iter=400,
                  angler_bfgs_conv=1e-14, 
                  mpm_prec = 512,
                  silent=True,
                  **kwargs                ):

        ### Define these always
        ###
        # self.name         =  ""
        # self.expr         =  "$f(x) = "
        # self.eps          =  eps
        # self.convention   =  "R"
        # self.phase_algorithm   =  phase_algorithm
        # self.expansion_method  =  expansion_method

        # self.x            =   ### function parameter
        # self.func         =   ### define function with one mandatory arg and others kwargs
        # self.params       =   ### dict to pass as kwargs for function paramters


        ## Define these for fitter
        ##
        #
        # self.deg          =   30
        # self.parity       =    0
        # self.kappa        = 10.0

        ## Always take these as kwargs
        ##

        self.precompute         =  precompute
        self.silent             =  silent

        self._mpm_prec          =  mpm_prec
        self._angler_bfgs_conv  =  angler_bfgs_conv 
        self._angler_max_iter   =  angler_max_iter

        ## Method to obtain polynomial expansion coefficients 
        ## prior to fitting (for Angler_opt method)
        ##
        if (self.expansion_method.lower() == "analytical"):
            self._angler_cheb_poly = self.expansion
        elif (self.expansion_method.lower() == "fit"):
            self._angler_cheb_poly = self.fit

        self._success_angler       =  -1
        self._success_fitter       =  -1
        self._success_computation  =  -1
        self._computed             =   False

        self.n_phases              =   0

        if(self.precompute):
            self.phis

        return


    ## Evaluate encoded function at value x (by passing instance arguments)
    ##
    def f(self,x):
        return(self.func(x,**self.params))
    

    ## Generate Chebyshev polynomial expansion
    ## (should be expanded to give access to Fourier equivalent)
    ##
    @property
    def expansion(self):
        expdr = Expander()
        return(expdr)
    

    ## Return Chebyshev or Fourier expansion coefficients
    ##
    @property
    def expansion_coeffs(self):
        return( self.expansion.coeffs )
    

    ## Evaluate error in expansion for target function (Chebyshev / Fourier)
    ## at designated value <x>
    ## (for assessing expansion error before fitting)
    ##
    def expansion_error(self,x):
        x = np.array(x)
        err = self.expansion.eval(x) - self.func(x,**self.params)
        return(err)
    

    ## Evaluate expansion for target function (Chebyshev / Fourier)
    ## at designated value <x>
    ## (for assessing expansion error before fitting)
    ##
    def expansion_eval(self,x):
        return(self.expansion.eval(np.array(x)))
    

    ## Plot error in approximating target function based on polynomial expansion
    ## alone (i.e., before phase fitting procedure)
    ##
    def plot_expansion_error(self,n_pts=100):

        kappa = self.tau

        lbound = -1.0
        ubound = 1.0
        ## change to range of designated parameter (wtih rescaling)

        xpts = np.linspace(lbound,ubound,n_pts)
        f_vals      =  self.func(xpts,**self.params)
        p_vals      =  self.expansion.eval(xpts)
        plot_func_error(xpts,f_vals,p_vals,self.eps,self.expr)



    ## Run angle generation and return angler object containg
    ## calculation output
    ##
    @property
    @cache 
    def phase_generator(self):

        if (self.convention.lower() == "r"):

            ## Instantiate required angler
            ##
            if (self.phase_algorithm.lower() == "optimization"):
                Ang = Angler_opt( cheb_poly=self._angler_cheb_poly, 
                                  bfgs_conv=self._angler_bfgs_conv, 
                                  max_iter=self._angler_max_iter,
                                  silent=self.silent  )
  
            elif (self.phase_algorithm.lower() == "random"):
                Ang = Angler_random( cheb_poly=self._angler_cheb_poly, silent=self.silent)

            ## Placeholder for fourier response class (port old methods to this)
            ##
        elif ( self.convention.lower() in ["fourier_response", "fourier", "fr"] ):

                fr_random = True if  (self.phase_algorithm.lower() == "random") else False

                Ang  =   Angler_fourier_response( tau=self.params['tau'], 
                                                  eps=self.eps,
                                                  silent=self.silent,
                                                  random=fr_random )
        else:
            raise Exception("Phase determination algorithm "+str(self.phase_algorithm)+" is not supported.")

        ## Run phase generation
        ##
        Ang.generate()
        
        ## Note success or failure of calculation
        ##
        self._success_angler           =  Ang._success
        if (self._success_angler == 1):
            self._computed             =   True


        return(Ang)
    


    ## Convert phases to desired convention and provide a convenient 
    ## access point for the data (while leaving external access to the Angler)
    ##
    @property
    @cache 
    def phis(self):

        phis = self.phase_generator.phases

        ## Switch based on optimization algorithm, as it will (usually)
        ## have a fixed output convention
        ##
        if (self.phase_algorithm.lower() == "optimization"):

            if (self.convention.lower() == 'r'):
                phis = phases_Wx_to_R(phis)
            elif (self.convention.lower() == 'wx'):
                pass
        
        self.n_phases = len(phis)
        return(phis)


    ## Instantiate a fitter instance, run fitting, and provide access
    ## (only fitting algo right now is Remez)
    ##
    @property
    @cache 
    def fit(self):
        theFit  = Fitter(self.f, self.deg, self.parity, kappa=self.kappa,silent=self.silent)
        theFit.remez()
        self._success_fitter =  theFit._success
        return(theFit)
    


    ## Print useful output regarding the phase factors
    ##
    def __str__(self):
        string = ""
        string += ("Phase angles:\t\t"+str(self.name)+"\n")
        string += ("Expression:\t\t"+str(str(self.expr))+"\n")
        string += ("# of Phases:\t\t"+str(self.n_phases)+"\n")
        string += ("Parameters:\n")
        for param in self.params.keys():
            string += "\t\t\t"+str(param)+"="+str(self.params[param])+"\n"
        string += ("Convention:\t\t"+str(self.convention)+"\n")


        # if (self.expansion_method.lower() == "analytical"):
        #     string += ("Expansion error:\t"+str(self.expansion_error)+"\n")
        

        string += ("Phase Algorithm:\t"+str(self.phase_algorithm)+"\n")
        string += ("Target Precision:\t"+str(self.eps)+"\n")

        if (self.expansion_method.lower() == "fit"):
            if (self._success_fitter == 0):
                fit_string = "Failed"
            elif (self._success_fitter == 1):
                fit_string = "Success"
            elif (self._success_fitter == -1):
                fit_string = "Not run"

            string += ("Fitter:\t"+fit_string+"\n")

        if (self.phase_algorithm.lower() == "optimization"):
            if (self._success_angler == 0):
                angle_string = "Failed"
            elif (self._success_angler == 1):
                angle_string = "Success"
            elif (self._success_angler == -1):
                angle_string = "Not run"

            string += ("Optimizer:\t\t"+angle_string+"\n")

        if (self._computed):
            string += "Angles Computed:\tTrue\n"
        else:
            string += "Angles Computed:\tFalse\n"


        return(string)
    

    ## Return degree of approximating Chebyshev polynomial
    ## or Fourier series
    ##
    @property
    @cache
    def expansion_degree(self):
        return(self._angler_cheb_poly.deg)


    ##  Return number of calculated phase angles.
    ##  
    @property 
    def num_phases(self):
        return(len(self.phis))


    ## Provide some colligative error measure over the fitting domain
    ##
    def expansion_error(self):
        return


    ## Provide a means for assessing error in a pointwise manner
    ##
    def _error(self,x):
        return
    
  