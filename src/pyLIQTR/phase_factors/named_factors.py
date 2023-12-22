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

from  pyLIQTR.phase_factors.angler                                  import  Angler_random
from  pyLIQTR.phase_factors.optimization.angler_optimization        import  Angler_opt

from  pyLIQTR.phase_factors.optimization.expander                   import  Expander
from  pyLIQTR.phase_factors.optimization.fitter                     import  Fitter
from  pyLIQTR.phase_factors.fourier_response.fourier_response       import  Angler_fourier_response, get_steps_from_logeps

from  pyLIQTR.phase_factors.phase_factors                           import  PhaseFactors
        
from  pyLIQTR.utils.plot_helpers                                    import  plot_func_error

from  pyLIQTR.phase_factors.conversions import  phases_Wx_to_R


import mpmath                            as mpm
mpm.mp.prec = 512

import abc

from functools import cache 







class f_sine(PhaseFactors):

    def __init__(self, tau=1.0, 
                       eps=1e-3, 
                       precompute=False, 
                       expansion_method="analytical", 
                       phase_algorithm="Optimization", 
                       **kwargs ):


        self.name         =  "sin"
        self.expr         =  "$f(x) = \sin [ tau \cdot x] / 2$"

        self.params       =  { 'tau'  :  tau}

        self.eps          =  eps
        self.tau          =  tau

        self.convention   =  "R"

        self.deg          =    4
        self.parity       =    1
        self.kappa        = 10.0

        self.phase_algorithm = phase_algorithm
        self.expansion_method=expansion_method

        kwargs['precompute']  =  precompute
        kwargs['eps']         =  self.eps
        kwargs['deg']         =  self.deg

        super().__init__(**kwargs)


        if(self.precompute):
            self.phis


        return

    def func(self,x,tau=1.0):
        return( np.sin(tau*x)/2  )


    @property
    def expansion(self):
        expdr = Expander()
        expdr.ja_sin(self.tau,self.eps)
        return(expdr)
    








class f_cosine(PhaseFactors):

    def __init__(self, tau=1.0, 
                       eps=1e-3, 
                       precompute=False, 
                       expansion_method="analytical", 
                       phase_algorithm="Optimization", 
                       **kwargs ):


        self.name         =  "cos"
        self.expr         =  "$f(x) = \cos [ tau \cdot x] / 2$"
        self.kappa        =  10.0
        self.defaults     =  { 'beta'  :  1.0 }
        self.eps          =  eps
        self.tau          =  tau

        self.params       =  { 'tau'  :  tau}


        self.convention   =  "R"

        self.deg          =    4
        self.parity       =    0
        self.kappa        = 10.0

        self.phase_algorithm = phase_algorithm
        self.expansion_method=expansion_method

        self.precompute = precompute

        kwargs['precompute']  =  self.precompute
        kwargs['eps']         =  self.eps
        kwargs['deg']         =  self.deg

        super().__init__(**kwargs)

        return


    def func(self,x,tau=1.0):
        return( np.cos(tau*x)/2  )


    @property
    def expansion(self):
        expdr = Expander()
        expdr.ja_cos(self.tau,self.eps)
        return(expdr)
    







class f_exp_fourier_response(PhaseFactors):

    def __init__(self, tau=1.0, 
                       eps=1e-2, 
                       expansion_method="analytical", 
                       phase_algorithm="random", 
                       convention="fr",
                       precompute=False, 
                       **kwargs ):


        self.name         =  "exp"
        self.expr         =  "$f(x) = \exp[i tau x]$"
        self.kappa        =  10.0
        self.defaults     =  { 'beta'  :  1.0 }
        self.eps          =  eps
        self.tau          =  tau

        self.params       =  { 'tau'  :  tau}

        self.convention   =  "fr"

        self.deg          =    4
        self.parity       =    0
        self.kappa        = 10.0

        self.phase_algorithm = "random"
        self.expansion_method="analytical"

        self.precompute = precompute

        kwargs['precompute']  =  self.precompute
        kwargs['eps']         =  self.eps
        kwargs['deg']         =  self.deg

        super().__init__(**kwargs)

        return


    def func(self,x,tau=1.0):
        return( np.exp(1.0j*tau*x)  )


    @property
    def expansion(self):
        expdr = Expander()
        expdr.ja_cos(self.tau,self.eps) 
        return(expdr)
    
