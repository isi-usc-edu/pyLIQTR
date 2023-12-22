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
import numpy        as np
from   scipy        import optimize as spo
from   scipy        import special as spc


from   pyLIQTR.phase_factors.angler                        import  Angler_abstract



## Angler for random phase angles
##
class Angler_fourier_response(Angler_abstract):

    def __init__(self,n=None,eps=1e-2,tau=1.0,silent=False,random=False):

        self._silent   =  silent
        self._success  =  False
        self._random   =  random

        if (n is not None):
            self.n_phases = n
        else:
            steps,closeval = get_steps_from_logeps(np.log(eps),tau,1)
            while getlogepsilon(tau,steps)>np.log(eps):
                steps+=4
            self.n_phases = steps+1

        self.phases      = None
        self.obj_value   = None

    def generate(self):

        if not self._silent:
            print("########     Random Phase Angles     ########\n\n")

            print("Fourier-response is deprecated: Genearting equivalent number of random angles.\n\n")
            print("Generated "+str(self.n_phases)+" random phases.\n")
            
        if self._random:
                  
            self.phases     =  np.pi*np.random.random(self.n_phases)
            self.obj_value  =  -1.0
            self._success   =  True





def getlogepsilon(tau,steps,nsegs=1):
    if np.isclose(tau,0):
        val = -np.inf
    else:
        q  = steps//2 + 1
        val  = np.log(32)
        val += np.log(tau/2) * q
        val -= spc.gammaln(q+1) # == log(q!)
        val -= np.log(nsegs) * (q-1)
    return val





def get_steps_from_logeps(true_eps,tau,nsegs=1):

    def f(x):
        return abs(true_eps - getlogepsilon(tau,steps=x,nsegs=1))
    root = spo.minimize_scalar(f)

    approx_steps = int(root.x)
   
    #find closest step divisible by 4
    steps1 = approx_steps - (approx_steps % 4)
    steps2 = (approx_steps + 4) - (approx_steps % 4)
    if (approx_steps - steps1) > (steps2-approx_steps):
        steps = steps2 
    else:
        steps = steps1
    
    return steps,f(steps)





def diffphis(phis):
    phis = np.append(phis[:1],np.append(np.diff(phis),-phis[-1:])+np.pi)
    phis = ((phis+np.pi) % (2*np.pi)) - np.pi
    return phis