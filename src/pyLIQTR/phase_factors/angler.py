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
###             ANGLER   -    Phase angles for QSP and QSVT                  ###
###                                                                          ###
################################################################################
###
###   REVISION: 11/21/2022
###
###   Justin E. Elenewski  -  MIT Lincoln Laboratory
###   justin dot elenewski at ll dot mit dot edu
###o

import abc


import numpy                as np
import scipy.special        as sfn
import copy                 as copy


from   pyLIQTR.phase_factors.optimization.ChebyshevPoly        import ChebyshevPoly




## Parent class for phase angle generators
##
class Angler_abstract:

    def __init__(self,**kwargs):
        return
        
    @abc.abstractmethod
    def generate(self):
        ...







## Angler for random phase angles
##
class Angler_random(Angler_abstract):

    def __init__(self,cheb_poly=None,n=None,silent=False):

        self._silent   =  silent
        self._success  =  False

        if (n is not None):
            self.n_phases = n
        elif (cheb_poly is not None):
            self.n_phases = 2*cheb_poly.deg + 1
        else:
            raise Exception("No polynomial or number of terms specified for random phase determination.")

        self.phases      = None
        self.obj_value   = None


    def generate(self):

        if not self._silent:
            print("########     Random Phase Angles     ########\n\n")
            print("Generated "+str(self.n_phases)+" random phases.\n")
            
                  
        self.phases     =  np.pi*np.random.random(self.n_phases)
        self.obj_value  =  -1.0
        self._success   =  True

    




    



