"""
Copyright (c) 2024 Massachusetts Institute of Technology 
SPDX-License-Identifier: BSD-2-Clause
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

    




    



