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
from abc import ABC, abstractmethod
from functools import cached_property 

from pyLIQTR.BlockEncodings import VALID_ENCODINGS


class ProblemInstance(ABC):

    def __init__(self,**kwargs):
        if "name" in kwargs:
            self._name = kwargs["name"]
        else:
            self._name = ""
        pass

        # terms  =  None


    """
    All classes that inherit from ProblemInstance need to optionally define one of these 
    methods in order to allow a ProblemInstance to be used in conjunction with.

    There may be future methods in the yield_X_Info() style (nominally, one/BlockEncoding)
    """

    def normalize_pauli_lcu(self):
        raise NotImplementedError()


    ## High-level details regarding operator (coefficient sets, etc.)
    def yield_Operator_Info(self):
        raise NotImplementedError()

    ## Matrix representing block encoded operator
    def yield_Operator_Matrix(self):
        raise NotImplementedError()

    def yield_PauliLCU_Info(self,doPad=0):
        raise NotImplementedError()
    
    def yield_Fermionic_Info(self):
        raise NotImplementedError()

    """
    Other methods that may be of use.
    """
    @abstractmethod
    def __str__(self):
        raise NotImplementedError()

    @abstractmethod
    def n_qubits(self):
        """
        Return the number of qubits that are required to implement this problem instance
        """
        raise NotImplementedError()
    

    def pauli_terms(self,return_as='arrays'):
        terms = [ string for string in self.yield_PauliLCU_Info(return_as=return_as) ]
        return(terms)

    @property
    def alpha(self):
        return(self.get_alpha())


    def matrix(self):
        raise NotImplementedError()

    def n_terms(self, encoding:VALID_ENCODINGS, **kwargs):
        """
        Return the number of terms that compose the chosen encoding
        """
        
        # for enc, gen in zip(VALID_ENCODINGS,VALID_GENERATORS):
        #     if (encoding == enc):
        #         info_gen = gen

        # return sum(1 for __ in info_gen(**kwargs))

        if encoding == VALID_ENCODINGS.PauliLCU:
            gen = self.yield_PauliLCU_Info
        if encoding == VALID_ENCODINGS.LinearT:
            gen = self.yield_LinearT_Info
        if encoding == VALID_ENCODINGS.Fermionic:
            gen = self.yield_Fermionic_Info

        return sum(1 for __ in gen(**kwargs))









