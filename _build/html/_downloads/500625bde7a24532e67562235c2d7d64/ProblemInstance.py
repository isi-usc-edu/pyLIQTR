"""
Copyright (c) 2024 Massachusetts Institute of Technology 
SPDX-License-Identifier: BSD-2-Clause
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









