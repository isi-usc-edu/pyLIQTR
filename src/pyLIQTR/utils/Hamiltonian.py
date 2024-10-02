"""
Copyright (c) 2024 Massachusetts Institute of Technology 
SPDX-License-Identifier: BSD-2-Clause
"""


from pyLIQTR.utils.qsp_helpers import QSPFilesIO
import numpy as np
from typing import Union, List, Tuple

class Hamiltonian:
    def __init__(self, ham_input, ham_type='lcu', N=0):
        # generic init. I don't like it, but can't figure out how to get the types to work!!!
        if ham_type == 'lcu':
            if isinstance(ham_input,str):
                rdr = QSPFilesIO()
                self.terms = QSPFilesIO.readHaml(ham_input)
            elif isinstance(ham_input,list):
                self.terms = ham_input
            else:
                raise ValueError("Unsupported hamiltonian input...")
        
            self.problem_size = len(self.terms[0][0])
            self.__ham_type = 'lcu'
        elif ham_type=='fermionic':
            self.terms = ham_input
            self.problem_size = N
            self.__ham_type = 'fermionic'
        else:
            self.terms = ham_input
            self.problem_size = len(self.terms)
            self.__ham_type = 'list'
            
    def __len__(self):
        return len(self.terms)

    def __str__(self):
        return str(self.terms)

    @property
    def problem_size(self):
        return self.__problem_size
    
    @problem_size.setter
    def problem_size(self,problem_size):
        self.__problem_size = problem_size

    @property
    def terms(self):
        return self.__ham
    
    @terms.setter
    def terms(self, new_val):
        self.__ham = new_val

    @property
    def is_lcu(self):
        return self.__ham_type == 'lcu'
    
    @property
    def is_fermionic(self):
        return self.__ham_type == 'fermionic'
    
    @property
    def loglen(self):
        return int(np.ceil(np.log2(len(self.terms))))

    @property
    def alphas(self):
        return [np.sqrt(np.abs(h[1])) for h in self.terms]

    @property
    def alpha(self):
        return sum([np.abs(h[1]) for h in self.terms])

    def adjust_hamiltonian(self):
        """
        Adjust the length of the hamiltonian to be of size 2^x by adding
        Identity/0 terms (used to help navigate binary trees)
        """
        """
        adjust_haml_c :: Hamiltonian -> Hamiltonian
        adjust_haml_c [] = []
        adjust_haml_c h  = (replicate lef (0,id)) ++ h ++ replicate rig (0,id)
        where
            id  = replicate (length $ snd $ head h) I
            dif = (2^(loglength h)) - (length h)
            lef = div (dif+1) 2
            rig = div (dif  ) 2

        TRASNLATES TO:
        id = replicate(length(snd(head(h))), I)
        dif = (2^loglength(h)) - (length(h))
        lef = floor((dif+1)/2)
        rig = floor(dif/2)
        hl = replicate(lef, (0,id))
        hr = replicate(rig, (0,id))
        return concat(hl,h,hr)
        """
        id = ''.join(['I' for __ in range(self.problem_size)])
        dif = 2**self.loglen - len(self)
        left = int(np.floor((dif+1)/2))
        right = int(np.floor(dif/2))

        #finally...
        hl = [(id,0) for __ in range(left)]
        hr = [(id,0)for __ in range(right)]
        self.terms = hl+self.terms[::-1]+hr
    
