"""
Copyright (c) 2024 Massachusetts Institute of Technology 
SPDX-License-Identifier: BSD-2-Clause
"""
from pyLIQTR.ProblemInstances.ProblemInstance import ProblemInstance
from pyLIQTR.BlockEncodings import VALID_ENCODINGS
from pyLIQTR.BlockEncodings.PauliStringLCU import PauliStringLCU
from pyLIQTR.BlockEncodings.fermi_hubbard_square_encoding import fermi_hubbard_square_encoding
from pyLIQTR.BlockEncodings.LinearT import Fermionic_LinearT
from pyLIQTR.BlockEncodings.DoubleFactorized import DoubleFactorized
from pyLIQTR.BlockEncodings.FirstQuantized import FirstQuantized

validEncodings = []

def getEncoding(encoding : VALID_ENCODINGS, instance =  None, prepare_type = None, **kwargs):

    instantiate = True if (instance is not None) else False

    if (encoding == VALID_ENCODINGS.PauliLCU):
        if instantiate:
            return PauliStringLCU(ProblemInstance=instance, prepare_type=prepare_type, **kwargs)
        else:
            return PauliStringLCU
    elif (encoding == VALID_ENCODINGS.FermiHubbardSquare):
        if instantiate:
            return fermi_hubbard_square_encoding(ProblemInstance=instance,**kwargs)
        else:
            return fermi_hubbard_square_encoding
    elif encoding == VALID_ENCODINGS.LinearT:
        if instantiate:
            return Fermionic_LinearT(ProblemInstance=instance,**kwargs)
        else:
            return Fermionic_LinearT
    elif encoding == VALID_ENCODINGS.DoubleFactorized:
        if instantiate:
            return DoubleFactorized(ProblemInstance=instance,**kwargs)
        else:
            return DoubleFactorized
    elif encoding == VALID_ENCODINGS.FirstQuantized:
        if instantiate:
            return FirstQuantized(ProblemInstance=instance,**kwargs)
        else:
            return FirstQuantized
    else:
        raise NotImplementedError("Sorry :'(")
