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
from pyLIQTR.ProblemInstances.ProblemInstance import ProblemInstance
from pyLIQTR.BlockEncodings import VALID_ENCODINGS
from pyLIQTR.BlockEncodings.PauliStringLCU import PauliStringLCU
from pyLIQTR.BlockEncodings.fermi_hubbard_square_encoding import fermi_hubbard_square_encoding
from pyLIQTR.BlockEncodings.LinearT import Fermionic_LinearT
from pyLIQTR.BlockEncodings.DoubleFactorized import DoubleFactorized

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
    else:
        raise NotImplementedError("Sorry :'(")
