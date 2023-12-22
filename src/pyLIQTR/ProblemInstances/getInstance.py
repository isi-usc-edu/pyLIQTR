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
from pyLIQTR.ProblemInstances.ProblemInstance      import  ProblemInstance,VALID_ENCODINGS
from pyLIQTR.ProblemInstances.LatticeInstance      import  LatticeInstance
from pyLIQTR.ProblemInstances.fermionic_models     import  FermiHubbard
from pyLIQTR.ProblemInstances.spin_models          import  Heisenberg, Transverse_Field_Ising
from pyLIQTR.ProblemInstances.ChemicalHamiltonian  import  ChemicalHamiltonian
from pyLIQTR.ProblemInstances.ElectronicStructure  import  ElectronicStructure

validProblemInstances = ["Lattice","FermiHubbard","Heisenberg","ElectronicStructure",\
                         "TransverseFieldIsing",\
                         "ChemicalHamiltonian"]


def getInstance(instance : str, instantiate = True, **kwargs):
    if instance in validProblemInstances:
        if instance == "Lattice":
            if instantiate:
                return LatticeInstance(**kwargs)
            else:
                return LatticeInstance
        elif instance == "FermiHubbard":
            if instantiate:
                return FermiHubbard(**kwargs)
            else:
                return FermiHubbard
        elif instance == "Heisenberg":
            if instantiate:
                return Heisenberg(**kwargs)
            else:
                return Heisenberg
        elif instance == "TransverseFieldIsing":
            if instantiate:
                return Transverse_Field_Ising(**kwargs)
            else:
                return Transverse_Field_Ising
        elif instance == "ChemicalHamiltonian":
            if instantiate:
                return ChemicalHamiltonian(**kwargs)
            else:
                return ChemicalHamiltonian
        elif instance == "ElectronicStructure":
            if instantiate:
                return ElectronicStructure(**kwargs)
            else:
                return ElectronicStructure
    else:
        raise ValueError(f"Unhandled problem instance ({instance}), must choose from: {validProblemInstances}")