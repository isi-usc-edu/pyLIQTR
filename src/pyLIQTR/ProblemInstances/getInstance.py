"""
Copyright (c) 2024 Massachusetts Institute of Technology 
SPDX-License-Identifier: BSD-2-Clause
"""
from pyLIQTR.ProblemInstances.ProblemInstance      import  ProblemInstance,VALID_ENCODINGS
from pyLIQTR.ProblemInstances.LatticeInstance      import  LatticeInstance
from pyLIQTR.ProblemInstances.fermionic_models     import  FermiHubbard
from pyLIQTR.ProblemInstances.spin_models          import  Heisenberg, Transverse_Field_Ising
from pyLIQTR.ProblemInstances.ChemicalHamiltonian  import  ChemicalHamiltonian
from pyLIQTR.ProblemInstances.ElectronicStructure  import  ElectronicStructure
from pyLIQTR.ProblemInstances.MoleculeParameters   import  MoleculeParameters

validProblemInstances = ["Lattice","FermiHubbard","Heisenberg","ElectronicStructure",\
                         "TransverseFieldIsing",\
                         "ChemicalHamiltonian","MoleculeParameters"]


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
        elif instance == "MoleculeParameters":
            if instantiate:
                return MoleculeParameters(**kwargs)
            else:
                return MoleculeParameters
    else:
        raise ValueError(f"Unhandled problem instance ({instance}), must choose from: {validProblemInstances}")