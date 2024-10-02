"""
Copyright (c) 2024 Massachusetts Institute of Technology 
SPDX-License-Identifier: BSD-2-Clause
"""
import pandas as pd
import numpy as np
import math
from openfermion.chem import MolecularData

from pyLIQTR.utils.utils import isWindows
if not isWindows():
    from openfermionpyscf import run_pyscf
from openfermion import jordan_wigner
import cirq

def get_hdf5(target_mol_no,filename='chemistry_instances.csv'):
    chem_data = pd.read_csv(filename)
    mol_name = chem_data['molecule']
    mol_basis = chem_data['basis']
    mol_geom = chem_data['geometry']
    mol_mult = chem_data['multiplicity']
    mol_charge = chem_data['charge']
    
    target_mol_name = mol_name[target_mol_no]
    target_mol_basis = 'sto-3g'

    #formatting for molecular geometry
    brack_atoms_1 = mol_geom[target_mol_no].replace("[", "")
    brack_atoms_2 = brack_atoms_1.replace("]", "")
    clean_atoms = brack_atoms_2.replace("'", "")
    at_atoms_1 = clean_atoms.replace("@1", "")
    at_atoms_2 = at_atoms_1.replace("@2", "")
    at_atoms_3 = at_atoms_2.replace("@3", "")
    split_atoms = at_atoms_3.split(',')
    
    target_mol_geom = []
    for atom in split_atoms:
        indv = atom.split()
        el = indv[0]
        x = float(indv[1])
        y = float(indv[2])
        z = float(indv[3])
        coords = (x, y, z)
        atom_geom = (el, coords)
        target_mol_geom.append(atom_geom)
    
    target_mol_mult = mol_mult[target_mol_no]
    target_mol_charge = mol_charge[target_mol_no]
    
    target_mol = MolecularData(target_mol_geom, target_mol_basis, target_mol_mult, target_mol_charge, target_mol_name)

    target_mol = run_pyscf(target_mol, run_scf=1, run_mp2=0, run_cisd=0, run_ccsd=0, run_fci=0, verbose=0)
    return target_mol

















