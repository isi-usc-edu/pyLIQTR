"""
Copyright (c) 2024 Massachusetts Institute of Technology 
SPDX-License-Identifier: BSD-2-Clause
"""
import numpy as np
import pandas as pd
import pyLIQTR.pest_interface.pest_python as pp
from functools import cached_property
from typing import Optional
from pyLIQTR.ProblemInstances.ProblemInstance import ProblemInstance

class MoleculeParameters(ProblemInstance):
    """
    This ProblemInstance captures the information about a molecular system that is needed for the FirstQuantized block encoding. The parameters ultimately needed are ``cell_volume``, ``num_electrons``, ``num_atoms``, ``sum_atomic_num`` and ``num_plane_waves``. A geometry file (.xyz) for the system of interest can be optionally used to set ``num_electrons``, ``num_atoms``, and ``sum_atomic_num``, by passing it in with ``geometry_file=``. Additionally, the number of plane waves (``num_plane_waves``) can be optionally calculated from the desired ``E_cut_off`` and ``lengths``.

    :param float cell_volume: The computational cell volume needed for the physical system of interest.
    :param Optional[int] num_electrons: The number of electrons in the system.
    :param Optional[int] num_atoms: The number of atoms in the system.
    :param Optional[int] sum_atomic_num: The sum of the atomic numbers.
    :param Optional[str] geometry_file: The path to the .xyz file containing information about the atoms in the system and their coordinates. This file can be used to set ``num_electrons``, ``num_atoms``, and ``sum_atomic_num``.
    :param Optional[int] num_plane_waves: The number of plane waves the make up the system basis.
    :param Optional[float] E_cut_off: The converged energy cut off in Rydberg units. This can be used together with ``lengths`` to set ``num_plane_waves``.
    :param Optional[list] lengths: A list containing the length of each dimension of the computational cell. This can be used together with ``E_cut_off`` to set ``num_plane_waves``.
    """

    def __init__(self,cell_volume:float,num_electrons:Optional[int]=None,num_atoms:Optional[int]=None,sum_atomic_num:Optional[int]=None,geometry_file:Optional[str]=None,num_plane_waves:Optional[int]=None,E_cut_off:Optional[float]=None,lengths:Optional[list]=None,**kwargs):

        self.cell_volume = cell_volume
        self.num_electrons = num_electrons
        self.num_atoms = num_atoms
        self.sum_atomic_num = sum_atomic_num
        self.num_plane_waves = num_plane_waves
        
        if not ((geometry_file is None) ^ ((num_electrons or num_atoms or sum_atomic_num) is None)):
            raise ValueError("provide only geometry_file or (num_electrons and num_atoms and sum_atomic_num")

        if not ((num_plane_waves is None) ^ ((E_cut_off or lengths) is None)):
            raise ValueError("provide only num_plane_waves or (E_cut_off and lengths")
            
        if geometry_file is not None:
            self.num_atoms, self.num_electrons, self.sum_atomic_num = self.get_params_from_geometry_file(geometry_file=geometry_file)
        
        if lengths is not None:
            assert np.prod(lengths) == cell_volume
            self.num_plane_waves = self.get_num_plane_waves(E_cut_off=E_cut_off,lengths=lengths)

        self._model = "MoleculeParameters"

        super(ProblemInstance, self).__init__(**kwargs)

    def __str__(self):
        return f"{self._model}\r\n\tNumber of electrons: {self.num_electrons}\n\r\tNumber of atoms: {self.num_atoms}\n\r\tSum of atomic numbers: {self.sum_atomic_num}\n\r\tComputational cell volume: {self.cell_volume}\n\r\tNumber of plane waves: {self.num_plane_waves}"

    def n_qubits(self):
        raise NotImplementedError

    def get_alpha(self):
        """returns sum of the absolute value of the hamiltonian coefficients."""
        raise NotImplementedError

    def get_params_from_geometry_file(self,geometry_file:str):
        """returns the number of atoms, number of electrons, and sum of the atomic numbers based on the information in ``geometry_file``."""
        molecule = pd.read_table(geometry_file, skiprows=2, sep='\s+', names=['atom'],usecols=[0])

        eta = 0
        for atom in molecule['atom']:
            eta += pp.dt_atomic_numbers[atom]

        return (len(molecule),eta,eta)
    
    def get_num_plane_waves(self,E_cut_off:float,lengths:list,gamma:float=1.0):
        """returns the number of plane waves calculated from the ``E_cut_off`` and ``lengths``."""
        a0 = pp.pw_to_dpw_cutoff(E_cut=E_cut_off,node_scale=gamma)
        vectors = np.array([ [ lengths[0], 0.0, 0.0 ], [ 0.0, lengths[1], 0.0 ], [ 0.0, 0.0, lengths[2] ] ]) / pp.bohr_to_angstrom
        N = 1
        for n in range(3):
            N *= int(np.ceil(vectors[n][n]/a0))
        return N

    def yield_FirstQuantized_Info(self,geometry_file:str=None):
        """yields parameters used for the FirstQuantized block encoding."""

        # number of plane waves (N)
        yield 'num_plane_waves', self.num_plane_waves

        # computational cell volume (Omega)
        yield 'cell_volume', self.cell_volume

        # number of atoms (L)
        yield 'num_atoms', self.num_atoms

        # number of electrons (eta)
        yield 'num_electrons', self.num_electrons

        # sum of atomic numbers (lambda_zeta)
        yield 'sum_atomic_num', self.sum_atomic_num