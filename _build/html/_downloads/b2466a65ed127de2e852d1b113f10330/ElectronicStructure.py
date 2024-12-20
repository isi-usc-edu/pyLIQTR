"""
Copyright (c) 2024 Massachusetts Institute of Technology 
SPDX-License-Identifier: BSD-2-Clause
"""
import numpy as np
import pandas as pd
import pyLIQTR.pest_interface.pest_python as pp
from functools import cached_property
from pyLIQTR.clam.operator_strings import op_strings
from pyLIQTR.ProblemInstances.ProblemInstance import ProblemInstance

class ElectronicStructure(ProblemInstance):
    """
    This ProblemInstance represents periodic electronic structure Hamiltonians generated using PEST. The Hamiltonians generally take on the form 
    
    .. math::

        H = \\sum_{p,q,\\sigma} T(p-q) a^\\dagger_{p,\\sigma}a_{q,\\sigma} + \\sum_{p,\\sigma} U(p) n_{p,\\sigma} + \\sum_{(p,\\alpha)\\neq(q,\\beta)} V(p-q)n_{p,\\alpha}n_{q,\\beta}
    
    where :math:`a^\dagger_{p,\sigma}` and :math:`a_{p,\sigma}` are fermionic creation and annihilation operators on spatial orbital :math:`p` with spin :math:`\sigma` in {up,down}, and :math:`n_{p,\sigma} = a^\dagger_{p,\sigma}a_{p,\sigma}` is the number operator.

    Given the PEST files, this ProblemInstance can yield the information needed to encode the Hamiltonian using the LinearT, PauliLCU or FirstQuantized block encoding. It also provides descriptions of the electronic structure system size with M_vals, the number of grid points along each spatial dimension, and N, the total number of orbitals.

    :param str filenameH: filename for PEST generated ham.hdf5 file containing the Hamiltonian coefficients
    :param str filenameG: filename for PEST generated grid.hdf5 file containing the basis grid
    """
    def __init__(self,filenameH:str, filenameG:str,**kwargs):

        self._filenameH = filenameH
        self._filenameG =filenameG
        self._H = pp.HamiltonianCoefficients()  
        try:
            self._H.open_grid(self._filenameG)
            self._H.open_hamiltonian(self._filenameH)
        except OSError:
            print(f"Unable to open {self._filenameH} and {self._filenameG}. Check filename and path. To generate new hdf5 files see 'pest-intro.ipynb' ")

        self._M_vals = self._H.n_points() # Number of grid points (orbitals) along each spatial dimension.
        self._N = np.prod(self._M_vals) # doesn't include spin
        self._model = "ElectronicStructure"

        super(ProblemInstance, self).__init__(**kwargs)

    def __str__(self):
        return f"{self._model}\r\n\tHamiltonian filename:{self._filenameH}\tGrid filename:{self._filenameG}\n\r\tN:{self._N}\tM_vals:{self._M_vals}"

    def n_qubits(self):
        # equal to number of basis functions including spin
        return int(2*self._N)

    @cached_property
    def _ops(self):
        _ops = op_strings(N_qb=2*self._N)
        spin_offset = self._N

        for p in range(self._N):

            # loop over q>p and include a factor of 2 since coeffs only depend on abs(p-q)
            for q in range(p+1,self._N):

                xx_str = 'X' + ('Z'*abs(p-q)) + 'X'
                yy_str = 'Y' + ('Z'*abs(p-q)) + 'Y'
                idx_set_up = tuple(np.arange(p,p+abs(p-q)+2,1))
                idx_set_down = tuple(np.arange(p+spin_offset,p+spin_offset+abs(p-q)+2,1))
                
                # NOTE: returns only non zero coefficients
                if self._H.c_xzx(p,q) != 0.0:
                    # spin up
                    _ops.append_tuple((idx_set_up, xx_str, 2*self._H.c_xzx(p,q))) # factor of 2 for q<p case
                    _ops.append_tuple((idx_set_up, yy_str, 2*self._H.c_xzx(p,q)))
                    # spin down
                    _ops.append_tuple((idx_set_down, xx_str, 2*self._H.c_xzx(p,q))) # factor of 2 for q<p case
                    _ops.append_tuple((idx_set_down, yy_str, 2*self._H.c_xzx(p,q)))
                if self._H.c_zz(p,q) != 0.0:
                    # alpha = beta = up, p \neq q
                    _ops.append_tuple(((p,q), 'ZZ', 2*self._H.c_zz(p,q))) # factor of 2 for q<p case
                    # alpha = beta = down, p \neq q
                    _ops.append_tuple(((p+spin_offset,q+spin_offset), 'ZZ', 2*self._H.c_zz(p,q)))
                    # alpha=up, beta=down, p \neq q
                    _ops.append_tuple(((p,q+spin_offset), 'ZZ', 2*self._H.c_zz(p,q)))
                    # alpha=down, beta=up, p \neq q
                    _ops.append_tuple(((p+spin_offset,q), 'ZZ', 2*self._H.c_zz(p,q)))

            if self._H.c_zz(p,p) != 0.0:
                # alpha=up, beta=down, p = q
                _ops.append_tuple(((p,p+spin_offset), 'ZZ', self._H.c_zz(p,p)))
                # alpha=down, beta=up, p = q
                _ops.append_tuple(((p+spin_offset,p), 'ZZ', self._H.c_zz(p,p)))

            if self._H.c_z(p) != 0.0:
                _ops.append_tuple(((p,),'Z',self._H.c_z(p)))

        return _ops

    def get_alpha(self,encoding:str='LinearT'):
        """returns sum of the absolute value of the hamiltonian coefficients."""
        # TODO: can this be more efficient
        if encoding == 'LinearT':
            T_pq_mag_sum = 0
            V_pq_mag_sum = 0
            U_p_mag_sum = 0
            for p in range(self._N):
                U_p_mag_sum += abs(self._H.U(p))
                for q in range(self._N):
                    T_pq_mag_sum += abs(self._H.T(p,q))
                    if p != q: # exclude p=q for V
                        V_pq_mag_sum += abs(self._H.V(p,q))    
            return T_pq_mag_sum + U_p_mag_sum + V_pq_mag_sum

    def yield_PauliLCU_Info(self,return_as='arrays',do_pad=0,pad_value=1.0):
        """yields parameters used for the PauliLCU block encoding."""

        if (return_as == 'arrays'):
            terms = self._ops.terms(do_pad=do_pad,pad_value=pad_value)
        elif (return_as == 'strings'):
            terms = self._ops.strings(do_pad=do_pad,pad_value=pad_value)

        for term in terms:
            yield term

    def yield_LinearT_Info(self,termSelect:str):
        """yields parameters used for the LinearT block encoding."""
        
        if termSelect == 'T':
            # this corresponds to tilde(T)**2 in https://arxiv.org/pdf/1805.03662.pdf
            for p in range(self._N):
                term = 2*self._H.c_xzx(p,0) # factor of 2 since coefficients occur for each xzx/yzy term only once depending on p<q or p>q
                sign = (1-np.sign(term))/2 if np.sign(term) else 0
                yield (int(sign),abs(term))

        if termSelect == 'U':
            for p in range(self._N):
                term = self._H.c_z(p)
                sign = (1-np.sign(term))/2 if np.sign(term) else 0
                yield (int(sign),abs(term))

        if termSelect == 'V':
            for p in range(self._N):
                term = self._H.c_zz(p,0)
                sign = (1-np.sign(term))/2 if np.sign(term) else 0
                yield (int(sign),abs(term))

    def yield_FirstQuantized_Info(self,geometry_file:str):
        """yields parameters used for the FirstQuantized block encoding."""

        molecule = pd.read_table(geometry_file, skiprows=2, sep='\s+', names=['atom'],usecols=[0])

        # number of plane waves (N)
        yield 'num_plane_waves', self._N

        # computational cell volume (Omega)
        yield 'cell_volume', abs(np.linalg.det(self._H.vecs))

        # number of atoms (L)
        yield 'num_atoms', len(molecule)

        # number of electrons (eta)
        eta = 0
        for atom in molecule['atom']:
            eta += pp.dt_atomic_numbers[atom]
        yield 'num_electrons', eta

        # sum of atomic numbers (lambda_zeta)
        yield 'sum_atomic_num', eta