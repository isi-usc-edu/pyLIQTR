"""
Copyright (c) 2024 Massachusetts Institute of Technology 
SPDX-License-Identifier: BSD-2-Clause
"""
import numpy as np
from functools import cached_property

from qualtran.bloqs.chemistry.pbc.first_quantization import SelectFirstQuantization
from pyLIQTR.circuits.operators.prepare_FirstQuantized import PrepareFirstQuantization

from pyLIQTR.BlockEncodings import *
from pyLIQTR.BlockEncodings.BlockEncoding import BlockEncoding_select_prepare

class FirstQuantized(BlockEncoding_select_prepare):
    '''
    Implements encoding from Section II of the reference.

    References:
        `Fault-Tolerant Quantum Simulations of Chemistry in First Quantization <https://arxiv.org/abs/2105.12767>`_

    :param ProblemInstance ProblemInstance: A pyLIQTR.ProblemInstance for the system of interest. Currently supports :class:`MoleculeParameters` and :class:`ElectronicStructure` instances.
    :param str geometry_file: File containing the atomic coordinates for the system of interest. Only needed if ProblemInstance is :class:`ElectronicStructure`.
    :param int num_bit_rot_aa: The number of precision bits used for rotating the ancilla qubit used for amplitude amplification in the uniform superpostion subroutine, equivalent to :math:`b_r` in the reference.
    :param float eps_M: The allowable error in the estimated energy due to imprecise preparation of the state with amplitudes :math:`1/||\\nu||`
    :param float eps_R: The allowable error in energies due to the finite precision of the atomic positions.
    :param float eps_T: Allowable error in energies due to the finite precision in the rotation of the qubit selecting between T and U + V.
    :param int n_M: The number of bits used in the inequality testing to prepare a state with amplitudes :math:`1/||\\nu||`, equivalent to :math:`n_{\\mathcal{M}}` in the reference.
    :param int n_T: The number of bits used for rotating the ancilla qubit used to select between T and U+V, equivalent to :math:`n_T` in the reference.
    :param int n_R: The number of bits used to store each component of the atomic position, equivalent to :math:`n_R` in the reference.
    '''
    def __init__(self,ProblemInstance, geometry_file:str=None,num_bits_rot_aa:int=8,eps_M:float=1e-4,eps_R:float=1e-4,eps_T:float=1e-4,n_M=None,n_T=None,n_R=None,**kwargs):
        super().__init__(ProblemInstance,**kwargs)

        self._encoding_type = VALID_ENCODINGS.FirstQuantized

        try:
            problem_parameters = dict(self.PI.yield_FirstQuantized_Info(geometry_file=geometry_file))
        except ValueError:
            raise ValueError(f'geometry_file parameter required for encoding {type(ProblemInstance)}')

        self.N = problem_parameters['num_plane_waves']
        self.Omega = problem_parameters['cell_volume']
        self.L = problem_parameters['num_atoms']
        self.eta = problem_parameters['num_electrons']
        self.lambda_zeta = problem_parameters['sum_atomic_num']

        self.num_bits_p = int(np.ceil(np.log2(self.N**(1/3)+1))) #eq 22
        self.lambda_nu = 2**self.num_bits_p - 1 #eq 25 and N**(1/3) below eq 28
        self.num_bits_rot_aa = num_bits_rot_aa

        if n_M is None:
            self.n_M = self.get_n_M(eps_M)
        else:
            self.n_M = n_M
        self.M_param = int(2**self.n_M)
        
        if n_T is None:
            self.n_T = self.get_n_T(eps_T)
        else:
            self.n_T = n_T

        if n_R is None:
            self.n_R = self.get_n_R(eps_R)
        else:
            self.n_R = n_R

        self._select_gate = SelectFirstQuantization(num_bits_p=self.num_bits_p,eta=self.eta,num_atoms=self.L,lambda_zeta=self.lambda_zeta,m_param=self.M_param,num_bits_nuc_pos=self.n_R,num_bits_t=self.n_T,num_bits_rot_aa=self.num_bits_rot_aa)

        self._prepare_gate = PrepareFirstQuantization(num_bits_p=self.num_bits_p,eta=self.eta,num_atoms=self.L,lambda_zeta=self.lambda_zeta,m_param=self.M_param,num_bits_nuc_pos=self.n_R,num_bits_t=self.n_T,num_bits_rot_aa=self.num_bits_rot_aa)

    @property
    def alpha(self):
        return self.lam
    
    @cached_property
    def lam(self):
        """Calculates value of lambda based on eq 127 of the reference."""
        numerator = self.lam_Tp() + self.lam_U() + self.lam_V()
        return numerator/self.Peq()

    def get_n_M(self,eps):
        """Calculates n_M based on eq 132 of the reference."""
        prefactor = 2*self.eta/(eps*np.pi*self.Omega**(1/3))
        factor1 = self.eta-1+2*self.lambda_zeta
        factor2 = 7*2**(self.num_bits_p+1)-9*self.num_bits_p-11-3*2**(-self.num_bits_p)
        return int(np.ceil(np.log2(prefactor*factor1*factor2)))

    def get_n_R(self,eps):
        """Calculates n_R based on eq 133 of the reference."""
        num = self.eta*self.lambda_zeta*self.lambda_nu
        den = eps*self.Omega**(1/3)
        return int(np.ceil(np.log2(num/den)))

    def get_n_T(self,eps):
        """Calculates n_T based on eq 134 of the reference."""
        return int(np.ceil(np.log2(np.pi*self.lam/eps)))

    def lam_V(self):
        """Calculates lambda_V based on eq 25 of the reference."""
        num = self.eta*(self.eta-1)*self.lambda_nu
        den = 2 * np.pi * self.Omega**(1/3)
        return num/den

    def lam_U(self):
        """Calculates lambda_U based on eq 25 of the reference."""
        num = self.eta * self.lambda_zeta * self.lambda_nu
        den = np.pi*self.Omega**(1/3)
        return num/den

    def lam_Tp(self):
        """Calculates lambda_T' based on eq 71 of the reference."""
        factor1 = 6*self.eta*np.pi**2 / self.Omega**(2/3)
        factor2 = (2**(self.num_bits_p-1))**2
        return factor1*factor2

    def Peq(self):
        """Calculates state preparation success probability based on eq 63 of the reference."""
        # TODO: special case when eta=lambda_zeta

        def Ps(n,br):
            # eq 59
            pow_logn = 2**np.ceil(np.log2(n))

            ## eq 60
            pref = 2**br / 2*np.pi
            to_round = pref * np.arcsin(np.sqrt(pow_logn/(4*n)))
            theta = np.round(to_round)/pref
            ##

            factor1 = 1 + (2- 4*n/pow_logn)*(np.sin(theta))**2
            factor2 = (np.sin(2*theta))**2
            return n*(factor1**2 + factor2)/pow_logn

        return Ps(3,8)*Ps(self.eta+2*self.lambda_zeta,self.num_bits_rot_aa)*(Ps(self.eta,self.num_bits_rot_aa))**2

