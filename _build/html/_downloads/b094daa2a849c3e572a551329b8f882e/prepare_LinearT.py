"""
Copyright (c) 2024 Massachusetts Institute of Technology 
SPDX-License-Identifier: BSD-2-Clause
"""
import cirq
import attr
import warnings
import numpy as np
import qualtran as qt
from functools import cached_property
from typing import List, Tuple, Sequence
from numpy.typing import NDArray
from qualtran import GateWithRegisters, Register, Signature, BoundedQUInt, QBit, QAny, QInt
from qualtran.linalg.lcu_util import preprocess_lcu_coefficients_for_reversible_sampling
from qualtran.bloqs.data_loading.qrom import QROM
from qualtran.bloqs.mcmt import MultiControlPauli
from qualtran.bloqs.state_preparation import PrepareUniformSuperposition

from pyLIQTR.circuits.operators.AddMod import Add

class FermionicPrepare_LinearT(GateWithRegisters):
    '''
    Implements circuit from Fig. 16 of https://arxiv.org/pdf/1805.03662.pdf.

    Initializes the state 
        
    .. math::
        (\\sum_{p,\\sigma} U(p) |\\theta_p\\rangle |1\\rangle_U |0\\rangle_V |p, \\sigma, p, \\sigma\\rangle  
        + \\sum_{p!=q, \\sigma} T(p-q) |\\theta_{p-q}^0\\rangle |0\\rangle_U |0\\rangle_V |p, \\sigma, q, \\sigma\\rangle 
        + \\sum_{(p,\\alpha)!=(q,\\beta)} V(p-q) |\\theta_{p-q}^1\\rangle |0\\rangle_U |1\\rangle_V  |p,\\alpha,q,\\beta\\rangle) |temp\\rangle
        
    where the coefficients :math:`U(d)`, :math:`T(d)`, and :math:`V(d)` are :math:`\\mu`-bit binary approximations to the true values. :math:`\\mu` is set according to the condition approx_error :math:`<= 1/(2^\\mu N)` where N is the total number of T,U, and V coefficients.

    :param List[Tuple[int,float]] T_array: The (XZX + YZY) operator coefficients, equivalent to :math:`\\tilde{T}^2` in the reference. Formatted such that the ith coefficient, Ti, is given by T_array[i] = ((1-sign(Ti))/2,abs(Ti)).
    :param List[Tuple[int,float]] U_array: The (Z) operator coefficients, equivalent to :math:`\\tilde{U}^2` in the reference. Formatted the same as T_array.
    :param List[Tuple[int,float]] V_array: The (ZZ) operator coefficients, equivalent to :math:`\\tilde{V}^2` in the reference. Formatted the same as T_array.
    :param NDArray[int] M_vals: Number of grid points (orbitals) along each spatial dimension.
    :param float approx_error: The desired accuracy to represent each coefficient which sets :math:`\\mu` size and keep/alt integers. See `qualtran.linalg.lcu_util.preprocess_lcu_coefficients_for_reversible_sampling` for more information.
    '''

    def __init__(self, T_array: List[Tuple[int,float]], U_array: List[Tuple[int,float]], V_array: List[Tuple[int,float]], M_vals: NDArray[np.int_], approx_error: float):

        if any(M_vals <= 2):
            # qualtran AdditionGate doesn't decompose properly for bitsize (logM) of 1 so need this warning for now
            warnings.warn(f"All M_vals must be greater than 2 for full circuit decomposition to work, currently M_vals={M_vals}", stacklevel=2)

        self.__T_array = T_array
        self.__U_array = U_array
        self.__V_array = V_array
        self.__M_vals = M_vals
        self.__approx_error = approx_error

        self.__logM_vals = np.ceil(np.log2(M_vals)).astype('int_').tolist()
        self.__N = 2*int(np.prod(M_vals)) # total number of spin orbitals. Factor of 2 for up and down spin
        self.__Np_bits = sum(self.__logM_vals)
        self.__D = len(M_vals)
        
        num_coeffs = len(T_array) + len(V_array) + len(U_array)
        self.__mu = max(0, int(np.ceil(-np.log2(approx_error * num_coeffs))))
        if self.__mu == 0:
            warnings.warn(f"approx_error={approx_error} may be too large for given basis size {np.prod(self.__M_vals)}",stacklevel=2)

        super(FermionicPrepare_LinearT, self)

    @cached_property
    def selection_registers(self) -> Tuple[Register]:
        theta_reg = Register(name="theta",dtype=QBit())
        U_reg = Register(name="U",dtype=BoundedQUInt(1,2))
        V_reg = Register(name="V",dtype=BoundedQUInt(1,2))
        p_reg = Register(name='p',dtype=BoundedQUInt(bitsize=self.__Np_bits,iteration_length=int(self.__N/2)))
        a_reg = Register(name="a",dtype=BoundedQUInt(1,2))
        q_reg = Register(name='q',dtype=BoundedQUInt(bitsize=self.__Np_bits,iteration_length=int(self.__N/2)))
        b_reg = Register(name="b",dtype=BoundedQUInt(1,2))
        return (theta_reg,U_reg,V_reg,p_reg,a_reg,q_reg,b_reg)

    @cached_property
    def alternates_bitsize(self) -> int:
        # factor of 2 comes from U and V bits
        return int(sum(self.__logM_vals) + 2)

    @cached_property
    def keep_bitsize(self) -> int:
        return self.__mu

    @cached_property
    def junk_registers(self) -> Tuple[Register]:
        # These make up the temp register. They are not perfectly uncomputed due to entanglement with the selection register and so must be retained and passed to the next instance of Subprepare/prepare.
        return ( Signature.build(
            sigma_mu=self.__mu,
            alt=self.alternates_bitsize,
            keep=self.keep_bitsize,
            theta_alt=1,
            less_than_equal=1,
        ) )

    @cached_property
    def signature(self) -> Signature:
        return Signature([*self.selection_registers, *self.junk_registers])
   
    def __repr__(self) -> str:
        T_repr = cirq._compat.proper_repr(self.__T_array)
        U_repr = cirq._compat.proper_repr(self.__U_array)
        V_repr = cirq._compat.proper_repr(self.__V_array)
        return (
            f'pyLIQTR.FermionicPrepare_LinearT('
            f'{T_repr}, '
            f'{U_repr}, '
            f'{V_repr}, '
            f'{self.__M_vals}, '
            f'{self.__approx_error})'
        )

    def decompose_from_registers(
        self, *, context: cirq.DecompositionContext, **quregs: Sequence[cirq.Qid]
    ) -> cirq.OP_TREE:
        
        U, V, p, q = quregs['U'], quregs['V'], quregs['p'], quregs['q']
        a, b = quregs['a'], quregs['b']
        theta, theta_alt = quregs['theta'], quregs['theta_alt']
        sigma_mu, alt, keep = quregs['sigma_mu'], quregs['alt'], quregs['keep']
        less_than_equal = quregs['less_than_equal']

        # seperate p and q registers into D registers each with sizes according to logM_vals
        i=0
        q_regs = []
        p_regs = []
        for logM in self.__logM_vals:
            q_regs.append(q[i:i+logM])
            p_regs.append(p[i:i+logM])
            i += logM

        subprepare = Subprepare_LinearT.from_TUV_arrays(T_array=self.__T_array, U_array=self.__U_array, V_array=self.__V_array, M_vals=self.__M_vals, approx_error=self.__approx_error)

        yield subprepare.on_registers(U=U,V=V,d=p,sigma_mu=sigma_mu,alt=alt,keep=keep,theta=theta,theta_alt=theta_alt,less_than_equal=less_than_equal)

        # prepare a uniform superposition over M for each dimension on the q register, zero-controlled on U
        for qi, q_reg in enumerate(q_regs):
            yield PrepareUniformSuperposition(int(self.__M_vals[qi]),cvs=(0,)).on_registers(ctrl=U,target=q_reg)

        yield cirq.H.on(*a)

        yield cirq.H.on(*b).controlled_by(*V)
        control_bits = [[bit] for bit in list(V)+list(p)]
        yield MultiControlPauli(cvs=(1,)+(0,)*self.__Np_bits,target_gate=cirq.H).on_registers(controls=control_bits,target=b)
        yield MultiControlPauli(cvs=(1,)+(0,)*self.__Np_bits,target_gate=cirq.X).on_registers(controls=control_bits,target=b)

        yield cirq.CNOT(*a,*b)

        yield qt.bloqs.basic_gates.swap.CSwap.make_on(ctrl=U, x=q, y=p)

        for Mi,logM in enumerate(self.__logM_vals):
            yield Add(a_dtype=QInt(logM)).on(*q_regs[Mi],*p_regs[Mi])


@cirq.value_equality()
@attr.frozen
class Subprepare_LinearT(GateWithRegisters):

    '''
    Implements circuit from Fig. 15 of https://arxiv.org/pdf/1805.03662.pdf. Code structure based on qualtran.StatePreparationAliasSampling. 

    Initializes the sate 
    
    .. math::
        \\sum_{d=0}^{N-1} ( U(d) |\\theta_d\\rangle |1\\rangle_U |0\\rangle_V  + T(d) |\\theta_d^0\\rangle |0\\rangle_U |0\\rangle_V  + V(d) |\\theta_d^1\\rangle |0\\rangle_U |1\\rangle_V ) |d\\rangle |temp_d\\rangle
        
    where the coefficients :math:`U(d)`, :math:`T(d)`, and :math:`V(d)` are :math:`\\mu`-bit binary approximations to the true values. The preparation involves classical alias sampling.
    '''

    M_vals: List[int] # number of p index values per dimension
    logM_vals: NDArray[np.int_] # ceil(log2) of the above
    D: int # spatial dimension
    alt: NDArray[np.int_] # alternate indices
    keep: NDArray[np.int_] # probability numerator for keeping the initially sampled index. Full probability is keep/2**mu
    theta: NDArray[np.int_] # represents signs (+/-) of input coefficients. 0 for positive, 1 for negative.
    theta_alt: NDArray[np.int_] # signs of coefficients at the alternate indices.
    mu: int # exponent of the denominator to divide keep by in order to get a probability. Related to precision of the approximation.
    

    @classmethod
    def from_TUV_arrays(
        cls, T_array: List[Tuple[int,float]],U_array: List[Tuple[int,float]],V_array: List[Tuple[int,float]], M_vals: NDArray[np.int_] ,*, approx_error: float = 1.0e-3
    ) -> 'Subprepare_LinearT':
        """Factory to construct the state subpreparation gate for a given set of T, U, and V coefficients.

        Args:
            T_array: The (XZX + YZY) operator coefficients, equivalent to tilde(T)**2 in the reference. Formatted such that the ith coefficient, Ti = T(i), is given by T_array[i] = ((1-sign(Ti))/2,abs(Ti)).
            U_array: The (Z) operator coefficients, equivalent to tilde(U)**2 in the reference. Formatted the same as T_array.
            V_array: The (ZZ) operator coefficients, equivalent to tilde(V)**2 in the reference. Formatted the same as T_array.
            M_vals: Number of grid points (spin orbitals) along each spatial dimension.
            approx_error: The desired accuracy to represent each coefficient
                (which sets mu size and keep/alt integers).
                See `qualtran.linalg.lcu_util.preprocess_lcu_coefficients_for_reversible_sampling`
                for more information.
        """

        # NOTE: the order of T, V, and U is important since T -> 00, V -> 01, U -> 10 in terms of the UV registers
        coefficients = [coeff[1] for coeff in T_array + V_array + U_array]
        theta = np.array([coeff[0] for coeff in T_array + V_array + U_array]) # entries should be 0 or 1

        alt, keep, mu = preprocess_lcu_coefficients_for_reversible_sampling(
            lcu_coefficients=coefficients, epsilon=approx_error
        )
        theta_alt = np.array([theta[i] for i in alt])

        logM_vals = np.ceil(np.log2(M_vals)).astype('int_')
        D = len(M_vals)

        return Subprepare_LinearT(
            M_vals=M_vals.tolist(), # convert numpy int to python int
            logM_vals=logM_vals,
            D=D,
            alt=np.array(alt),
            keep=np.array(keep),
            theta=theta,
            theta_alt = theta_alt,
            mu=mu,
        )

    @cached_property
    def alternates_bitsize(self) -> int:
        # factor of 2 comes from U and V bits
        return int(sum(self.logM_vals) + 2)

    @cached_property
    def keep_bitsize(self) -> int:
        return self.mu

    @cached_property
    def selection_bitsize(self) -> int:
        # factor of 2 comes from U and V bits
        return int(sum(self.logM_vals) + 2)

    @cached_property
    def junk_registers(self) -> Tuple[Register]:
        # These (except for theta) make up the temp register. They are not perfectly uncomputed due to entanglement with the selection register and so must be retained and passed to the next instance of Subprepare/prepare.
        return ( Signature.build(
            sigma_mu=self.mu,
            alt=self.alternates_bitsize,
            keep=self.keep_bitsize,
            theta=1,
            theta_alt=1,
            less_than_equal=1,
        ) )

    @cached_property
    def selection_registers(self) -> Tuple[Register]:
        U_reg = Register(name="U",dtype=BoundedQUInt(1,2))
        V_reg = Register(name="V",dtype=BoundedQUInt(1,2))
        d_reg = Register("d",dtype=QAny(int(sum(self.logM_vals))))
        return (
            U_reg,
            V_reg,
            d_reg,
        )

    @cached_property
    def signature(self) -> Signature:
        return Signature([*self.selection_registers, *self.junk_registers])


    def __repr__(self) -> str:
        M_repr = cirq._compat.proper_repr(self.M_vals)
        logM_repr = cirq._compat.proper_repr(self.logM_vals)
        alt_repr = cirq._compat.proper_repr(self.alt)
        keep_repr = cirq._compat.proper_repr(self.keep)
        theta_repr = cirq._compat.proper_repr(self.theta)
        theta_alt_repr = cirq._compat.proper_repr(self.theta_alt)
        return (
            f'pyLIQTR.Subprepare_LinearT('
            f'{M_repr}, '
            f'{logM_repr}, '
            f'{self.D}, '
            f'{alt_repr}, '
            f'{keep_repr}, '
            f'{theta_repr}, '
            f'{theta_alt_repr}, '
            f'{self.mu})'
        )

    def _value_equality_values_(self):
        # NOTE: needed to make qualtran.t_complexity() work. Returns values used to determine when two objects are equal. See https://github.com/quantumlib/Cirq/blob/v1.2.0/cirq-core/cirq/value/value_equality_attr.py#L26 for more info
        return (
            tuple(self.M_vals),
            tuple(self.logM_vals),
            self.D,
            tuple(self.alt),
            tuple(self.keep),
            tuple(self.theta),
            tuple(self.theta_alt),
            self.mu,
        )

    def decompose_from_registers(
        self, *, context: cirq.DecompositionContext, **quregs: Sequence[cirq.Qid]
    ) -> cirq.OP_TREE:
        
        U, V, d = quregs['U'], quregs['V'], quregs['d']
        sigma_mu, alt, keep = quregs['sigma_mu'], quregs['alt'], quregs['keep']
        theta, theta_alt = quregs['theta'], quregs['theta_alt']
        less_than_equal = quregs['less_than_equal']

        # seperate d register into D registers with sizes according to logM_vals
        i=0
        d_regs = []
        for logM in self.logM_vals:
            d_regs.append(d[i:i+logM])
            i += logM

        # prepare a uniform superposition on the UV bits to produce |00>_UV -> (|00>_UV + |01>_UV + |10>_UV)/sqrt(3)
        yield PrepareUniformSuperposition(3).on_registers(target=[*quregs['U'],*quregs['V']],controls=[])
        # prepare a uniform superposition over M for each dimension on the d register
        for di, d_reg in enumerate(d_regs):
            yield PrepareUniformSuperposition(self.M_vals[di]).on(*d_reg)
        # prepare uniform superposition on sigma_mu register for comparing to keep during alias sampling procedure
        yield cirq.H.on_each(*sigma_mu)
        # use QROM to iterate over combined UV-d registers to load theta, alt and keep data
        qrom_gate = qt.bloqs.data_loading.QROM(
            [self.alt, self.keep, self.theta, self.theta_alt],
            (self.selection_bitsize,),
            (self.alternates_bitsize, self.keep_bitsize, 1, 1),
        )
        yield qrom_gate.on_registers(selection=[*quregs['U'],*quregs['V'],*quregs['d']], target0_=alt, target1_=keep, target2_=theta, target3_=theta_alt)
        # flip less_than_equal bit when keep is less than sigma_mu
        yield qt.bloqs.arithmetic.comparison.LessThanEqual(self.mu, self.mu).on(
            *keep, *sigma_mu, *less_than_equal
        )
        # swap alt data controlled on less_than_equal_bit
        yield qt.bloqs.basic_gates.swap.CSwap.make_on(
            ctrl=less_than_equal, x=theta_alt, y=theta
        )
        ## the zero indexed alt bit corresponds to U
        yield qt.bloqs.basic_gates.swap.CSwap.make_on(
            ctrl=less_than_equal, x=[alt[0]], y=U
        )
        ## the one indexed alt bit corresponds to V
        yield qt.bloqs.basic_gates.swap.CSwap.make_on(
            ctrl=less_than_equal, x=[alt[1]], y=V
        )
        ## the remaining alt bits correspond to d
        yield qt.bloqs.basic_gates.swap.CSwap.make_on(
            ctrl=less_than_equal, x=alt[2:], y=d
        )
        # undo the less than comparison so the less_than_equal_bit returns to |0>
        yield qt.bloqs.arithmetic.comparison.LessThanEqual(self.mu, self.mu).on(
            *keep, *sigma_mu, *less_than_equal
        )