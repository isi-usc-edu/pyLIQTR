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
from typing import List, Tuple
from numpy.typing import NDArray

import attr
import cirq
import qualtran as qt
import numpy as np
from cirq._compat import cached_property
from qualtran.linalg.lcu_util import preprocess_lcu_coefficients_for_reversible_sampling
from pyLIQTR.circuits.operators.ControlledUniformSuperposition import ControlledPrepareUniformSuperposition

@cirq.value_equality()
@attr.frozen
class InnerPrepare(qt.GateWithRegisters):
    '''
    Implements [In-prep_l] block from Fig 16 of https://arxiv.org/abs/2011.03494. Outlined in step 3 pgs 52-53

    Prepares sum_p sqrt(T_pq') |p>|0> + sum_{l=1}^L (sum_{p=1}^Xi_l sqrt(f_p^l) |p>|succ_p=1> + sum_{p>Xi_l}junk_p|succ_p=0>)|l>

    where succ_p is used to flag when the condition p<=Xi_l is met
    '''
    n_Xi: int # number of Xi bits = int(np.ceil(np.log2(max(Xi_vals)))) see Eq C14
    n_LXi: int # int(np.ceil(np.log2(L*Xi+N/2)))
    alt_data: NDArray
    keep_data: NDArray
    signs_data: NDArray
    alt_signs_data: NDArray
    keep_bitsize: int # bitsize of keep data (mu from alias sampling)
    br: int # precision of rotation in controlled uniform superposition
    bphi: int # number of phase gradient bits (>=br)

    @classmethod
    def from_Tf_arrays(
        cls, T_coeffs: List[float], T_signs: List[int], fpl_coeffs: List[List[float]], fpl_signs: List[List[int]], Xi_vals: List, keep_bitsize:int=8, br:int=7,bphi:int=8
    ) -> 'InnerPrepare':
        '''
        Method to construct InnerPrepare gate for a given set of one-body and double factorization two-body coefficients.

        Args:
            - T_coeffs: should have N/2 elements, corresponds to absolute value of Eq A9
            - T_signs: should have N/2 elements, corresponds to signs of T_coeffs -- 0 for negative, 1 for positive
            - fpl_coeffs: should have LxN/2 elements, corresponds to absolute value of f_p^l in Eq C1
            - fpl_signs: should have LxN/2 elements, corresponds to signs of fpl_coeffs -- 0 for negative, 1 for positive
            - Xi_vals: list of cutoffs for truncating the double factorization, limit on inner sum in Eq C11
            - keep_bitsize: the number of precision bits for alias sampling
            - br: number of bits storing angle for aa rotation used in ControlledPrepareUniformSuperposition
        '''
        L = len(fpl_coeffs)
        max_Xi = len(T_coeffs)#max(Xi_vals)
        num_fpl = sum(Xi_vals)
        num_one_body_coeffs = len(T_coeffs)

        # if all coeffs are 0, don't compute alt vals
        if sum(T_coeffs):
            eps = 2**(-keep_bitsize)/len(T_coeffs)
            alt_data, keep_data, _ = preprocess_lcu_coefficients_for_reversible_sampling(lcu_coefficients=T_coeffs, epsilon=eps)
            alt_signs = [T_signs[i] for i in alt_data]
        else:
            alt_data = [0]*len(T_coeffs)
            keep_data = [0]*len(T_coeffs)
            alt_signs = [0]*len(T_signs)

        signs = T_signs[:]
        for l in range(L):
            Xi_l = Xi_vals[l]
            f_coeffs = fpl_coeffs[l][:Xi_l]
            f_signs = fpl_signs[l][:Xi_l]

            eps_l = 2**(-keep_bitsize)/len(f_coeffs)
            alt_l, keep_l, _ = preprocess_lcu_coefficients_for_reversible_sampling(lcu_coefficients=f_coeffs,epsilon=eps_l)
            alt_signs_l = [f_signs[i] for i in alt_l]

            alt_data.extend(alt_l)
            alt_signs.extend(alt_signs_l)
            keep_data.extend(keep_l)
            signs.extend(f_signs)
        
        return InnerPrepare(
                n_Xi=int(np.ceil(np.log2(max_Xi))),
                n_LXi=int(np.ceil(np.log2(num_fpl+num_one_body_coeffs))),
                alt_data=np.array(alt_data),
                keep_data=np.array(keep_data),
                signs_data=np.array(signs),
                alt_signs_data=np.array(alt_signs),
                keep_bitsize=keep_bitsize,
                br=br,
                bphi=bphi
        )

    @cached_property
    def extra_registers(self)-> Tuple[qt._infra.registers.Register]:
        return ( qt._infra.registers.Signature.build(
            unary_ancilla = self.n_Xi,
            zero_padding = self.n_LXi-self.n_Xi,
            less_than_equal_ancilla = 1,
            alt = self.n_Xi,
            keep = self.keep_bitsize,
            alt_sign = 1,
            sigma = self.keep_bitsize,
            Xi_l = self.n_Xi,
            offset = self.n_LXi,
            rot_data = self.br, # for phase gradient rotation
            rot_ancilla = 1, # target for aa rotation
            succ_p = 1,
            sign = 1,
            phi = self.bphi
        ) )

    @cached_property
    def selection_registers(self) -> Tuple[qt._infra.registers.SelectionRegister]:
        contiguous_reg = qt._infra.registers.SelectionRegister(name="contiguous_index",bitsize=self.n_LXi+1)
        p_reg = qt._infra.registers.SelectionRegister(name="p",bitsize=self.n_Xi)
        return (contiguous_reg, p_reg)

    @cached_property
    def signature(self) -> qt._infra.registers.Signature:
        return qt._infra.registers.Signature([*self.selection_registers, *self.extra_registers])

    def _value_equality_values_(self):
        # NOTE: needed to make t_complexity work. Returns values used to determine when two objects are equal. See https://github.com/quantumlib/Cirq/blob/v1.2.0/cirq-core/cirq/value/value_equality_attr.py#L26 for more info
        return (
            self.n_Xi,
            self.n_LXi,
            tuple(self.alt_data),
            tuple(self.keep_data),
            tuple(self.signs_data),
            tuple(self.alt_signs_data),
            self.keep_bitsize,
            self.br,
            self.bphi,
        )
    
    def decompose_from_registers(
        self,
        *,
        context: cirq.DecompositionContext,
        **quregs: NDArray[cirq.Qid],  # type:ignore[type-var]
    ) -> cirq.OP_TREE:

        index_reg, Xi_l, offset, contiguous_index = quregs['p'], quregs['Xi_l'], quregs['offset'], quregs['contiguous_index']
        unary_ancilla, less_than_equal_ancilla = quregs['unary_ancilla'], quregs['less_than_equal_ancilla']
        alt, keep, sign_reg, alt_sign_reg = quregs['alt'], quregs['keep'], quregs['sign'], quregs['alt_sign']
        sigma = quregs['sigma']
        zero_padding = quregs['zero_padding']

        # prepare equal superposition over Xi_l states on |p> (rounded up to the nearest power of 2, where states with p<Xi_l are flagged on succ_p qubit)
        yield ControlledPrepareUniformSuperposition(br=self.br,bphi=self.bphi,n_Xi=self.n_Xi).on_registers(**quregs)

        # create contiguous index register by adding offset to Xi_l. should cost (n_L,Xi - 1) Toffolis since the contiguous index should iterate over all coefficients which corresponds to the range [0, sum(Xi_l)+N/2-1]
        yield qt.bloqs.arithmetic.addition.OutOfPlaceAdder(bitsize=self.n_LXi).on_registers(a=list(zero_padding)+list(index_reg),b=offset,c=contiguous_index)

        # qrom for alias sampling preparation, output is selected by contiguous index
        qrom_gate = qt.bloqs.qrom.QROM(
            [self.alt_data, self.keep_data, self.signs_data, self.alt_signs_data],
            (self.n_LXi+1,),
            (self.n_Xi,self.keep_bitsize,1,1)
        )
        yield qrom_gate.on_registers(selection=contiguous_index,target0_=alt,target1_=keep,target2_=sign_reg,target3_=alt_sign_reg)

        # prepare uniform superposition on sigma register for comparing to keep during alias sampling procedure
        yield cirq.H.on_each(*sigma)

        # compare keep to sigma for alias sampling
        # TODO: is there a way to explicitly check less_than ancilla can be reused?
        yield qt.bloqs.arithmetic.comparison.LessThanEqual(self.keep_bitsize,self.keep_bitsize).on(*keep, *sigma, *less_than_equal_ancilla)

        # swap alt index values into register where coefficients are to be prepared (p reg and sign)
        yield qt.bloqs.basic_gates.swap.CSwap.make_on(ctrl=less_than_equal_ancilla, x=alt_sign_reg, y=sign_reg)
        yield qt.bloqs.basic_gates.swap.CSwap.make_on(ctrl=less_than_equal_ancilla, x=alt, y=index_reg)

        #uncompute less than -- frees less_than_equal_ancilla
        yield qt.bloqs.arithmetic.comparison.LessThanEqual(self.keep_bitsize,self.keep_bitsize).on(*keep, *sigma, *less_than_equal_ancilla)

        #uncompute add -- uses only Cliffords
        yield qt.bloqs.arithmetic.addition.OutOfPlaceAdder(bitsize=self.n_LXi,adjoint=True).on_registers(a=list(zero_padding)+list(index_reg),b=offset,c=contiguous_index)