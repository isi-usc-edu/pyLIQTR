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
import qualtran as qt
import cirq

import numpy as np
from cirq._compat import cached_property
from numpy.typing import NDArray
from typing import List, Tuple

from pyLIQTR.BlockEncodings import *
from pyLIQTR.BlockEncodings.BlockEncoding import BlockEncoding

from pyLIQTR.circuits.operators.DF_OuterPrepare import OuterPrepare
from pyLIQTR.circuits.operators.DF_InnerPrepare import InnerPrepare
from pyLIQTR.circuits.operators.DF_RotationsBlock import RotationsBlock
from pyLIQTR.circuits.operators.RotationsQROM import RotationsQROM

from qualtran.linalg.lcu_util import _differences, _partial_sums
from qualtran.cirq_interop.bit_tools import iter_bits_fixed_point
from qualtran.bloqs.multi_control_multi_target_pauli import MultiControlPauli
from qualtran.bloqs.rotations.phase_gradient import PhaseGradientState

class DoubleFactorized(BlockEncoding):
    '''
    Implements encoding from Appendix C Figure 16 of Ref [1].

    Args:
        - ProblemInstance: A pyLIQTR.ProblemInstance for the system of interest. Currently supports ChemicalHamiltonian instances.
        - df_error_threshold: The threshold used to throw out factors from the double factorization. Truncation is carried out as described in Appendix C Section 3 of Ref [1]. This parameter corresponds to the RHS of Eq C41.
        - sf_error_threshold: The threshold used to throw out factors from the first eigendecomposition. Terms with eigenvalues less than or equal to sf_error_threshold are thrown out. This decreases L, the rank of the two body tensor in eq C10.
        - br: The number of precision bits to use for the rotation angles output by the QROM in step 2 and used in step 3a)iv on page 52 of Ref [1]. Note, this is not used in step 1a which does not currently use a phase gradient rotation.
        - phase_gradient_eps: Overall error in gradient state preparation, ie each rotation done to prepare the phase gradient state will be performed with error phase_gradient_eps/bits_rot_givens
        - energy_error: The allowable error in phase estimation energy. Used to set bits_rot_givens, keep_bitsize, and outer_prep_eps if step_error is not passed as input.

        Optional inputs:
        - step_error: Error in spectral norm of walk operator. Corresponds to epsilon in eq C24 of Ref [1]. Used to set bits_rot_givens, keep_bitsize, and outer_prep_eps if those are not passed as inputs.
        - bits_rot_givens: The number of precision bits to use for the Givens rotations in step 4 on page 53 of Ref [1]. Called $\beth$ in Ref [1].
        - keep_bitsize: Number of precision bits to use for preparing the coefficients on the second selection register (p). N2 in Eq C31 of Ref [1].
        - outer_prep_eps: Precision of normalized coefficients prepared on first selection register (l). Used to set N1 in Eq C27 of Ref [1].

    References:
    [1] https://arxiv.org/abs/2011.03494
    [2] https://arxiv.org/abs/2007.14460

    '''
    def __init__(self,ProblemInstance,df_error_threshold:float=1e-3,sf_error_threshold=1e-8,br:int=7,phase_gradient_eps=1e-10,energy_error=1e-3,step_error=None,bits_rot_givens=None,keep_bitsize=None,outer_prep_eps=None,**kwargs):
        super().__init__(ProblemInstance,**kwargs)

        one_body_array, two_body_array, self.Xi_l_data, self.givens_angle_tensor = self.PI.yield_DF_Info(df_error_threshold=df_error_threshold,sf_error_threshold=sf_error_threshold)
        
        self._encoding_type = VALID_ENCODINGS.DoubleFactorized
        self.N = len(one_body_array[0])*2 # number of spin orbitals
        self.L = len(two_body_array) # rank of two body tensor
        self.one_body_signs = one_body_array[0]
        self.one_body_mags = one_body_array[1] # these correspond to |T_pq'| in the ref
        self.two_body_signs = [two_body_array[l][0] for l in range(self.L)]
        self.two_body_mags = [two_body_array[l][1] for l in range(self.L)] # these correspond to |f_p^l| in the ref
        

        # define bit lengths for qubit registers
        self.nL = (self.L+1-1).bit_length() 
        self.nXi = (int(np.ceil(self.N/2)-1)).bit_length() #ref says bitlength of max(Xi_l) but |p> reg is used to index over one body coeffs too
        self.nLXi = (sum(self.Xi_l_data)+int(np.ceil(self.N/2))-1).bit_length()

        # define error/precision related parameters
        self.phase_gradient_eps = phase_gradient_eps  #overall error in gradient state preparation, ie each rotation done to prepare the phase gradient state will be performed with error eps/bgrad where here bgrad = bits_rot_givens
        self.df_error_threshold = df_error_threshold
        self.sf_error_threshold = sf_error_threshold
        self.br = br # bits precision for single qubit rotation in amplitude amplification in outer and inner prep
        self.energy_error = energy_error

        ## defaults for optional input parameters based on energy_error if step_error not provided
        if step_error is None:
            # default based on paragraph below Eq 1 of Ref [2]
            self.step_error = 0.1*energy_error/self.alpha 
        else:
            self.step_error = step_error
        if bits_rot_givens is None:
            # default based on Eq C24 of Ref [1]
            self.bits_rot_givens = int(np.ceil(5.652 + np.log2(self.N/(2*self.step_error))))# bits precision for givens angle rotations in controlled rotations block, called $\beth$ in ref Eq C24
        else:
            self.bits_rot_givens = bits_rot_givens
        if keep_bitsize is None:
            self.keep_bitsize = int(np.ceil(2.5+np.log2(1/self.step_error))) # N in Eq C14
        else:
            self.keep_bitsize = keep_bitsize
        if outer_prep_eps is None:
            self.outer_prep_eps = self.step_error
        else:
            self.outer_prep_eps = outer_prep_eps

        self.keep_bitsize_outer = max(0, int(np.ceil(-np.log2(self.outer_prep_eps * (self.L+1))))) # N1 in Eq C27

    @cached_property
    def alpha(self):
        return self.PI.get_alpha(encoding='DF',df_error_threshold=self.df_error_threshold,sf_error_threshold=self.sf_error_threshold)
    
    @cached_property
    def control_registers(self) -> Tuple[qt._infra.registers.Register]:
        registers = () if self._control_val is None else (qt._infra.registers.Register('control', 1),)
        return registers

    @cached_property
    def selection_registers(self) -> Tuple[qt._infra.registers.SelectionRegister]:
        l_reg = qt._infra.registers.SelectionRegister(name='l',bitsize=self.nL,iteration_length=self.L + 1)
        p_reg = qt._infra.registers.SelectionRegister(name='p',bitsize=self.nXi)
        return (l_reg, p_reg)

    @cached_property
    def extra_registers(self) -> Tuple[qt._infra.registers.Register]:
        return ( qt._infra.registers.Signature.build(
            succ_l = 1,
            l_neq_0 = 1,
            Xi_l = self.nXi,
            offset = self.nLXi,
            rot_data = self.br,
            succ_p = 1,
            rotations = int((np.ceil(self.N/2)-1)*self.bits_rot_givens),
            spin_select = 1,
            zero_padding = self.nLXi-self.nXi,
            sign = 1,
            phi = self.bits_rot_givens, # phase gradient state 
            less_than_equal_ancilla = 1,
        ))
    
    @cached_property
    def inner_prep_extra_registers(self)-> Tuple[qt._infra.registers.Register]:
        return ( qt._infra.registers.Signature.build(
            contiguous_index = self.nLXi+1,
            rot_ancilla = 1, # target for aa rotation
            unary_ancilla = self.nXi,
            alt = self.nXi,
            keep = self.keep_bitsize,
            alt_sign = 1,
            sigma = self.keep_bitsize,
        ))

    @cached_property
    def outer_prep_extra_registers(self)-> Tuple[qt._infra.registers.Register]:
        return ( qt._infra.registers.Signature.build(
            alt_l = self.nL,
            keep_l = self.keep_bitsize_outer,
            sigma_l = self.keep_bitsize_outer,
        ))

    @cached_property
    def target_registers(self) -> Tuple[qt._infra.registers.Register]:
        halfN = int(np.ceil(self.N/2))
        return ( qt._infra.registers.Signature.build(
            target_spin_up = halfN,
            target_spin_down = halfN
        ))

    @cached_property
    def signature(self) -> qt._infra.registers.Signature:
        return qt._infra.registers.Signature(
            [*self.control_registers, *self.selection_registers, *self.extra_registers,*self.inner_prep_extra_registers, *self.outer_prep_extra_registers, *self.target_registers]
        )

    def compute_data_l(self):
        l_neq_0_data = np.ones(self.L+1)
        l_neq_0_data[0] = 0 # 0 for first entry, one otherwise

        halfN = np.ceil(self.N/2)
        Xi_l_data_with_1B = [halfN]+self.Xi_l_data # first element should be number of one-electron terms
        Xi_l_data_with_1B = np.array(Xi_l_data_with_1B)-1 # shift down 1 since summation indexing starts at 0

        # compute offsets
        offset_data = np.array(list(_partial_sums(Xi_l_data_with_1B)))[:-1]

        # compute rotation angles based on truncations (Xi_l)
        floor_n_Xi = np.floor(np.log2(Xi_l_data_with_1B))
        rotation_angles = 2*np.arccos(1-2**floor_n_Xi / Xi_l_data_with_1B)
        # account for num bits rotation
        rot_data = approx_angles_as_ints_with_br_bits(rotation_angles,self.br)

        return (l_neq_0_data,np.array(Xi_l_data_with_1B),offset_data,rot_data)

    def get_givens_angles(self):
        # B = bits_rot_givens
        # There are N/2-1 rotations per coefficient, each rotation angle uses B bits 
        # qrom indexes over number of coefficients (call it M) so
        # for m in M:
        # givens_angles[m] = np.zeros(B*(N/2-1))
        # such that there are N/2-1 batches of B bits, ie 
        # for i in range(N/2-1)
        #   givens_angles[m,i*B:(i+1)*B] = ith rotation angle
        halfN = int(np.ceil(self.N/2))
        p_sum_limits = [halfN]+self.Xi_l_data
        num_coeffs = int(sum(p_sum_limits))
        givens_angles = np.zeros((num_coeffs,self.bits_rot_givens*(halfN-1)))
        m = 0
        for l in range(self.L+1):
            for p in range(p_sum_limits[l]): 
                for i,theta in enumerate(self.givens_angle_tensor[l,p,:]):
                    theta_normalized = theta/(2*np.pi) % 1
                    binary_theta = list(iter_bits_fixed_point(theta_normalized,width=self.bits_rot_givens))
                    givens_angles[m,i*self.bits_rot_givens:(i+1)*self.bits_rot_givens] = binary_theta #lsb is last element in list
                m += 1
        return givens_angles


    def decompose_from_registers(self, context, **quregs):
        control = quregs.get('control', ())
        succ_l, l_reg = quregs['succ_l'], quregs['l']
        l_neq_0, Xi_l, offset, rot = quregs['l_neq_0'], quregs['Xi_l'], quregs['offset'], quregs['rot_data']
        succ_p, p_reg = quregs['succ_p'], quregs['p']
        rotations, spin_select = quregs['rotations'], quregs['spin_select']
        target_up, target_down = quregs['target_spin_up'], quregs['target_spin_down']
        zero_padding = quregs['zero_padding']
        sign_qb = quregs['sign']
        phase_gradient_state = quregs['phi']
        rot_aa_ancilla, sigma = quregs['rot_ancilla'], quregs['sigma']
        sigma_l, alt_l, keep_l = quregs['sigma_l'], quregs['alt_l'], quregs['keep_l']
        less_than_equal_ancilla = quregs['less_than_equal_ancilla']

        # prepare phase gradient state
        # TODO what should eps be for preparing phase gradient state (eps is overall error in gradient state preparation, ie each rotation will be performed with error eps/bgrad where here bgrad = bits_rot_givens)
        yield PhaseGradientState(bitsize=self.bits_rot_givens, eps=self.phase_gradient_eps).on(*phase_gradient_state)
        
        # prepare superposition over l
        # calculate outer coeffs. First element should be l=0 term (ie sum(Tpq)), then l=1 to L terms correspond to sum_p(fp^l)
        outer_coefficients = np.concatenate(([sum(self.one_body_mags)],np.sum(self.two_body_mags,axis=1)))
        outer_prep = OuterPrepare.from_lcu_probs(lcu_probabilities=outer_coefficients,probability_epsilon=self.outer_prep_eps) #epsilon should be consistent with keep_bitsize_outer
        yield outer_prep.on_registers(success=succ_l,selection=l_reg,sigma_mu=sigma_l,alt=alt_l,keep=keep_l,less_than_equal=less_than_equal_ancilla)

        # load l != 0, Xi^l (truncation), offset, and rot (amplitude amplification) data
        l_neq_0_data,Xi_l_data_with_1B,offset_data,rot_data = self.compute_data_l()
        # 
        qrom_gate = qt.bloqs.qrom.QROM(
            data=[l_neq_0_data,Xi_l_data_with_1B,offset_data,rot_data],
            selection_bitsizes=(self.nL,),
            target_bitsizes=(1,self.nXi,self.nLXi,self.br)
            )
        yield qrom_gate.on_registers(selection=l_reg,target0_=l_neq_0,target1_=Xi_l,target2_=offset,target3_=rot)

        # prepare superposition over p
        In_prep_l = InnerPrepare.from_Tf_arrays(T_coeffs=self.one_body_mags,T_signs=self.one_body_signs, fpl_coeffs=self.two_body_mags, fpl_signs=self.two_body_signs,Xi_vals=self.Xi_l_data,keep_bitsize=self.keep_bitsize,br=self.br,bphi=self.bits_rot_givens)
        yield In_prep_l.on_registers(**quregs)

        # add p and offset to create contiguous register. contiguous index should iterate over all coefficients which corresponds to the range [0, sum(Xi_l)+N/2-1]
        yield qt.bloqs.arithmetic.Add(bitsize=self.nLXi).on_registers(a=offset,b=list(zero_padding)+list(p_reg))

        # QROM for Givens rotation angles
        givens_angles = self.get_givens_angles()
        qrom_rotations = RotationsQROM(
            data = [givens_angles],
            selection_bitsizes=(self.nLXi,),
            target_bitsizes=(len(rotations),)
        )
        # TODO: implement optimal data loading with clean ancilla based on https://arxiv.org/pdf/1902.02134.pdf appendix B
        yield qrom_rotations.on_registers(selection0=list(zero_padding)+list(p_reg),target0_=rotations)

        # prepare system registers
        yield cirq.H.on(*spin_select)
        yield qt.bloqs.basic_gates.swap.CSwap.make_on(ctrl=spin_select, x=target_down, y=target_up)
        
        # controlled rotations
        controlled_rotations = RotationsBlock(num_data_bits=len(rotations),num_target_bits=len(target_down),precision_bits=self.bits_rot_givens,phase_gradient_bits=self.bits_rot_givens)
        yield controlled_rotations.on_registers(angle_data=rotations,target=target_down,phi=phase_gradient_state)

        Z1 = cirq.ZPowGate(exponent=-1,global_shift=-1/2)
        if not self._controlled:
            # controlled Z1 
            controlled_Z1 = MultiControlPauli(cvs=(1,1),target_gate=Z1)
            yield controlled_Z1.on_registers(controls=[*succ_l,*succ_p],target=target_down[0])

            # controlled Z for sign qubit
            sign_controlled_Z = MultiControlPauli(cvs=(1,1),target_gate=cirq.Z)
            yield sign_controlled_Z.on_registers(controls=[*succ_l,*succ_p],target=sign_qb)
        else:
            # controlled Z1
            controlled_Z1 = MultiControlPauli(cvs=(self._control_val,1,1),target_gate=Z1)
            yield controlled_Z1.on_registers(controls=[*control,*succ_l,*succ_p],target=target_down[0])

            # controlled Z for sign qubit
            sign_controlled_Z = MultiControlPauli(cvs=(self._control_val,1,1),target_gate=cirq.Z)
            yield sign_controlled_Z.on_registers(controls=[*control,*succ_l,*succ_p],target=sign_qb)
        #################################### partial uncompute ####################################

        ## undo controlled rotations
        yield controlled_rotations.on_registers(angle_data=rotations,target=target_down,phi=phase_gradient_state)

        ## undo givens qrom using measurement based uncompute
        yield qrom_rotations.measurement_uncompute(selection=list(zero_padding)+list(p_reg),data=rotations,measurement_key='first_qrom_data_measurement')

        ## undo prepare system registers
        yield qt.bloqs.basic_gates.swap.CSwap.make_on(ctrl=spin_select, x=target_down, y=target_up)
        yield cirq.H.on(*spin_select)

        ## undo contiguous register addition
        yield cirq.inverse(qt.bloqs.arithmetic.Add(bitsize=self.nLXi).on_registers(a=offset,b=list(zero_padding)+list(p_reg)))

        ## undo InnerPrepare
        yield cirq.inverse(In_prep_l.on_registers(**quregs))

        #################################### reflect ####################################
        if not self._controlled:
            reflect = MultiControlPauli(cvs=(1,1,)+(0,)*(self.nXi+1+self.keep_bitsize),target_gate=cirq.Z)
            yield reflect.on_registers(controls=[*succ_l,*l_neq_0,*p_reg,*rot_aa_ancilla,*sigma],target=spin_select)
        else:
            reflect = MultiControlPauli(cvs=(self._control_val,1,1,)+(0,)*(self.nXi+1+self.keep_bitsize),target_gate=cirq.Z)
            yield reflect.on_registers(controls=[*control,*succ_l,*l_neq_0,*p_reg,*rot_aa_ancilla,*sigma],target=spin_select)

        #################################### redo steps BUT for two-body term only ####################################

        # prepare superposition over p
        # one-body coefficients aren't needed so set T'=0
        In_prep_l_2b = InnerPrepare.from_Tf_arrays(T_coeffs=[0]*len(self.one_body_mags),T_signs=[0]*len(self.one_body_signs), fpl_coeffs=self.two_body_mags, fpl_signs=self.two_body_signs,Xi_vals=self.Xi_l_data,keep_bitsize=self.keep_bitsize,br=self.br,bphi=self.bits_rot_givens)
        yield In_prep_l_2b.on_registers(**quregs)

        # add p and offset to create contiguous register. contiguous index should iterate over all coefficients which corresponds to the range [0, sum(Xi_l)+N/2-1]
        yield qt.bloqs.arithmetic.Add(bitsize=self.nLXi).on_registers(a=offset,b=list(zero_padding)+list(p_reg))

        # QROM for Givens rotation angles
        # TODO: only need to load angles for two body tensor
        yield qrom_rotations.on_registers(selection0=list(zero_padding)+list(p_reg),target0_=rotations)

        # prepare system registers
        yield cirq.H.on(*spin_select)
        yield qt.bloqs.basic_gates.swap.CSwap.make_on(ctrl=spin_select, x=target_down, y=target_up)
        
        # controlled rotations
        yield controlled_rotations.on_registers(angle_data=rotations,target=target_down,phi=phase_gradient_state)

        # controlled Z1 
        if not self._controlled:
            # controlled Z1 
            more_controlled_Z1 = MultiControlPauli(cvs=(1,1,1),target_gate=Z1)
            yield more_controlled_Z1.on_registers(controls=[*succ_l,*succ_p,*l_neq_0],target=target_down[0])

            # controlled Z for sign qubit
            sign_more_controlled_Z = MultiControlPauli(cvs=(1,1,1),target_gate=cirq.Z)
            yield sign_more_controlled_Z.on_registers(controls=[*succ_l,*succ_p,*l_neq_0],target=sign_qb)
        else:
            # controlled Z1
            more_controlled_Z1 = MultiControlPauli(cvs=(self._control_val,1,1,1),target_gate=Z1)
            yield more_controlled_Z1.on_registers(controls=[*control,*succ_l,*succ_p,*l_neq_0],target=target_down[0])

            # controlled Z for sign qubit
            sign_more_controlled_Z = MultiControlPauli(cvs=(self._control_val,1,1,1),target_gate=cirq.Z)
            yield sign_more_controlled_Z.on_registers(controls=[*control,*succ_l,*succ_p,*l_neq_0],target=sign_qb)

        #################################### full uncompute ####################################

        ## undo controlled rotations
        yield controlled_rotations.on_registers(angle_data=rotations,target=target_down,phi=phase_gradient_state)

        ## undo givens qrom using measurement based uncompute
        yield qrom_rotations.measurement_uncompute(selection=list(zero_padding)+list(p_reg),data=rotations,measurement_key='second_qrom_data_measurement') 

        ## undo prepare system registers
        yield qt.bloqs.basic_gates.swap.CSwap.make_on(ctrl=spin_select, x=target_down, y=target_up)
        yield cirq.H.on(*spin_select)

        ## undo contiguous register addition
        yield cirq.inverse(qt.bloqs.arithmetic.Add(bitsize=self.nLXi).on_registers(a=offset,b=list(zero_padding)+list(p_reg)))

        ## undo InnerPrepare
        yield cirq.inverse(In_prep_l_2b.on_registers(**quregs))

        ## undo In_l - data_l
        yield qrom_gate.on_registers(selection=l_reg,target0_=l_neq_0,target1_=Xi_l,target2_=offset,target3_=rot)

        ## undo OuterPrepare
        yield cirq.inverse(outer_prep.on_registers(success=succ_l,selection=l_reg,sigma_mu=sigma_l,alt=alt_l,keep=keep_l,less_than_equal=less_than_equal_ancilla))

def approx_angles_as_ints_with_br_bits(angles:NDArray[float],br:int=10):
    angles_normalized = angles / (2*np.pi) % 1
    approx_ints = np.zeros(len(angles),dtype=int)
    for i,angle in enumerate(angles_normalized):
        binary_angle = [*iter_bits_fixed_point(angle,width=br,signed=False)]
        approx_ints[i] = int(''.join(str(b) for b in binary_angle), 2)
    return approx_ints
