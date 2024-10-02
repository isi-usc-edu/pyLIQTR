"""
Copyright (c) 2024 Massachusetts Institute of Technology 
SPDX-License-Identifier: BSD-2-Clause
"""
from functools import cached_property
from typing import Dict, Optional, Tuple

from attrs import frozen

from qualtran import (
    Bloq,
    BloqBuilder,
    BoundedQUInt,
    QAny,
    QBit,
    Register,
    Signature,
    Soquet,
    SoquetT,
    Side
)
from qualtran.bloqs.block_encoding.lcu_select_and_prepare import PrepareOracle
from qualtran.bloqs.chemistry.pbc.first_quantization.select_and_prepare import PrepareTUVSuperpositions
from qualtran.bloqs.basic_gates import Hadamard

from pyLIQTR.circuits.operators.UniformSuperpositionIJ import UniformSuperpositionIJFirstQuantization
from pyLIQTR.circuits.operators.PrepareT_FirstQuantized import PrepareT_FirstQuantized
from pyLIQTR.circuits.operators.PrepareUV_FirstQuantized import PrepareUVFirstQuantization

@frozen
class PrepareFirstQuantization(PrepareOracle):
    """State preparation for the first quantized chemistry Hamiltonian.

    Prepares the state in Eq. 48 of the reference.

    .. math::
        &(\\cos\\theta|0\\rangle+\\sin\\theta|1\\rangle)|+\\rangle\\frac{1}{\\sqrt{\\eta}}\\left( \\sqrt{\eta-1}|0\\rangle\\sum_{i\\neq j=1}^\\eta |i\\rangle|j\\rangle + |1\\rangle\\sum_{j=1}^\\eta |j\\rangle|j\\rangle \\right) \\left( \\frac{1}{\\sqrt{3}}\\sum_{w=0}^2|w\\rangle \\right)\\\\
        &\\otimes \\left( \\frac{1}{2^{n_p-1}-1} \\sum_{r,s=0}^{n_p-2} 2^{(r+s)/2}|r\\rangle|s\\rangle \\right) \\left( \\sqrt{\\frac{\\lambda_U}{\\lambda_U+\\lambda_V}}|0\\rangle + \\sqrt{\\frac{\\lambda_V}{\\lambda_U+\\lambda_V}}|1\\rangle\\right)\\\\
        &\\otimes \\left( \\sqrt{\\frac{p_\\nu}{\\lambda_\\nu}}|0\\rangle\\sum_{\\nu\\in G_0}\\frac{1}{||\\nu||}|\\nu\\rangle + \\sqrt{1-p_\\nu}|1\\rangle|\\nu^\\perp\\rangle \\right) \\left( \\frac{1}{\\sqrt{\\sum_\\ell\\zeta_\\ell}} \\sum_{\\ell=1}^L \\sqrt{\\zeta_\\ell}|\\ell\\rangle \\right)

    Registers:

    .. line-block::
        tuv: Flag register for selecting between kinetic and potential terms in the Hamiltonian.
        uv: Flag register for selecting between the different potential components of the Hamiltonian.
        i: A register for selecting electronic registers.
        j: A register for selecting electronic registers.
        w: A register for selecting x, y and z components of the momentum register.
        r: A register for controlling elements of the momentum register. Used for block encodiding kinetic energy operator.
        s: A register for controlling elements of the momentum register. Used for block encodiding kinetic energy operator.
        mu: A register used for implementing nested boxes for the momentum state preparation.
        nu_x: x component of the momentum register for Coulomb potential.
        nu_y: y component of the momentum register for Coulomb potential.
        nu_z: z component of the momentum register for Coulomb potential.
        m: an ancilla register in a uniform superposition.
        l: The register for selecting the nuclei.
        succ_nu: Flag for success of the UV state preparation.
        plus_t: Qubit used for phasing.
        succ_i:Qubit flagging success of the uniform superposition preparation for register i.
        succ_j: Qubit flagging success of the uniform superposition preparation for register j.
        flag_equal:Qubit flagging the states where :math:`i=j`.
        less_than_ancilla: Qubits used for inequality test in uniform superposition preparation on i and j.
        rotation_ancilla: Qubits rotated for amplitude amplification on i and j.
        T_less_than_ancilla: Qubits used for inequality test in uniform superposition preparation in PrepareT.
        T_rotation_ancilla: Qubits rotated for amplitude amplification in PrepareT.
        catalytic: The catalytic state for controlled Hadamards.

    References:
        `Fault-Tolerant Quantum Simulations of Chemistry in First Quantization <https://arxiv.org/abs/2105.12767>`_

    :param int num_bits_p: The number of bits to represent each dimension of the momentum register.
    :param int eta: The number of electrons.
    :param int num_atoms: The number of atoms. $L$ in the reference.
    :param int lambda_zeta: The sum of nuclear charges.
    :param int m_param: $\mathcal{M}$ in the reference.
    :param int num_bits_nuc_pos: The number of bits of precision for representing the nuclear coordinates.
    :param int num_bits_t: The number of bits of precision for the state preparation
        over the register selecting between the different components of the
        Hamiltonian.
    :param int num_bits_rot_aa: The number of bits of precision for the rotation for
        amplitude amplification.
    """

    num_bits_p: int
    eta: int
    num_atoms: int
    lambda_zeta: int
    m_param: int = 2**8
    num_bits_nuc_pos: int = 16
    num_bits_t: int = 16
    num_bits_rot_aa: int = 8

    @property
    def selection_registers(self) -> Tuple[Register, ...]:
        n_nu = self.num_bits_p + 1
        n_eta = (self.eta - 1).bit_length()
        n_at = (self.num_atoms - 1).bit_length()
        n_m = (self.m_param - 1).bit_length()
        return (
            Register('tuv', QBit()),
            Register('uv', QBit()),
            Register('i', BoundedQUInt(bitsize=n_eta, iteration_length=self.eta)),
            Register('j', BoundedQUInt(bitsize=n_eta, iteration_length=self.eta)),
            Register('w', BoundedQUInt(bitsize=3)),
            Register("r", BoundedQUInt(bitsize=self.num_bits_p)),
            Register("s", BoundedQUInt(bitsize=self.num_bits_p)),
            Register("mu", BoundedQUInt(bitsize=self.num_bits_p)),
            Register("nu_x", BoundedQUInt(bitsize=n_nu)),
            Register("nu_y", BoundedQUInt(bitsize=n_nu)),
            Register("nu_z", BoundedQUInt(bitsize=n_nu)),
            Register("m", BoundedQUInt(bitsize=n_m)),
            Register("l", BoundedQUInt(bitsize=n_at)),
        )

    @cached_property
    def junk_registers(self) -> Tuple[Register, ...]:
        return (
            Register("succ_nu", QBit()), 
            Register("plus_t", QBit()),
            Register('succ_i', QBit(), side=Side.RIGHT),
            Register('succ_j', QBit(), side=Side.RIGHT),
            Register('flag_equal', QBit(), side=Side.RIGHT),
            Register('less_than_ancilla', QBit(), shape=(2,1), side=Side.RIGHT),
            Register('rotation_ancilla', QBit(), shape=(2,1), side=Side.RIGHT),
            Register('T_less_than_ancilla',QBit()),
            Register('T_rotation_ancilla',QBit()),
            Register('catalytic',QBit()))


    def pretty_name(self) -> str:
        return r'PREP'

    def build_composite_bloq(
        self,
        bb: BloqBuilder,
        tuv: SoquetT,
        uv: SoquetT,
        plus_t: SoquetT,
        i: SoquetT,
        j: SoquetT,
        w: SoquetT,
        r: SoquetT,
        s: SoquetT,
        T_less_than_ancilla: SoquetT,
        T_rotation_ancilla: SoquetT,
        catalytic: SoquetT,
        mu: SoquetT,
        nu_x: Soquet,
        nu_y: Soquet,
        nu_z: Soquet,
        m: SoquetT,
        succ_nu: SoquetT,
        l: SoquetT,
    ) -> Dict[str, 'SoquetT']:
        tuv, uv = bb.add(
            PrepareTUVSuperpositions(
                self.num_bits_t, self.eta, self.lambda_zeta, self.num_bits_rot_aa
            ),
            tuv=tuv,
            uv=uv,
        )
        i, j, succ_i, succ_j, flag_equal, less_than_ancilla, rotation_ancilla = bb.add(
            UniformSuperpositionIJFirstQuantization(self.eta), i=i, j=j,
        )
        # # |+>
        plus_t = bb.add(Hadamard(), q=plus_t)

        w, r, s, T_less_than_ancilla, T_rotation_ancilla, catalytic = bb.add(
            PrepareT_FirstQuantized(self.num_bits_p),
            w=w,
            r=r,
            s=s,
            less_than_ancilla=T_less_than_ancilla,
            rot_ancilla=T_rotation_ancilla,
            catalytic=catalytic,
        )

        mu, [nu_x, nu_y, nu_z], m, l, succ_nu, catalytic = bb.add(
            PrepareUVFirstQuantization(
                self.num_bits_p,
                self.eta,
                self.num_atoms,
                self.m_param,
                self.lambda_zeta,
                self.num_bits_nuc_pos,
            ),
            mu=mu,
            nu=[nu_x, nu_y, nu_z],
            m=m,
            l=l,
            flag_nu=succ_nu,
            catalytic=catalytic
        )
        return {
            'tuv': tuv,
            'uv': uv,
            'plus_t': plus_t,
            'i': i,
            'j': j,
            'succ_i':succ_i,
            'succ_j':succ_j,
            'flag_equal':flag_equal,
            'less_than_ancilla':less_than_ancilla,
            'rotation_ancilla':rotation_ancilla,
            'w': w,
            'r': r,
            's': s,
            'T_less_than_ancilla':T_less_than_ancilla,
            'T_rotation_ancilla':T_rotation_ancilla,
            'catalytic':catalytic,
            'mu': mu,
            'nu_x': nu_x,
            'nu_y': nu_y,
            'nu_z': nu_z,
            'm': m,
            'l': l,
            'succ_nu': succ_nu,
        }