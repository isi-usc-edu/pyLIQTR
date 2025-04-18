"""
Copyright (c) 2024 Massachusetts Institute of Technology 
SPDX-License-Identifier: BSD-2-Clause
"""
from cirq import MeasurementGate, ClassicallyControlledOperation
from functools import cached_property
from typing import Dict, Tuple, Optional, Callable, Union, Sequence

from qualtran import BloqBuilder, Soquet, SoquetT, BoundedQUInt, Register, Side, QBit
from qualtran.bloqs.chemistry.pbc.first_quantization.select_and_prepare import SelectFirstQuantization
from qualtran.bloqs.chemistry.pbc.first_quantization.select_and_prepare import MultiplexedCSwap3D

from pyLIQTR.circuits.operators.SelectUV_FirstQuantized import SelectUVFirstQuantizationPYL
from pyLIQTR.circuits.operators.SelectT_FirstQuantized import SelectT_FirstQuantized

class SelectFirstQuantizationPYL(SelectFirstQuantization):
    """SELECT operation for the first quantized chemistry Hamiltonian.

    Registers:

    .. line-block::
        tuv: Flag register for selecting between kinetic and potential terms in the Hamiltonian.
        uv: Flag register for selecting between the different potential
            components of the Hamiltonian.
        i_ne_j: Register flagging $i \ne j$
        plus_t: A register prepared in the $|+\rangle$ state.
        i: A register for selecting electronic registers.
        j: A register for selecting electronic registers.
        w: A register for selecting x, y and z components of the momentum register.
        r: A register for controlling elements of the momentum register. Used
            for block encodiding kinetic energy operator.
        s: A register for controlling elements of the momentum register. Used
            for block encodiding kinetic energy operator.
        mu: A register used for implementing nested boxes for the momentum state preparation.
        nu_x: x component of the momentum register for Coulomb potential.
        nu_y: y component of the momentum register for Coulomb potential.
        nu_z: z component of the momentum register for Coulomb potential.
        m: an ancilla register in a uniform superposition.
        l: The register for selecting the nuclei.
        sys: The system register. Will store $\eta$ registers (x, y and z)
            compents of size num_bits_p.

    References:
        `Fault-Tolerant Quantum Simulations of Chemistry in First Quantization <https://arxiv.org/abs/2105.12767>`_

    :param int num_bits_p: The number of bits to represent each dimension of the momentum register.
    :param int eta: The number of electrons.
    :param int num_atoms: The number of atoms. $L$ in the reference.
    :param int lambda_zeta: sum of nuclear charges.
    :param int m_param: $\mathcal{M}$ in the reference.
    :param int num_bits_nuc_pos: The number of bits of precision for representing the nuclear coordinates.
    :param int num_bits_t: The number of bits of precision for the state preparation
        over the register selecting between the different components of the
        Hamiltonian.
    :param int num_bits_rot_aa: The number of bits of precision for the rotation for
        amplitude amplification.
    """
    @cached_property
    def control_registers(self) -> Tuple[Register, ...]:
        return (
            Register("tuv", QBit()),
            Register("i_ne_j", QBit()),
            Register("plus_t", QBit()),
        )

    @cached_property
    def selection_registers(self) -> Tuple[Register, ...]:
        n_nu = self.num_bits_p + 1
        n_eta = (self.eta - 1).bit_length()
        n_at = (self.num_atoms - 1).bit_length()
        n_m = (self.m_param - 1).bit_length()
        return (
            Register("uv", QBit(),side=Side.RIGHT),
            Register('i', BoundedQUInt(bitsize=n_eta, iteration_length=self.eta)),
            Register('j', BoundedQUInt(bitsize=n_eta, iteration_length=self.eta)),
            Register("w", BoundedQUInt(bitsize=2,iteration_length=3)),
            Register("r", BoundedQUInt(bitsize=self.num_bits_p)),
            Register("s", BoundedQUInt(bitsize=self.num_bits_p)),
            Register("mu", BoundedQUInt(bitsize=self.num_bits_p)),
            Register("nu_x", BoundedQUInt(bitsize=n_nu)),
            Register("nu_y", BoundedQUInt(bitsize=n_nu)),
            Register("nu_z", BoundedQUInt(bitsize=n_nu)),
            Register("m", BoundedQUInt(bitsize=n_m)),
            Register("l", BoundedQUInt(bitsize=n_at)),
        )

    def build_composite_bloq(
        self,
        bb: BloqBuilder,
        tuv: SoquetT,
        i_ne_j: SoquetT,
        plus_t: SoquetT,
        i: SoquetT,
        j: SoquetT,
        w: SoquetT,
        r: SoquetT,
        s: SoquetT,
        mu: SoquetT,
        nu_x: Soquet,
        nu_y: Soquet,
        nu_z: Soquet,
        m: SoquetT,
        l: SoquetT,
        sys: SoquetT,
    ) -> Dict[str, 'SoquetT']:
        # ancilla for swaps from electronic system registers.
        # we assume these are left in a clean state after SELECT operations
        p = [bb.allocate(self.num_bits_p) for _ in range(3)]
        q = [bb.allocate(self.num_bits_p) for _ in range(3)]
        rl = bb.allocate(self.num_bits_nuc_pos)
        i, sys, p = bb.add(
            MultiplexedCSwap3D(self.num_bits_p, self.eta), sel=i, targets=sys, junk=p
        )
        j, sys, q = bb.add(
            MultiplexedCSwap3D(self.num_bits_p, self.eta), sel=j, targets=sys, junk=q
        )
        tuv, plus_t, w, r, s, p = bb.add(
            SelectT_FirstQuantized(self.num_bits_p),
            plus=plus_t,
            flag_T=tuv,
            w=w,
            r=r,
            s=s,
            sys=p,
        )
        tuv, uv, l, rl, [nu_x, nu_y, nu_z], p, q = bb.add(
            SelectUVFirstQuantizationPYL(
                self.num_bits_p, self.eta, self.num_atoms, self.num_bits_nuc_pos
            ),
            flag_tuv=tuv,
            l=l,
            rl=rl,
            nu=[nu_x, nu_y, nu_z],
            p=p,
            q=q,
        )
        i, sys, p = bb.add(
            MultiplexedCSwap3D(self.num_bits_p, self.eta), sel=i, targets=sys, junk=p
        )
        j, sys, q = bb.add(
            MultiplexedCSwap3D(self.num_bits_p, self.eta), sel=j, targets=sys, junk=q
        )
        for pi in p:
            bb.free(pi)
        for qi in q:
            bb.free(qi)
        bb.free(rl)
        return {
            'tuv': tuv,
            'uv': uv,
            'plus_t': plus_t,
            'i_ne_j': i_ne_j,
            'i': i,
            'j': j,
            'w': w,
            'r': r,
            's': s,
            'mu': mu,
            'nu_x': nu_x,
            'nu_y': nu_y,
            'nu_z': nu_z,
            'm': m,
            'l': l,
            'sys': sys,
        }

    def call_graph(
        self,
        generalizer: Optional[Union['GeneralizerT', Sequence['GeneralizerT']]] = None,
        keep: Optional[Callable[['Bloq'], bool]] = None,
        max_depth: Optional[int] = None,
    ) -> Tuple['nx.DiGraph', Dict['Bloq', Union[int, 'sympy.Expr']]]:
        """Get the bloq call graph and call totals.

        The call graph has edges from a parent bloq to each of the bloqs that it calls in
        its decomposition. The number of times it is called is stored as an edge attribute.
        To specify the bloq call counts for a specific node, override `Bloq.build_call_graph()`.

        Args:
            generalizer: If provided, run this function on each (sub)bloq to replace attributes
                that do not affect resource estimates with generic sympy symbols. If the function
                returns `None`, the bloq is omitted from the counts graph. If a sequence of
                generalizers is provided, each generalizer will be run in order.
            keep: If this function evaluates to True for the current bloq, keep the bloq as a leaf
                node in the call graph instead of recursing into it.
            max_depth: If provided, build a call graph with at most this many layers.

        Returns:
            g: A directed graph where nodes are (generalized) bloqs and edge attribute 'n' reports
                the number of times successor bloq is called via its predecessor.
            sigma: Call totals for "leaf" bloqs. We keep a bloq as a leaf in the call graph
                according to `keep` and `max_depth` (if provided) or if a bloq cannot be
                decomposed.
        """
        from qualtran.resource_counting import get_bloq_call_graph

        if keep is None:
            def keep_measurements(op):
                if isinstance(op,MeasurementGate):
                    return True
                elif isinstance(op,ClassicallyControlledOperation):
                    return True
                return False

            keep = keep_measurements

        return get_bloq_call_graph(self, generalizer=generalizer, keep=keep, max_depth=max_depth)