"""
Copyright (c) 2024 Massachusetts Institute of Technology 
SPDX-License-Identifier: BSD-2-Clause
"""
import cirq
from cirq import MeasurementGate, ClassicallyControlledOperation
from functools import cached_property
from attrs import frozen
from typing import Dict, Tuple, Optional, Callable, Union, Sequence
from numpy.typing import NDArray

from qualtran import BoundedQUInt, Register, Side, QBit, QAny, Signature
from qualtran.bloqs.block_encoding.lcu_select_and_prepare import SelectOracle
from qualtran.bloqs.chemistry.pbc.first_quantization.select_and_prepare import MultiplexedCSwap3D

from pyLIQTR.circuits.operators.SelectUV_FirstQuantized import SelectUVFirstQuantization
from pyLIQTR.circuits.operators.SelectT_FirstQuantized import SelectT_FirstQuantized

@frozen
class SelectFirstQuantizationPYL(SelectOracle):
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
        m: An ancilla register in a uniform superposition.
        Rl: A register storing the value of :math:`R_\\ell`.
        overflow: Carry bits resulting from addition/subtraction.
        phase_gradient_state: The phase gradient state register.
        sys: The system register. Will store $\eta$ registers (x, y and z)
            components of size num_bits_p.

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
    num_bits_p: int
    eta: int
    num_atoms: int
    lambda_zeta: int
    m_param: int = 2**8
    num_bits_nuc_pos: int = 16
    num_bits_t: int = 16
    num_bits_rot_aa: int = 8

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
        n_m = (self.m_param - 1).bit_length()
        bphi = max(self.num_bits_t,self.num_bits_nuc_pos)
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
            Register("Rl", QAny(bitsize=self.num_bits_nuc_pos), shape=(3,)),
            Register("overflow", QAny(bitsize=2), shape=(3,)),
            Register("phase_gradient_state", QAny(bitsize=bphi))
        )
    
    @cached_property
    def target_registers(self) -> Tuple[Register, ...]:
        return (Register("sys", QAny(bitsize=self.num_bits_p), shape=(self.eta, 3)),)

    @cached_property
    def signature(self) -> Signature:
        return Signature(
            [*self.control_registers, *self.selection_registers, *self.target_registers]
        )

    def pretty_name(self) -> str:
        return r'SELECT'
    
    def decompose_from_registers(
        self,
        *,
        context: cirq.DecompositionContext,
        **quregs: NDArray[cirq.Qid],
    ) -> cirq.OP_TREE:
        i = quregs['i']
        j = quregs['j']
        sys = quregs['sys']
        plus_t, tuv, uv = quregs['plus_t'], quregs['tuv'], quregs['uv']
        w, r, s = quregs['w'], quregs['r'], quregs['s']
        Rl = quregs['Rl']
        nu_x, nu_y, nu_z = quregs['nu_x'], quregs['nu_y'], quregs['nu_z']
        phi = quregs['phase_gradient_state']
        overflow = quregs['overflow']
        bphi = max(self.num_bits_t,self.num_bits_nuc_pos)

        p = [context.qubit_manager.qalloc(self.num_bits_p) for _ in range(3)]
        yield MultiplexedCSwap3D(self.num_bits_p, self.eta).on_registers( sel=i, targets=sys, junk=p)

        q = [context.qubit_manager.qalloc(self.num_bits_p) for _ in range(3)]
        yield MultiplexedCSwap3D(self.num_bits_p, self.eta).on_registers(sel=j, targets=sys, junk=q)

        yield SelectT_FirstQuantized(self.num_bits_p).on_registers(plus=plus_t, flag_T=tuv, w=w, r=r, s=s, sys=p)

        yield SelectUVFirstQuantization(self.num_bits_p, self.eta, self.num_atoms, self.num_bits_nuc_pos, bphi=bphi).on_registers(flag_tuv=tuv, uv = uv, Rl=Rl, nu=[nu_x, nu_y, nu_z], phi=phi, p=p, q=q, overflow=overflow)

        yield MultiplexedCSwap3D(self.num_bits_p, self.eta).on_registers( sel=i, targets=sys, junk=p)
        for i in range(3):
            context.qubit_manager.qfree(p[i])

        yield MultiplexedCSwap3D(self.num_bits_p, self.eta).on_registers(sel=j, targets=sys, junk=q)
        for i in range(3):
            context.qubit_manager.qfree(q[i])


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