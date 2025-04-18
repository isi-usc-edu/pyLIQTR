"""
Copyright (c) 2024 Massachusetts Institute of Technology 
SPDX-License-Identifier: BSD-2-Clause
"""
import cirq
import attr
import sympy
from typing import Set
from numpy.typing import NDArray
from qualtran import DecomposeNotImplementedError
from qualtran.bloqs.rotations import AddIntoPhaseGrad
from qualtran.bloqs.basic_gates import CNOT, ZGate, CZPowGate, XGate, OnEach
from qualtran.bloqs.mcmt import And

@attr.s
class AddIntoPhaseGradwithDecompose(AddIntoPhaseGrad):
    """Quantum-quantum addition into a phase gradient register using $b_{phase} - 2$ Toffolis

    .. math::
        U|x\\rangle|\\text{phase\\_grad}\\rangle = |x\\rangle|\\text{phase\\_grad} + x\\rangle

    Registers:

    .. line-block::
        x: Input THRU register storing input value x to be added to the phase gradient register.
        phase_grad: Phase gradient THRU register.

    References:
        `Compilation of Fault-Tolerant Quantum Heuristics for Combinatorial Optimization <https://arxiv.org/abs/2007.07391>`_
        Appendix A: Addition for controlled rotations, Figure 13

    :param int x_bitsize: Size of input register.
    :param int phase_bitsize: Size of phase gradient register to which the input value should be added. Note, currently decomposition is only supported for phase_bitsize=x_bitsize.
    :param int sign: Whether the input register x should be added (+1) or subtracted (-1) from the phase gradient register. Default is +1.
    """
    def __attrs_post_init__(self):
        if isinstance(self.x_bitsize, sympy.Expr) or isinstance(self.phase_bitsize, sympy.Expr):
            pass
        elif self.x_bitsize != self.phase_bitsize:
            raise DecomposeNotImplementedError(f"Decomposition not implemented for {self.x_bitsize=} != {self.phase_bitsize=}")
        if self.right_shift != 0 or self.sign not in (1,-1) or self.controlled_by != None:
            raise DecomposeNotImplementedError(f"Decomposition not implemented for {self} with right_shift!=0, sign!=(1 or -1), or controlled_by!=None")

    def _left_building_block(self, inp, out, anc, depth):
        if depth == self.phase_bitsize - 2:
            return
        else:
            if depth < 1:
                raise ValueError(f"{depth=} is not a positive integer")
            if depth < len(inp):
                yield CNOT().on(anc[depth - 1], inp[depth])
                control = inp[depth]
            else:
                # If inp[depth] doesn't exist, we treat it as a |0>,
                # and therefore applying CNOT().on(anc[depth - 1], inp[depth])
                # essentially "copies" anc[depth - 1] into inp[depth]
                # in the classical basis. So therefore, on future operations,
                # we can use anc[depth - 1] in its place.
                control = anc[depth - 1]
            yield CNOT().on(anc[depth - 1], out[depth])
            yield And().on(control, out[depth], anc[depth])
            yield CNOT().on(anc[depth - 1], anc[depth])
            yield from self._left_building_block(inp, out, anc, depth + 1)

    def _right_building_block(self, inp, out, anc, depth):
        if depth == 0:
            return
        else:
            yield CNOT().on(anc[depth - 1], anc[depth])
            if depth < len(inp):
                yield And().adjoint().on(inp[depth], out[depth], anc[depth])
                yield CNOT().on(anc[depth - 1], inp[depth])
                yield CNOT().on(inp[depth], out[depth])
            else:
                yield And().adjoint().on(anc[depth - 1], out[depth], anc[depth])
            yield from self._right_building_block(inp, out, anc, depth - 1)

    def decompose_from_registers(
        self,
        *,
        context: cirq.DecompositionContext,
        **quregs: NDArray[cirq.Qid],
    ) -> cirq.OP_TREE:

        # reverse the order of qubits for big endian-ness (0th element of input_bits is lsb)
        input_bits = quregs['x'][::-1]
        output_bits = quregs['phase_grad'][::-1]

        # special cases for bitsize < 3
        if self.x_bitsize == 1:
            # for bitsize=1 we have |phi>=|+> -> e^{pi*i*ell}|+> which gives a phase of -1 for ell=+/-1 and a phase of 1 for ell=0
            yield ZGate().on(input_bits[0])
            return
        elif self.x_bitsize == 2:
            if self.sign==-1:
                yield OnEach(self.phase_bitsize, XGate()).on_registers(q=output_bits)
            yield CZPowGate(exponent=1.0,global_shift=0.0).on(input_bits[0],output_bits[0]) 
            yield CNOT().on(input_bits[0],output_bits[0])
            yield ZGate().on(input_bits[-1])
            if self.sign==-1:
                yield OnEach(self.phase_bitsize, XGate()).on_registers(q=output_bits)
            return

        # for bitsize >= 3
        if self.sign==-1:
            # apply X to phi register for subtraction, see reference Section II.A.1
            yield OnEach(self.phase_bitsize, XGate()).on_registers(q=output_bits)
        ancillas = context.qubit_manager.qalloc(self.phase_bitsize - 2)[::-1]
        # Start off the addition by anding into the ancilla
        yield And().on(input_bits[0], output_bits[0], ancillas[0])
        # iterate up to msb using left building block
        yield from self._left_building_block(input_bits, output_bits, ancillas, 1)
        
        # middle segment
        yield CNOT().on(ancillas[self.phase_bitsize - 3], input_bits[self.phase_bitsize - 2])
        yield CNOT().on(ancillas[self.phase_bitsize - 3], output_bits[self.phase_bitsize - 2])
        yield ZGate().on(ancillas[-1])
        yield CZPowGate(exponent=1.0,global_shift=0.0).on(input_bits[self.phase_bitsize - 2],output_bits[self.phase_bitsize - 2])
        yield CNOT().on(ancillas[self.phase_bitsize - 3], input_bits[self.phase_bitsize - 2])
        yield CNOT().on(input_bits[self.phase_bitsize - 2],output_bits[self.phase_bitsize - 2])
        yield ZGate().on(input_bits[-1])
        
        # iterate back down to lsb using right building block
        yield from self._right_building_block(
            input_bits, output_bits, ancillas, self.phase_bitsize - 3
        )
        yield And().adjoint().on(input_bits[0], output_bits[0], ancillas[0])
        yield CNOT().on(input_bits[0], output_bits[0])
        context.qubit_manager.qfree(ancillas)

        if self.sign==-1:
            # un-apply X to phi register
            yield OnEach(self.phase_bitsize, XGate()).on_registers(q=quregs['phase_grad'])

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        sign_counts = {(OnEach(self.phase_bitsize, XGate()),2)} if self.sign==-1 else set()

        if self.x_bitsize==1:
            return {(ZGate(),1)}
        elif self.x_bitsize==2:
            return {(CZPowGate(exponent=1.0,global_shift=0.0),1),(CNOT(),1),(ZGate(),1)} | sign_counts
        else:
            num_toffoli = self.phase_bitsize - 2
            return {(And(), num_toffoli),(And().adjoint(), num_toffoli), (CNOT(),6*(num_toffoli-1) + 5),(ZGate(),2),(CZPowGate(exponent=1.0,global_shift=0.0),1)} | sign_counts

    def adjoint(self) -> 'Bloq':
        return AddIntoPhaseGradwithDecompose(
            self.x_bitsize,
            self.phase_bitsize,
            self.right_shift,
            sign=-1 * self.sign,
            controlled_by=self.controlled_by,
        )

    def __pow__(self, power):
        if power == 1:
            return self
        if power == -1:
            return self.adjoint()
        raise NotImplementedError("AddIntoPhaseGrad.__pow__ defined only for powers +1/-1.")

    def _t_complexity_(self) -> 'TComplexity':
        # overwriting parent class with default
        return NotImplemented