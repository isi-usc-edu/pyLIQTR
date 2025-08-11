from functools import cached_property
from typing import Union

from attrs import field, frozen

from qualtran import Bloq, QBit, QInt, QMontgomeryUInt, QUInt, Signature, BloqBuilder, Soquet, SoquetT
from qualtran.bloqs.arithmetic.addition import Add
from qualtran.bloqs.basic_gates import XGate, OnEach


@frozen
class ControlledAddOrSubtract(Bloq):
    r"""Adds or subtracts in-place into the target, based on a control bit.

    Applies the transformation

    $$
        |1\rangle |a\rangle |b\rangle \mapsto |1\rangle |a\rangle |b + a\rangle \\
        |0\rangle |a\rangle |b\rangle \mapsto |0\rangle |a\rangle |b - a\rangle
    $$

    Given two numbers `a`, `b` and a control bit `ctrl`, this bloq computes:

    - the sum `b + a` when `ctrl=1`,
    - the difference `b - a` when `ctrl=0`,

    and stores the result in the second register (`b`).

    This uses an uncontrolled `Add` surrounded by controlled bitwise `XGate`s, and only
    the `Add` requires T gates, which has half the T-cost of a controlled `Add`.


    Args:
        a_dtype: dtype of the lhs `a`
        b_dtype: dtype of the rhs `b`. If it is not big enough to store the
                 result, the most significant bits are dropped on overflow.
        add_when_ctrl_is_on: If True (default), add when `ctrl=1` and subtract when
                             `ctrl=0`. If False, do the opposite: subtract when `ctrl=0`
                             and add when `ctrl=1`.

    Registers:
        ctrl: a single control bit
        a: an integer value.
        b: an integer value.

    References:
        [Compilation of Fault-Tolerant Quantum Heuristics for Combinatorial Optimization](https://arxiv.org/abs/2007.07391).
        Sanders et. al. Section II-A-1, Algorithm 1.
    """

    a_dtype: Union[QInt, QUInt, QMontgomeryUInt] = field()
    b_dtype: Union[QInt, QUInt, QMontgomeryUInt] = field()
    add_when_ctrl_is_on = True

    @b_dtype.default
    def b_dtype_default(self):
        return self.a_dtype

    @cached_property
    def signature(self) -> Signature:
        return Signature.build_from_dtypes(ctrl=QBit(), a=self.a_dtype, b=self.b_dtype)

    def build_composite_bloq(
        self, bb: BloqBuilder, ctrl: Soquet, a: Soquet, b: Soquet
    ) -> dict[str, SoquetT]:
        if self.add_when_ctrl_is_on:
            # flip the control bit
            ctrl = bb.add(XGate(), q=ctrl)

        # subcircuit to add when ctrl=0 and subtract when ctrl=1.
        # (0, a, b) or (1, a, b)
        ctrl, b = bb.add(OnEach(self.b_dtype.num_qubits, XGate()).controlled(), ctrl=ctrl, q=b)
        # -> (0, a, b) or (1, a, -1 - b)
        a, b = bb.add(Add(self.a_dtype, self.b_dtype), a=a, b=b)
        # -> (0, a, b + a) or (1, a, -1 - b + a)
        ctrl, b = bb.add(OnEach(self.b_dtype.num_qubits, XGate()).controlled(), ctrl=ctrl, q=b)
        # -> (0, a, b + a) or (1, a, b - a)

        if self.add_when_ctrl_is_on:
            ctrl = bb.add(XGate(), q=ctrl)

        return {'ctrl': ctrl, 'a': a, 'b': b}