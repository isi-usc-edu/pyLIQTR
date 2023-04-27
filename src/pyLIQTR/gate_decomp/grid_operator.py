"""
DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.

This material is based upon work supported by the Under Secretary of Defense for
Research and Engineering under Air Force Contract No. FA8702-15-D-0001. Any opinions,
findings, conclusions or recommendations expressed in this material are those of the
author(s) and do not necessarily reflect the views of the Under Secretary of Defense
for Research and Engineering.

© 2022 Massachusetts Institute of Technology.

The software/firmware is provided to you on an As-Is basis

Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS Part
252.227-7013 or 7014 (Feb 2014). Notwithstanding any copyright notice, U.S. Government
rights in this work are defined by DFARS 252.227-7013 or DFARS 252.227-7014 as detailed
above. Use of this work other than as specifically authorized by the U.S. Government
may violate any copyrights that exist in this work.
"""

from decimal import Decimal as D

import numpy as np

from pyLIQTR.gate_decomp.rings import Z_OMEGA


class GridOperator:
    """
    A class defining linear operators G such that for all u ∈ Z[ω], Gu ∈ Z[ω].
    """

    def __init__(
        self, a: int, ap: int, b: int, bp: int, c: int, cp: int, d: int, dp: int
    ) -> None:
        self.a = a
        self.ap = ap
        self.b = b
        self.bp = bp
        self.c = c
        self.cp = cp
        self.d = d
        self.dp = dp

    @classmethod
    def I(cls):
        return cls(1, 0, 0, 0, 0, 0, 1, 0)

    @classmethod
    def R(cls):
        return cls(0, 1, 0, -1, 0, 1, 0, 1)

    @classmethod
    def A(cls):
        return cls(1, 0, -2, 0, 0, 0, 1, 0)

    @classmethod
    def APowN(cls, n: int):
        return cls(1, 0, -2 * n, 0, 0, 0, 1, 0)

    @classmethod
    def B(cls):
        return cls(1, 0, 0, 2, 0, 0, 1, 0)

    @classmethod
    def BPowN(cls, n: int):
        return cls(1, 0, 0, 2 * n, 0, 0, 1, 0)

    @classmethod
    def K(cls):
        return cls(-1, 1, 0, -1, 1, 1, 0, 1)

    @classmethod
    def X(cls):
        return cls(0, 0, 1, 0, 1, 0, 0, 0)

    @classmethod
    def Z(cls):
        return cls(1, 0, 0, 0, 0, 0, -1, 0)

    def inverse(self):
        det_int = (
            self.a * self.d
            - self.b * self.c
            + (self.ap * self.dp - self.bp * self.cp) // 2
        )
        det_sqrt2 = (
            self.a * self.dp + self.ap * self.d - self.b * self.cp - self.bp * self.c
        )
        if det_sqrt2 != 0 or (det_int != 1 and det_int != -1):
            raise ValueError(
                "Grid operator is not special (i.e. does not have determinant of +/- 1). Cannot guarantee the existance of an inverse"
            )
        return GridOperator(
            self.d // det_int,
            self.dp // det_int,
            -self.b // det_int,
            -self.bp // det_int,
            -self.c // det_int,
            -self.cp // det_int,
            self.a // det_int,
            self.ap // det_int,
        )

    def __eq__(self, other) -> bool:
        if (
            self.a == other.a
            and self.ap == other.ap
            and self.b == other.b
            and self.bp == other.bp
            and self.c == other.c
            and self.cp == other.cp
            and self.d == other.d
            and self.dp == other.dp
        ):
            return True
        else:
            return False

    def __mul__(self, other):
        a = (
            self.a * other.a
            + self.b * other.c
            + (self.ap * other.ap + self.bp * other.cp) // 2
        )
        ap = (
            self.a * other.ap
            + self.ap * other.a
            + self.b * other.cp
            + self.bp * other.c
        )
        b = (
            self.a * other.b
            + self.b * other.d
            + (self.ap * other.bp + self.bp * other.dp) // 2
        )
        bp = (
            self.a * other.bp
            + self.ap * other.b
            + self.b * other.dp
            + self.bp * other.d
        )
        c = (
            self.c * other.a
            + self.d * other.c
            + (self.cp * other.ap + self.dp * other.cp) // 2
        )
        cp = (
            self.c * other.ap
            + self.cp * other.a
            + self.d * other.cp
            + self.dp * other.c
        )
        d = (
            self.c * other.b
            + self.d * other.d
            + (self.cp * other.bp + self.dp * other.dp) // 2
        )
        dp = (
            self.c * other.bp
            + self.cp * other.b
            + self.d * other.dp
            + self.dp * other.d
        )

        return GridOperator(a, ap, b, bp, c, cp, d, dp)

    def __imul__(self, other):
        self.a = (
            self.a * other.a
            + self.b * other.c
            + (self.ap * other.ap + self.bp * other.cp) // 2
        )
        self.ap = (
            self.a * other.ap
            + self.ap * other.a
            + self.b * other.cp
            + self.bp * other.c
        )
        self.b = (
            self.a * other.b
            + self.b * other.d
            + (self.ap * other.bp + self.bp * other.dp) // 2
        )
        self.bp = (
            self.a * other.bp
            + self.ap * other.b
            + self.b * other.dp
            + self.bp * other.d
        )
        self.c = (
            self.c * other.a
            + self.d * other.c
            + (self.cp * other.ap + self.dp * other.cp) // 2
        )
        self.cp = (
            self.c * other.ap
            + self.cp * other.a
            + self.d * other.cp
            + self.dp * other.c
        )
        self.d = (
            self.c * other.b
            + self.d * other.d
            + (self.cp * other.bp + self.dp * other.dp) // 2
        )
        self.dp = (
            self.c * other.bp
            + self.cp * other.b
            + self.d * other.dp
            + self.dp * other.d
        )

    def __str__(self) -> str:
        first = str(self.a) + " + " + str(self.ap) + "/√2"
        second = str(self.b) + " + " + str(self.bp) + "/√2"
        third = str(self.c) + " + " + str(self.cp) + "/√2"
        fourth = str(self.d) + " + " + str(self.dp) + "/√2"
        return "[[" + first + ", " + second + "], [" + third + ", " + fourth + "]]"

    def transpose(self):
        a = self.a
        ap = self.ap
        self.a = self.d
        self.ap = self.dp
        self.d = a
        self.dp = ap

    def conj2(self):
        return GridOperator(
            self.a, -self.ap, self.b, -self.bp, self.c, -self.cp, self.d, -self.dp
        )

    # TODO: Implement via repeated squaring
    def __pow__(self, other: int):
        result = GridOperator.I()
        for _ in range(other):
            result = self * result
        return result

    def multiply_z_omega(self, other: Z_OMEGA) -> Z_OMEGA:
        x1 = other.a4
        y1 = other.a2
        x2 = other.a3 - other.a1
        y2 = other.a3 + other.a1
        alpha = self.a * x2 + self.ap * x1 + self.b * y2 + self.bp * y1
        beta = self.c * x2 + self.cp * x1 + self.d * y2 + self.dp * y1
        new_d = self.a * x1 + self.b * y1 + (self.ap * x2 + self.bp * y2) // 2
        new_b = self.c * x1 + self.d * y1 + (self.cp * x2 + self.dp * y2) // 2
        new_a = (beta - alpha) // 2
        new_c = (beta + alpha) // 2
        return Z_OMEGA(new_a, new_b, new_c, new_d)

    def to_numpy_arr(self) -> np.array:
        SQRT_2 = D(2).sqrt()
        a11 = self.a + self.ap / SQRT_2
        a12 = self.b + self.bp / SQRT_2
        a21 = self.c + self.cp / SQRT_2
        a22 = self.d + self.dp / SQRT_2
        return np.array([[float(a11), float(a12)], [float(a21), float(a22)]])
