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

from pyLIQTR.gate_decomp.rings import Z_OMEGA, Z_SQRT2
import math
import numpy as np
from typing import List


class MAT_D_OMEGA:
    """
    A class of 2x2 matrices over the ring D[ω]
    """

    def __init__(self, a00, a01, a10, a11, k):
        if (
            not isinstance(a00, Z_OMEGA)
            or not isinstance(a01, Z_OMEGA)
            or not isinstance(a10, Z_OMEGA)
            or not isinstance(a11, Z_OMEGA)
        ):
            raise TypeError(
                f"Entries must be instances of the type Z_OMEGA, not {type(a00)}"
            )
        self.a00 = a00
        self.a01 = a01
        self.a10 = a10
        self.a11 = a11
        self.k = k

    def __str__(self):
        return (
            "[\n\t["
            + str(self.a00)
            + " "
            + str(self.a01)
            + "]\n\t["
            + str(self.a10)
            + " "
            + str(self.a11)
            + "]\n] k = "
            + str(self.k)
        )

    def __eq__(self, other):
        if (
            self.a00 == other.a00
            and self.a01 == other.a01
            and self.a10 == other.a10
            and self.a11 == other.a11
            and self.k == other.k
        ):
            return True
        else:
            return False

    def __matmul__(self, other):
        return MAT_D_OMEGA(
            self.a00 * other.a00 + self.a01 * other.a10,
            self.a00 * other.a01 + self.a01 * other.a11,
            self.a10 * other.a00 + self.a11 * other.a10,
            self.a10 * other.a01 + self.a11 * other.a11,
            self.k + other.k,
        )

    def __mul__(self, other):
        return MAT_D_OMEGA(
            self.a00 * other,
            self.a01 * other,
            self.a10 * other,
            self.a11 * other,
            self.k,
        )

    def __neg__(self):
        return MAT_D_OMEGA(-self.a00, -self.a01, -self.a10, -self.a11, self.k)

    def to_array(self):
        return (
            np.array(
                [
                    [complex(self.a00), complex(self.a01)],
                    [complex(self.a10), complex(self.a11)],
                ]
            )
            / math.sqrt(2) ** self.k
        )

    def mult_pow_2(self, n):
        assert isinstance(n, int)
        return MAT_D_OMEGA(
            self.a00 * int(2 ** n),
            self.a01 * int(2 ** n),
            self.a10 * int(2 ** n),
            self.a11 * int(2 ** n),
            self.k + int(2 * n),
        )

    def reduce_by_sqrt2(self):
        return MAT_D_OMEGA(
            self.a00.mul_by_sqrt2(),
            self.a01.mul_by_sqrt2(),
            self.a10.mul_by_sqrt2(),
            self.a11.mul_by_sqrt2(),
            self.k - 1,
        )

    def adjoint(self):
        return MAT_D_OMEGA(
            self.a00.conj(), self.a10.conj(), self.a01.conj(), self.a11.conj(), self.k
        )

    def elements(self):
        return [self.a00, self.a01, self.a10, self.a11]

    def convert_to_so3(self):
        increase_k = False
        for elt in self.elements():
            if is_odd(elt):
                increase_k = True
        if increase_k:
            elts = [convert_to_zsqrti_increase_k(elt) for elt in self.elements()]
            k = 2 * self.k + 2
        else:
            elts = [convert_to_zsqrti(elt) for elt in self.elements()]
            k = 2 * self.k

        a1 = elts[0][0]
        a2 = elts[0][1]
        b1 = elts[1][0]
        b2 = elts[1][1]
        c1 = elts[2][0]
        c2 = elts[2][1]
        d1 = elts[3][0]
        d2 = elts[3][1]
        u00 = a1 * d1 + a2 * d2 + b1 * c1 + b2 * c2
        u01 = a2 * d1 + b1 * c2 - b2 * c1 - a1 * d2
        u02 = a1 * c1 + a2 * c2 - b1 * d1 - b2 * d2
        u10 = a1 * d2 - a2 * d1 + b1 * c2 - b2 * c1
        u11 = a1 * d1 + a2 * d2 - b1 * c1 - b2 * c2
        u12 = a1 * c2 - a2 * c1 - b1 * d2 + b2 * d1
        u20 = Z_SQRT2(2, 0) * (a1 * b1 + a2 * b2)
        u21 = Z_SQRT2(2, 0) * (a2 * b1 - a1 * b2)
        u22 = a1 ** 2 + a2 ** 2 - b1 ** 2 - b2 ** 2
        so3 = SO3(u00, u01, u02, u10, u11, u12, u20, u21, u22, k)
        so3.reduce()
        return so3

    @classmethod
    def I(cls):
        return cls(
            Z_OMEGA(0, 0, 0, 1),
            Z_OMEGA(0, 0, 0, 0),
            Z_OMEGA(0, 0, 0, 0),
            Z_OMEGA(0, 0, 0, 1),
            0,
        )

    @classmethod
    def X(cls):
        return cls(
            Z_OMEGA(0, 0, 0, 0),
            Z_OMEGA(0, 0, 0, 1),
            Z_OMEGA(0, 0, 0, 1),
            Z_OMEGA(0, 0, 0, 0),
            0,
        )

    @classmethod
    def Y(cls):
        return cls(
            Z_OMEGA(0, 0, 0, 0),
            Z_OMEGA(0, -1, 0, 0),
            Z_OMEGA(0, 1, 0, 0),
            Z_OMEGA(0, 0, 0, 0),
            0,
        )

    @classmethod
    def Z(cls):
        return cls(
            Z_OMEGA(0, 0, 0, 1),
            Z_OMEGA(0, 0, 0, 0),
            Z_OMEGA(0, 0, 0, 0),
            Z_OMEGA(0, 0, 0, -1),
            0,
        )

    @classmethod
    def H(cls):
        return cls(
            Z_OMEGA(0, 0, 0, 1),
            Z_OMEGA(0, 0, 0, 1),
            Z_OMEGA(0, 0, 0, 1),
            Z_OMEGA(0, 0, 0, -1),
            1,
        )

    @classmethod
    def S(cls):
        return cls(
            Z_OMEGA(0, 0, 0, 1),
            Z_OMEGA(0, 0, 0, 0),
            Z_OMEGA(0, 0, 0, 0),
            Z_OMEGA(0, 1, 0, 0),
            0,
        )

    @classmethod
    def Sd(cls):
        return cls(
            Z_OMEGA(0, 0, 0, 1),
            Z_OMEGA(0, 0, 0, 0),
            Z_OMEGA(0, 0, 0, 0),
            Z_OMEGA(0, -1, 0, 0),
            0,
        )

    @classmethod
    def T(cls):
        return cls(
            Z_OMEGA(0, 0, 0, 1),
            Z_OMEGA(0, 0, 0, 0),
            Z_OMEGA(0, 0, 0, 0),
            Z_OMEGA(0, 0, 1, 0),
            0,
        )

    @classmethod
    def Td(cls):
        return cls(
            Z_OMEGA(0, 0, 0, 1),
            Z_OMEGA(0, 0, 0, 0),
            Z_OMEGA(0, 0, 0, 0),
            Z_OMEGA(-1, 0, 0, 0),
            0,
        )


class SO3:
    """A class representing the Bloch sphere representation of a unitary operator.

    Given an operator U ∈ U(2), define its Bloch sphere representation B ∈ SO(3) as
    the operator such that if
    U(xX + yY + zZ)U^† = x'X + y'Y + z'Z, then
    B(x y z)^T = (x' y' z')^T
    """

    def __init__(self, u00, u01, u02, u10, u11, u12, u20, u21, u22, k=0):
        self.u00 = u00
        self.u01 = u01
        self.u02 = u02
        self.u10 = u10
        self.u11 = u11
        self.u12 = u12
        self.u20 = u20
        self.u21 = u21
        self.u22 = u22
        self.k = k

    def __eq__(self, other):
        self_elts = self.elements()
        other_elts = other.elements()
        for i in range(9):
            if self_elts[i] != other_elts[i]:
                return False
        if self.k != other.k:
            return False
        return True

    def __neg__(self):
        return SO3(
            -self.u00,
            -self.u01,
            -self.u02,
            -self.u10,
            -self.u11,
            -self.u12,
            -self.u20,
            -self.u21,
            -self.u22,
            self.k,
        )

    def elements(self) -> List[Z_SQRT2]:
        return [
            self.u00,
            self.u01,
            self.u02,
            self.u10,
            self.u11,
            self.u12,
            self.u20,
            self.u21,
            self.u22,
        ]

    def reduce(self):
        while self.reduce_by_2():
            self.reduce_by_2()
        while self.reduce_by_1():
            self.reduce_by_1()

    def reduce_by_2(self) -> bool:
        for elt in self.elements():
            if elt.a % 2 != 0 or elt.b % 2 != 0:
                return False
        else:
            self.k -= 2
            self.u00.a //= 2
            self.u01.a //= 2
            self.u02.a //= 2
            self.u10.a //= 2
            self.u11.a //= 2
            self.u12.a //= 2
            self.u20.a //= 2
            self.u21.a //= 2
            self.u22.a //= 2
            self.u00.b //= 2
            self.u01.b //= 2
            self.u02.b //= 2
            self.u10.b //= 2
            self.u11.b //= 2
            self.u12.b //= 2
            self.u20.b //= 2
            self.u21.b //= 2
            self.u22.b //= 2
            return True

    def reduce_by_1(self) -> bool:
        for elt in self.elements():
            if elt.a % 2 != 0:
                return False
        else:
            self.k -= 1
            for elt in self.elements():
                tmp = elt.a
                elt.a = elt.b
                elt.b = tmp // 2
            return True

    def __matmul__(self, other):
        ans = SO3(
            self.u00 * other.u00 + self.u01 * other.u10 + self.u02 * other.u20,
            self.u00 * other.u01 + self.u01 * other.u11 + self.u02 * other.u21,
            self.u00 * other.u02 + self.u01 * other.u12 + self.u02 * other.u22,
            self.u10 * other.u00 + self.u11 * other.u10 + self.u12 * other.u20,
            self.u10 * other.u01 + self.u11 * other.u11 + self.u12 * other.u21,
            self.u10 * other.u02 + self.u11 * other.u12 + self.u12 * other.u22,
            self.u20 * other.u00 + self.u21 * other.u10 + self.u22 * other.u20,
            self.u20 * other.u01 + self.u21 * other.u11 + self.u22 * other.u21,
            self.u20 * other.u02 + self.u21 * other.u12 + self.u22 * other.u22,
            self.k + other.k,
        )
        ans.reduce()
        return ans

    def __str__(self):
        return (
            "k = "
            + str(self.k)
            + "\n["
            + str(self.u00)
            + " "
            + str(self.u01)
            + " "
            + str(self.u02)
            + "]\n["
            + str(self.u10)
            + " "
            + str(self.u11)
            + " "
            + str(self.u12)
            + "]\n["
            + str(self.u20)
            + " "
            + str(self.u21)
            + " "
            + str(self.u22)
            + "]"
        )

    def get_parity(self):
        arr = np.zeros((3, 3), dtype=np.uint8)
        for i in range(9):
            a = self.elements()[i]
            if a.a % 2 != 0:
                row = i // 3
                col = i % 3
                arr[row, col] = 1

        return arr


def is_odd(a: Z_OMEGA):
    if (a.a3 + a.a1) % 2 == 0:
        return False
    else:
        return True


def convert_to_zsqrti(a: Z_OMEGA):
    return Z_SQRT2(a.a4, (a.a3 - a.a1) // 2), Z_SQRT2(a.a2, (a.a3 + a.a1) // 2)


def convert_to_zsqrti_increase_k(a: Z_OMEGA):
    return Z_SQRT2(a.a3 - a.a1, a.a4), Z_SQRT2(a.a3 + a.a1, a.a2)
