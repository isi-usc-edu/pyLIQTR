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

import math
from decimal import Decimal

import numpy as np

D = Decimal


class ComplexAP:
    """
    A class with arbitrary precision complex numbers.
    """

    def __init__(self, a, b):
        """
        a + bi
        """
        self.real: Decimal
        self.imaginary: Decimal
        if isinstance(a, Decimal):
            self.real = a
        elif isinstance(a, int):
            self.real = D(a)
        else:
            raise TypeError("Entries must be instances of decimal.Decimal or int")
        if isinstance(b, Decimal):
            self.imaginary = b
        elif isinstance(b, int):
            self.imaginary = D(b)
        else:
            raise TypeError("Entries must be instances of decimal.Decimal or int")

    def __eq__(self, other):
        if self.real == other.real and self.imaginary == other.imaginary:
            return True
        else:
            return False

    def __add__(self, other):
        return ComplexAP(self.real + other.real, self.imaginary + other.imaginary)

    def __sub__(self, other):
        return ComplexAP(self.real - other.real, self.imaginary - other.imaginary)

    def __mul__(self, other):
        if isinstance(other, ComplexAP):
            return ComplexAP(
                self.real * other.real - self.imaginary * other.imaginary,
                self.imaginary * other.real + self.real * other.imaginary,
            )

        elif isinstance(other, Decimal):
            return ComplexAP(self.real * other, self.imaginary * other)

        else:
            raise TypeError(
                "Can only multiply ComplexAP with ComplexAP or decimal.Decimal, not {}".format(
                    type(other)
                )
            )

    def __abs__(self):
        return (self.real**2 + self.imaginary**2).sqrt()

    def conj(self):
        return ComplexAP(self.real, -self.imaginary)

    def __str__(self):
        return "{} + {}j".format(self.real, self.imaginary)

    def __truediv__(self, other):
        scale = other.real**2 + other.imaginary**2
        real = self.real * other.real + self.imaginary * other.imaginary
        imaginary = self.imaginary * other.real - self.real * other.imaginary
        return ComplexAP(real / scale, imaginary / scale)

    def __pow__(self, other):
        result = ComplexAP(1, 0)
        for _ in range(other):
            result *= self

        return result


class Z_SQRT2:
    """
    A class representing the ring
    Z[√2] = {a + b√2 | a, b ∈ Z}
    """

    def __init__(self, a, b) -> None:
        if not isinstance(a, int) or not isinstance(b, int):
            raise TypeError("Entries must be integers")
        self.a = a
        self.b = b

    def __eq__(self, other):
        if other == 0:
            if self.a == 0 and self.b == 0:
                return True
            else:
                return False
        if self.a == other.a and self.b == other.b:
            return True
        else:
            return False

    def __str__(self):
        return "{} + {}\u221A2".format(self.a, self.b)

    def __add__(self, other):
        return Z_SQRT2(self.a + other.a, self.b + other.b)

    def __sub__(self, other):
        return Z_SQRT2(self.a - other.a, self.b - other.b)

    def __mul__(self, other):
        if isinstance(other, int):
            return Z_SQRT2(self.a * other, self.b * other)
        first = self.a * other.a + 2 * self.b * other.b
        second = self.b * other.a + self.a * other.b
        return Z_SQRT2(first, second)

    def __neg__(self):
        return Z_SQRT2(-self.a, -self.b)

    def __truediv__(self, other):
        if other.a == 0:
            return self.divide_sqrt2(other.b)
        denominator = other.a**2 - 2 * other.b**2
        numerator = self.b * other.a - self.a * other.b

        if numerator % denominator != 0:
            raise ValueError("{} does not divide {}".format(other, self))
        second = numerator // denominator
        if (self.a - 2 * second * other.b) % other.a != 0:
            raise ValueError("{} does not divide {}".format(other, self))
        first = (self.a - 2 * second * other.b) // other.a
        return Z_SQRT2(first, second)

    def divide_sqrt2(self, other):
        first = 0
        second = 0
        if self.b % other == 0:
            first = self.b // other
        else:
            raise ValueError("{}√2 does not divide {}".format(other, self))
        if self.a % (2 * other) == 0:
            second = self.a // (2 * other)
        else:
            raise ValueError("{}√2 does not divide {}".format(other, self))
        return Z_SQRT2(first, second)

    def __mod__(self, other):
        x = abs(other)
        first = round((self.a * other.a - 2 * self.b * other.b) / x)
        second = round((self.b * other.a - self.a * other.b) / x)
        q = Z_SQRT2(first, second)
        return self - q * other

    def divide(self, other):
        x = abs(other)
        first = (self.a * other.a - 2 * self.b * other.b) / x
        first = round(first)
        second = (self.b * other.a - self.a * other.b) / x
        second = round(second)
        q = Z_SQRT2(first, second)
        r = self - q * other
        return q, r

    def conj(self):
        return Z_SQRT2(self.a, -self.b)

    def __abs__(self):
        return self.a**2 - 2 * self.b**2

    def __float__(self):
        return float(self.a + self.b * D("2").sqrt())

    def to_decimal(self, precise: bool = False, num_decimal_places: int = 16):
        """
        Return the number as a Decimal.

        """
        return D("2").sqrt().fma(self.b, self.a)

    # TODO: Implement via repeated squaring
    def __pow__(self, other: int):
        if other < 0:
            raise ValueError("Can only use positive exponents")
        result = Z_SQRT2(1, 0)
        for _ in range(other):
            result = result * self
        return result

    def sqrt(self):
        b_squared_plus = (self.a + D(self.a**2 - 2 * self.b**2).sqrt()) / 4
        b_squared_minus = (self.a - D(self.a**2 - 2 * self.b**2).sqrt()) / 4
        if (D(b_squared_plus).sqrt()).to_integral_value() ** 2 == b_squared_plus:
            b = D(b_squared_plus).sqrt()
        elif (D(b_squared_minus).sqrt()).to_integral_value() ** 2 == b_squared_minus:
            b = D(b_squared_minus).sqrt()
        else:
            raise ValueError(f"{self} is not square")
        if b == 0:
            a = D(self.a).sqrt()
            assert a.to_integral_value() ** 2 == self.a
        else:
            a = self.b / (2 * b)
        return Z_SQRT2(round(a), round(b))

    def to_zomega(self):
        return Z_OMEGA(-self.b, 0, self.b, self.a)


class Z_OMEGA:
    """
    A class representing the ring
    Z[ω] = {a1ω^3 + a2ω^2 + a3ω + a4 | a1, a2, a3, a4 ∈ Z}
    with ω = exp(iπ/4)
    """

    def __init__(self, a1, a2, a3, a4):
        if (
            not isinstance(a1, int)
            or not isinstance(a2, int)
            or not isinstance(a3, int)
            or not isinstance(a4, int)
        ):
            raise TypeError("Entries must be integers")
        self.a1 = a1
        self.a2 = a2
        self.a3 = a3
        self.a4 = a4

    @classmethod
    def from_Z_SQRT2(cls, a: Z_SQRT2, b: Z_SQRT2):
        """
        Initialize an element of the ring Z[ω] using two elements from the ring Z[√2]
        such that c = a + bi ∈ Z[ω]
        """
        a1 = b.b - a.b
        a2 = b.a
        a3 = a.b + b.b
        a4 = a.a
        return cls(a1, a2, a3, a4)

    def __eq__(self, other):
        if isinstance(other, int):
            if self.a4 == other and self.a1 == 0 and self.a2 == 0 and self.a3 == 0:
                return True
            else:
                return False
        else:
            if (
                self.a1 == other.a1
                and self.a2 == other.a2
                and self.a3 == other.a3
                and self.a4 == other.a4
            ):
                return True
            else:
                return False

    def __str__(self):
        return "{}\u03C9^3 + {}\u03C9^2 + {}\u03C9 + {}".format(
            self.a1, self.a2, self.a3, self.a4
        )

    def __add__(self, other):
        return Z_OMEGA(
            self.a1 + other.a1,
            self.a2 + other.a2,
            self.a3 + other.a3,
            self.a4 + other.a4,
        )

    def __sub__(self, other):
        return Z_OMEGA(
            self.a1 - other.a1,
            self.a2 - other.a2,
            self.a3 - other.a3,
            self.a4 - other.a4,
        )

    def __mul__(self, other):
        if isinstance(other, Z_OMEGA):
            return Z_OMEGA(
                self.a1 * other.a4
                + self.a2 * other.a3
                + self.a3 * other.a2
                + self.a4 * other.a1,
                -self.a1 * other.a1
                + self.a2 * other.a4
                + self.a3 * other.a3
                + self.a4 * other.a2,
                -self.a1 * other.a2
                - self.a2 * other.a1
                + self.a3 * other.a4
                + self.a4 * other.a3,
                -self.a1 * other.a3
                - self.a2 * other.a2
                - self.a3 * other.a1
                + self.a4 * other.a4,
            )
        elif isinstance(other, int) or isinstance(other, Decimal):
            return Z_OMEGA(
                self.a1 * other, self.a2 * other, self.a3 * other, self.a4 * other
            )

    def __truediv__(self, other):
        A = np.array(
            [
                [other.a4, other.a3, other.a2, other.a1],
                [-other.a1, other.a4, other.a3, other.a2],
                [-other.a2, -other.a1, other.a4, other.a3],
                [-other.a3, -other.a2, -other.a1, other.a4],
            ],
            dtype=np.int64,
        )
        b = np.array([self.a1, self.a2, self.a3, self.a4], dtype=np.int64)
        x = np.linalg.solve(A, b)
        return Z_OMEGA(round(x[0]), round(x[1]), round(x[2]), round(x[3]))

    def __mod__(self, other):
        n = self * other.conj() * ((other * other.conj()).conj2())
        k = abs(other)
        a1 = (n.a1 + k // 2) // k
        a2 = (n.a2 + k // 2) // k
        a3 = (n.a3 + k // 2) // k
        a4 = (n.a4 + k // 2) // k
        q = Z_OMEGA(a1, a2, a3, a4)
        assert q * other - (q * other - self) == self
        return q * other - self

    def __complex__(self):
        return complex(
            (self.a1 + 0j) * np.exp(1j * math.pi / 4) ** 3
            + (self.a2 + 0j) * np.exp(1j * math.pi / 4) ** 2
            + (self.a3 + 0j) * np.exp(1j * math.pi / 4)
            + self.a4
        )

    def __neg__(self):
        return Z_OMEGA(-self.a1, -self.a2, -self.a3, -self.a4)

    def __pow__(self, other):
        result = Z_OMEGA(0, 0, 0, 1)
        for i in range(other):
            result *= self
        return result

    def to_complexAP(self):
        real = (self.a3 - self.a1) / D("2").sqrt() + self.a4
        imaginary = (self.a3 + self.a1) / D("2").sqrt() + self.a2
        return ComplexAP(real, imaginary)

    def conj(self):
        return Z_OMEGA(-self.a3, -self.a2, -self.a1, self.a4)

    def conj2(self):
        return Z_OMEGA(-self.a1, self.a2, -self.a3, self.a4)

    def __abs__(self):
        return (self.a1**2 + self.a2**2 + self.a3**2 + self.a4**2) ** 2 - 2 * (
            self.a1 * self.a2
            + self.a2 * self.a3
            + self.a3 * self.a4
            - self.a4 * self.a1
        ) ** 2

    def magnitude_squared(self):
        return self * self.conj()

    def to_zsqrt(self):
        assert self.a1 + self.a3 == 0
        assert self.a2 == 0
        return Z_SQRT2(self.a4, self.a3)

    def mul_by_sqrt2(self):
        assert self.a2 == 0 and self.a4 == 0
        assert (self.a3 + self.a1) % 2 == 0 and (self.a3 - self.a1) % 2 == 0
        return Z_OMEGA(0, (self.a1 + self.a3) // 2, 0, (self.a3 - self.a1) // 2)


## static functions


def is_reducible(u: Z_OMEGA) -> bool:
    """
    Given a value u ∈ Z[ω], determine that supposing if we had some
    w = (1/√2^k)u ∈ D[ω], we could re-write w as w = (1/√2^(k-1))u' ∈ D[ω]
    for some u' ∈ Z[ω].
    """
    if (u.a1 + u.a3) % 2 == 0 and (u.a2 + u.a4) % 2 == 0:
        return True

    else:
        return False


def reduce(u: Z_OMEGA) -> Z_OMEGA:
    """
    Given a u ∈ Z[ω] such that is_reducible(u) is True, return u' such that
    (1/√2^k)u = (1/√2^(k-1))u'.
    """
    a1p = (u.a2 - u.a4) // 2
    a2p = (u.a1 + u.a3) // 2
    a3p = (u.a2 + u.a4) // 2
    a4p = (u.a3 - u.a1) // 2
    return Z_OMEGA(a1p, a2p, a3p, a4p)


def increase(u: Z_OMEGA) -> Z_OMEGA:
    a1p = u.a2 - u.a4
    a2p = u.a1 + u.a3
    a3p = u.a2 + u.a4
    a4p = u.a3 - u.a1
    return Z_OMEGA(a1p, a2p, a3p, a4p)
