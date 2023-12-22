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
from __future__ import annotations
import math

import numpy as np
import gmpy2
from gmpy2 import mpfr, mpc, mpz


class Z_SQRT2:
    """
    A class representing the ring
    Z[√2] = {a + b√2 | a, b ∈ Z}
    """

    def __init__(self, a: int, b: int) -> None:
        self.a = a
        self.b = b

    def __eq__(self, other: Z_SQRT2):
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

    def __add__(self, other: Z_SQRT2) -> Z_SQRT2:
        return Z_SQRT2(self.a + other.a, self.b + other.b)

    def __sub__(self, other: Z_SQRT2) -> Z_SQRT2:
        return Z_SQRT2(self.a - other.a, self.b - other.b)

    def __mul__(self, other: Z_SQRT2) -> Z_SQRT2:
        if isinstance(other, int):
            return Z_SQRT2(self.a * other, self.b * other)
        first = self.a * other.a + 2 * self.b * other.b
        second = self.b * other.a + self.a * other.b
        return Z_SQRT2(first, second)

    def __neg__(self) -> Z_SQRT2:
        return Z_SQRT2(-self.a, -self.b)

    def __truediv__(self, other: Z_SQRT2) -> Z_SQRT2:
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

    def divide_sqrt2(self, other: Z_SQRT2) -> Z_SQRT2:
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

    def __mod__(self, other: Z_SQRT2) -> Z_SQRT2:
        x = abs(other)
        first = round((self.a * other.a - 2 * self.b * other.b) / x)
        second = round((self.b * other.a - self.a * other.b) / x)
        q = Z_SQRT2(first, second)
        return self - q * other

    def divide(self, other: Z_SQRT2) -> Z_SQRT2:
        x = abs(other)
        first = (self.a * other.a - 2 * self.b * other.b) / x
        first = round(first)
        second = (self.b * other.a - self.a * other.b) / x
        second = round(second)
        q = Z_SQRT2(first, second)
        r = self - q * other
        return q, r

    def conj(self) -> Z_SQRT2:
        return Z_SQRT2(self.a, -self.b)

    def __abs__(self) -> Z_SQRT2:
        return self.a**2 - 2 * self.b**2

    def __float__(self) -> float:
        return float(self.a + self.b * math.sqrt(2))

    def __mpfr__(self) -> mpfr:
        """
        Return the number as an mpfr object
        """
        res = gmpy2.fma(gmpy2.sqrt(mpfr(2)), self.b, self.a)
        return res

    # TODO: Implement via repeated squaring
    def __pow__(self, other: int) -> Z_SQRT2:
        if other < 0:
            raise ValueError("Can only use positive exponents")
        result = Z_SQRT2(1, 0)
        for _ in range(other):
            result = result * self
        return result

    def sqrt(self) -> Z_SQRT2:
        """
        Compute A = a + b√2 such that A*A = self

        Raises
        ------
        ValueError
            If the object is not a square
        """
        c = mpz(self.a)
        d = mpz(self.b)
        if d == 0:
            return self._simple_sqrt()
        a = self._compute_a(c, d)
        b = self._compute_b(a, c, d)
        ans = Z_SQRT2(int(a), int(b))
        assert (ans * ans == self)
        return ans

    def _compute_a(self, c: mpz, d: mpz) -> mpz:
        tmp_num1, tmp_bool1 = gmpy2.iroot(c**2 - 2*d**2, 2)
        if tmp_bool1:
            if gmpy2.is_even(c + tmp_num1):
                tmp_num2, tmp_bool2 = gmpy2.iroot(gmpy2.divexact(c + tmp_num1, 2), 2)
            if gmpy2.is_even(c - tmp_num1):
                tmp_num3, tmp_bool3 = gmpy2.iroot(gmpy2.divexact(c - tmp_num1, 2), 2)
            if tmp_bool2 or tmp_bool3:
                if tmp_bool2:
                    a = tmp_num2
                else:
                    a = tmp_num3
                return a
        raise ValueError(f"{self} is not square")

    def _compute_b(self, a: mpz, c: mpz, d: mpz) -> mpz:
        q, r = gmpy2.f_divmod(d, 2*a)
        if r == 0:
            return q
        raise ValueError(f"{self} is not square")

    def _simple_sqrt(self):
        c = mpz(self.a)
        t1 = gmpy2.iroot(c, 2)
        if t1[1]:
            ans = Z_SQRT2(int(t1[0]), 0)
            assert (ans * ans == self)
            return ans
        else:
            if c.is_even():
                t2 = gmpy2.iroot(gmpy2.divexact(c, 2), 2)
                if t2[1]:
                    ans = Z_SQRT2(int(t1[1]), 0)
                    assert (ans * ans == self)
                    return ans
        raise ValueError(f"{self} is not square")

    def to_zomega(self) -> Z_OMEGA:
        return Z_OMEGA(-self.b, 0, self.b, self.a)


class Z_OMEGA:
    """
    A class representing the ring
    Z[ω] = {a1ω^3 + a2ω^2 + a3ω + a4 | a1, a2, a3, a4 ∈ Z}
    with ω = exp(iπ/4)
    """

    def __init__(self, a1, a2, a3, a4):
        self.a1 = a1
        self.a2 = a2
        self.a3 = a3
        self.a4 = a4

    @classmethod
    def from_Z_SQRT2(cls, a: Z_SQRT2, b: Z_SQRT2) -> Z_OMEGA:
        """
        Initialize an element of the ring Z[ω] using two elements from the ring Z[√2]
        such that c = a + bi ∈ Z[ω]
        """
        a1 = b.b - a.b
        a2 = b.a
        a3 = a.b + b.b
        a4 = a.a
        return cls(a1, a2, a3, a4)

    def __eq__(self, other: Z_OMEGA):
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

    def __str__(self) -> str:
        return "{}\u03C9^3 + {}\u03C9^2 + {}\u03C9 + {}".format(
            self.a1, self.a2, self.a3, self.a4
        )

    def __add__(self, other: Z_OMEGA) -> Z_OMEGA:
        return Z_OMEGA(
            self.a1 + other.a1,
            self.a2 + other.a2,
            self.a3 + other.a3,
            self.a4 + other.a4,
        )

    def __sub__(self, other: Z_OMEGA) -> Z_OMEGA:
        return Z_OMEGA(
            self.a1 - other.a1,
            self.a2 - other.a2,
            self.a3 - other.a3,
            self.a4 - other.a4,
        )

    def __mul__(self, other: Z_OMEGA) -> Z_OMEGA:
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
        elif isinstance(other, int):
            return Z_OMEGA(
                self.a1 * other, self.a2 * other, self.a3 * other, self.a4 * other
            )

    def __truediv__(self, other: Z_OMEGA) -> Z_OMEGA:
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

    def __mod__(self, other: Z_OMEGA) -> Z_OMEGA:
        n = self * other.conj() * ((other * other.conj()).conj2())
        k = abs(other)
        a1 = (n.a1 + k // 2) // k
        a2 = (n.a2 + k // 2) // k
        a3 = (n.a3 + k // 2) // k
        a4 = (n.a4 + k // 2) // k
        q = Z_OMEGA(a1, a2, a3, a4)
        assert q * other - (q * other - self) == self
        return q * other - self

    def __complex__(self) -> complex:
        return complex(
            (self.a1 + 0j) * np.exp(1j * math.pi / 4) ** 3
            + (self.a2 + 0j) * np.exp(1j * math.pi / 4) ** 2
            + (self.a3 + 0j) * np.exp(1j * math.pi / 4)
            + self.a4
        )

    def __neg__(self) -> Z_OMEGA:
        return Z_OMEGA(-self.a1, -self.a2, -self.a3, -self.a4)

    def __pow__(self, other: Z_OMEGA) -> Z_OMEGA:
        result = Z_OMEGA(0, 0, 0, 1)
        for _ in range(other):
            result *= self
        return result

    def __mpc__(self) -> mpc:
        real = (self.a3 - self.a1) / gmpy2.sqrt(2) + self.a4
        imaginary = (self.a3 + self.a1) / gmpy2.sqrt(2) + self.a2
        return mpc(real, imaginary)

    def conj(self) -> Z_OMEGA:
        return Z_OMEGA(-self.a3, -self.a2, -self.a1, self.a4)

    def conj2(self) -> Z_OMEGA:
        return Z_OMEGA(-self.a1, self.a2, -self.a3, self.a4)

    def __abs__(self) -> int:
        return (self.a1**2 + self.a2**2 + self.a3**2 + self.a4**2) ** 2 - 2 * (
            self.a1 * self.a2
            + self.a2 * self.a3
            + self.a3 * self.a4
            - self.a4 * self.a1
        ) ** 2

    def magnitude_squared(self) -> Z_OMEGA:
        return self * self.conj()

    def to_zsqrt(self) -> Z_SQRT2:
        assert self.a1 + self.a3 == 0
        assert self.a2 == 0
        return Z_SQRT2(self.a4, self.a3)

    def mul_by_sqrt2(self) -> Z_OMEGA:
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
