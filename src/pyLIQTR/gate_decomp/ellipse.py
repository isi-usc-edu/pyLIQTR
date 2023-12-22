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

from typing import Tuple, Union

import gmpy2
from gmpy2 import mpfr


class Ellipse:
    """A class representing ellipses as positive definite matrices"""

    def __init__(
        self,
        a: mpfr,
        b: mpfr,
        d: mpfr,
        x: mpfr = mpfr(0),
        y: mpfr = mpfr(0),
        semi_major: Union[mpfr, None] = None,
        semi_minor: Union[mpfr, None] = None,
    ) -> None:
        self.x = x
        self.y = y
        self.a = a
        self.b = b
        self.d = d
        self.z = None
        self.e = None
        self.semi_major = semi_major
        self.semi_minor = semi_minor

    @classmethod
    def from_axes(
        cls, x: mpfr, y: mpfr, theta: mpfr, semi_major: mpfr, semi_minor: mpfr
    ):
        delta = gmpy2.exp2(-50)
        with gmpy2.local_context(gmpy2.get_context(), round=gmpy2.RoundUp):
            semi_major *= 1 + delta
            semi_minor *= 1 + delta
        a = (gmpy2.sin(theta) / semi_minor) ** 2 + (gmpy2.cos(theta) / semi_major) ** 2
        b = (
            gmpy2.cos(theta)
            * gmpy2.sin(theta)
            * (1 / (semi_major) ** 2 - 1 / (semi_minor) ** 2)
        )
        d = (gmpy2.sin(theta) / semi_major) ** 2 + (gmpy2.cos(theta) / semi_minor) ** 2
        return cls(a, b, d, x, y, semi_major=semi_major, semi_minor=semi_minor)

    def _calc_z_and_e(self):
        """For an ellipse described by the matrix [[a, b], [b, d]], find z and e such
        that a = eλ^{-z}, d = eλ^z
        """
        lmbda = 1 + gmpy2.sqrt(2)
        self.z = mpfr(0.5) * gmpy2.log2(self.d / self.a) / gmpy2.log2(lmbda)
        self.e = gmpy2.sqrt(self.a * self.d)

    def contains(self, x: mpfr, y: mpfr) -> bool:
        """Determine if the ellipse contains the point (x, y)"""
        magnitude = (
            self.a * (x - self.x) ** 2
            + 2 * self.b * (x - self.x) * (y - self.y)
            + self.d * (y - self.y) ** 2
        )
        if magnitude <= 1:
            return True
        else:
            False

    def descriminant(self) -> mpfr:
        gmpy2.get_context().precision += 10
        descriminant = (self.a + self.d) ** 2 - 4 * (self.a * self.d - self.b**2)
        gmpy2.get_context().precision -= 10
        return +descriminant

    def is_positive_semi_definite(self) -> bool:
        descriminant = self.descriminant()
        if descriminant < 0:
            raise ValueError(
                "Ellipse has imaginary eigenvalues. Descriminant ="
                f" {descriminant} Ellipse = {self}"
            )
        return ((self.a + self.d - gmpy2.sqrt(descriminant))) > 0

    def determinant(self) -> mpfr:
        return self.a * self.d - self.b**2

    def compute_y_points(self, x: mpfr) -> Tuple[mpfr, mpfr]:
        x -= self.x
        first = pow(self.b, 2) * pow(x, 2)
        second = self.d * (self.a * pow(x, 2) - 1)
        descriminant = first - second

        if descriminant < 0:
            raise ValueError("x value is outside ellipse")
        descriminant_sqrt = gmpy2.sqrt(descriminant)
        y1 = (-self.b * x - descriminant_sqrt) / self.d
        y2 = (-self.b * x + descriminant_sqrt) / self.d
        return y1 + self.y, y2 + self.y

    def compute_x_points(self, y: mpfr) -> Tuple[mpfr, mpfr]:
        y -= self.y
        descriminant = pow(y, 2) * (pow(self.b, 2) - self.a * self.d) + self.a
        if descriminant < 0:
            raise ValueError("y value is outside ellipse")
        descriminant_sqrt = gmpy2.sqrt(descriminant)
        x1 = (-self.b * y - descriminant_sqrt) / self.a
        x2 = (-self.b * y + descriminant_sqrt) / self.a
        return x1 + self.x, x2 + self.x

    def __str__(self) -> str:
        return (
            f"Ellipse: a = {self.a}, b = {self.b}, d = {self.d}, z = {self.z}, e ="
            f" {self.e}, x = {self.x}, y = {self.y}"
        )


def calculate_skew(a: Ellipse, b: Ellipse):
    return a.b**2 + b.b**2


def calculate_bias(a: Ellipse, b: Ellipse):
    if a.z is None:
        a._calc_z_and_e()
    if b.z is None:
        b._calc_z_and_e()
    return b.z - a.z


def force_det_one(a: Ellipse):
    det = a.determinant()
    scale = gmpy2.sqrt(det)
    return scale, Ellipse(
        a.a / scale,
        a.b / scale,
        a.d / scale,
        a.x,
        a.y,
        semi_major=a.semi_major / gmpy2.sqrt(scale),
        semi_minor=a.semi_minor / gmpy2.sqrt(scale),
    )


def scale_ellipse(a: Ellipse, scale: int):
    return Ellipse(
        a.a * scale,
        a.b * scale,
        a.d * scale,
        a.x,
        a.y,
        semi_major=a.semi_major * gmpy2.sqrt(scale),
        semi_minor=a.semi_minor * gmpy2.sqrt(scale),
    )
