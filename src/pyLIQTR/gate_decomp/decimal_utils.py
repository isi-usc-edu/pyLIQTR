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

A collection of functions that are calculated to arbitrary precision using the decimal package
"""

from decimal import Decimal, getcontext, localcontext
import math
from pyLIQTR.gate_decomp.rings import ComplexAP

D = Decimal


def exp_c_ap(theta):
    """
    Complex exponential function in arbitrary precision.
    """
    return ComplexAP(prec_cos(theta), prec_sin(theta))


def prec_sin(x):
    """
    Calculate sin(x) to arbitrary precision using its Taylor expansion.

    From https://docs.python.org/3/library/decimal.html
    """
    s = D("0")
    prev = 1
    i = 0
    while s != prev:
        prev = s
        s += (-D("1")) ** i * (x ** (2 * i + 1)) / (math.factorial(2 * i + 1))
        i += 1
    return s


def prec_cos(x):
    """
    Calculate cos(x) to arbitrary precision using its Taylor expansion.

    From https://docs.python.org/3/library/decimal.html
    """
    s = D("1")
    prev = 0
    i = 1
    while s != prev:
        prev = s
        s += (-D("1")) ** i * (x ** (2 * i)) / (math.factorial(2 * i))
        i += 1
    return s


def prec_arcsin(x: D):
    """
    Calculate arcsin(x) to arbitrary precision using its Taylor expansion for |x| < 1
    """
    if abs(x) >= 1:
        raise ValueError(f"{x} not in domain of arcsin ([-1, 1])")
    s = 0
    prev = 1
    i = 0
    while s != prev:
        prev = s
        num = math.factorial(2 * i) * pow(x, 2 * i + 1)
        den = pow(pow(2, i) * math.factorial(i), 2) * (2 * i + 1)
        s += num / D(den)
        i += 1
    return s


def prec_arctan(x: D):
    """
    Calculate arctan(x) to current precision using Eulers infinite series [1]

    For large values of x the series converges slowly, so when x > 1 we make use
    of the identity

    arctan(a) + arctan(b) = arctan[(a+b)/(1-ab)]
    (https://en.wikipedia.org/wiki/Inverse_trigonometric_functions#Arctangent_addition_formula)

    and note that for a = x / (x+1), b = x**2 / (x**2 + x + 1), a,b < 1 and so
    arctan(a) + arctan(b) = arctan(x)

    [1] Chien-Lih, H. (2005). 89.67 An elementary derivation of Euler’s series for the
    arctangent function. The Mathematical Gazette, 89(516), 469-470.
    doi:10.1017/S0025557200178404
    """
    prec = getcontext().prec
    with localcontext() as ctx:
        ctx.prec = prec + 1
        if x > 1:
            a = x / (x + 1)
            b = (x ** 2) / (x ** 2 + x + 1)
            return prec_arctan(a) + prec_arctan(b)
        s = 0
        prev = 1
        i = 0
        while s != prev:
            prev = s
            num = pow(2, 2 * i) * pow(math.factorial(i), 2) * pow(x, 2 * i + 1)
            den = math.factorial(2 * i + 1) * pow(1 + x ** 2, i + 1)
            s += num / den
            i += 1
    return +s


def prec_arctan2(x: D, y: D):
    # special values
    if x == 0:
        if y > 0:
            return prec_pi() / 2
        else:
            return -prec_pi / 2
    make_neg = False
    sub_from_pi = False
    if y < 0:
        make_neg = True
        y *= -1
    if x < 0:
        sub_from_pi = True
        x *= -1
    angle = prec_arctan(y / x)
    if sub_from_pi:
        angle = prec_pi() - angle
    if make_neg:
        angle *= -1
    return angle


def prec_log2(x):
    """
    Calculate the base-2 logarithm to arbitrary precision
    """
    return x.ln() / D("2").ln()


def prec_pi():
    """
    Compute Pi to the current precision.

    From https://docs.python.org/3/library/decimal.html
    """
    getcontext().prec += 2  # extra digits for intermediate steps
    three = Decimal(3)  # substitute "three=3.0" for regular floats
    lasts, t, s, n, na, d, da = 0, three, 3, 1, 0, 0, 24
    while s != lasts:
        lasts = s
        n, na = n + na, na + 8
        d, da = d + da, da + 32
        t = (t * n) / d
        s += t
    getcontext().prec -= 2
    return +s


def arg(u: ComplexAP):
    return prec_arctan2(u.real, u.imaginary)
