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

Functions used in solving the equation t^* t = E given E ∈ Z[√2], with t ∈ Z[ω]. 
The approach used for solving the equation is described in [1].
[1] - arXiv:1403.2975
"""

from pyLIQTR.gate_decomp.rings import Z_OMEGA, Z_SQRT2
from typing import List
from random import randint
import math
from math import gcd as int_gcd


def modular_sqrtm1(p):
    """
    Find the square root of -1 mod p.

    Return None if no h is found. Algorithm described in remark 11 of arXiv:1212.6253.
    """
    exponent = (p - 1) // 2
    num_tries = 25
    for _ in range(num_tries):
        b = randint(0, 50)
        soln = pow(b, exponent, p)
        if soln == p - 1:
            exponent = (p - 1) // 4
            new_soln = pow(b, exponent, p)
            return new_soln
    return None


def ring_gcd(a, b):
    """
    Calculate the gcd of a and b

    Works for any type that has the __mod__ operator defined, but should not be used
    when a and b are integers, as the built-in in math.gcd() is much faster
    """
    if b == 0:
        return a
    else:
        return ring_gcd(b, a % b)


def solveDiophantine(xi: Z_SQRT2):
    """
    Given some E = x + y√2 ∈ Z[√2], try to solve the Diophantine equation
    t^* t = E
    for t ∈ Z[ω]. Return None if no solution is found.
    """
    p = abs(xi)
    if p < 2:
        return None
    factors = find_prime_factors_integers(p)
    if factors is None:
        return None
    s = Z_OMEGA(0, 0, 0, 1)
    alt_xi = xi
    for factor in factors:
        if factor % 8 == 7:
            return None
        prime_factors_Z_SQRT2 = find_prime_factors_Z_SQRT2(factor)
        if prime_factors_Z_SQRT2 is None:
            return None
        for eta in prime_factors_Z_SQRT2:
            # TODO Make a function to check for divisibility instead of try/except
            try:
                alt_xi = alt_xi / eta
                t = find_prime_factor_Z_OMEGA(eta, factor)
                if t is None:
                    return None
                else:
                    s *= t
            except ValueError:
                pass
    u = xi / (s.conj() * s).to_zsqrt()
    if abs(u) ** 2 != 1:
        return None
    assert (s.conj() * s).to_zsqrt() * u == xi
    v = u.sqrt()
    return v.to_zomega() * s


def find_prime_factor_Z_OMEGA(eta: Z_SQRT2, p: int) -> Z_OMEGA:
    """
    Given η ∈ Z[√2] such that η | p where p is a prime integer, find a prime
    factor of η in Z[ω]
    """
    if p == 2:
        return Z_OMEGA(0, 0, 1, 1)
    a = p % 8
    if a % 2 == 0 and p != 2:
        return None
    elif a == 7:
        return None
    elif a == 1 or a == 5:
        h = modular_sqrtm1(p)
        if h is None:
            return None
        return ring_gcd(Z_OMEGA(0, 1, 0, h), Z_OMEGA(-eta.b, 0, eta.b, eta.a))
    elif a == 3:
        h = modular_square_root(-2, p)
        if h is None:
            return None
        return ring_gcd(Z_OMEGA(1, 0, 1, h), Z_OMEGA(-eta.b, 0, eta.b, eta.a))


def find_prime_factors_Z_SQRT2(p: int) -> List[Z_SQRT2]:
    """Given a prime integer p, find its prime factorization in the ring Z[√2]"""
    if p == 2:
        return [Z_SQRT2(0, 1), Z_SQRT2(0, 1)]
    r = p % 8
    if r == 3 or r == 5:
        return [Z_SQRT2(p, 0)]
    if r == 1 or r == 7:
        x = modular_square_root(2, p)
        if x is None:
            return None
        eta = ring_gcd(Z_SQRT2(p, 0), Z_SQRT2(x, 1))
        return [eta, eta.conj()]


def g_plus(x, n):
    return (x ** 2 + 1) % n


def g_minus(x, n):
    return (x ** 2 - 1) % n


def pollard_rho(n: int, use_plus=True, start=2):
    max_tries = 10000
    if use_plus:
        g = g_plus
    else:
        g = g_minus
    x = start
    y = start
    d = 1
    i = 0
    while d == 1:
        x = g(x, n)
        y = g(g(y, n), n)
        d = int_gcd(abs(x - y), n)
        i += 1
        if i > max_tries:
            return None

    if d == n:
        return None
    else:
        return d


def find_prime_factors_integers(n: int):
    factors = []
    stack = []
    stack.insert(0, n)
    while len(stack) != 0:
        p = stack.pop()
        if is_prime(p):
            # can't factor xi if xi | p and p = 7 (mod 8)
            if p % 8 == 7:
                return None
            factors.append(p)
        else:
            starts = [2]
            for start in starts:
                factor = pollard_rho(p, start)
                if factor == 7:
                    return None
                if factor is not None:
                    stack.insert(0, factor)
                    stack.insert(0, p // factor)
                    break
                if factor is None:
                    # try one more time
                    factor = pollard_rho(p, False, start)
                    if factor == 7:
                        return None
                    if factor is not None:
                        stack.insert(0, factor)
                        stack.insert(0, p // factor)
                        break
            if factor is None:
                return None

    return factors


def is_prime(n: int):
    """Use the fermat primaility test to determine if n is prime"""
    if n == 2 or n == 3:
        return True
    a = randint(2, n - 1)

    if int_gcd(a, n) != 1:
        return False
    return pow(a, n - 1, n) == 1


def modular_square_root(a: int, p: int):
    """Calculate the modular square root with the Tonelli-Shanks algorithm"""
    q = p - 1
    s = 0
    while q % 2 == 0:
        q //= 2
        s += 1

    num_tries = int(math.log(p) ** 2)
    # find quadratic non-residue of p. Should succeed on average in 2 tries
    for i in range(num_tries):
        z = randint(2, p - 1)
        if pow(z, (p - 1) // 2, p) != 1:
            break
    if i == num_tries:
        # odds of this happening should be incredibly small!
        return None

    m = s
    c = pow(z, q, p)
    t = pow(a, q, p)
    r = pow(a, (q + 1) // 2, p)
    depth = 1
    while t != 0 and t != 1:
        i = 0
        while pow(t, 2 ** i, p) != 1:
            i += 1
            if i == m:
                # if we get to i = m, this means 2 is not a quadratic residue, likely
                # because p isn't actually prime
                return None

        b = pow(c, 2 ** (m - i - 1), p)
        m = i
        c = pow(b, 2, p)
        t = (t * b ** 2) % p
        r = (r * b) % p

        if depth > 20:
            # don't spend too much effort on this
            print("Hit the depth limit for modular square root solving")
            return None
        depth += 1
    if t == 0:
        return 0

    if t == 1:
        return r
