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

from decimal import Decimal, getcontext
from pyLIQTR.gate_decomp.decimal_utils import prec_pi, exp_c_ap, arg
from pyLIQTR.gate_decomp.rings import Z_OMEGA, Z_SQRT2, ComplexAP
from pyLIQTR.gate_decomp.point_enumeration import candidate_generator_direct, candidate_generator_fallback
from pyLIQTR.gate_decomp.solve_diophantine import solveDiophantine
from time import perf_counter
from typing import Tuple, Union
from pyLIQTR.gate_decomp.exact_decomp import (
    exact_decomp_to_matrix_string,
    exact_decomp_to_qasm,
)
from math import gcd

D = Decimal
PI = prec_pi()


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


def corrections(theta):
    """
    Find the multiple of pi/2 that when added to theta, results in
    theta+correction ∈ [-pi/4, pi/4]
    """
    u_correction = Z_OMEGA(0, 0, 0, 1)
    theta_correction = D("0")
    if theta >= PI / 4 and theta < 3 * PI / 4:
        u_correction = Z_OMEGA(0, 1, 0, 0)
        theta_correction = -prec_pi() / D("2")
    if theta >= 3 * PI / 4 and theta < 5 * PI / 4:
        u_correction = Z_OMEGA(0, 0, 0, -1)
        theta_correction = -prec_pi()
    if theta >= 5 * PI / 4 and theta < 7 * PI / 4:
        u_correction = Z_OMEGA(0, -1, 0, 0)
        theta_correction = -3 * prec_pi() / D("2")
    return u_correction, theta_correction


def approxRzAnalysis(eps: D, theta: D) -> Tuple[Z_OMEGA, Z_OMEGA, int, int, D, float]:
    """
    Approximate Rz(theta) as a unitary with entries from the ring D[ω] where
    D[ω] = {1/√2^k(aω^3 + bω^2 + cω + d) | k ∈ N, a, b, c, d ∈ Z}

    This version of the function returns additional information about the solution,
    such as the number of candidates tried, the true error of the approximation,
    and the time it took to find the approximation.
    """
    start = perf_counter()
    u_correction, theta_correction = corrections(theta)
    u_generator = candidate_generator_direct(eps, theta + theta_correction)
    ii = 0
    z = exp_c_ap(theta)
    for u, k in u_generator:
        ii += 1
        while is_reducible(u):
            print("u = ", u, " is reducible")
            k -= 1
            u = reduce(u)
        xi = Z_SQRT2(2 ** k, 0) - (u * u.conj()).to_zsqrt()
        t = solveDiophantine(xi)
        if t is not None:
            end = perf_counter()
            if k % 2 == 0:
                scale = 2 ** (k // 2)
            else:
                scale = 2 ** (k // 2) * D(2).sqrt()
            uc = (u * u_correction).to_complexAP() * (1 / D(scale))
            print("uc = ", uc)
            print("z  = ", z)
            assert (t.conj() * t).to_zsqrt() == xi
            err = get_error(uc, z)
            return u * u_correction, t * u_correction, k, ii, err, end - start


def approxRz_direct(eps: D, theta: D) -> Tuple[Z_OMEGA, Z_OMEGA, int]:
    """
    Approximate Rz(theta) as a unitary with entries from the ring D[ω]
    where D[ω] = {1/√2^k(aω^3 + bω^2 + cω + d) | k ∈ N, a, b, c, d ∈ Z}
    """
    u_correction, theta_correction = corrections(theta)
    u_generator = candidate_generator_direct(eps, theta + theta_correction)
    for u, k in u_generator:
        while is_reducible(u):
            k -= 1
            u = reduce(u)
        xi = Z_SQRT2(2 ** k, 0) - (u * u.conj()).to_zsqrt()
        t = solveDiophantine(xi)
        if t is not None:
            assert (t.conj() * t).to_zsqrt() == xi
            return u * u_correction, t * u_correction, k


def approxRz_fallback(eps: D, theta: D, r: D) -> Tuple[Z_OMEGA, Z_OMEGA, int]:
    """"""
    u_correction, theta_correction = corrections(theta)
    u_generator = candidate_generator_fallback(eps, theta + theta_correction, r)
    for u, k in u_generator:
        while is_reducible(u):
            k -= 1
            u = reduce(u)
        xi = Z_SQRT2(2 ** k, 0) - (u * u.conj()).to_zsqrt()
        t = solveDiophantine(xi)
        if t is not None:
            assert (t.conj() * t).to_zsqrt() == xi
            return u * u_correction, t * u_correction, k


def get_ring_elts_analysis(
    init_theta: Union[float, D], prec: int
) -> Tuple[Z_OMEGA, Z_OMEGA, int, int, float]:
    """A version of get_ring_elts used for profiling"""
    getcontext().prec = max(2 * prec + 3, 28)
    eps = D("1e-{}".format(prec))
    theta = -D(init_theta)
    while theta < D("0"):
        theta += 2 * prec_pi()
    while theta > 2 * prec_pi():
        theta -= 2 * prec_pi()
    u, t, k, num_cands, err, time = approxRzAnalysis(eps, theta)
    print("u = ", u)
    print(
        "u = ",
        u.to_complexAP().real / D(2).sqrt() ** k,
        " + ",
        u.to_complexAP().imaginary / D(2).sqrt() ** k,
        "i",
    )
    print("t = ", t)
    print(
        "t = ",
        t.to_complexAP().real / D(2).sqrt() ** k,
        " + ",
        t.to_complexAP().imaginary / D(2).sqrt() ** k,
        "i",
    )
    print("t*t = ", (t.conj() * t).to_zsqrt())
    print("k = ", k)
    print("2^k - u*u = ", Z_SQRT2(2 ** k, 0) - (u.conj() * u).to_zsqrt())
    print("num cands = ", num_cands)
    print("Total time: {:.3f}s".format(time))
    print("Time per candidate: {:.3f}s".format((time) / num_cands))
    print("Actual error: {:.3e}".format(err))
    return u, t, k, num_cands, time


def get_ring_elts_direct(
    init_theta: Union[float, D], prec: int, eps: Union[D, None] = None
) -> Tuple[Z_OMEGA, Z_OMEGA, int]:
    """
    Given an angle θ, find elements u, t ∈ Z[ω], and an integer k such that the matrix
    (1/√2^k) | u -t^* |
             | t  u^* |
    approximates a Z-rotation over the angle theta, i.e. the operator
    | exp(-iθ/2)  0 |
    |   0 exp(iθ/2) |
    """
    if prec == 0:
        prec = abs(eps.log10().to_integral_value("ROUND_UP"))
    getcontext().prec = max(2 * int(prec) + 4, 28)
    if eps is None:
        eps = D("1e-{}".format(prec))
    theta = -D(init_theta) / D(2)
    while theta < D("0"):
        theta += 2 * prec_pi()
    while theta > 2 * prec_pi():
        theta -= 2 * prec_pi()
    u, t, k = approxRz_direct(eps, theta)
    return u, t, k


def get_ring_elts_fallback(
    init_theta: D, prec: int, r: D
) -> Tuple[Z_OMEGA, Z_OMEGA, int, Z_OMEGA, Z_OMEGA, int]:
    getcontext().prec = max(2 * prec + 4, 28)
    eps = D("1e-{}".format(prec))
    theta = D(init_theta) / D(2)
    while theta < D("0"):
        theta += 2 * prec_pi()
    while theta > 2 * prec_pi():
        theta -= 2 * prec_pi()
    u1, t1, k1 = approxRz_fallback(eps / 2, theta, r)
    u1 = u1.conj()
    t1 = -t1.conj()
    # if the projective rotation "fails", we will instead apply
    # the rotation exp(i*Arg(t)*Z). When this happens, need
    # to use the "fallback" correction theta - Arg(t)
    t1c = t1.to_complexAP()
    t1_mag_squared = (t1c.real ** 2 + t1c.imaginary ** 2) / (2 ** k1)
    theta_fallback = theta - arg(t1c)
    u2, t2, k2 = get_ring_elts_direct(theta_fallback, 0, eps / (2 * t1_mag_squared))
    return u1, t1, k1, u2, t2, k2


def get_error(u: ComplexAP, z: ComplexAP) -> D:
    err = 2 - 2 * (z.real * u.real + z.imaginary * u.imaginary)
    return err.sqrt()


def check_common_cases(
    numerator: int, denominator: int, use_qasm: bool
) -> Union[str, None]:
    # check some common cases
    if denominator == 4:
        if numerator == 1:
            return "T"
        elif numerator == 3:
            return "TS"
        elif numerator == 5:
            return "TZ"
        elif numerator == 7:
            return "TSZ"
    elif denominator == 2:
        if numerator == 1:
            return "S"
        elif numerator == 3:
            return "SZ"
    elif denominator == 1:
        return "Z"
    else:
        return None


def approximate_rz_direct(
    numerator: int, denominator: int, precision: int, use_qasm: bool = False
) -> str:
    """
    Approximate a Z-rotation of the angle numerator*pi/denominator to the precision
    10^{-prec} in the operator norm.
    """
    if numerator > 2 * denominator or numerator < 1:
        raise ValueError("Angle must be in range (0, 2*pi)")

    factor = gcd(numerator, denominator)
    numerator = numerator // factor
    denominator = denominator // factor

    simple_sol = check_common_cases(numerator, denominator, use_qasm)

    if simple_sol is not None:
        return simple_sol

    getcontext().prec = 2 * precision + 4
    u, t, k = get_ring_elts_direct(D(numerator) * prec_pi() / D(denominator), precision)
    if use_qasm:
        return exact_decomp_to_qasm(u, t, k)
    else:
        return exact_decomp_to_matrix_string(u, t, k)


def approximate_rz_fallback(
    numerator: int,
    denominator: int,
    precision: int,
    r: D = D("0.999"),
    use_qasm: bool = False,
) -> Union[Tuple[str, int, str, int], Tuple[str, int, int]]:
    """
    Approximate a Z-rotation of the angle numerator*pi/denominator to the precision
    10^{-prec} in the diamond norm using the fallback protocol.

    The return type depends on the parameter use_qasm. If set to False, this function
    returns a tuple containing the primary gate sequence, its T count, the secondary
    gate sequence, and its T count. If set to True, the function returns a tuple with
    the fallback circuit as a qasm circuit, the primary T count, and the secondary T
    count. In the qasm circuit, q0 is the qubit upon which we are performing the
    rotation, and q1 is the ancilla qubit initialized in the zero state.
    """
    if numerator > 2 * denominator or numerator < 1:
        raise ValueError("Angle must be in range (0, 2*pi)")

    factor = gcd(numerator, denominator)
    numerator = numerator // factor
    denominator = denominator // factor

    simple_sol = check_common_cases(numerator, denominator, use_qasm)

    if simple_sol is not None:
        return simple_sol

    getcontext().prec = 2 * (precision + 1) + 4

    u1, t1, k1, u2, t2, k2 = get_ring_elts_fallback(
        D(numerator) * prec_pi() / D(denominator), precision, r
    )

    if use_qasm:
        primary_gate_sequence = exact_decomp_to_qasm(u1, t1, k1)
        fallback_gate_sequence = exact_decomp_to_qasm(u2, t2, k2, conditional=True)
        qasm_string = (
            "initzero q1\ncx q1, q0\n"
            + primary_gate_sequence[0]
            + "\ncx q1, q0\nmzr q1, r1\n@classcomp"
            + " print_classical_bits input r1 output\nif ( r1 == 1 ) y q0\n"
            + fallback_gate_sequence[0]
        )

        return qasm_string
    else:
        primary_gate_sequence = exact_decomp_to_matrix_string(u1, t1, k1)
        fallback_gate_sequence = exact_decomp_to_matrix_string(u2, t2, k2)
        return primary_gate_sequence + fallback_gate_sequence
