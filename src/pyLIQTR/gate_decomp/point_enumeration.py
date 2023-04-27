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

Functions for enumerating points of the ring D[ω] in various subregions of the unit
disk. Uses methods described in [1].
[1] - arXiv:1403.2975
"""

import math
import sys
from copy import deepcopy
from decimal import ROUND_FLOOR, Decimal
from typing import Iterable, List, Tuple, Union

from pyLIQTR.gate_decomp.decimal_utils import *
from pyLIQTR.gate_decomp.ellipse import (
    Ellipse,
    calculate_bias,
    calculate_skew,
    force_det_one,
    scale_ellipse,
)
from pyLIQTR.gate_decomp.grid_operator import GridOperator
from pyLIQTR.gate_decomp.rings import Z_OMEGA, Z_SQRT2, is_reducible, reduce

D = Decimal


# General methods


def scaled_one_dim_grid_problem(x0: D, x1: D, y0: D, y1: D) -> Iterable[Z_SQRT2]:
    """
    Enumerate solutions to the scaled one dimensional grid problem for intervals [x0, x1],
    [y0, y1], i.e solve the one dimensional grid problem in the specific case that
    -1 + √2 <= x1 - x0 < 1.
    """
    num_places = abs(x0.as_tuple().exponent - y0.as_tuple().exponent)
    max_digits = 0
    for num in [x0, x1, y0, y1]:
        if len(num.as_tuple().digits) > max_digits:
            max_digits = len(num.as_tuple().digits)
    current_prec = getcontext().prec
    with localcontext() as ctx:
        ctx.prec = max(current_prec, max_digits + num_places)
        SQRT_2 = D("2").sqrt()
        assert x1 - x0 >= 1 - SQRT_2
        lower_bound_b = (x0 - y1) / (2 * SQRT_2)
        upper_bound_b = (x1 - y0) / (2 * SQRT_2)

        for b in range(math.floor(upper_bound_b), math.ceil(lower_bound_b) - 1, -1):
            lower_bound_a = x0 - b * SQRT_2
            upper_bound_a = x1 - b * SQRT_2
            if math.ceil(lower_bound_a) == math.floor(upper_bound_a):
                a = math.ceil(lower_bound_a)
                alpha = a + b * SQRT_2
                beta = a - b * SQRT_2
                if alpha >= x0 and alpha <= x1 and beta >= y0 and beta <= y1:
                    yield Z_SQRT2(math.ceil(lower_bound_a), b)


def get_num_pot_sols_scaled(x0: D, x1: D, y0: D, y1: D) -> int:
    num_places = abs(x0.as_tuple().exponent - y0.as_tuple().exponent)
    max_digits = 0
    for num in [x0, x1, y0, y1]:
        if len(num.as_tuple().digits) > max_digits:
            max_digits = len(num.as_tuple().digits)
    current_prec = getcontext().prec
    with localcontext() as ctx:
        ctx.prec = max(current_prec, max_digits + num_places)
        SQRT_2 = D("2").sqrt()
        assert x1 - x0 >= 1 - SQRT_2
        lower_bound_b = (x0 - y1) / (2 * SQRT_2)
        upper_bound_b = (x1 - y0) / (2 * SQRT_2)
        return int(upper_bound_b - lower_bound_b) + 1


def get_num_pot_sols(
    x0: D,
    x1: D,
    y0: D,
    y1: D,
) -> int:
    delta = x1 - x0
    Delta = y1 - y0

    l = Z_SQRT2(1, 1)
    linv = Z_SQRT2(-1, 1)
    kd = find_k(delta)
    kD = find_k(Delta)
    if abs(kd) <= abs(kD):
        k = find_k(delta)
        if k < 0:
            x_scale = (l ** (-k)).to_decimal()
            y_scale = (-D(1)) ** (k) * (linv ** (-k)).to_decimal()
        else:
            x_scale = (linv**k).to_decimal()
            y_scale = ((-l) ** k).to_decimal()
        x0_scaled = x_scale * x0
        x1_scaled = x_scale * x1
        y0_scaled = y_scale * y0
        y1_scaled = y_scale * y1
        if y0_scaled < y1_scaled:
            num_potential_sols = get_num_pot_sols_scaled(
                x0_scaled, x1_scaled, y0_scaled, y1_scaled
            )
        else:
            num_potential_sols = get_num_pot_sols_scaled(
                x0_scaled, x1_scaled, y1_scaled, y0_scaled
            )
        return num_potential_sols
    else:
        return get_num_pot_sols(y0, y1, x0, x1)


def find_k(delta: D) -> int:
    """
    Given some value δ >= 1, find the smallest integer k such that δ * (λ^-1)^k < 1, where
    λ = 1 + √2
    """
    l = Z_SQRT2(1, 1).to_decimal()
    log_result = delta.log10() / l.log10()
    result = int((log_result + 1).to_integral_value(rounding=ROUND_FLOOR))
    return result


def solve_one_dim_grid_problem(x0: D, x1: D, y0: D, y1: D) -> Iterable[Z_SQRT2]:
    """
    Enumerate solutions to the one dimensional grid problem given intervals [x0, x1] and
    [y0, y1].

    Given two real intervals [x0, x1] and [y0, y1] such that
    (x1 - x0)*(y1 - y0) >= (1 + √2)^2, enumerate all numbers of the form a + b√2 such
    that a + b√2 ∈ [x0, x1] and a - b√2 ∈ [y0, y1].
    """
    delta = x1 - x0
    Delta = y1 - y0
    l = Z_SQRT2(1, 1)
    linv = Z_SQRT2(-1, 1)
    kd = find_k(delta)
    kD = find_k(Delta)
    if abs(kd) <= abs(kD):
        k = find_k(delta)
        if k < 0:
            x_scale = (l ** (-k)).to_decimal()
            y_scale = (-D(1)) ** (k) * (linv ** (-k)).to_decimal()
        else:
            x_scale = (linv**k).to_decimal()
            y_scale = ((-l) ** k).to_decimal()
        x0_scaled = x_scale * x0
        x1_scaled = x_scale * x1
        y0_scaled = y_scale * y0
        y1_scaled = y_scale * y1
        if y0_scaled < y1_scaled:
            scaled_solutions = scaled_one_dim_grid_problem(
                x0_scaled, x1_scaled, y0_scaled, y1_scaled
            )
        else:
            scaled_solutions = scaled_one_dim_grid_problem(
                x0_scaled, x1_scaled, y1_scaled, y0_scaled
            )
        if k >= 0:
            return (sol * Z_SQRT2(1, 1) ** k for sol in scaled_solutions)
        else:
            return (sol / Z_SQRT2(1, 1) ** -k for sol in scaled_solutions)
    else:
        return (sol.conj() for sol in solve_one_dim_grid_problem(y0, y1, x0, x1))


def solve_two_dim_grid_problem_upright_rectangles(
    Ax0: D,
    Ax1: D,
    Ay0: D,
    Ay1: D,
    Bx0: D,
    Bx1: D,
    By0: D,
    By1: D,
    ellipse1: Union[Ellipse, None] = None,
    ellipse2: Union[Ellipse, None] = None,
) -> Iterable[Z_OMEGA]:
    """
    Given two subregions A and B, of R^2, of the form A,B = [x0, x1] x [y0, y1], find
    all u ∈ Z[ω] such that u ∈ A and u.conj2() ∈ B (where conj2 is sqrt(2) conjugation)
    """
    solutions = []
    num_x_grid_points = get_num_pot_sols(Ax0, Ax1, Bx0, Bx1)
    num_y_grid_points = get_num_pot_sols(Ay0, Ay1, By0, By1)
    if num_x_grid_points > 100 * num_y_grid_points and ellipse1 is not None:
        beta_solutions1 = solve_one_dim_grid_problem(Ay0, Ay1, By0, By1)
        for beta in beta_solutions1:
            assert beta.to_decimal() >= Ay0
            assert beta.to_decimal() <= Ay1
            Ax0_tmp, Ax1_tmp = ellipse1.compute_x_points(beta.to_decimal())
            assert beta.conj().to_decimal() <= By1
            assert beta.conj().to_decimal() >= By0
            Bx0_tmp, Bx1_tmp = ellipse2.compute_x_points(beta.conj().to_decimal())
            if Ax1_tmp - Ax0_tmp > 0 and Bx1_tmp - Bx0_tmp > 0:
                new_alpha_solutions = solve_one_dim_grid_problem(
                    Ax0_tmp, Ax1_tmp, Bx0_tmp, Bx1_tmp
                )
                for alpha in new_alpha_solutions:
                    yield Z_OMEGA.from_Z_SQRT2(alpha, beta)
    elif num_y_grid_points > 100 * num_x_grid_points and ellipse1 is not None:
        alpha_solutions1 = solve_one_dim_grid_problem(Ax0, Ax1, Bx0, Bx1)
        for alpha in alpha_solutions1:
            Ay0_tmp, Ay1_tmp = ellipse1.compute_y_points(alpha.to_decimal())
            By0_tmp, By1_tmp = ellipse2.compute_y_points(alpha.conj().to_decimal())
            if Ay1_tmp - Ay0_tmp > 0 and By1_tmp - By0_tmp > 0:
                new_beta_solutions = solve_one_dim_grid_problem(
                    Ay0_tmp, Ay1_tmp, By0_tmp, By1_tmp
                )
                for beta in new_beta_solutions:
                    yield Z_OMEGA.from_Z_SQRT2(alpha, beta)
    else:
        alpha_solutions1 = solve_one_dim_grid_problem(Ax0, Ax1, Bx0, Bx1)
        beta_solutions1 = solve_one_dim_grid_problem(Ay0, Ay1, By0, By1)
        found_beta1_solutions = []
        for alpha in alpha_solutions1:
            if len(found_beta1_solutions) == 0:
                for beta in beta_solutions1:
                    found_beta1_solutions.append(beta)
                    yield Z_OMEGA.from_Z_SQRT2(alpha, beta)
            else:
                for beta in found_beta1_solutions:
                    yield Z_OMEGA.from_Z_SQRT2(alpha, beta)

    offset = 1 / D("2").sqrt()
    Ax0 -= offset
    Ax1 -= offset
    Ay0 -= offset
    Ay1 -= offset
    Bx0 += offset
    Bx1 += offset
    By0 += offset
    By1 += offset

    if ellipse1 is not None:
        ellipse1_offset = Ellipse(
            ellipse1.a, ellipse1.b, ellipse1.d, ellipse1.x - offset, ellipse1.y - offset
        )
        ellipse2_offset = Ellipse(
            ellipse2.a, ellipse2.b, ellipse2.d, ellipse2.x - offset, ellipse2.y - offset
        )

    num_x_grid_points = get_num_pot_sols(Ax0, Ax1, Bx0, Bx1)
    num_y_grid_points = get_num_pot_sols(Ay0, Ay1, By0, By1)
    if num_x_grid_points > 100 * num_y_grid_points and ellipse1 is not None:
        beta_solutions1 = solve_one_dim_grid_problem(Ay0, Ay1, By0, By1)
        for beta in beta_solutions1:
            Ax0_tmp, Ax1_tmp = ellipse1_offset.compute_x_points(beta.to_decimal())
            Bx0_tmp, Bx1_tmp = ellipse2_offset.compute_x_points(
                beta.conj().to_decimal()
            )
            if Ax1_tmp - Ax0_tmp > 0 and Bx1_tmp - Bx0_tmp > 0:
                new_alpha_solutions = solve_one_dim_grid_problem(
                    Ax0_tmp, Ax1_tmp, Bx0_tmp, Bx1_tmp
                )
                for alpha in new_alpha_solutions:
                    yield Z_OMEGA.from_Z_SQRT2(alpha, beta) + Z_OMEGA(0, 0, 1, 0)
    elif num_y_grid_points > 100 * num_x_grid_points and ellipse1 is not None:
        alpha_solutions1 = solve_one_dim_grid_problem(Ax0, Ax1, Bx0, Bx1)
        for alpha in alpha_solutions1:
            Ay0_tmp, Ay1_tmp = ellipse1_offset.compute_y_points(alpha.to_decimal())
            By0_tmp, By1_tmp = ellipse2_offset.compute_y_points(
                alpha.conj().to_decimal()
            )
            if Ay1_tmp - Ay0_tmp > 0 and By1_tmp - By0_tmp > 0:
                new_beta_solutions = solve_one_dim_grid_problem(
                    Ay0_tmp, Ay1_tmp, By0_tmp, By1_tmp
                )
                for beta in new_beta_solutions:
                    yield Z_OMEGA.from_Z_SQRT2(alpha, beta) + Z_OMEGA(0, 0, 1, 0)
    else:
        alpha_solutions2 = solve_one_dim_grid_problem(Ax0, Ax1, Bx0, Bx1)
        beta_solutions2 = solve_one_dim_grid_problem(Ay0, Ay1, By0, By1)
        found_beta2_solutions = []
        for alpha in alpha_solutions2:
            if len(found_beta2_solutions) == 0:
                for beta in beta_solutions2:
                    found_beta2_solutions.append(beta)
                    yield Z_OMEGA.from_Z_SQRT2(alpha, beta) + Z_OMEGA(0, 0, 1, 0)
            else:
                for beta in found_beta2_solutions:
                    yield Z_OMEGA.from_Z_SQRT2(alpha, beta) + Z_OMEGA(0, 0, 1, 0)


def find_ellipse_bounding_box(ellipse: Ellipse) -> Tuple[D, D, D, D]:
    """
    Given the matrix elements of an ellipse defined as a 2x2 positive definite matrix:\n
    | a  b |\n
    | b  d |\n
    return its bounding box in the form [[x0, x1], [y0, y1]]
    """
    x = ((-4 * ellipse.d) / ((2 * ellipse.b) ** 2 - 4 * ellipse.a * ellipse.d)).sqrt()
    y = ((-4 * ellipse.a) / ((2 * ellipse.b) ** 2 - 4 * ellipse.a * ellipse.d)).sqrt()
    return (-x, x, -y, y)


def solve_2_dim_grid_problem_ellipse(
    ellipse1: Ellipse, ellipse2: Ellipse
) -> Iterable[Z_OMEGA]:
    """Given the matrix elements of two ellipses A and B centered at (x1, y1) and
    (x2, y2) defined as a 2x2 positive definite matrices:]\n
    | a  b |\n
    | b  d |\n
    return the list of elements u ∈ Z[ω] such that u ∈ A and u.conj2 ∈ B (where conj2 is sqrt(2) conjugation)
    """
    boxA = find_ellipse_bounding_box(ellipse1)
    boxA = (
        boxA[0] + ellipse1.x,
        boxA[1] + ellipse1.x,
        boxA[2] + ellipse1.y,
        boxA[3] + ellipse1.y,
    )
    boxB = find_ellipse_bounding_box(ellipse2)
    boxB = (
        boxB[0] + ellipse2.x,
        boxB[1] + ellipse2.x,
        boxB[2] + ellipse2.y,
        boxB[3] + ellipse2.y,
    )
    params = boxA + boxB
    potential_solutions = solve_two_dim_grid_problem_upright_rectangles(
        *params, ellipse1, ellipse2
    )
    for solution in potential_solutions:
        x1 = solution.to_complexAP().real
        y1 = solution.to_complexAP().imaginary
        x2 = solution.conj2().to_complexAP().real
        y2 = solution.conj2().to_complexAP().imaginary
        if ellipse1.contains(x1, y1) and ellipse2.contains(x2, y2):
            yield solution


def determine_shift_operator(
    ellipse1: Ellipse, ellipse2: Ellipse
) -> Tuple[int, Ellipse, Ellipse]:
    lmbda = 1 + D(2).sqrt()
    bias = calculate_bias(ellipse1, ellipse2)
    k = math.floor((1 - bias) / 2)
    lmbda_pow_k = lmbda**k
    lmbda_pow_neg_k = lmbda**-k
    ellipse1.a *= lmbda_pow_k
    ellipse1.d *= lmbda_pow_neg_k
    ellipse1.z -= k
    ellipse2.a *= lmbda_pow_neg_k
    ellipse2.b *= D((-1) ** k)
    ellipse2.d *= lmbda_pow_k
    ellipse2.z += k
    return k


def apply_shift_operator(k: int, grid_op: GridOperator) -> GridOperator:
    if k > 0:
        for _ in range(k):
            grid_op = GridOperator(
                grid_op.a + grid_op.ap,
                2 * grid_op.a + grid_op.ap,
                grid_op.b,
                grid_op.bp,
                grid_op.c,
                grid_op.cp,
                grid_op.dp - grid_op.d,
                2 * grid_op.d - grid_op.dp,
            )
    if k < 0:
        for _ in range(-k):
            grid_op = GridOperator(
                grid_op.ap - grid_op.a,
                2 * grid_op.a - grid_op.ap,
                grid_op.b,
                grid_op.bp,
                grid_op.c,
                grid_op.cp,
                grid_op.dp + grid_op.d,
                2 * grid_op.d + grid_op.dp,
            )
    return grid_op


def reduce_skew(
    ellipse1: Ellipse, ellipse2: Ellipse
) -> Tuple[GridOperator, Ellipse, Ellipse]:
    ellipse1._calc_z_and_e()
    ellipse2._calc_z_and_e()
    if not ellipse1.is_positive_semi_definite():
        raise ValueError("ellipse1 is not positive semi-definite")
    if not ellipse2.is_positive_semi_definite():
        raise ValueError("ellipse2 is not positive semi-definite")
    lmbda = 1 + D(2).sqrt()
    grid_op = GridOperator.I()
    initial_bias = calculate_bias(ellipse1, ellipse2)
    k = 0
    new_ellipse1 = deepcopy(ellipse1)
    new_ellipse2 = deepcopy(ellipse2)
    if abs(initial_bias) > 1:
        k = determine_shift_operator(new_ellipse1, new_ellipse2)
    if new_ellipse2.b < 0:
        grid_op = grid_op * GridOperator.Z()
    new_ellipse1._calc_z_and_e()
    new_ellipse2._calc_z_and_e()
    if (new_ellipse1.z + new_ellipse2.z) < 0:
        grid_op = grid_op * GridOperator.X()
    new_ellipse1, new_ellipse2 = apply_grid_operator(
        grid_op, new_ellipse1, new_ellipse2
    )
    new_ellipse1._calc_z_and_e()
    new_ellipse2._calc_z_and_e()
    # CASES:
    if new_ellipse1.b >= 0:
        if (
            new_ellipse1.z >= D("-0.8")
            and new_ellipse1.z <= D("0.8")
            and new_ellipse2.z >= D("-0.8")
            and new_ellipse2.z <= D("0.8")
        ):
            grid_op = grid_op * GridOperator.R()
        elif new_ellipse1.z <= D("0.3") and new_ellipse2.z >= D("0.8"):
            grid_op = grid_op * GridOperator.K()
        elif new_ellipse1.z >= D("0.3") and new_ellipse2.z >= D("0.3"):
            c = min(new_ellipse1.z, new_ellipse2.z)
            n = max(1, math.floor((lmbda**c) / 2))
            grid_op = grid_op * GridOperator.APowN(n)
        elif new_ellipse1.z >= D("0.8") and new_ellipse2.z <= D("0.3"):
            grid_op = grid_op * GridOperator.K().conj2()
        else:
            raise ValueError(
                f"Ellipse pair did not match any case. First ellipse = {new_ellipse1}. Second ellipse = {new_ellipse2}"
            )
    else:
        if (new_ellipse1.z >= D("-0.8") and new_ellipse1.z <= D("0.8")) and (
            new_ellipse2.z >= D("-0.8") and new_ellipse2.z <= D("0.8")
        ):
            grid_op = grid_op * GridOperator.R()
        elif new_ellipse1.z >= D("-0.2") and new_ellipse2.z >= D("-0.2"):
            c = min(new_ellipse1.z, new_ellipse2.z)
            n = max(1, math.floor((lmbda**c) / 2))
            grid_op = grid_op * GridOperator.BPowN(n)
        else:
            raise ValueError(
                f"Ellipse pair did not match any case. First ellipse = {new_ellipse1}. Second ellipse = {new_ellipse2}"
            )

    if k != 0:
        grid_op = apply_shift_operator(k, grid_op)
    final_ellipse1, final_ellipse2 = apply_grid_operator(grid_op, ellipse1, ellipse2)
    return grid_op, final_ellipse1, final_ellipse2


def apply_op_to_ellipse(grid_op: GridOperator, ellipse: Ellipse) -> Ellipse:
    """
    Given a grid op G and ellipse E, performs the matrix multiplication
    (G^T)EG
    """
    SQRT_2 = D(2).sqrt()
    alpha = D(grid_op.a) + D(grid_op.ap) / SQRT_2
    beta = D(grid_op.b) + D(grid_op.bp) / SQRT_2
    gamma = D(grid_op.c) + D(grid_op.cp) / SQRT_2
    delta = D(grid_op.d) + D(grid_op.dp) / SQRT_2
    u1 = alpha * ellipse.a + gamma * ellipse.b
    u2 = alpha * ellipse.b + gamma * ellipse.d
    b1 = beta * ellipse.a + delta * ellipse.b
    b2 = beta * ellipse.b + delta * ellipse.d
    new_a = u1 * alpha + u2 * gamma
    new_b = u1 * beta + u2 * delta
    new_d = b1 * beta + b2 * delta
    inverse = grid_op.inverse()
    new_x = (D(inverse.a) + D(inverse.ap) / SQRT_2) * ellipse.x + (
        D(inverse.b) + D(inverse.bp) / SQRT_2
    ) * ellipse.y
    new_y = (D(inverse.c) + D(inverse.cp) / SQRT_2) * ellipse.x + (
        D(inverse.d) + D(inverse.dp) / SQRT_2
    ) * ellipse.y

    return Ellipse(new_a, new_b, new_d, new_x, new_y)


def apply_grid_operator(grid_op: GridOperator, ellipse1: Ellipse, ellipse2: Ellipse):
    new_ellipse1 = apply_op_to_ellipse(grid_op, ellipse1)
    new_ellipse2 = apply_op_to_ellipse(grid_op.conj2(), ellipse2)
    if grid_op != GridOperator.I():
        assert ellipse1 != new_ellipse1
    return new_ellipse1, new_ellipse2


def find_grid_operator(ellipse1: Ellipse, ellipse2: Ellipse) -> GridOperator:
    grid_op = GridOperator.I()
    old_skew = calculate_skew(ellipse1, ellipse2)
    i = 0
    while calculate_skew(ellipse1, ellipse2) > 15:
        i += 1
        new_grid_op, new_ellipse1, new_ellipse2 = reduce_skew(ellipse1, ellipse2)
        new_skew = calculate_skew(new_ellipse1, new_ellipse2)
        grid_op = grid_op * new_grid_op
        if new_skew > D("0.9") * old_skew:
            sys.exit(
                "Skew was not decreased, ellipse1 = "
                + str(ellipse1)
                + " ellips2 = "
                + str(ellipse2)
            )
        old_skew = new_skew
        ellipse1 = new_ellipse1
        ellipse2 = new_ellipse2
    return grid_op


def get_uprightness(ellipse: Ellipse) -> D:
    ellipse_area = ellipse.area()
    bbox = find_ellipse_bounding_box(ellipse)
    box_area = (bbox[1] - bbox[0]) * (bbox[3] - bbox[2])
    return ellipse_area / box_area


# Methods specifically for direct approximation


def find_bounding_ellipse_direct(epsilon: D, phi: D, k: int = 0) -> Ellipse:
    """
    Given an epsilon and a phi, find x0, y0, a, b, and d such that the ellipse defined by
    a(x-x0)^2 + 2b(x-x0)(y-y0) + d(y-y0)^2 = (ab)^2
    is the smallest ellipse that bounds the region
    {u | u•z >= 1 - ε^2 / 2}
    where z = exp(iφ)
    """
    sqrt2 = D(2).sqrt()
    c = 1 - epsilon**2 / 2
    x = c * prec_cos(phi)
    y = c * prec_sin(phi)
    semi_major = epsilon**2 / 2
    if k % 2 == 0:
        scale = 2 ** (k // 2)
    else:
        scale = 2 ** (k // 2) * sqrt2
    shift = semi_major / 3
    x += prec_cos(phi) * shift
    y += prec_sin(phi) * shift
    semi_major = semi_major * D(2) / D(3)
    semi_minor = (epsilon**2 - epsilon**4 / 4).sqrt()
    semi_minor = semi_minor * D(2) / D(3).sqrt()
    semi_major = semi_major * scale
    semi_minor = semi_minor * scale
    x = x * scale
    y = y * scale
    return Ellipse.from_axes(x, y, phi, semi_major, semi_minor)


def candidate_generator_direct(epsilon: D, phi: D) -> Iterable[Tuple[Z_OMEGA, int]]:
    k_min = 3 * prec_log2(1 / epsilon) // 2
    k = int(k_min)
    found_solution = False
    max_iters = 300
    i = 0
    z = ComplexAP(prec_cos(phi), prec_sin(phi))
    grid_op = GridOperator.I()
    while not found_solution and i < max_iters:
        if k % 2 == 0:
            radius = D(2 ** (k // 2))
        else:
            radius = 2 ** (k // 2) * D(2).sqrt()
        ellipse1 = find_bounding_ellipse_direct(epsilon, phi, k)
        ellipse2 = Ellipse.from_axes(0, 0, 0, D(1), D(1))
        # find the grid operator (only have to do this once)
        if k == k_min:
            # need to have both ellipses have det 1 in order to find the grid operator. Will
            # scale the ellipse back afterwards
            scale, ellipse1 = force_det_one(ellipse1)
            grid_op = find_grid_operator(ellipse1, ellipse2)
            # scale ellipse back
            ellipse1 = scale_ellipse(ellipse1, scale)
        ellipse2 = Ellipse.from_axes(0, 0, 0, radius, radius)
        ellipse1, ellipse2 = apply_grid_operator(grid_op, ellipse1, ellipse2)

        potential_solutions = solve_2_dim_grid_problem_ellipse(ellipse1, ellipse2)
        # now we have to rotate the solutions back and check if they work
        for sol in potential_solutions:
            scaled_sol = grid_op.multiply_z_omega(sol)
            tmp_k = k

            while is_reducible(scaled_sol):
                scaled_sol = reduce(scaled_sol)
                tmp_k -= 1

            sol_complex = scaled_sol.to_complexAP()
            magnitude_squared = scaled_sol.magnitude_squared().to_zsqrt().to_decimal()
            dot_prod = z.real * sol_complex.real + z.imaginary * sol_complex.imaginary
            dot_prod = dot_prod / radius
            if abs(magnitude_squared) <= 2**k and dot_prod >= (1 - epsilon**2 / 2):
                yield scaled_sol, tmp_k

        k += 1
        i += 1


def find_u_candidates_direct(epsilon: D, phi: D) -> List[Tuple[Z_OMEGA, int]]:
    """Find all of the u candidates for the lowest k at which a candidate exists"""
    initial_k = -1
    solutions = []
    for i, sol in enumerate(candidate_generator_direct(epsilon, phi)):
        if i == 0:
            initial_k = sol[1]
        if sol[1] == initial_k:
            solutions.append(sol)
        else:
            break
    return solutions


# Methods for fallback approximation


def find_bounding_ellipse_fallback(epsilon: D, phi: D, r: D, k: int) -> Ellipse:
    """
    Given an epsilon, phi, r, and k, find x0, y0, a, b, and d such that the ellipse defined by
    a(x-x0)^2 + 2b(x-x0)(y-y0) + d(y-y0)^2 = 1
    return the (approximately) smallest ellipse that bounds the region

    C = {u | |u| >= r and Arg(u)∈[θ - δ, θ + δ]}

    for δ = arcsin(eps/2)
    """
    delta = prec_arcsin(epsilon / 2)
    sqrt2 = D(2).sqrt()
    if k % 2 == 0:
        scale = 2 ** (k // 2)
    else:
        scale = 2 ** (k // 2) * sqrt2

    x_c = (1 + r) / 2 * prec_cos(phi) * scale
    y_c = (1 + r) / 2 * prec_sin(phi) * scale

    semi_major = ((1 - r * prec_cos(delta)) * sqrt2 / 2) * scale
    semi_minor = (prec_sin(delta) * sqrt2) * scale

    return Ellipse.from_axes(x_c, y_c, phi, semi_major, semi_minor)


def candidate_generator_fallback(
    epsilon: D, phi: D, r: D
) -> Iterable[Tuple[Z_OMEGA, int]]:
    # k_min = 3 * prec_log2(1 / epsilon) // 2
    k_min = 0
    k = int(k_min)
    found_solution = False
    max_iters = 300
    i = 0
    grid_op = GridOperator.I()
    delta = prec_arcsin(epsilon / 2)
    while not found_solution and i < max_iters:
        if k % 2 == 0:
            radius = D(2 ** (k // 2))
        else:
            radius = 2 ** (k // 2) * D(2).sqrt()
        ellipse1 = find_bounding_ellipse_fallback(epsilon, phi, r, k)
        ellipse2 = Ellipse.from_axes(0, 0, 0, D(1), D(1))
        # find the grid operator (only have to do this once)
        if k == k_min:
            # need to have both ellipses have det 1 in order to find the grid operator. Will
            # scale the ellipse back afterwards
            scale, ellipse1 = force_det_one(ellipse1)
            grid_op = find_grid_operator(ellipse1, ellipse2)
            # scale ellipse back
            ellipse1 = scale_ellipse(ellipse1, scale)
        ellipse2 = Ellipse.from_axes(0, 0, 0, radius, radius)
        ellipse1, ellipse2 = apply_grid_operator(grid_op, ellipse1, ellipse2)

        potential_solutions = solve_2_dim_grid_problem_ellipse(ellipse1, ellipse2)
        # now we have to rotate the solutions back and check if they work
        for sol in potential_solutions:
            sol = grid_op.multiply_z_omega(sol)
            is_reducible = False
            if (sol.a1 + sol.a3) % 2 == 0 and (sol.a2 + sol.a4) % 2 == 0:
                is_reducible = True
            if not is_reducible:
                sol_complex = sol.to_complexAP()
                arg_u = arg(sol_complex)
                magnitude_squared = sol_complex.conj() * sol_complex

                if (
                    abs(magnitude_squared) <= 2**k
                    and abs(magnitude_squared) >= pow(2, k) * r**2
                    and abs(arg_u - phi) <= delta
                ):
                    yield sol, k

        k += 1
        i += 1


def find_u_candidates_fallback(epsilon: D, phi: D, r: D) -> List[Tuple[Z_OMEGA, int]]:
    """Find all of the u candidates for the lowest k at which a candidate exists"""
    initial_k = -1
    solutions = []
    for i, sol in enumerate(candidate_generator_fallback(epsilon, phi, r)):
        if i == 0:
            initial_k = sol[1]
        if sol[1] == initial_k:
            solutions.append(sol)
        else:
            break
    return solutions
