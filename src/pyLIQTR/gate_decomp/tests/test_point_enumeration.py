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

import unittest
from decimal import Decimal

import numpy as np

from pyLIQTR.gate_decomp.ellipse import Ellipse, calculate_bias, calculate_skew
from pyLIQTR.gate_decomp.point_enumeration import *
from pyLIQTR.gate_decomp.rings import Z_SQRT2

D = Decimal


class TestApproximationFunctions(unittest.TestCase):
    def test_scaled_one_dim_grid_problem_1(self):
        x0 = D("0")
        x1 = D("0.7")
        y0 = D(10)
        y1 = D(20)
        solutions = [sol for sol in scaled_one_dim_grid_problem(x0, x1, y0, y1)]
        self.assertGreater(len(solutions), 0)
        for solution in solutions:
            self.assertLessEqual(solution.to_decimal(), x1)
            self.assertGreaterEqual(solution.to_decimal(), x0)
            self.assertLessEqual(solution.conj().to_decimal(), y1)
            self.assertGreaterEqual(solution.conj().to_decimal(), y0)

    def test_scaled_one_dim_grid_problem_2(self):
        x0 = D("0")
        x1 = D("0.99999999999999999999999")
        y0 = D(8)
        l = 1 + D(2).sqrt()
        y1 = y0 + (l**2) / x1
        solutions = [sol for sol in scaled_one_dim_grid_problem(x0, x1, y0, y1)]
        self.assertGreater(len(solutions), 0)
        for solution in solutions:
            self.assertLessEqual(solution.to_decimal(), x1)
            self.assertGreaterEqual(solution.to_decimal(), x0)
            self.assertLessEqual(solution.conj().to_decimal(), y1)
            self.assertGreaterEqual(solution.conj().to_decimal(), y0)

    def test_scaled_one_dim_grid_problem_3(self):
        x0 = D("63887.49858757059024747894997")
        x1 = D("63888.37471944214825992600264")
        y0 = D("-3.031041453191973743961856950")
        y1 = D("3.621412719467655328944981621")
        solutions = [sol for sol in scaled_one_dim_grid_problem(x0, x1, y0, y1)]
        self.assertGreater(len(solutions), 0)
        for solution in solutions:
            self.assertLessEqual(solution.to_decimal(), x1)
            self.assertGreaterEqual(solution.to_decimal(), x0)
            self.assertLessEqual(solution.conj().to_decimal(), y1)
            self.assertGreaterEqual(solution.conj().to_decimal(), y0)

    def test_find_k_1(self):
        delta = Z_SQRT2(1, 1).to_decimal()
        linv = 1 / delta
        k = find_k(delta)
        scaled_delta = (linv**k) * delta
        self.assertGreaterEqual(scaled_delta, linv)
        self.assertLess(scaled_delta, 1)

    def test_find_k_2(self):
        lmbda = 1 + D(2).sqrt()
        delta = Z_SQRT2(1, 1).to_decimal() ** 2
        k = find_k(delta)
        scaled_delta = (lmbda ** (-k)) * delta
        self.assertGreaterEqual(scaled_delta, lmbda ** (-1))
        self.assertLess(scaled_delta, 1)

    def test_find_k_3(self):
        lmbda = 1 + D(2).sqrt()
        delta = Z_SQRT2(1, 1).to_decimal() ** 10
        k = find_k(delta)
        scaled_delta = (lmbda ** (-k)) * delta
        self.assertGreaterEqual(scaled_delta, lmbda ** (-1))
        self.assertLess(scaled_delta, 1)

    def test_find_k_4(self):
        lmbda = 1 + D(2).sqrt()
        delta = D("6725.999851323220026050415779")
        k = find_k(delta)
        scaled_delta = (lmbda ** (-k)) * delta
        self.assertGreaterEqual(scaled_delta, lmbda ** (-1))
        self.assertLess(scaled_delta, 1)

    def test_find_k_5(self):
        lmbda = 1 + D(2).sqrt()
        delta = D("1.4")
        k = find_k(delta)
        scaled_delta = (lmbda ** (-k)) * delta
        self.assertGreaterEqual(scaled_delta, lmbda ** (-1))
        self.assertLess(scaled_delta, 1)

    def test_solve_one_dim_grid_problem_1(self):
        x0 = D("9.1")
        x1 = D("9.2")
        y0 = D(-20)
        y1 = D(-19)
        solutions = solve_one_dim_grid_problem(x0, x1, y0, y1)
        self.assertIn(Z_SQRT2(-5, 10), solutions)
        for sol in solutions:
            self.assertGreaterEqual(sol.to_decimal(), x0)
            self.assertLessEqual(sol.to_decimal(), x1)
            self.assertGreaterEqual(sol.conj().to_decimal(), y0)
            self.assertLessEqual(sol.conj().to_decimal(), y1)

    def test_solve_one_dim_grid_problem_2(self):
        y0 = D("9.1")
        y1 = D("9.2")
        x0 = D(-20)
        x1 = D(-19)
        solutions = solve_one_dim_grid_problem(x0, x1, y0, y1)
        self.assertIn(Z_SQRT2(-5, -10), solutions)
        for sol in solutions:
            self.assertGreaterEqual(sol.to_decimal(), x0)
            self.assertLessEqual(sol.to_decimal(), x1)
            self.assertGreaterEqual(sol.conj().to_decimal(), y0)
            self.assertLessEqual(sol.conj().to_decimal(), y1)

    def test_solve_one_dim_grid_problem_3(self):
        x0 = D("12634400")
        x1 = D("12634500")
        y0 = D("-12754089.17")
        y1 = D("-12754089.16")
        solutions = solve_one_dim_grid_problem(x0, x1, y0, y1)
        self.assertIn(Z_SQRT2(-59807, 8976213), solutions)
        for sol in solutions:
            self.assertGreaterEqual(sol.to_decimal(), x0)
            self.assertLessEqual(sol.to_decimal(), x1)
            self.assertGreaterEqual(sol.conj().to_decimal(), y0)
            self.assertLessEqual(sol.conj().to_decimal(), y1)

    def test_solve_one_dim_grid_problem_4(self):
        y0 = D("12634400")
        y1 = D("12634500")
        x0 = D("-12754089.17")
        x1 = D("-12754089.16")
        solutions = solve_one_dim_grid_problem(x0, x1, y0, y1)
        self.assertIn(Z_SQRT2(-59807, -8976213), solutions)
        for sol in solutions:
            self.assertGreaterEqual(sol.to_decimal(), x0)
            self.assertLessEqual(sol.to_decimal(), x1)
            self.assertGreaterEqual(sol.conj().to_decimal(), y0)
            self.assertLessEqual(sol.conj().to_decimal(), y1)

    def test_solve_one_dim_grid_problem_5(self):
        x0 = D(-297)
        x1 = D("248.5823575862261")
        y0 = D(779)
        y1 = D("779.0106829464767380079485168")
        solutions = solve_one_dim_grid_problem(x0, x1, y0, y1)
        for sol in solutions:
            self.assertGreaterEqual(sol.to_decimal(), x0)
            self.assertLessEqual(sol.to_decimal(), x1)
            self.assertGreaterEqual(sol.conj().to_decimal(), y0)
            self.assertLessEqual(sol.conj().to_decimal(), y1)

    def test_solve_one_dim_grid_problem_6(self):
        x0 = D("314248.7172986995797533837621")
        x1 = D("314248.7172986995797852939267")
        y0 = D("-0.3492727719167084114226836111")
        y1 = D("0.3492727719167084114226836111")
        solutions = solve_one_dim_grid_problem(x0, x1, y0, y1)
        for sol in solutions:
            self.assertGreaterEqual(sol.to_decimal(), x0)
            self.assertLessEqual(sol.to_decimal(), x1)
            self.assertGreaterEqual(sol.conj().to_decimal(), y0)
            self.assertLessEqual(sol.conj().to_decimal(), y1)

    def test_one_dim_grid_problem_rand(self):
        for _ in range(10):
            delta = np.random.rand() * 1000
            Delta = ((1 + D("2").sqrt()) ** 2) / D(delta)
            x0 = np.random.randint(-1000, 1000)
            y0 = np.random.randint(-1000, 1000)
            x1 = x0 + delta
            y1 = y0 + Delta
            solutions = [
                sol for sol in solve_one_dim_grid_problem(D(x0), D(x1), D(y0), D(y1))
            ]
            failure_string = f"Falied for delta = {delta}, x0 = {x0}, and y0 = {y0}"
            self.assertGreater(len(solutions), 0)
            for solution in solutions:
                self.assertLessEqual(solution.to_decimal(), x1, failure_string)
                self.assertGreaterEqual(solution.to_decimal(), x0, failure_string)
                self.assertLessEqual(solution.conj().to_decimal(), y1, failure_string)
                self.assertGreaterEqual(
                    solution.conj().to_decimal(), y0, failure_string
                )

    def test_two_dim_upright_rectangle1(self):
        correct_ans = Z_OMEGA(1, 2, 3, 4)
        solutions = [
            sol
            for sol in solve_two_dim_grid_problem_upright_rectangles(
                D(5), D(6), D(4), D(5), D(2), D(3), D(-1), D(0)
            )
        ]
        self.assertGreater(len(solutions), 0)
        self.assertIn(correct_ans, solutions)

    def test_two_dim_upright_rectangle2(self):
        correct_ans = Z_OMEGA(-2, 3, 1, -6)
        solutions = [
            sol
            for sol in solve_two_dim_grid_problem_upright_rectangles(
                D(-4), D("-3.8"), D("2.2"), D("2.4"), D(-9), D(-8), D("2.5"), D("4.2")
            )
        ]
        self.assertGreater(len(solutions), 0)
        self.assertIn(correct_ans, solutions)

    def test_solve_two_dim_grid_problem_ellipse(self):
        a1 = D(3)
        b1 = D(0.5)
        d1 = D(1)
        a2 = D(3)
        b2 = D(0.3)
        d2 = D(0.3)
        x1 = D(-1.7)
        y1 = D(13.95)
        x2 = D(-12.3)
        y2 = D(-7.9)
        ellipse1 = Ellipse(a1, b1, d1, x1, y1)
        ellipse2 = Ellipse(a2, b2, d2, x2, y2)
        true_sol1 = Z_OMEGA(4, 3, 12, -7)
        true_sol2 = Z_OMEGA(4, 3, 11, -7)
        solutions = [
            sol for sol in solve_2_dim_grid_problem_ellipse(ellipse1, ellipse2)
        ]
        self.assertGreater(len(solutions), 0)
        self.assertIn(true_sol1, solutions)
        self.assertIn(true_sol2, solutions)

    def test_find_bounding_ellipse(self):
        phi = np.random.rand() * 2 * np.pi
        exponent = np.random.randint(1, 15)
        exponent = 15
        epsilon = 10**-exponent
        epsilon = D(f"1e-{exponent}")
        getcontext().prec = 33
        ellipse = find_bounding_ellipse_direct(epsilon, D(phi))
        c = (1 - epsilon**2 / 2 + 1) / 2
        x1 = D(c) * prec_cos(D(phi))
        y1 = D(c) * prec_sin(D(phi))
        z = (
            ellipse.a * (x1 - ellipse.x) ** 2
            + 2 * ellipse.b * (x1 - ellipse.x) * (y1 - ellipse.y)
            + ellipse.d * (y1 - ellipse.y) ** 2
        )
        self.assertLessEqual(z, 1)

    def test_find_e_and_z(self):
        a = np.random.random()
        b = np.random.random()
        d = (1 - b**2) / a
        a = D(a)
        b = D(b)
        d = D(d)
        ellipse = Ellipse(a, b, d)
        ellipse._calc_z_and_e()
        lmbda = D(1) + D(2).sqrt()
        self.assertAlmostEqual(ellipse.e * lmbda ** (-ellipse.z), a)
        self.assertAlmostEqual(ellipse.e * lmbda ** (ellipse.z), d)

    def test_grid_op_mult(self):
        first = GridOperator(0, 1, 0, -1, 0, 1, 0, 1)
        second = GridOperator(1, 0, 0, 2, 0, 0, 1, 0)
        result = first * second
        correct_result = GridOperator(0, 1, 1, -1, 0, 1, 1, 1)
        self.assertEqual(result, correct_result)

    def test_inverse_grid_op(self):
        I = GridOperator(1, 0, 0, 0, 0, 0, 1, 0)
        self.assertEqual(I, GridOperator.R() * GridOperator.R().inverse())
        self.assertEqual(I, GridOperator.A() * GridOperator.A().inverse())
        self.assertEqual(I, GridOperator.B() * GridOperator.B().inverse())
        self.assertEqual(I, GridOperator.K() * GridOperator.K().inverse())
        self.assertEqual(I, GridOperator.X() * GridOperator.X().inverse())
        self.assertEqual(I, GridOperator.Z() * GridOperator.Z().inverse())

    def test_R_op(self):
        lmbda = 1 + D(2).sqrt()
        z1 = D(0.75)
        z2 = D(0.7)
        b1 = D(-3)
        b2 = D(3.5)
        e1 = (b1**2 + 1).sqrt()
        e2 = (b2**2 + 1).sqrt()
        E1 = Ellipse(e1 * lmbda ** (-z1), b1, e1 * lmbda**z1)
        E2 = Ellipse(e2 * lmbda ** (-z2), b2, e2 * lmbda**z2)
        initial_skew = calculate_skew(E1, E2)
        self.assertGreaterEqual(initial_skew, 15)
        E1, E2 = apply_grid_operator(GridOperator.R(), E1, E2)
        final_skew = calculate_skew(E1, E2)
        self.assertLessEqual(final_skew, D(0.9) * initial_skew)

    def test_K_op(self):
        lmbda = 1 + D(2).sqrt()
        z1 = D(0.25)
        z2 = D(-0.7)
        b1 = D(3)
        b2 = D(3.5)
        e1 = (b1**2 + 1).sqrt()
        e2 = (b2**2 + 1).sqrt()
        E1 = Ellipse(e1 * lmbda ** (-z1), b1, e1 * lmbda**z1)
        E2 = Ellipse(e2 * lmbda ** (-z2), b2, e2 * lmbda**z2)
        initial_skew = calculate_skew(E1, E2)
        self.assertGreaterEqual(initial_skew, 15)
        E1, E2 = apply_grid_operator(GridOperator.K(), E1, E2)
        final_skew = calculate_skew(E1, E2)
        self.assertLessEqual(final_skew, D(0.9) * initial_skew)

    def test_A_op(self):
        lmbda = 1 + D(2).sqrt()
        z1 = D(0.4)
        z2 = D(0.5)
        c = min(z1, z2)
        n = max(1, math.floor((lmbda**c) / 2))
        b1 = D(3)
        b2 = D(3.5)
        e1 = (b1**2 + 1).sqrt()
        e2 = (b2**2 + 1).sqrt()
        E1 = Ellipse(e1 * lmbda ** (-z1), b1, e1 * lmbda**z1)
        E2 = Ellipse(e2 * lmbda ** (-z2), b2, e2 * lmbda**z2)
        initial_skew = calculate_skew(E1, E2)
        self.assertGreaterEqual(initial_skew, 15)
        for _ in range(n):
            E1, E2 = apply_grid_operator(GridOperator.A(), E1, E2)
        final_skew = calculate_skew(E1, E2)
        self.assertLessEqual(final_skew, D(0.9) * initial_skew)

    def test_B_op(self):
        lmbda = 1 + D(2).sqrt()
        z1 = D(-0.1)
        z2 = D(0.2)
        c = min(z1, z2)
        n = max(1, math.floor((lmbda**c) / 2))
        b1 = D(-3)
        b2 = D(3.5)
        e1 = (b1**2 + 1).sqrt()
        e2 = (b2**2 + 1).sqrt()
        E1 = Ellipse(e1 * lmbda ** (-z1), b1, e1 * lmbda**z1)
        E2 = Ellipse(e2 * lmbda ** (-z2), b2, e2 * lmbda**z2)
        initial_skew = calculate_skew(E1, E2)
        self.assertGreaterEqual(initial_skew, 15)
        for _ in range(n):
            E1, E2 = apply_grid_operator(GridOperator.B(), E1, E2)
        final_skew = calculate_skew(E1, E2)
        self.assertLessEqual(final_skew, D(0.9) * initial_skew)

    def test_shift_operator(self):
        lmbda = 1 + D(2).sqrt()
        z1 = D(1.3)
        z2 = D(-0.9)
        b1 = 0
        b2 = 0
        while (b1**2 + b2**2) < 15:
            b1 = D(np.random.rand() - 0.5) * 10
            b2 = D(np.random.rand() - 0.5) * 10
        e1 = D(b1**2 + 1).sqrt()
        e2 = D(b2**2 + 1).sqrt()
        E1 = Ellipse(e1 * lmbda ** (-z1), b1, e1 * lmbda**z1)
        E2 = Ellipse(e2 * lmbda ** (-z2), b2, e2 * lmbda**z2)
        initial_bias = calculate_bias(E1, E2)
        self.assertGreater(abs(initial_bias), 1)
        k = determine_shift_operator(E1, E2)
        final_bias = calculate_bias(E1, E2)
        self.assertLessEqual(abs(final_bias), 1)

    def test_reduce_skew_large_bias1(self):
        lmbda = 1 + D(2).sqrt()
        z1 = D(1.3)
        z2 = D(-0.9)
        b1 = D("4.8")
        b2 = D("2.4")
        e1 = D(b1**2 + 1).sqrt()
        e2 = D(b2**2 + 1).sqrt()
        E1 = Ellipse(e1 * lmbda ** (-z1), b1, e1 * lmbda**z1)
        E2 = Ellipse(e2 * lmbda ** (-z2), b2, e2 * lmbda**z2)
        initial_skew = calculate_skew(E1, E2)
        self.assertGreaterEqual(initial_skew, 15)
        grid_op, _, _ = reduce_skew(E1, E2)
        E1, E2 = apply_grid_operator(grid_op, E1, E2)
        final_skew = calculate_skew(E1, E2)
        self.assertLessEqual(
            final_skew, D(0.9) * initial_skew, f"Failed with b1 = {b1}, b2 = {b2}"
        )

    def test_reduce_skew_large_bias2(self):
        lmbda = 1 + D(2).sqrt()
        z1 = D(2.3)
        z2 = D(-0.9)
        b1 = D("4.8")
        b2 = D("2.4")
        e1 = D(b1**2 + 1).sqrt()
        e2 = D(b2**2 + 1).sqrt()
        E1 = Ellipse(e1 * lmbda ** (-z1), b1, e1 * lmbda**z1)
        E2 = Ellipse(e2 * lmbda ** (-z2), b2, e2 * lmbda**z2)
        initial_skew = calculate_skew(E1, E2)
        self.assertGreaterEqual(initial_skew, 15)
        grid_op, _, _ = reduce_skew(E1, E2)
        E1, E2 = apply_grid_operator(grid_op, E1, E2)
        final_skew = calculate_skew(E1, E2)
        self.assertLessEqual(
            final_skew, D(0.9) * initial_skew, f"Failed with b1 = {b1}, b2 = {b2}"
        )

    def test_reduce_skew_large_bias3(self):
        lmbda = 1 + D(2).sqrt()
        z1 = D(4.3)
        z2 = D(-2.9)
        b1 = D("4.8")
        b2 = D("2.4")
        e1 = D(b1**2 + 1).sqrt()
        e2 = D(b2**2 + 1).sqrt()
        E1 = Ellipse(e1 * lmbda ** (-z1), b1, e1 * lmbda**z1)
        E2 = Ellipse(e2 * lmbda ** (-z2), b2, e2 * lmbda**z2)
        initial_skew = calculate_skew(E1, E2)
        self.assertGreaterEqual(initial_skew, 15)
        grid_op, _, _ = reduce_skew(E1, E2)
        E1, E2 = apply_grid_operator(grid_op, E1, E2)
        final_skew = calculate_skew(E1, E2)
        self.assertLessEqual(
            final_skew, D(0.9) * initial_skew, f"Failed with b1 = {b1}, b2 = {b2}"
        )

    def test_reduce_skew_large_bias4(self):
        lmbda = 1 + D(2).sqrt()
        z1 = D("-4.3")
        z2 = D("2.9")
        b1 = D("4.8")
        b2 = D("2.4")
        e1 = D(b1**2 + 1).sqrt()
        e2 = D(b2**2 + 1).sqrt()
        E1 = Ellipse(e1 * lmbda ** (-z1), b1, e1 * lmbda**z1)
        E2 = Ellipse(e2 * lmbda ** (-z2), b2, e2 * lmbda**z2)
        initial_skew = calculate_skew(E1, E2)
        self.assertGreaterEqual(initial_skew, 15)
        grid_op, _, _ = reduce_skew(E1, E2)
        E1, E2 = apply_grid_operator(grid_op, E1, E2)
        final_skew = calculate_skew(E1, E2)
        self.assertLessEqual(
            final_skew, D(0.9) * initial_skew, f"Failed with b1 = {b1}, b2 = {b2}"
        )

    def test_reduce_skew_negative_beta(self):
        lmbda = 1 + D(2).sqrt()
        z1 = D(0.7)
        z2 = D(0.3)
        b1 = D(5)
        b2 = D(-4.1)
        e1 = (b1**2 + 1).sqrt()
        e2 = (b2**2 + 1).sqrt()
        E1 = Ellipse(e1 * lmbda ** (-z1), b1, e1 * lmbda**z1)
        E2 = Ellipse(e2 * lmbda ** (-z2), b2, e2 * lmbda**z2)
        initial_skew = calculate_skew(E1, E2)
        self.assertGreaterEqual(initial_skew, 15)
        grid_op, _, _ = reduce_skew(E1, E2)
        E1, E2 = apply_grid_operator(grid_op, E1, E2)
        final_skew = calculate_skew(E1, E2)
        self.assertLessEqual(final_skew, D(0.9) * initial_skew)

    def test_reduce_skew_negative_exps(self):
        lmbda = 1 + D(2).sqrt()
        z1 = D(-0.5)
        z2 = D(0.3)
        b1 = D(5)
        b2 = D(3)
        e1 = (b1**2 + 1).sqrt()
        e2 = (b2**2 + 1).sqrt()
        E1 = Ellipse(e1 * lmbda ** (-z1), b1, e1 * lmbda**z1)
        E2 = Ellipse(e2 * lmbda ** (-z2), b2, e2 * lmbda**z2)
        initial_skew = calculate_skew(E1, E2)
        self.assertGreaterEqual(initial_skew, 15)
        grid_op, _, _ = reduce_skew(E1, E2)
        E1, E2 = apply_grid_operator(grid_op, E1, E2)
        final_skew = calculate_skew(E1, E2)
        self.assertLessEqual(final_skew, D(0.9) * initial_skew)

    def test_reduce_skew_negative_exps2(self):
        lmbda = 1 + D(2).sqrt()
        z1 = D(-0.2)
        z2 = D(-1.1)
        b1 = D(5)
        b2 = D(3)
        e1 = (b1**2 + 1).sqrt()
        e2 = (b2**2 + 1).sqrt()
        E1 = Ellipse(e1 * lmbda ** (-z1), b1, e1 * lmbda**z1)
        E2 = Ellipse(e2 * lmbda ** (-z2), b2, e2 * lmbda**z2)
        initial_skew = calculate_skew(E1, E2)
        self.assertGreaterEqual(initial_skew, 15)
        grid_op, _, _ = reduce_skew(E1, E2)
        E1, E2 = apply_grid_operator(grid_op, E1, E2)
        final_skew = calculate_skew(E1, E2)
        self.assertLessEqual(final_skew, D(0.9) * initial_skew)

    def test_reduce_skew_rand(self):
        lmbda = 1 + D(2).sqrt()
        z1 = D(np.random.rand() - 0.5) * 10
        z2 = D(np.random.rand() - 0.5) * 10
        b1 = 0
        b2 = 0
        while (b1**2 + b2**2) < 15:
            b1 = D(np.random.rand() - 0.5) * 10
            b2 = D(np.random.rand() - 0.5) * 10
        e1 = (b1**2 + 1).sqrt()
        e2 = (b2**2 + 1).sqrt()
        E2 = Ellipse(e1 * lmbda ** (-z1), b1, e1 * lmbda**z1)
        E1 = Ellipse(e2 * lmbda ** (-z2), b2, e2 * lmbda**z2)
        ## CHECK AND SEE IF THE BIAS IS CHANGED BY THE SHIFT OPERATOR
        initial_skew = calculate_skew(E1, E2)
        self.assertGreaterEqual(initial_skew, 15)
        grid_op, _, _ = reduce_skew(E1, E2)
        E1, E2 = apply_grid_operator(grid_op, E1, E2)
        final_skew = calculate_skew(E1, E2)
        self.assertLessEqual(
            final_skew,
            D(0.9) * initial_skew,
            f"Random reduce skew test fail.\nz1 = {z1}\nz2 = {z2}\nb1 = {b1}\nb2 = {b2}\ne1 = {e1}\ne2 = {e2}",
        )

    def test_reduce_skew_case_1_1_1(self):
        lmbda = 1 + D(2).sqrt()
        b1 = D(4)
        z1 = D(-0.4)
        z2 = D(0.5)
        b2 = D(8)
        e1 = (b1**2 + 1).sqrt()
        e2 = (b2**2 + 1).sqrt()
        E1 = Ellipse(e1 * lmbda ** (-z1), b1, e1 * lmbda**z1)
        E2 = Ellipse(e2 * lmbda ** (-z2), b2, e2 * lmbda**z2)
        initial_skew = calculate_skew(E1, E2)
        self.assertGreaterEqual(initial_skew, 15)
        grid_op, _, _ = reduce_skew(E1, E2)
        E1, E2 = apply_grid_operator(grid_op, E1, E2)
        final_skew = calculate_skew(E1, E2)
        self.assertLessEqual(final_skew, D(0.9) * initial_skew)

    def test_reduce_skew_case_1_1_2(self):
        lmbda = 1 + D(2).sqrt()
        b1 = D(2)
        z1 = D(0.7)
        z2 = D(0.75)
        b2 = D(8)
        e1 = (b1**2 + 1).sqrt()
        e2 = (b2**2 + 1).sqrt()
        E1 = Ellipse(e1 * lmbda ** (-z1), b1, e1 * lmbda**z1)
        E2 = Ellipse(e2 * lmbda ** (-z2), b2, e2 * lmbda**z2)
        initial_skew = calculate_skew(E1, E2)
        self.assertGreaterEqual(initial_skew, 15)
        grid_op, _, _ = reduce_skew(E1, E2)
        E1, E2 = apply_grid_operator(grid_op, E1, E2)
        final_skew = calculate_skew(E1, E2)
        self.assertLessEqual(final_skew, D(0.9) * initial_skew)

    def test_reduce_skew_case_1_2(self):
        lmbda = 1 + D(2).sqrt()
        b1 = D(2)
        z1 = D(-0.05)
        z2 = D(0.9)
        b2 = D(8)
        e1 = (b1**2 + 1).sqrt()
        e2 = (b2**2 + 1).sqrt()
        E1 = Ellipse(e1 * lmbda ** (-z1), b1, e1 * lmbda**z1)
        E2 = Ellipse(e2 * lmbda ** (-z2), b2, e2 * lmbda**z2)
        initial_skew = calculate_skew(E1, E2)
        self.assertGreaterEqual(initial_skew, 15)
        grid_op, _, _ = reduce_skew(E1, E2)
        E1, E2 = apply_grid_operator(grid_op, E1, E2)
        final_skew = calculate_skew(E1, E2)
        self.assertLessEqual(final_skew, D(0.9) * initial_skew)

    def test_reduce_skew_case_1_3(self):
        lmbda = 1 + D(2).sqrt()
        b1 = D(2)
        z1 = D(0.4)
        z2 = D(1.2)
        b2 = D(8)
        e1 = (b1**2 + 1).sqrt()
        e2 = (b2**2 + 1).sqrt()
        E1 = Ellipse(e1 * lmbda ** (-z1), b1, e1 * lmbda**z1)
        E2 = Ellipse(e2 * lmbda ** (-z2), b2, e2 * lmbda**z2)
        initial_skew = calculate_skew(E1, E2)
        self.assertGreaterEqual(initial_skew, 15)
        grid_op, _, _ = reduce_skew(E1, E2)
        E1, E2 = apply_grid_operator(grid_op, E1, E2)
        final_skew = calculate_skew(E1, E2)
        self.assertLessEqual(final_skew, D(0.9) * initial_skew)

    def test_reduce_skew_case_1_4(self):
        lmbda = 1 + D(2).sqrt()
        b1 = D(2)
        z1 = D(0.9)
        z2 = D(-0.08)
        b2 = D(8)
        e1 = (b1**2 + 1).sqrt()
        e2 = (b2**2 + 1).sqrt()
        E1 = Ellipse(e1 * lmbda ** (-z1), b1, e1 * lmbda**z1)
        E2 = Ellipse(e2 * lmbda ** (-z2), b2, e2 * lmbda**z2)
        initial_skew = calculate_skew(E1, E2)
        self.assertGreaterEqual(initial_skew, 15)
        grid_op, _, _ = reduce_skew(E1, E2)
        E1, E2 = apply_grid_operator(grid_op, E1, E2)
        final_skew = calculate_skew(E1, E2)
        self.assertLessEqual(final_skew, D(0.9) * initial_skew)

    def test_reduce_skew_case_2_1(self):
        lmbda = 1 + D(2).sqrt()
        b1 = D(-2)
        z1 = D(0.7)
        z2 = D(-0.15)
        b2 = D(8)
        e1 = (b1**2 + 1).sqrt()
        e2 = (b2**2 + 1).sqrt()
        E1 = Ellipse(e1 * lmbda ** (-z1), b1, e1 * lmbda**z1)
        E2 = Ellipse(e2 * lmbda ** (-z2), b2, e2 * lmbda**z2)
        initial_skew = calculate_skew(E1, E2)
        self.assertGreaterEqual(initial_skew, 15)
        grid_op, _, _ = reduce_skew(E1, E2)
        E1, E2 = apply_grid_operator(grid_op, E1, E2)
        final_skew = calculate_skew(E1, E2)
        self.assertLessEqual(final_skew, D(0.9) * initial_skew)

    def test_reduce_skew_case_2_2(self):
        lmbda = 1 + D(2).sqrt()
        b1 = D(-2)
        z1 = D(1.1)
        z2 = D(0.2)
        b2 = D(8)
        e1 = (b1**2 + 1).sqrt()
        e2 = (b2**2 + 1).sqrt()
        E1 = Ellipse(e1 * lmbda ** (-z1), b1, e1 * lmbda**z1)
        E2 = Ellipse(e2 * lmbda ** (-z2), b2, e2 * lmbda**z2)
        initial_skew = calculate_skew(E1, E2)
        self.assertGreaterEqual(initial_skew, 15)
        grid_op, _, _ = reduce_skew(E1, E2)
        E1, E2 = apply_grid_operator(grid_op, E1, E2)
        final_skew = calculate_skew(E1, E2)
        self.assertLessEqual(final_skew, D(0.9) * initial_skew)

    def test_reduce_skew_extra_1(self):
        lmbda = 1 + D(2).sqrt()
        b1 = D("-27.347913755282596045130981677")
        z1 = D("0.199230493135169674872069292126")
        z2 = D("-0.167360607388364229438710764534")
        b2 = D("0.585786437626904951198311275784")
        e1 = (b1**2 + 1).sqrt()
        e2 = (b2**2 + 1).sqrt()
        E1 = Ellipse(e1 * lmbda ** (-z1), b1, e1 * lmbda**z1)
        E2 = Ellipse(e2 * lmbda ** (-z2), b2, e2 * lmbda**z2)
        initial_skew = calculate_skew(E1, E2)
        self.assertGreaterEqual(initial_skew, 15)
        grid_op, _, _ = reduce_skew(E1, E2)
        E1, E2 = apply_grid_operator(grid_op, E1, E2)
        final_skew = calculate_skew(E1, E2)
        self.assertLessEqual(final_skew, D(0.9) * initial_skew)

    def test_mult_grid_op_z_omega(self):
        u = Z_OMEGA(3, -4, 1, 9)
        G = GridOperator.A() * GridOperator.K()
        result = G.multiply_z_omega(u)
        u_arr = np.array(
            [float(u.to_complexAP().real), float(u.to_complexAP().imaginary)]
        )
        approx_result = G.to_numpy_arr() @ u_arr
        self.assertAlmostEqual(approx_result[0], float(result.to_complexAP().real))
        self.assertAlmostEqual(approx_result[1], float(result.to_complexAP().imaginary))

    def test_find_u_candidate_1(self):
        epsilon = D("0.2")
        phi = D(np.pi / 6)
        true_k = 3
        true_u = Z_OMEGA(0, 0, 2, 1)
        solutions = find_u_candidates_direct(epsilon, phi)
        found_sol = False
        for sol in solutions:
            if sol[0] == true_u and sol[1] == true_k:
                found_sol = True
        self.assertTrue(found_sol)

    def test_find_u_candidate_2(self):
        epsilon = D("0.00000001")
        phi = D(np.pi / 6)
        zx = prec_cos(phi)
        zy = prec_sin(phi)
        solutions = find_u_candidates_direct(epsilon, phi)
        for sol in solutions:
            k = sol[1]
            ux = sol[0].to_complexAP().real
            uy = sol[0].to_complexAP().imaginary
            u_conj = sol[0].conj2().to_complexAP()
            dot_product = D(2).sqrt() ** (-k) * (ux * zx + uy * zy)
            self.assertGreaterEqual(dot_product, 1 - (epsilon**2) / 2)
            self.assertLessEqual(u_conj.real**2 + u_conj.imaginary**2, 2**k)

    def test_find_u_candidate_3(self):
        getcontext().prec = 28
        epsilon = D("0.0000000001")
        phi = prec_pi() / D(9)
        zx = prec_cos(phi)
        zy = prec_sin(phi)
        solutions = find_u_candidates_direct(epsilon, phi)
        for sol in solutions:
            k = sol[1]
            ux = sol[0].to_complexAP().real
            uy = sol[0].to_complexAP().imaginary
            u_conj = sol[0].conj2().to_complexAP()
            dot_product = D(2).sqrt() ** (-k) * (ux * zx + uy * zy)
            self.assertGreaterEqual(dot_product, 1 - (epsilon**2) / 2)
            self.assertLessEqual(u_conj.real**2 + u_conj.imaginary**2, 2**k)

    def test_find_u_candidate_4(self):
        getcontext().prec = 28
        epsilon = D("0.001")
        phi = prec_pi() / D(128)
        zx = prec_cos(phi)
        zy = prec_sin(phi)
        solutions = find_u_candidates_direct(epsilon, phi)
        for sol in solutions:
            k = sol[1]
            ux = sol[0].to_complexAP().real
            uy = sol[0].to_complexAP().imaginary
            u_conj = sol[0].conj2().to_complexAP()
            dot_product = D(2).sqrt() ** (-k) * (ux * zx + uy * zy)
            self.assertGreaterEqual(dot_product, 1 - (epsilon**2) / 2)
            self.assertLessEqual(u_conj.real**2 + u_conj.imaginary**2, 2**k)

    def test_find_u_cand_fallback_1(self):
        getcontext().prec = 28
        phi = D("0.4")
        eps = D("0.1")
        r = D("0.9")
        true_u = Z_OMEGA(0, 0, 1, 1)
        true_k = 2
        solutions = find_u_candidates_fallback(eps, phi, r)
        found_sol = False
        self.assertEqual(len(solutions), 1)
        for sol in solutions:
            if sol[0] == true_u and sol[1] == true_k:
                found_sol = True
        self.assertTrue(found_sol)

    def test_find_u_cand_fallback_2(self):
        getcontext().prec = 28
        phi = D("0.34")
        eps = D("0.01")
        r = D("0.97")
        true_u = Z_OMEGA(-1, 2, 0, 3)
        true_k = 4
        solutions = find_u_candidates_fallback(eps, phi, r)
        found_sol = False
        self.assertEqual(len(solutions), 1)
        for sol in solutions:
            if sol[0] == true_u and sol[1] == true_k:
                found_sol = True
        self.assertTrue(found_sol)


if __name__ == "__main__":
    unittest.main()
