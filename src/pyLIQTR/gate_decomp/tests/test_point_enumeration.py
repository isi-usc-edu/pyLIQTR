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
import time

from gmpy2 import mpfr
import random

from pyLIQTR.gate_decomp.point_enumeration import *


class TestApproximationFunctions(unittest.TestCase):
    def test_scaled_one_dim_grid_problem_1(self):
        x0_m = mpfr("0")
        x1_m = mpfr("0.7")
        y0_m = mpfr(10)
        y1_m = mpfr(20)
        solutions_m = [
            sol for sol in scaled_one_dim_grid_problem(x0_m, x1_m, y0_m, y1_m)
        ]
        self.assertGreater(len(solutions_m), 0)
        for solution_m in solutions_m:
            self.assertLessEqual(solution_m, x1_m)
            self.assertGreaterEqual(solution_m, x0_m)
            self.assertLessEqual(solution_m.conj(), y1_m)
            self.assertGreaterEqual(solution_m.conj(), y0_m)

    def test_scaled_one_dim_grid_problem_2(self):
        x0_m = mpfr("0")
        x1_m = mpfr("0.99999999999999999999999")
        y0_m = mpfr(8)
        l_m = 1 + gmpy2.sqrt(mpfr(2))
        y1_m = y0_m + (l_m**2) / x1_m
        solutions_m = [
            sol for sol in scaled_one_dim_grid_problem(x0_m, x1_m, y0_m, y1_m)
        ]
        self.assertGreater(len(solutions_m), 0)
        for solution in solutions_m:
            self.assertLessEqual(solution, x1_m)
            self.assertGreaterEqual(solution, x0_m)
            self.assertLessEqual(solution.conj(), y1_m)
            self.assertGreaterEqual(solution.conj(), y0_m)

    def test_scaled_one_dim_grid_problem_3(self):
        gmpy2.get_context().precision = 100

        x0_m = mpfr("63887.49858757059024747894997")
        x1_m = mpfr("63888.37471944214825992600264")
        y0_m = mpfr("-3.031041453191973743961856950")
        y1_m = mpfr("3.621412719467655328944981621")
        solutions_m = [
            sol for sol in scaled_one_dim_grid_problem(x0_m, x1_m, y0_m, y1_m)
        ]
        self.assertGreater(len(solutions_m), 0)
        for solution in solutions_m:
            self.assertLessEqual(solution, x1_m)
            self.assertGreaterEqual(solution, x0_m)
            self.assertLessEqual(solution.conj(), y1_m)
            self.assertGreaterEqual(solution.conj(), y0_m)

    def test_scaled_one_dim_grid_problem_4(self):
        gmpy2.get_context().precision = 53
        x0 = mpfr("1414.2135623730951")
        x1 = mpfr(1415.0)
        y0 = mpfr("5.6568542494923806")
        y1 = mpfr("5.8")
        solutions = [sol for sol in scaled_one_dim_grid_problem(x0, x1, y0, y1)]

    def test_find_k_1(self):
        delta_m = mpfr(Z_SQRT2(1, 1))
        linv_m = 1 / delta_m
        k_m = find_k(delta_m)
        scaled_delta = (linv_m**k_m) * delta_m
        self.assertGreaterEqual(scaled_delta, linv_m)
        self.assertLess(scaled_delta, 1)

    def test_find_k_2(self):
        gmpy2.get_context().precision = 53
        lmbda_m = 1 + gmpy2.sqrt(2)
        delta_m = Z_SQRT2(1, 1) ** 2
        k_m = find_k(delta_m)
        scaled_delta_m = (lmbda_m ** (-k_m)) * delta_m
        self.assertGreaterEqual(scaled_delta_m, lmbda_m ** (-1))
        self.assertLess(scaled_delta_m, 1)

    def test_find_k_3(self):
        gmpy2.get_context().precision = 53
        lmbda_m = 1 + gmpy2.sqrt(2)
        delta_m = Z_SQRT2(1, 1) ** 10
        k_m = find_k(delta_m)
        scaled_delta_m = (lmbda_m ** (-k_m)) * delta_m
        self.assertGreaterEqual(scaled_delta_m, lmbda_m ** (-1))
        self.assertLess(scaled_delta_m, 1)


    def test_find_k_4(self):
        gmpy2.get_context().precision = 53
        lmbda_m = 1 + gmpy2.sqrt(2)
        delta_m = mpfr("6725.999851323220026050415779")
        k_m = find_k(delta_m)
        scaled_delta_m = (lmbda_m ** (-k_m)) * delta_m
        self.assertGreaterEqual(scaled_delta_m, lmbda_m ** (-1))
        self.assertLess(scaled_delta_m, 1)


    def test_find_k_5(self):
        lmbda_m = 1 + gmpy2.sqrt(2)
        delta_m = mpfr("1.4")
        k_m = find_k(delta_m)
        scaled_delta_m = (lmbda_m ** (-k_m)) * delta_m
        self.assertGreaterEqual(scaled_delta_m, lmbda_m ** (-1))
        self.assertLess(scaled_delta_m, 1)


    def test_solve_one_dim_grid_problem_1(self):
        x0_m = mpfr("9.1")
        x1_m = mpfr("9.2")
        y0_m = mpfr(-20)
        y1_m = mpfr(-19)
        solutions_m = [s for s in solve_one_dim_grid_problem(x0_m, x1_m, y0_m, y1_m)]
        self.assertIn(Z_SQRT2(-5, 10), solutions_m)
        for sol in solutions_m:
            self.assertGreaterEqual(sol, x0_m)
            self.assertLessEqual(sol, x1_m)
            self.assertGreaterEqual(sol.conj(), y0_m)
            self.assertLessEqual(sol.conj(), y1_m)


    def test_solve_one_dim_grid_problem_2(self):
        y0_m = mpfr("9.1")
        y1_m = mpfr("9.2")
        x0_m = mpfr(-20)
        x1_m = mpfr(-19)
        solutions_m = [s for s in solve_one_dim_grid_problem(x0_m, x1_m, y0_m, y1_m)]
        self.assertIn(Z_SQRT2(-5, -10), solutions_m)
        for sol in solutions_m:
            self.assertGreaterEqual(sol, x0_m)
            self.assertLessEqual(sol, x1_m)
            self.assertGreaterEqual(sol.conj(), y0_m)
            self.assertLessEqual(sol.conj(), y1_m)


    def test_solve_one_dim_grid_problem_3(self):
        x0_m = mpfr("12634400")
        x1_m = mpfr("12634500")
        y0_m = mpfr("-12754089.17")
        y1_m = mpfr("-12754089.16")
        solutions_m = [s for s in solve_one_dim_grid_problem(x0_m, x1_m, y0_m, y1_m)]
        self.assertIn(Z_SQRT2(-59807, 8976213), solutions_m)
        for sol in solutions_m:
            self.assertGreaterEqual(sol, x0_m)
            self.assertLessEqual(sol, x1_m)
            self.assertGreaterEqual(sol.conj(), y0_m)
            self.assertLessEqual(sol.conj(), y1_m)


    def test_solve_one_dim_grid_problem_4(self):
        y0_m = mpfr("12634400")
        y1_m = mpfr("12634500")
        x0_m = mpfr("-12754089.17")
        x1_m = mpfr("-12754089.16")
        solutions_m = [s for s in solve_one_dim_grid_problem(x0_m, x1_m, y0_m, y1_m)]
        self.assertIn(Z_SQRT2(-59807, -8976213), solutions_m)
        for sol in solutions_m:
            self.assertGreaterEqual(sol, x0_m)
            self.assertLessEqual(sol, x1_m)
            self.assertGreaterEqual(sol.conj(), y0_m)
            self.assertLessEqual(sol.conj(), y1_m)


    def test_solve_one_dim_grid_problem_5(self):
        x0_m = mpfr(-297)
        x1_m = mpfr("248.5823575862261")
        y0_m = mpfr(779)
        y1_m = mpfr("779.0106829464767380079485168")
        solutions_m = [s for s in solve_one_dim_grid_problem(x0_m, x1_m, y0_m, y1_m)]
        for sol in solutions_m:
            self.assertGreaterEqual(sol, x0_m)
            self.assertLessEqual(sol, x1_m)
            self.assertGreaterEqual(sol.conj(), y0_m)
            self.assertLessEqual(sol.conj(), y1_m)

    def test_solve_one_dim_grid_problem_6(self):
        # need this precision for x1 - x0 to not be 0
        gmpy2.get_context().precision = 65
        x0_m = mpfr("314248.7172986995797533837621")
        x1_m = mpfr("314248.7172986995797852939267")
        y0_m = mpfr("-0.3492727719167084114226836111")
        y1_m = mpfr("0.3492727719167084114226836111")
        solutions_m = [s for s in solve_one_dim_grid_problem(x0_m, x1_m, y0_m, y1_m)]
        for sol in solutions_m:
            self.assertGreaterEqual(sol.to_mpfr(), x0_m)
            self.assertLessEqual(sol.to_mpfr(), x1_m)
            self.assertGreaterEqual(sol.conj().to_mpfr(), y0_m)
            self.assertLessEqual(sol.conj().to_mpfr(), y1_m)

    def test_one_dim_grid_problem_rand(self):
        gmpy2.get_context().precision = 85
        seed = int(time.time())
        random.seed(seed)
        for _ in range(10):
            delta = random.random() * 1000
            Delta_m = ((1 + gmpy2.sqrt(2)) ** 2) / mpfr(delta)
            x0 = mpfr(random.randint(-1000, 1000))
            y0 = mpfr(random.randint(-1000, 1000))
            x1 = mpfr(x0 + delta)
            y1 = mpfr(y0 + Delta_m)
            solutions = [sol for sol in solve_one_dim_grid_problem(x0, x1, y0, y1)]
            failure_string = (
                f"Falied for seed = {seed}, delta = {delta}, x0 = {x0}, and y0 = {y0}"
            )
            self.assertGreater(len(solutions), 0)
            for solution in solutions:
                self.assertLessEqual(solution, x1, failure_string)
                self.assertGreaterEqual(solution, x0, failure_string)
                self.assertLessEqual(solution.conj(), y1, failure_string)
                self.assertGreaterEqual(solution.conj(), y0, failure_string)


    def test_solve_one_dim_grid_problem_tmp(self):
        gmpy2.get_context().precision = 60
        delta = 615.4848888981296
        Delta = ((1 + gmpy2.sqrt(2)) ** 2) / mpfr(delta)
        x0 = 403
        y0 = 954
        x1 = mpfr(x0 + delta)
        y1 = mpfr(y0 + Delta)
        solutions = [sol for sol in solve_one_dim_grid_problem(x0, x1, y0, y1)]
        for solution in solutions:
            self.assertLessEqual(mpfr(solution), x1)
            self.assertGreaterEqual(mpfr(solution), x0)
            self.assertLessEqual(mpfr(solution.conj()), y1)
            self.assertGreaterEqual(mpfr(solution.conj()), y0)

    def test_two_dim_upright_rectangle1(self):
        correct_ans = Z_OMEGA(1, 2, 3, 4)
        solutions_m = [
            sol
            for sol in solve_two_dim_grid_problem_upright_rectangles(
                mpfr(5), mpfr(6), mpfr(4), mpfr(5), mpfr(2), mpfr(3), mpfr(-1), mpfr(0)
            )
        ]
        self.assertGreater(len(solutions_m), 0)
        self.assertIn(correct_ans, solutions_m)

    def test_two_dim_upright_rectangle2(self):
        correct_ans = Z_OMEGA(-2, 3, 1, -6)
        solutions_m = [
            sol
            for sol in solve_two_dim_grid_problem_upright_rectangles(
                mpfr(-4),
                mpfr("-3.8"),
                mpfr("2.2"),
                mpfr("2.4"),
                mpfr(-9),
                mpfr(-8),
                mpfr("2.5"),
                mpfr("4.2"),
            )
        ]
        self.assertGreater(len(solutions_m), 0)
        self.assertIn(correct_ans, solutions_m)

    def test_solve_two_dim_grid_problem_ellipse(self):
        a1 = mpfr(3)
        b1 = mpfr(0.5)
        d1 = mpfr(1)
        a2 = mpfr(3)
        b2 = mpfr(0.3)
        d2 = mpfr(0.3)
        x1 = mpfr(-1.7)
        y1 = mpfr(13.95)
        x2 = mpfr(-12.3)
        y2 = mpfr(-7.9)
        ellipse1 = Ellipse(a1, b1, d1, x1, y1)
        ellipse2 = Ellipse(a2, b2, d2, x2, y2)
        true_sol1 = Z_OMEGA(4, 3, 12, -7)
        true_sol2 = Z_OMEGA(4, 3, 11, -7)
        solutions_m = [
            sol for sol in solve_two_dim_grid_problem_ellipse(ellipse1, ellipse2)
        ]
        self.assertGreater(len(solutions_m), 0)
        self.assertIn(true_sol1, solutions_m)
        self.assertIn(true_sol2, solutions_m)


    def test_find_bounding_ellipse(self):
        gmpy2.get_context().precision += 10
        phi = random.random() * 2 * gmpy2.const_pi()
        exponent = 15
        epsilon = 10**-exponent
        epsilon = mpfr(f"1e-{exponent}")
        ellipse = find_bounding_ellipse_direct(epsilon, phi)
        c = (1 - epsilon**2 / 2 + 1) / 2
        x1 = mpfr(c) * gmpy2.cos(phi)
        y1 = mpfr(c) * gmpy2.sin(phi)
        z = (
            ellipse.a * (x1 - ellipse.x) ** 2
            + 2 * ellipse.b * (x1 - ellipse.x) * (y1 - ellipse.y)
            + ellipse.d * (y1 - ellipse.y) ** 2
        )
        self.assertLessEqual(z, 1)
        gmpy2.get_context().precision -= 10

    def test_find_e_and_z(self):
        a = random.random()
        b = random.random()
        d = (1 - b**2) / a
        a = mpfr(a)
        b = mpfr(b)
        d = mpfr(d)
        ellipse = Ellipse(a, b, d)
        ellipse._calc_z_and_e()
        lmbda = 1 + gmpy2.sqrt(2)
        self.assertAlmostEqual(ellipse.e * lmbda ** (-ellipse.z), a)
        self.assertAlmostEqual(ellipse.e * lmbda ** (ellipse.z), d)

    def test_R_op(self):
        lmbda = 1 + gmpy2.sqrt(2)
        z1 = mpfr(0.75)
        z2 = mpfr(0.7)
        b1 = mpfr(-3)
        b2 = mpfr(3.5)
        e1 = gmpy2.sqrt(b1**2 + 1)
        e2 = gmpy2.sqrt(b2**2 + 1)
        E1 = Ellipse(e1 * lmbda ** (-z1), b1, e1 * lmbda**z1)
        E2 = Ellipse(e2 * lmbda ** (-z2), b2, e2 * lmbda**z2)
        initial_skew = calculate_skew(E1, E2)
        self.assertGreaterEqual(initial_skew, 15)
        E1, E2 = apply_grid_operator(GridOperator.R(), E1, E2)
        final_skew = calculate_skew(E1, E2)
        self.assertLessEqual(final_skew, mpfr(0.9) * initial_skew)

    def test_K_op(self):
        lmbda = 1 + gmpy2.sqrt(2)
        z1 = mpfr(0.25)
        z2 = mpfr(-0.7)
        b1 = mpfr(3)
        b2 = mpfr(3.5)
        e1 = gmpy2.sqrt(b1**2 + 1)
        e2 = gmpy2.sqrt(b2**2 + 1)
        E1 = Ellipse(e1 * lmbda ** (-z1), b1, e1 * lmbda**z1)
        E2 = Ellipse(e2 * lmbda ** (-z2), b2, e2 * lmbda**z2)
        initial_skew = calculate_skew(E1, E2)
        self.assertGreaterEqual(initial_skew, 15)
        E1, E2 = apply_grid_operator(GridOperator.K(), E1, E2)
        final_skew = calculate_skew(E1, E2)
        self.assertLessEqual(final_skew, mpfr(0.9) * initial_skew)

    def test_A_op(self):
        lmbda = 1 + gmpy2.sqrt(2)
        z1 = mpfr(0.4)
        z2 = mpfr(0.5)
        c = min(z1, z2)
        n = max(1, gmpy2.floor((lmbda**c) / 2))
        b1 = mpfr(3)
        b2 = mpfr(3.5)
        e1 = gmpy2.sqrt(b1**2 + 1)
        e2 = gmpy2.sqrt(b2**2 + 1)
        E1 = Ellipse(e1 * lmbda ** (-z1), b1, e1 * lmbda**z1)
        E2 = Ellipse(e2 * lmbda ** (-z2), b2, e2 * lmbda**z2)
        initial_skew = calculate_skew(E1, E2)
        self.assertGreaterEqual(initial_skew, 15)
        for _ in range(n):
            E1, E2 = apply_grid_operator(GridOperator.A(), E1, E2)
        final_skew = calculate_skew(E1, E2)
        self.assertLessEqual(final_skew, mpfr(0.9) * initial_skew)

    def test_B_op(self):
        lmbda = 1 + gmpy2.sqrt(2)
        z1 = mpfr(-0.1)
        z2 = mpfr(0.2)
        c = min(z1, z2)
        n = max(1, gmpy2.floor((lmbda**c) / 2))
        b1 = mpfr(-3)
        b2 = mpfr(3.5)
        e1 = gmpy2.sqrt(b1**2 + 1)
        e2 = gmpy2.sqrt(b2**2 + 1)
        E1 = Ellipse(e1 * lmbda ** (-z1), b1, e1 * lmbda**z1)
        E2 = Ellipse(e2 * lmbda ** (-z2), b2, e2 * lmbda**z2)
        initial_skew = calculate_skew(E1, E2)
        self.assertGreaterEqual(initial_skew, 15)
        for _ in range(n):
            E1, E2 = apply_grid_operator(GridOperator.B(), E1, E2)
        final_skew = calculate_skew(E1, E2)
        self.assertLessEqual(final_skew, mpfr(0.9) * initial_skew)

    def test_shift_operator(self):
        lmbda = 1 + gmpy2.sqrt(2)
        z1 = mpfr(1.3)
        z2 = mpfr(-0.9)
        b1 = 0
        b2 = 0
        while (b1**2 + b2**2) < 15:
            b1 = mpfr(random.random() - 0.5) * 10
            b2 = mpfr(random.random() - 0.5) * 10
        e1 = gmpy2.sqrt(b1**2 + 1)
        e2 = gmpy2.sqrt(b2**2 + 1)
        E1 = Ellipse(e1 * lmbda ** (-z1), b1, e1 * lmbda**z1)
        E2 = Ellipse(e2 * lmbda ** (-z2), b2, e2 * lmbda**z2)
        initial_bias = calculate_bias(E1, E2)
        self.assertGreater(abs(initial_bias), 1)
        k = determine_shift_operator(E1, E2)
        final_bias = calculate_bias(E1, E2)
        self.assertLessEqual(abs(final_bias), 1)

    def test_reduce_skew_large_bias1(self):
        lmbda = 1 + gmpy2.sqrt(2)
        z1 = mpfr(1.3)
        z2 = mpfr(-0.9)
        b1 = mpfr("4.8")
        b2 = mpfr("2.4")
        e1 = gmpy2.sqrt(b1**2 + 1)
        e2 = gmpy2.sqrt(b2**2 + 1)
        E1 = Ellipse(e1 * lmbda ** (-z1), b1, e1 * lmbda**z1)
        E2 = Ellipse(e2 * lmbda ** (-z2), b2, e2 * lmbda**z2)
        initial_skew = calculate_skew(E1, E2)
        self.assertGreaterEqual(initial_skew, 15)
        grid_op, _, _ = reduce_skew(E1, E2)
        E1, E2 = apply_grid_operator(grid_op, E1, E2)
        final_skew = calculate_skew(E1, E2)
        self.assertLessEqual(
            final_skew, mpfr(0.9) * initial_skew, f"Failed with b1 = {b1}, b2 = {b2}"
        )

    def test_reduce_skew_large_bias2(self):
        lmbda = 1 + gmpy2.sqrt(2)
        z1 = mpfr(2.3)
        z2 = mpfr(-0.9)
        b1 = mpfr("4.8")
        b2 = mpfr("2.4")
        e1 = gmpy2.sqrt(b1**2 + 1)
        e2 = gmpy2.sqrt(b2**2 + 1)
        E1 = Ellipse(e1 * lmbda ** (-z1), b1, e1 * lmbda**z1)
        E2 = Ellipse(e2 * lmbda ** (-z2), b2, e2 * lmbda**z2)
        initial_skew = calculate_skew(E1, E2)
        self.assertGreaterEqual(initial_skew, 15)
        grid_op, _, _ = reduce_skew(E1, E2)
        E1, E2 = apply_grid_operator(grid_op, E1, E2)
        final_skew = calculate_skew(E1, E2)
        self.assertLessEqual(
            final_skew, mpfr(0.9) * initial_skew, f"Failed with b1 = {b1}, b2 = {b2}"
        )

    def test_reduce_skew_extra_1(self):
        lmbda = 1 + gmpy2.sqrt(2)
        b1 = mpfr("-27.347913755282596045130981677")
        z1 = mpfr("0.199230493135169674872069292126")
        z2 = mpfr("-0.167360607388364229438710764534")
        b2 = mpfr("0.585786437626904951198311275784")
        e1 = gmpy2.sqrt(b1**2 + 1)
        e2 = gmpy2.sqrt(b2**2 + 1)
        E1 = Ellipse(e1 * lmbda ** (-z1), b1, e1 * lmbda**z1)
        E2 = Ellipse(e2 * lmbda ** (-z2), b2, e2 * lmbda**z2)
        initial_skew = calculate_skew(E1, E2)
        self.assertGreaterEqual(initial_skew, 15)
        grid_op, _, _ = reduce_skew(E1, E2)
        E1, E2 = apply_grid_operator(grid_op, E1, E2)
        final_skew = calculate_skew(E1, E2)
        self.assertLessEqual(final_skew, mpfr(0.9) * initial_skew)

    def test_find_u_candidate_1(self):
        epsilon = mpfr("0.2")
        phi = gmpy2.const_pi() / 6
        true_k = 3
        true_u = Z_OMEGA(0, 0, 2, 1)
        solutions = find_u_candidates_direct(epsilon, phi)
        found_sol = False
        for sol in solutions:
            if sol[0] == true_u and sol[1] == true_k:
                found_sol = True
        self.assertTrue(found_sol)

    def test_find_u_candidate_2(self):
        gmpy2.get_context().precision = 57
        epsilon = mpfr("0.00000001")
        phi = gmpy2.const_pi() / 6
        zx = gmpy2.cos(phi)
        zy = gmpy2.sin(phi)
        solutions = find_u_candidates_direct(epsilon, phi)
        for sol in solutions:
            k = sol[1]
            ux = mpc(sol[0]).real
            uy = mpc(sol[0]).imag
            u_conj = mpc(sol[0].conj2())
            dot_product = gmpy2.sqrt(2) ** (-k) * (ux * zx + uy * zy)
            self.assertGreaterEqual(dot_product, 1 - (epsilon**2) / 2)
            self.assertLessEqual(u_conj.real**2 + u_conj.imag**2, 2**k)

    def test_find_u_candidate_3(self):
        gmpy2.get_context().precision = 80
        epsilon = mpfr("0.0000000001")
        phi = gmpy2.const_pi() / 9
        zx = gmpy2.cos(phi)
        zy = gmpy2.sin(phi)
        solutions = find_u_candidates_direct(epsilon, phi)
        for sol in solutions:
            k = sol[1]
            ux = mpc(sol[0]).real
            uy = mpc(sol[0]).imag
            u_conj = mpc(sol[0].conj2())
            dot_product = gmpy2.sqrt(2) ** (-k) * (ux * zx + uy * zy)
            self.assertGreaterEqual(dot_product, 1 - (epsilon**2) / 2)
            self.assertLessEqual(u_conj.real**2 + u_conj.imag**2, 2**k)

    def test_find_u_candidate_4(self):
        gmpy2.get_context().precision = 80
        epsilon = mpfr("0.001")
        phi = gmpy2.const_pi() / 128
        zx = gmpy2.cos(phi)
        zy = gmpy2.sin(phi)
        solutions = find_u_candidates_direct(epsilon, phi)
        for sol in solutions:
            k = sol[1]
            ux = mpc(sol[0]).real
            uy = mpc(sol[0]).imag
            u_conj = mpc(sol[0].conj2())
            dot_product = gmpy2.sqrt(2) ** (-k) * (ux * zx + uy * zy)
            self.assertGreaterEqual(dot_product, 1 - (epsilon**2) / 2)
            self.assertLessEqual(u_conj.real**2 + u_conj.imag**2, 2**k)


if __name__ == "__main__":
    unittest.main()
