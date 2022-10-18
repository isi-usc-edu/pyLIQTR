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
from pyLIQTR.gate_decomp.exact_decomp import *
from decimal import Decimal as D, getcontext
from pyLIQTR.gate_decomp.gate_approximation import get_ring_elts_direct, get_ring_elts_fallback
from pyLIQTR.gate_decomp.decimal_utils import prec_pi, arg
from pyLIQTR.gate_decomp.matrices import MAT_D_OMEGA
from random import randint
from math import gcd


class TestExactDecomp(unittest.TestCase):
    # for angles from π/120 to 239π/120, (except for multiples of π/2, π/4, and π/8), make sure that
    # the gate decomposition is equal to the approximate unitary found
    def test_exact_decomp_prec10(self):
        prec = 10
        PI = prec_pi()
        for i in range(239):
            i += 1
            if i % 15 != 0:
                u, t, k = get_ring_elts_direct(D(i * PI / 120), prec)
                circuit, _ = exact_decomp_to_matrix_string(u, t, k)
                mat = MAT_D_OMEGA(u, -t.conj(), t, u.conj(), k)
                self.assertTrue(
                    are_equivalent(circuit, mat, False), f"Failed for {i}π/120"
                )

    # for higher precisions just pick random gates instead of looping through everything
    def test_exact_decomp_prec20(self):
        prec = 20
        PI = prec_pi()
        for _ in range(10):
            i = randint(1, 240)
            while i % 15 == 0:
                i = randint(1, 240)
            u, t, k = get_ring_elts_direct(D(i * PI / 120), prec)
            circuit, _ = exact_decomp_to_matrix_string(u, t, k)
            mat = MAT_D_OMEGA(u, -t.conj(), t, u.conj(), k)
            self.assertTrue(are_equivalent(circuit, mat, False), f"Failed for {i}π/120")

    def test_exact_decomp_prec30(self):
        prec = 30
        PI = prec_pi()
        for _ in range(10):
            i = randint(1, 240)
            while i % 15 == 0:
                i = randint(1, 240)
            u, t, k = get_ring_elts_direct(D(i * PI / 120), prec)
            circuit, _ = exact_decomp_to_matrix_string(u, t, k)
            mat = MAT_D_OMEGA(u, -t.conj(), t, u.conj(), k)
            self.assertTrue(are_equivalent(circuit, mat, False), f"Failed for {i}π/120")

    def test_exact_decomp_prec40(self):
        prec = 40
        PI = prec_pi()
        for _ in range(10):
            i = randint(1, 240)
            while i % 15 == 0:
                i = randint(1, 240)
            u, t, k = get_ring_elts_direct(D(i * PI / 120), prec)
            circuit, _ = exact_decomp_to_matrix_string(u, t, k)
            mat = MAT_D_OMEGA(u, -t.conj(), t, u.conj(), k)
            self.assertTrue(are_equivalent(circuit, mat, False), f"Failed for {i}π/120")

    def test_rand_angles_prec50(self):
        getcontext().prec = 104
        for _ in range(5):
            denom = 10000
            num = randint(1, 20000)
            common_factor = gcd(num, denom)
            num //= common_factor
            denom //= common_factor
            while denom in [1, 2, 4, 8]:
                num = randint(1, 20000)
                denom = 10000
                common_factor = gcd(num, denom)
                num //= common_factor
                denom //= common_factor
            u, t, k = get_ring_elts_direct(D(num) * prec_pi() / D(denom), 50)
            circuit, _ = exact_decomp_to_matrix_string(u, t, k)
            mat = MAT_D_OMEGA(u, -t.conj(), t, u.conj(), k)
            self.assertTrue(
                are_equivalent(circuit, mat, False), f"Failed for {num}π/{denom}"
            )

    def test_exact_decomp_fallback_prec10(self):
        prec = 10
        PI = prec_pi()
        r = D("0.999")
        for _ in range(10):
            i = randint(1, 240)
            while i % 15 == 0:
                i = randint(1, 240)
            angle = D(i * PI / 120)
            u1, t1, k1, u2, t2, k2 = get_ring_elts_fallback(angle, prec, r)
            circuit1, _ = exact_decomp_to_matrix_string(u1, t1, k1)
            mat1 = MAT_D_OMEGA(u1, -t1.conj(), t1, u1.conj(), k1)
            self.assertTrue(
                are_equivalent(circuit1, mat1, False), f"Failed for {i}π/{120}"
            )
            circuit2, _ = exact_decomp_to_matrix_string(u2, t2, k2)
            mat2 = MAT_D_OMEGA(u2, -t2.conj(), t2, u2.conj(), k2)
            self.assertTrue(
                are_equivalent(circuit2, mat2, False), f"Failed for {i}π/{120}"
            )

    def test_exact_decomp_fallback_prec20(self):
        prec = 20
        PI = prec_pi()
        r = D("0.999")
        for _ in range(10):
            i = randint(1, 240)
            while i % 15 == 0:
                i = randint(1, 240)
            angle = D(i * PI / 120)
            u1, t1, k1, u2, t2, k2 = get_ring_elts_fallback(angle, prec, r)
            circuit1, _ = exact_decomp_to_matrix_string(u1, t1, k1)
            mat1 = MAT_D_OMEGA(u1, -t1.conj(), t1, u1.conj(), k1)
            self.assertTrue(
                are_equivalent(circuit1, mat1, False), f"Failed for {i}π/{120}"
            )
            circuit2, _ = exact_decomp_to_matrix_string(u2, t2, k2)
            mat2 = MAT_D_OMEGA(u2, -t2.conj(), t2, u2.conj(), k2)
            self.assertTrue(
                are_equivalent(circuit2, mat2, False), f"Failed for {i}π/{120}"
            )

    def test_exact_decomp_fallback_prec30(self):
        prec = 30
        PI = prec_pi()
        r = D("0.999")
        for _ in range(10):
            i = randint(1, 240)
            while i % 15 == 0:
                i = randint(1, 240)
            angle = D(i * PI / 120)
            u1, t1, k1, u2, t2, k2 = get_ring_elts_fallback(angle, prec, r)
            circuit1, _ = exact_decomp_to_matrix_string(u1, t1, k1)
            mat1 = MAT_D_OMEGA(u1, -t1.conj(), t1, u1.conj(), k1)
            self.assertTrue(
                are_equivalent(circuit1, mat1, False), f"Failed for {i}π/{120}"
            )
            circuit2, _ = exact_decomp_to_matrix_string(u2, t2, k2)
            mat2 = MAT_D_OMEGA(u2, -t2.conj(), t2, u2.conj(), k2)
            self.assertTrue(
                are_equivalent(circuit2, mat2, False), f"Failed for {i}π/{120}"
            )

    def test_exact_decomp_fallback_prec40(self):
        prec = 40
        PI = prec_pi()
        r = D("0.999")
        for _ in range(5):
            i = randint(1, 240)
            while i % 15 == 0:
                i = randint(1, 240)
            angle = D(i * PI / 120)
            u1, t1, k1, u2, t2, k2 = get_ring_elts_fallback(angle, prec, r)
            circuit1, _ = exact_decomp_to_matrix_string(u1, t1, k1)
            mat1 = MAT_D_OMEGA(u1, -t1.conj(), t1, u1.conj(), k1)
            self.assertTrue(
                are_equivalent(circuit1, mat1, False), f"Failed for {i}π/{120}"
            )
            circuit2, _ = exact_decomp_to_matrix_string(u2, t2, k2)
            mat2 = MAT_D_OMEGA(u2, -t2.conj(), t2, u2.conj(), k2)
            self.assertTrue(
                are_equivalent(circuit2, mat2, False), f"Failed for {i}π/{120}"
            )

    def test_exact_decomp_fallback_prec50(self):
        prec = 50
        PI = prec_pi()
        r = D("0.999")
        for _ in range(5):
            i = randint(1, 240)
            while i % 15 == 0:
                i = randint(1, 240)
            angle = D(i * PI / 120)
            u1, t1, k1, u2, t2, k2 = get_ring_elts_fallback(angle, prec, r)
            circuit1, _ = exact_decomp_to_matrix_string(u1, t1, k1)
            mat1 = MAT_D_OMEGA(u1, -t1.conj(), t1, u1.conj(), k1)
            self.assertTrue(
                are_equivalent(circuit1, mat1, False), f"Failed for {i}π/{120}"
            )
            circuit2, _ = exact_decomp_to_matrix_string(u2, t2, k2)
            mat2 = MAT_D_OMEGA(u2, -t2.conj(), t2, u2.conj(), k2)
            self.assertTrue(
                are_equivalent(circuit2, mat2, False), f"Failed for {i}π/{120}"
            )


if __name__ == "__main__":
    unittest.main()
