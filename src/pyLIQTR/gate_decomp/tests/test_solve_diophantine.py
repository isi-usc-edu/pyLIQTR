"""
Copyright (c) 2024 Massachusetts Institute of Technology 
SPDX-License-Identifier: BSD-2-Clause
"""


import unittest
from pyLIQTR.gate_decomp.solve_diophantine import *


class TestFactoringFunctions(unittest.TestCase):
    def test_factor1(self):
        a = 1513
        b = pollard_rho(a)
        self.assertEqual(a % b, 0)

    def test_factor2(self):
        a = 25
        b = pollard_rho(a, False)
        self.assertEqual(a % b, 0)

    def test_modular_square_root1(self):
        a = -2
        p = 13177
        soln = modular_square_root(a, p)
        self.assertEqual(pow(soln, 2, p), (p + a) % p)

    def test_modular_square_root2(self):
        a = 5
        p = 41
        soln = modular_square_root(a, p)
        self.assertEqual(pow(soln, 2, p), (p + a) % p)

    def test_modular_square_root3(self):
        a = 2
        p = 13177
        soln = modular_square_root(a, p)
        self.assertEqual(pow(soln, 2, p), (p + a) % p)

    def test_modular_square_root4(self):
        a = 2
        from sympy import sieve

        sieve.extend_to_no(10000)
        for p in sieve._list:
            if p % 8 == 1 or p % 8 == 7:
                soln = modular_square_root(a, p)
                self.assertEqual(
                    pow(soln, 2, p),
                    (p + a) % p,
                    f"Failed to find modular square root of 2 mod {p}",
                )

    def test_modular_square_root5(self):
        a = 2
        p = 1009
        soln = modular_square_root(a, p)
        self.assertEqual(pow(soln, 2, p), (p + a) % p)


if __name__ == "__main__":
    unittest.main()
