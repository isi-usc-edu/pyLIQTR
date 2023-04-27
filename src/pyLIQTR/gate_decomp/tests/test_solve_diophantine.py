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
