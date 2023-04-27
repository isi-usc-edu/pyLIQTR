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
from pyLIQTR.gate_decomp.decimal_utils import *
from decimal import Decimal, getcontext

D = Decimal


class TestDecimalUtils(unittest.TestCase):
    """
    General principal is that lower precision answer should equal the rounded answer
    calculated at higher precision
    """

    def test_arctan_precision1(self):
        x = D("0.1")
        getcontext().prec = 50
        res1 = prec_arctan(x)
        getcontext().prec = 55
        res2 = prec_arctan(x)
        getcontext().prec = 50
        self.assertEqual(res1, +res2)

    def test_arctan_precision2(self):
        x = D("0.5")
        getcontext().prec = 50
        res1 = prec_arctan(x)
        getcontext().prec = 55
        res2 = prec_arctan(x)
        getcontext().prec = 50
        self.assertEqual(res1, +res2)

    def test_arctan_precision3(self):
        x = D("0.8")
        getcontext().prec = 50
        res1 = prec_arctan(x)
        getcontext().prec = 55
        res2 = prec_arctan(x)
        getcontext().prec = 50
        self.assertEqual(res1, +res2)

    def test_arctan_precision4(self):
        x = D("1.1")
        getcontext().prec = 50
        res1 = prec_arctan(x)
        getcontext().prec = 55
        res2 = prec_arctan(x)
        getcontext().prec = 50
        self.assertEqual(+res1, +res2)

    def test_arctan_precision5(self):
        x = D("15")
        getcontext().prec = 50
        res1 = prec_arctan(x)
        getcontext().prec = 55
        res2 = prec_arctan(x)
        getcontext().prec = 50
        self.assertEqual(+res1, +res2)

    def test_arctan_precision6(self):
        x = D("1000")
        getcontext().prec = 50
        res1 = prec_arctan(x)
        getcontext().prec = 55
        res2 = prec_arctan(x)
        getcontext().prec = 50
        self.assertEqual(+res1, +res2)


if __name__ == "__main__":
    unittest.main()