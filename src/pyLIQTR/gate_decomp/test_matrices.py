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
from pyLIQTR.gate_decomp.matrices import *


z1 = Z_OMEGA(0, 0, 0, 1)
zm1 = Z_OMEGA(0, 0, 0, -1)
zw = Z_OMEGA(0, 0, 1, 0)
zwconj = Z_OMEGA(-1, 0, 0, 0)
z0 = Z_OMEGA(0, 0, 0, 0)
zi = Z_OMEGA(0, 1, 0, 0)
zmi = Z_OMEGA(0, -1, 0, 0)

Im = MAT_D_OMEGA(z1, z0, z0, z1, 0)
Zm = MAT_D_OMEGA(z1, z0, z0, zm1, 0)
Sm = MAT_D_OMEGA(z1, z0, z0, zi, 0)
Sdagm = MAT_D_OMEGA(z1, z0, z0, zmi, 0)
Tm = MAT_D_OMEGA(z1, z0, z0, zw, 0)
Tdagm = MAT_D_OMEGA(z1, z0, z0, zwconj, 0)
Hm = MAT_D_OMEGA(z1, z1, z1, zm1, 1)
Xm = MAT_D_OMEGA(z0, z1, z1, z0, 0)
Ym = MAT_D_OMEGA(z0, -zi, zi, z0, 0)

Hs = SO3(
    Z_SQRT2(0, 0),
    Z_SQRT2(0, 0),
    Z_SQRT2(1, 0),
    Z_SQRT2(0, 0),
    Z_SQRT2(-1, 0),
    Z_SQRT2(0, 0),
    Z_SQRT2(1, 0),
    Z_SQRT2(0, 0),
    Z_SQRT2(0, 0),
    0,
)
Ss = SO3(
    Z_SQRT2(0, 0),
    Z_SQRT2(-1, 0),
    Z_SQRT2(0, 0),
    Z_SQRT2(1, 0),
    Z_SQRT2(0, 0),
    Z_SQRT2(0, 0),
    Z_SQRT2(0, 0),
    Z_SQRT2(0, 0),
    Z_SQRT2(1, 0),
    0,
)
Ts = SO3(
    Z_SQRT2(1, 0),
    Z_SQRT2(-1, 0),
    Z_SQRT2(0, 0),
    Z_SQRT2(1, 0),
    Z_SQRT2(1, 0),
    Z_SQRT2(0, 0),
    Z_SQRT2(0, 0),
    Z_SQRT2(0, 0),
    Z_SQRT2(0, 1),
    1,
)
Is = SO3(
    Z_SQRT2(1, 0),
    Z_SQRT2(0, 0),
    Z_SQRT2(0, 0),
    Z_SQRT2(0, 0),
    Z_SQRT2(1, 0),
    Z_SQRT2(0, 0),
    Z_SQRT2(0, 0),
    Z_SQRT2(0, 0),
    Z_SQRT2(1, 0),
    0,
)


class TestMatrices(unittest.TestCase):
    def test_mult_SO3_1(self):
        self.assertEqual(Hs @ Hs, Is)

    def test_mult_SO3_2(self):
        self.assertEqual(Ss @ Ss @ Ss @ Ss, Is)

    def test_mult_SO3_3(self):
        res = Ts @ Ts
        res.reduce_by_2()
        self.assertEqual(res, Ss)

    def test_convert_H(self):
        self.assertEqual(Hm.convert_to_so3(), Hs)

    def test_convert_S(self):
        self.assertEqual(Sm.convert_to_so3(), Ss)

    def test_convert_T(self):
        self.assertEqual(Tm.convert_to_so3(), Ts)

    def test_mult_MATDOMEGA_1(self):
        self.assertEqual(Tm @ Tm, Sm)

    def test_mult_MATDOMEGA_2(self):
        self.assertEqual(Tm @ Tdagm, Im)

    def test_convert_T2(self):
        res = Tm @ Tm
        self.assertEqual(res.convert_to_so3(), Ss)

    def test_convert_HT(self):
        res = Hm @ Tm
        res = res.convert_to_so3()
        self.assertEqual(res, Hs @ Ts)

    def test_convert_complicated(self):
        res = (
            Hm
            @ Tm
            @ Sm
            @ Hm
            @ Tm
            @ Sm
            @ Hm
            @ Tm
            @ Sm
            @ Hm
            @ Tm
            @ Hm
            @ Tm
            @ Hm
            @ Tm
            @ Hm
            @ Tm
            @ Sm
            @ Hm
            @ Tm
            @ Hm
        )
        res = res.convert_to_so3()
        soln = (
            Hs
            @ Ts
            @ Ss
            @ Hs
            @ Ts
            @ Ss
            @ Hs
            @ Ts
            @ Ss
            @ Hs
            @ Ts
            @ Hs
            @ Ts
            @ Hs
            @ Ts
            @ Hs
            @ Ts
            @ Ss
            @ Hs
            @ Ts
            @ Hs
        )
        self.assertEqual(res, soln)


if __name__ == "__main__":
    unittest.main()
