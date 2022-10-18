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

Get a list of all single qubit clifford operators (up to phase)
"""

from pyLIQTR.gate_decomp.matrices import SO3, MAT_D_OMEGA

Xm = MAT_D_OMEGA.X()
Ym = MAT_D_OMEGA.Y()
Zm = MAT_D_OMEGA.Z()
Sdagm = MAT_D_OMEGA.Sd()
Sm = MAT_D_OMEGA.S()
Im = MAT_D_OMEGA.I()
Hm = MAT_D_OMEGA.H()

Xs = Xm.convert_to_so3()
Ys = Ym.convert_to_so3()
Zs = Zm.convert_to_so3()
Sds = Sdagm.convert_to_so3()
Ss = Sm.convert_to_so3()
Is = Im.convert_to_so3()
Hs = Hm.convert_to_so3()


clifford_string_list = [
    "I",
    "S",
    "Z",
    "Sd",
    "H",
    "H S",
    "H Z",
    "H Sd",
    "S H",
    "S H S",
    "S H Z",
    "S H Sd",
    "Z H",
    "Z H S",
    "Z H Z",
    "Z H Sd",
    "Sd H",
    "Sd H S",
    "Sd H Z",
    "Sd H Sd",
    "X",
    "X S",
    "Y",
    "Y S",
]


def create_operator_op(clifford_list: str) -> SO3:
    op = Is
    for char in clifford_list.split(sep=" "):
        if char == "S":
            op = op @ Ss
        elif char == "Sd":
            op = op @ Sds
        elif char == "H":
            op = op @ Hs
        elif char == "X":
            op = op @ Xs
        elif char == "Y":
            op = op @ Ys
        elif char == "Z":
            op = op @ Zs
    return op


# create a tuple where each element is a tuple of the form (string, SO3)
clifford_tuple = tuple(
    [("".join(x.split(sep=" ")), create_operator_op(x)) for x in clifford_string_list]
)
