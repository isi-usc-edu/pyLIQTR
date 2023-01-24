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

Transformers for decomposing single qubit rotations in cirq circuits into
Clifford+T gates using the approximation method described in arXiv:1403.2975v3,
and an exact synthesis algorithm based on the Matsumoto-Amano normal form (see
arXiv:0806.3834, arXiv:1312.6584, and https://www.cs.umd.edu/~amchilds/qa/qa.pdf
Section 3)
"""

import cirq
from typing import Iterable, Tuple, List, Union
from decimal import getcontext, Decimal
from pyLIQTR.gate_decomp.exact_decomp import exact_decomp
from pyLIQTR.gate_decomp.decimal_utils import prec_pi
from pyLIQTR.gate_decomp.gate_approximation import get_ring_elts_direct
from functools import lru_cache

D = Decimal

# Single qubit gates that don't get decomposed.
clifford_plus_T_gates = [
    "H",
    "T",
    "X",
    "X**-1",
    "Y",
    "Y**-1",
    "Z",
    "Z**-1",
    "S",
    "S**-1",
    "T**-1",
    "reset",
]
clifford_plus_T_ops = [cirq.S, cirq.S ** -1, cirq.H, cirq.X, cirq.Y, cirq.Z, cirq.T]


def clifford_plus_t_direct_transform(
    circuit: cirq.AbstractCircuit, precision=10
) -> cirq.AbstractCircuit:
    """Decompose every single qubit unitary in the circuit into a Clifford+T
    approximation

    Parameters
    ----------
    circuit: cirq.AbstractCircuit
        The circuit to be decomposed. Currently, the circuit can only use the following
        gates:
        - Multi-qubit gates (ignored by the decomposition function)
        - Single qubit gates from the list clifford_plus_T_gates above (ignored by the
        decomposition function
        - Rx, Ry, and Rz gates, which are decomposed into clifford+T approximations
    precision: int
        Used to determine the maximum approximation error for each gate approximation.
        The decomposition returned by the function will be within a distance of
        10^{-precision} of the given rotation in the operator norm

    Returns
    -------
    cirq.AbstractCircuit:
        A new circuit where the single qubit rotations have been replaced by approximations
        using the gates listed in the list clifford_plus_T_ops above
    """
    operations = []
    for moment in circuit:
        for op in moment:
            if len(op.qubits) != 1 or is_gate_clifford(op):
                operations.append(op)
            else:
                qubit = op.qubits[0]
                gate_list = parse_gate(op.gate)
                for gate_tuple in gate_list:
                    # if Z rotation, can directly approximate
                    if gate_tuple[0] == "z":
                        operations.append(
                            decompose_diagonal_cirq(D(gate_tuple[1]), precision, qubit)
                        )
                    # if Y gate, can approximate with RY(θ) = (SH)RZ(θ)(SH)dag
                    # = (SH)RZ(θ)(HSdag)
                    if gate_tuple[0] == "y":
                        operations.append(cirq.S(qubit) ** -1)
                        operations.append(cirq.H(qubit))
                        operations.append(
                            decompose_diagonal_cirq(D(gate_tuple[1]), precision, qubit)
                        )
                        operations.append(cirq.H(qubit))
                        operations.append(cirq.S(qubit))
                    # if X gate, can approximate with RX(θ) = H RZ(θ) H
                    if gate_tuple[0] == "x":
                        operations.append(cirq.H(qubit))
                        operations.append(
                            decompose_diagonal_cirq(D(gate_tuple[1]), precision, qubit)
                        )
                        operations.append(cirq.H(qubit))
    return cirq.Circuit(operations)


def parse_gate(gate: cirq.Gate) -> List[Tuple[str, float]]:
    """Parse cirq gate into something that can be decomposed

    Parameters
    ----------
    gate: cirq.Gate
        The gate to be decomposed. Currently only supports for Rx, Ry, and Rz gates.

    Returns
    -------
    List[Tuple[str, float]]:
        A list of gates represented by a tuple. The tuple elements are
        - str: rotation axis (x/y/z)
        - float: rotation angle

    """
    axis = str(gate)[1]
    if axis in ["x", "y", "z"]:
        angle = gate._rads
    else:
        print(f"{gate} is not a rotation gate, cannot decompose")
    return [(axis, angle)]


def check_common_angles(angle: D, precision: int) -> Union[None, cirq.Gate]:
    """
    Note that if θ2 is our target angle and θ1 is one of the rotations we can perform
    exactly (i.e a multiple of π/8), and |θ1 - θ2|/2 is less than our desired
    error we can just use that gate, as err = sqrt(2 - 2Re(u*z)), so if we let
    u=exp(i*θ1/2) and z=exp(i*θ2/2), then
    err = sqrt(2-2Re(u*z))
        = sqrt(2-2cos((θ1-θ2)/2))
        = 2*sin((θ1-θ2)/4)
        < (θ1-θ2)/2

    NOTE: Assumes incoming rotation angles are in the range [-π, π]
    """
    eps = D(f"2e-{precision}")
    pi = prec_pi()
    gates = [
        [cirq.Z],
        [cirq.S ** -1, cirq.T ** -1],
        [cirq.S ** -1],
        [cirq.T ** -1],
        [cirq.I],
        [cirq.T],
        [cirq.S],
        [cirq.S, cirq.T],
        [cirq.Z],
    ]
    coeffs = [
        D(-1),
        D("-0.75"),
        D("-0.5"),
        D("-0.25"),
        D(0),
        D("0.25"),
        D("0.5"),
        D("0.75"),
        D(1),
    ]
    for gate, coeff in zip(gates, coeffs):
        if abs(pi * coeff - angle) < eps:
            return gate
    return None


@lru_cache
def decompose_diagonal_cirq(
    angle: D, precision: int, qubit: cirq.Qid
) -> Iterable[cirq.Operation]:
    """Decompose an rz rotation into cirq gates"""
    getcontext().prec = 2 * precision + 4
    easy_approx = check_common_angles(angle, precision)
    if easy_approx is not None:
        return [x.on(qubit) for x in easy_approx]
    else:
        u, t, k = get_ring_elts_direct(angle, precision)
        gates, _ = exact_decomp(u, t, k, clifford_plus_T_ops)
        return [x.on(qubit) for x in gates]


def is_gate_clifford(op):
    if str(op.gate).strip() in clifford_plus_T_gates:
        return True
    return False
