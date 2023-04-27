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

import math
import random
import re
import warnings
from decimal import Decimal, getcontext
from functools import lru_cache
from typing import FrozenSet, Iterable, List, Tuple, Union

import cirq

from pyLIQTR.gate_decomp.decimal_utils import prec_sin
from pyLIQTR.gate_decomp.exact_decomp import exact_decomp
from pyLIQTR.gate_decomp.gate_approximation import get_ring_elts_direct
from pyLIQTR.gate_decomp.rotation_gates import (
    T_COUNT_CONST,
    T_COUNT_SLOPE,
    T_COUNT_STD_DEV,
    check_common_angles,
    clifford_gates,
    clifford_plus_T_ops,
    rx_decomp,
    ry_decomp,
    rz_decomp,
)

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
directCpT_gates = [
    "CCX",
    "ccx",
    "CCZ",
    "ccz",
    "TOFFOLI",
    "toffoli",
]
filtered_gates = [
    "RESET",
    "reset",
]

def parse_and_reconstruct(line):
    #The Clifford + T representation can be reconstructed by:
    def reconstruct(leading_T,gate_sequence,sequence_length, clifford_part):
        cirq_cliffs = {"S":cirq.S,"Sd":cirq.inverse(cirq.S), "H": cirq.H, "Z":cirq.Z,"X":cirq.X, "Y":cirq.Y,"T":cirq.T}
        def fixS(g):
            if "S" in g and len(g) > 1:
                return "Sd"
            else:
                return g

        ops = [cirq_cliffs[fixS(g.strip())] for g in clifford_part]
        gate_seq = gate_sequence
        while gate_seq > 0:
            ops += [cirq.T, cirq.H]
            if gate_seq % 2 == 1:
                ops += [cirq.S]
            gate_seq //= 2
            sequence_length -= 1
        ops += sequence_length * [cirq.T, cirq.H]
        if leading_T:
            ops += [cirq.T]
        return ops
    parsed_line = line
    for rmv in ["// ","(",")","[","]"]:
        parsed_line = parsed_line.replace(rmv,"")
    parsed_line = parsed_line.split(', ')
    print(parsed_line)
    leadingT = bool(parsed_line[0])
    gate_sequence = int(parsed_line[1])
    sequence_length = int(parsed_line[2])
    clifford_part = parsed_line[3:]
    return reconstruct(leadingT,gate_sequence,sequence_length, clifford_part)



# direct decomposition of a gate into Clifford+T
def decompose_cirq_directCpT(op, rev=False):
    gate = str(op.gate).strip()
    q0, q1, q2 = op.qubits
    gate_to_gen = None
    for dGate in ["CCX", "ccx", "TOFFOLI", "toffoli"]:
        if gate.startswith(dGate):
            gate_to_gen = "ccx"
    for dGate in ["CCZ", "ccz"]:
        if gate.startswith(dGate):
            gate_to_gen = "ccz"

    # construct the gate
    gate_list = []
    if gate_to_gen == "ccx":
        gate_list += [cirq.H(q2)]
    if rev:
        gate_list += [
            cirq.S(q1) ** -1,
            cirq.T(q0) ** -1,
            cirq.CX(q0, q1),
            cirq.T(q1),
            cirq.CX(q0, q1),
            cirq.T(q2) ** -1,
            cirq.T(q1),
            cirq.CX(q0, q2),
            cirq.T(q2),
            cirq.CX(q1, q2),
            cirq.T(q2) ** -1,
            cirq.CX(q0, q2),
            cirq.T(q2),
            cirq.CX(q1, q2),
        ]
    else:
        gate_list += [
            cirq.CX(q1, q2),
            cirq.T(q2) ** -1,
            cirq.CX(q0, q2),
            cirq.T(q2),
            cirq.CX(q1, q2),
            cirq.T(q2) ** -1,
            cirq.CX(q0, q2),
            cirq.T(q1) ** -1,
            cirq.T(q2),
            cirq.CX(q0, q1),
            cirq.T(q1) ** -1,
            cirq.CX(q0, q1),
            cirq.T(q0),
            cirq.S(q1),
        ]
    if gate_to_gen == "ccx":
        gate_list += [cirq.H(q2)]

    return gate_list


def count_rotation_gates(circuit: cirq.AbstractCircuit):
    total_gates = 0
    for moment in circuit:
        for op in moment:
            if is_filtered_gate(op):
                pass
            elif (len(op.qubits) == 1) and (not is_gate_clifford(op)):
                m = re.search("R[xyz]\(([^\)]+)\)", str(op))
                if m:
                    angle_str = m.groups()[0]
                    angle = float(
                        "".join(
                            [
                                s
                                for s in angle_str
                                if (s in ["e", "-", "."] or s.isnumeric())
                            ]
                        )
                    )
                    if (
                        (abs(angle) < 1e-10)
                        or (abs(abs(angle) - 0.5) < 1e-10)
                        or (abs(abs(angle) - 1.0) < 1e-10)
                    ):
                        pass
                    else:
                        total_gates += 1

    return total_gates


def clifford_plus_t_direct_transform(
    circuit: cirq.AbstractCircuit,
    precision: Union[int, float, None] = 10,
    circuit_precision: Union[int, float, None] = None,
    num_rotation_gates: Union[int, None] = None,
    use_rotation_decomp_gates: bool = False,
    use_random_decomp: bool = False,
    warn_if_not_decomposed: bool = True,
) -> cirq.AbstractCircuit:
    """Decompose every single qubit unitary in the circuit into a Clifford+T
    approximation.

    Setting the Precision
    ---------------------
    There are two ways to specify the desired approximation error due to decomposition:
    `gate_precision` and `circuit_precision`. If `circuit_precision` is not None, then
    each gate will have an approximation error bounded by `circuit_precision` divided by
    the number of rotation gates, so that the approximation error of the entire circuit
    will be bounded by `circuit_precision`. If `circuit_precision` is None, and
    `gate_precision` is not, then each rotation gate in the circuit will be replaced by
    an approximation upper bounded by `gate_precision` in the operator norm.

    Output Circuit
    --------------
    There are also two ways in which the circuit can be output, which is specificed via the
    `use_rotation_decomp_gates` parameter (`False` by default). In the default mode, this function
    returns the fully decomposed circuit, with all of the single qubit rotations replaced with
    Clifford and T gates.

    When `use_rotation_decomp_gates` is set to `True`, the single qubit rotation gates in
    the circuit get replaced with custom gates (`r<x/y/z>_decomp`) that know their
    Clifford+T decomposition, and can be queried for their number of T gates. Users can
    obtain a fully decomposed circuit using `cirq.decompose`, for example, if `circuit`
    is a `cirq.Circuit` containing the custom rotation gates, they can be expanding via

    >>> expanded_circuit = cirq.expand_composite(
            circuit, no_decomp=lambda g: g.gate in [cirq.H]
        )

    Note that we have to specify that we do not want H gates to be decomposed.

    Parameters
    ----------
    circuit: cirq.AbstractCircuit
        The circuit to be decomposed. Currently, the circuit can only use the following
        gates:
        - Multi-qubit gates. CCX and CCZ gates are decomposed into Clifford+T exactly.
        The rest are ignored by the decomposition function.
        - Single qubit gates from the list clifford_plus_T_gates above (ignored by the
        decomposition function.
        - Rx, Ry, and Rz gates, which are decomposed into clifford+T approximations.

    gate_precision: Union[int, float, None]
        Used to determine the maximum approximation error for each gate decomposition.
        If given an int, the decomposition returned by the function will be within a
        distance of 10^{-precision} of the given rotation in the operator norm, whereas
        if given a float the max error in the operator norm will simply be
        gate_precision.
    circuit_precision: Union[int, float, None]
        If not None, the approximation error for each gate will be bounded by either
        `circuit_precision` divided by the number of rotation gates (if given a float),
        or 10^{-circuit_precision} (if given an int). Note that this requires doing an
        initial pass over the circuit to count the number of rotation gates, unless
        `num_rotation_gates` is specified by the caller.
    num_rotation_gates: Union[int, None]:
        The number of rotation gates in the circuit.
    use_rotation_decomp_gates: bool
        If true, replace `cirq.r<x/y/z>` gates with
        `pyLIQTR.gate_decomp.r<x/y/z>_decomp` gates, which can be expanded later into a
        Clifford+T circuit. If false, return a fully expanded Clifford+T circuit.
    use_random_decomp: bool
        If true, rotations will be "decomposed" into randomly selected Clifford+T
        sequences, with the length of the sequence depending on the given precision.
        For a precision of ε, the T count is sampled from a normal distribution with
        a mean of 3.02 * log2(1/ε) + 0.77 and a standard deviation of 2.06 (these
        numbers were obtained from decomposing 1000 random angles at varying
        precisions).
    warn_if_not_decomposed: bool
        If true, raises a warning when the circuit includes a non-clifford+T gate that
        is not an X, Y, or Z rotation. Such gates are not decomposed.

    Returns
    -------
    cirq.AbstractCircuit:
        A new circuit where the single qubit rotations have been replaced by approximations
        using the gates listed in the list clifford_plus_T_ops above
    """
    gate_precision = determine_gate_precision(
        circuit, precision, circuit_precision, num_rotation_gates
    )
    operations = []
    for moment in circuit:
        for op in moment:
            if is_directCpT_gate(op):
                # these gates have known optimal decompositions into Clifford+T
                if "**-1.0" in str(op):
                    # We want the inverse C+T circuit to be the inverted gate-wise
                    # reverse of the base circuit. This should allow for better
                    # optimization. This is valid for this limited set of gates that we
                    # know has a unitary decomposition and are self inverses. If we
                    # allow cirq to decompose the inverse circuit then it will produce
                    # an inverse circuit that has the form of the forward circuit
                    # (11/23/22 KMO)
                    new_gates = decompose_cirq_directCpT(op, rev=True)
                else:
                    new_gates = decompose_cirq_directCpT(op)

                operations.append(new_gates)
            elif (
                len(op.qubits) != 1
                or is_gate_clifford(op.without_classical_controls())
                or "Measurement" in str(op)
            ):
                operations.append(op)
            else:
                classical_controls = None
                if isinstance(op, cirq.ClassicallyControlledOperation):
                    classical_controls = op.classical_controls
                    op = op.without_classical_controls()

                qubit = op.qubits[0]
                gate_list = parse_gate(op.gate, warn_if_not_decomposed)
                if isinstance(gate_list, cirq.Gate):
                    operations.append(op)
                else:
                    for gate_tuple in gate_list:
                        if gate_tuple[1] == 0:
                            continue
                        exp = math.frexp(float(gate_tuple[1]))[1]
                        # subtract 1 b/c frexp returns an exponent such that
                        # x = 0.1b_1b_2...b_52 x 2^E
                        # instead of
                        # x = 1.b_1b_2...b_52 x 2^E (1)
                        # and double precision rounding error is 2^(E-53) for E in
                        # equation 1
                        # error due to floating point rep
                        rounding_err_angle = pow(2, exp - 53 - 1)
                        # error in operator norm due to floating point rounding in angle
                        rounding_err_rotation = 2 * prec_sin(D(rounding_err_angle / 4))
                        if D(gate_precision) < rounding_err_rotation:
                            gate_precision = D(
                                f"1e-{math.floor(abs(math.log10(rounding_err_rotation)))}"
                            )
                            warning_string = (
                                "Desired precision of decomposition exceeds the"
                                " precision from floating point representation of"
                                " angle. Setting the precision of this gate to ε ="
                                + str(gate_precision)
                            )
                            warnings.warn(warning_string)

                        # if Z rotation, can directly approximate
                        if gate_tuple[0] == "z":
                            if use_rotation_decomp_gates:
                                if classical_controls is not None:
                                    operations.append(
                                        rz_decomp(
                                            gate_tuple[1],
                                            gate_precision,
                                            use_random_decomp,
                                        )
                                        .on(qubit)
                                        .with_classical_controls(*classical_controls)
                                    )
                                else:
                                    operations.append(
                                        rz_decomp(
                                            gate_tuple[1],
                                            gate_precision,
                                            use_random_decomp,
                                        ).on(qubit)
                                    )
                            else:
                                operations.append(
                                    decompose_diagonal_cirq(
                                        D(gate_tuple[1]),
                                        gate_precision,
                                        qubit,
                                        classical_controls,
                                        use_random_decomp,
                                    )
                                )
                        # if Y gate, can approximate with RY(θ) = (SH)RZ(θ)(SH)dag
                        # = (SH)RZ(θ)(HSdag)
                        if gate_tuple[0] == "y":
                            if use_rotation_decomp_gates:
                                if classical_controls is not None:
                                    operations.append(
                                        ry_decomp(
                                            gate_tuple[1],
                                            gate_precision,
                                            use_random_decomp,
                                        )
                                        .on(qubit)
                                        .with_classical_controls(*classical_controls)
                                    )
                                else:
                                    operations.append(
                                        ry_decomp(
                                            gate_tuple[1],
                                            gate_precision,
                                            use_random_decomp,
                                        ).on(qubit)
                                    )
                            else:
                                operations.append(cirq.S(qubit) ** -1)
                                operations.append(cirq.H(qubit))
                                operations.append(
                                    decompose_diagonal_cirq(
                                        D(gate_tuple[1]),
                                        gate_precision,
                                        qubit,
                                        classical_controls,
                                        use_random_decomp,
                                    )
                                )
                                operations.append(cirq.H(qubit))
                                operations.append(cirq.S(qubit))
                        # if X gate, can approximate with RX(θ) = H RZ(θ) H
                        if gate_tuple[0] == "x":
                            if use_rotation_decomp_gates:
                                if classical_controls is not None:
                                    operations.append(
                                        rx_decomp(
                                            gate_tuple[1],
                                            gate_precision,
                                            use_random_decomp,
                                        )
                                        .on(qubit)
                                        .with_classical_controls(*classical_controls)
                                    )
                                else:
                                    operations.append(
                                        rx_decomp(
                                            gate_tuple[1],
                                            gate_precision,
                                            use_random_decomp,
                                        ).on(qubit)
                                    )
                            else:
                                operations.append(cirq.H(qubit))
                                operations.append(
                                    decompose_diagonal_cirq(
                                        D(gate_tuple[1]),
                                        gate_precision,
                                        qubit,
                                        classical_controls,
                                        use_random_decomp,
                                    )
                                )
                                operations.append(cirq.H(qubit))

    return cirq.Circuit(operations)


def determine_gate_precision(
    circuit: cirq.AbstractCircuit,
    gate_precision: Union[int, float, None] = 10,
    circuit_precision: Union[int, float, None] = None,
    num_rotation_gates: Union[int, None] = None,
) -> float:
    if circuit_precision is not None:
        if num_rotation_gates is None:
            num_rotation_gates = get_num_rotation_gates(circuit)
        if isinstance(circuit_precision, float):
            return circuit_precision / num_rotation_gates
        else:
            return pow(10, -circuit_precision) / num_rotation_gates
    else:
        if isinstance(gate_precision, float):
            return gate_precision
        else:
            return pow(10, -gate_precision)


def get_num_rotation_gates(
    circuit: cirq.AbstractCircuit,
) -> int:
    num_rotation_gates = 0
    for moment in circuit:
        for op in moment:
            if parse_gate(op.gate) is not None:
                num_rotation_gates += 1
    return num_rotation_gates


def parse_gate(
    gate: cirq.Gate, warn_if_not_decomposed: bool = True
) -> List[Tuple[str, float]]:
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
    gate_string = str(gate)
    if "R" in gate_string:
        axis = gate_string[1]
        if axis in ["x", "y", "z"]:
            angle = gate._rads
    elif ("X" in gate_string or "Y" in gate_string or "Z" in gate_string) and (
        "**" in gate_string
    ):
        axis = gate_string[0].lower()
        angle = math.pi * gate.exponent
    else:
        if warn_if_not_decomposed:
            warnings.warn(
                f"{gate} is not a rotation gate, cannot decompose. Set"
                " warn_if_not_decomposed=False to silence this warning.",
                RuntimeWarning,
            )
        return gate
    return [(axis, angle)]


@lru_cache
def decompose_diagonal_cirq(
    angle: D,
    precision: float,
    qubit: cirq.Qid,
    classical_control: Union[None, FrozenSet[cirq.Condition]] = None,
    use_random_decomp: bool = False,
) -> Iterable[cirq.Operation]:
    """Decompose an rz rotation into cirq gates

    If the original operation was classically controlled, all of the gates from the
    decomposition will be classically controlled as well.
    """
    eps = D(precision)
    getcontext().prec = int(2 * abs(eps.log10().to_integral_value("ROUND_UP"))) + 4
    easy_approx = check_common_angles(angle, precision)
    if classical_control is None:
        if easy_approx is not None:
            return [x.on(qubit) for x in easy_approx]
        else:
            if use_random_decomp:
                gates = random_decomp(eps)
                return [x.on(qubit) for x in gates]
            else:
                u, t, k = get_ring_elts_direct(angle, prec=0, eps=eps)
                gates, _ = exact_decomp(u, t, k, clifford_plus_T_ops)
                return [x.on(qubit) for x in gates]
    else:
        if easy_approx is not None:
            return [
                cirq.ClassicallyControlledOperation(x.on(qubit), classical_control)
                for x in easy_approx
            ]
        else:
            if use_random_decomp:
                gates = random_decomp(eps)
                return [
                    cirq.ClassicallyControlledOperation(x.on(qubit), classical_control)
                    for x in gates
                ]
            else:
                u, t, k = get_ring_elts_direct(angle, prec=0, eps=eps)
                gates, _ = exact_decomp(u, t, k, clifford_plus_T_ops)
                return [
                    cirq.ClassicallyControlledOperation(x.on(qubit), classical_control)
                    for x in gates
                ]


def is_gate_clifford(op):
    if str(op.gate).strip() in clifford_plus_T_gates:
        return True
    return False


def is_directCpT_gate(op):
    gate = str(op.gate).strip()
    for dGate in directCpT_gates:
        if gate.startswith(dGate):
            return True

    return False


def is_filtered_gate(op):
    gate = str(op.gate).strip()
    for dGate in filtered_gates:
        if gate.startswith(dGate):
            return True

    return False


def random_decomp(eps: D):
    gates = clifford_gates[random.randint(0, 23)]
    sequence_length = int(
        random.gauss(
            T_COUNT_SLOPE * math.log2(1 / eps) + T_COUNT_CONST,
            T_COUNT_STD_DEV,
        )
    )
    leading_T = random.random() < 0.5
    if leading_T:
        sequence_length -= 1
    gate_seq = random.getrandbits(sequence_length)
    while gate_seq > 0:
        gates += [cirq.T, cirq.H]
        if gate_seq % 2 == 1:
            gates += [cirq.S]
        gate_seq //= 2
        sequence_length -= 1
    gates += sequence_length * [cirq.T, cirq.H]
    if leading_T:
        gates += [cirq.T]
    return gates
