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

Custom rotation gates that calculate their Clifford+T decomposition when initialized,
and then implement that when cirq.decompose() is called.
"""

import math
import copy
import random
from functools import lru_cache
from typing import Dict, List, Optional, Tuple, Union

import cirq
from cirq import protocols, value
from cirq._compat import proper_repr
import gmpy2
from gmpy2 import mpfr

from pyLIQTR.gate_decomp.exact_decomp import exact_decomp_compressed
from pyLIQTR.gate_decomp.gate_approximation import (
    get_ring_elts_direct,
)

# these vales were determined by finding the T counts at varying precisions of 1000
# random angles.
T_COUNT_SLOPE = 3.02
T_COUNT_CONST = 0.77
T_COUNT_STD_DEV = 2.06
# not the optimal decomposition subroutine as of 2025 (but that's ok for now)


clifford_plus_T_ops = [cirq.S, cirq.S**-1, cirq.H, cirq.X, cirq.Y, cirq.Z, cirq.T]

CLIFFORD_GATES = [
    [],
    [cirq.S],
    [cirq.Z],
    [cirq.S**-1],
    [cirq.H],
    [cirq.S, cirq.H],
    [cirq.Z, cirq.H],
    [cirq.S**-1, cirq.H],
    [cirq.H, cirq.S],
    [cirq.S, cirq.H, cirq.S],
    [cirq.Z, cirq.H, cirq.S],
    [cirq.S**-1, cirq.H, cirq.S],
    [cirq.H, cirq.Z],
    [cirq.S, cirq.H, cirq.Z],
    [cirq.Z, cirq.H, cirq.Z],
    [cirq.S**-1, cirq.H, cirq.Z],
    [cirq.H, cirq.S**-1],
    [cirq.S, cirq.H, cirq.S**-1],
    [cirq.Z, cirq.H, cirq.S**-1],
    [cirq.S**-1, cirq.H, cirq.S**-1],
    [cirq.X],
    [cirq.S, cirq.X],
    [cirq.Y],
    [cirq.S, cirq.Y],
]


def check_common_angles(angle: mpfr, precision: mpfr) -> Union[None, cirq.Gate]:
    """
    Note that if θ2 is our target angle and θ1 is one of the rotations we can perform exactly (i.e a multiple of π/8), and |θ1 - θ2|/2 is less than our desired error we can just use that gate, as err = sqrt(2 - 2Re(u*z)), so if we let u=exp(i*θ1/2) and z=exp(i*θ2/2), then

    err = sqrt(2-2Re(u*z))
        = sqrt(2-2cos((θ1-θ2)/2))
        = 2*sin((θ1-θ2)/4)

    NOTE: Assumes incoming rotation angles are in the range [-π, π]
    """
    pi = gmpy2.const_pi()
    pi = gmpy2.const_pi()
    while angle < -pi:
        angle += 2 * pi
    while angle > pi:
        angle -= 2 * pi
    gates = [
        [cirq.Z],
        [cirq.S**-1, cirq.T**-1],
        [cirq.S**-1],
        [cirq.T**-1],
        [cirq.I],
        [cirq.T],
        [cirq.S],
        [cirq.S, cirq.T],
        [cirq.Z],
    ]
    coeffs = [
        mpfr(-1),
        mpfr("-0.75"),
        mpfr("-0.5"),
        mpfr("-0.25"),
        mpfr(0),
        mpfr("0.25"),
        mpfr("0.5"),
        mpfr("0.75"),
        mpfr(1),
        mpfr(-1),
        mpfr("-0.75"),
        mpfr("-0.5"),
        mpfr("-0.25"),
        mpfr(0),
        mpfr("0.25"),
        mpfr("0.5"),
        mpfr("0.75"),
        mpfr(1),
    ]
    for gate, coeff in zip(gates, coeffs):
        if abs(gmpy2.sin((pi * coeff - angle) / 4)) <= precision:
            return gate
    return None


@lru_cache
def _cliff_plus_t_decomp(
    rads: Union[float, mpfr], eps: Union[mpfr, None]
) -> Tuple[bool, int, int, List[cirq.Gate]]:
    """
    Compute the Clifford+T decomposition given an angle and precision.

    Made its own function in order to implement a cache.
    """
    u, t, k = get_ring_elts_direct(rads, prec=0, eps=eps)
    return exact_decomp_compressed(u, t, k, clifford_plus_T_ops)


class decomp_mixin(cirq.Gate):
    def __init__(
        self, rads, precision=1e-10, use_random_decomp=False, mismatch_counts=False, *args, **kwargs
    ) -> None:
        self.mismatch = mismatch_counts
        eps = mpfr(precision)
        gmpy2.get_context().precision = max(
            int(gmpy2.ceil(gmpy2.log2(10 ** (2.5 * precision + 15)))), 100
        )
        self.clifford_part: List[cirq.Gate] = check_common_angles(
            mpfr(rads), precision
        )
        eps = mpfr(precision)
        gmpy2.get_context().precision = max(
            int(gmpy2.ceil(gmpy2.log2(10 ** (2.5 * precision + 15)))), 100
        )
        self.clifford_part: List[cirq.Gate] = check_common_angles(
            mpfr(rads), precision
        )
        self.leading_T: bool = False
        self.gate_sequence: int = 0
        self.sequence_length: int = 0
        self.precision: float = precision
        if self.clifford_part is None and not use_random_decomp:
            (
                self.leading_T,
                self.gate_sequence,
                self.sequence_length,
                self.clifford_part,
            ) = _cliff_plus_t_decomp(mpfr(rads), eps)
        elif self.clifford_part is None and use_random_decomp:
            self.sequence_length = int(
                random.gauss(
                    T_COUNT_SLOPE * math.log2(1 / precision) + T_COUNT_CONST,
                    T_COUNT_STD_DEV,
                )
            )
            self.leading_T = random.random() < 0.5
            if self.leading_T:
                self.sequence_length -= 1
            self.gate_sequence = random.getrandbits(self.sequence_length)
            self.clifford_part = copy.copy(CLIFFORD_GATES[random.randint(0, 23)])

        self.num_T_gates = int(self.leading_T) + self.sequence_length
        self.num_Clifford_gates = len(self.clifford_part) + \
                                    self.sequence_length + \
                                    self.gate_sequence.bit_count()
        super().__init__(*args, **kwargs)

    def _decompose_(self, qubits) -> List[cirq.Operation]:
        q = qubits[0]
        ops = [g.on(q) for g in self.clifford_part]
        gate_seq = self.gate_sequence
        sequence_length = self.sequence_length
        while gate_seq > 0:
            ops += [cirq.T.on(q), cirq.H.on(q)]
            if gate_seq % 2 == 1:
                ops += [cirq.S.on(q)]
            gate_seq //= 2
            sequence_length -= 1
        ops += sequence_length * [cirq.T.on(q), cirq.H.on(q)]
        if self.leading_T:
            ops += [cirq.T.on(q)]
        return ops

    def get_resouces(self) -> Dict:
        return {"T count": self.num_T_gates, "Clifford count":self.num_Clifford_gates}

    def get_T_count(self) -> int:
        if self.mismatch == True:
            return 1.15 * math.log2(1 / self.precision) + 9.2
        else:
            return self.num_T_gates

    def get_Clifford_count(self) -> int:
        return self.num_Clifford_gates

    def num_qubits(self) -> int:
        return 1

    def _num_qubits_(self) -> int:
        return 1


class rz_decomp(cirq.Rz, decomp_mixin):
    """
    A subclass of cirq.Rz that knows its own clifford+T decomposition.
    """

    def __init__(
        self, rads: value.TParamVal, precision: float = 1e-10, use_random_decomp=False
    ):
        cirq.Rz.__init__(self, rads=rads)
        decomp_mixin.__init__(
            self, rads=rads, precision=precision, use_random_decomp=use_random_decomp
        )

    def _circuit_diagram_info_(
        self, args: "cirq.CircuitDiagramInfoArgs"
    ) -> Union[str, "protocols.CircuitDiagramInfo"]:
        angle_str = self._format_exponent_as_angle(args)
        return "Rz_d({}, prec={:.1e})".format(angle_str, self.precision)

    def __str__(self) -> str:
        if self._exponent == 1:
            return "Rz_d(π)"
        return f"Rz_d({self._exponent}π)"

    def __repr__(self) -> str:
        return f"rz_decomp(rads={proper_repr(self._rads)}, precision={self.precision})"

    def _decompose_(self, qubits):
        return super()._decompose_(qubits)

    def _qasm_(
        self, args: "cirq.QasmArgs", qubits: Tuple["cirq.Qid", ...]
    ) -> Optional[str]:
        init_string = super()._qasm_(args, qubits)
        qasm_string_list = init_string.splitlines()
        rot_qasm_string = qasm_string_list[-1]
        qasm_string_list[-1] = rot_qasm_string[:2] + "_d" + rot_qasm_string[2:]
        tuple_string = str(
            (
                self.leading_T,
                self.gate_sequence,
                self.sequence_length,
                self.clifford_part,
            )
        )
        tuple_string = tuple_string.replace("cirq.", "")
        qasm_string_list.append("// " + tuple_string)
        return "\n".join(qasm_string_list)


class ry_decomp(cirq.Ry, decomp_mixin):
    """
    A subclass of cirq.Ry that knows its own clifford+T decomposition.
    """

    def __init__(
        self, rads: value.TParamVal, precision: float = 1e-10, use_random_decomp=False
    ) -> None:
        cirq.Ry.__init__(self, rads=rads)
        decomp_mixin.__init__(
            self, rads=rads, precision=precision, use_random_decomp=use_random_decomp
        )

    def _decompose_(self, qubits) -> List[cirq.Operation]:
        q = qubits[0]
        main_sequence = super()._decompose_(qubits)
        if main_sequence != [cirq.I]:
            return [cirq.S(q) ** -1, cirq.H(q)] + main_sequence + [cirq.H(q), cirq.S(q)]
        else:
            return []

    def _circuit_diagram_info_(
        self, args: "cirq.CircuitDiagramInfoArgs"
    ) -> Union[str, "protocols.CircuitDiagramInfo"]:
        angle_str = self._format_exponent_as_angle(args)
        return "Ry_d({}, prec={:.1e})".format(angle_str, self.precision)

    def __str__(self) -> str:
        if self._exponent == 1:
            return "Ry_d(π)"
        return f"Ry_d({self._exponent}π)"

    def __repr__(self) -> str:
        return f"ry_decomp(rads={proper_repr(self._rads)}, precision={self.precision})"

    def _qasm_(
        self, args: "cirq.QasmArgs", qubits: Tuple["cirq.Qid", ...]
    ) -> Optional[str]:
        init_string = super()._qasm_(args, qubits)
        qasm_string_list = init_string.splitlines()
        rot_qasm_string = qasm_string_list[-1]
        qubit_string = rot_qasm_string[rot_qasm_string.find("q[") : -1]
        qasm_string_list[-1] = rot_qasm_string[:1] + "z_d" + rot_qasm_string[2:]
        qasm_string_list.insert(-1, "sdg " + qubit_string + ";")
        qasm_string_list.insert(-1, "h " + qubit_string + ";")
        tuple_string = str(
            (
                self.leading_T,
                self.gate_sequence,
                self.sequence_length,
                self.clifford_part,
            )
        )
        tuple_string = tuple_string.replace("cirq.", "")
        qasm_string_list.append("// " + tuple_string)
        qasm_string_list.append("h " + qubit_string + ";")
        qasm_string_list.append("s " + qubit_string + ";")
        return "\n".join(qasm_string_list)


class rx_decomp(cirq.Rx, decomp_mixin):
    """
    A subclass of cirq.Rx that knows its own clifford+T decomposition.
    """

    def __init__(
        self, rads: value.TParamVal, precision: float = 1e-10, use_random_decomp=False
    ) -> None:
        cirq.Rx.__init__(self, rads=rads)
        decomp_mixin.__init__(
            self, rads=rads, precision=precision, use_random_decomp=use_random_decomp
        )

    def _circuit_diagram_info_(
        self, args: "cirq.CircuitDiagramInfoArgs"
    ) -> Union[str, "protocols.CircuitDiagramInfo"]:
        angle_str = self._format_exponent_as_angle(args)
        return "Rx_d({}, prec={:.1e})".format(angle_str, self.precision)

    def _decompose_(self, qubits) -> List[cirq.Operation]:
        q = qubits[0]
        main_sequence = super()._decompose_(qubits)
        if main_sequence != [cirq.I]:
            return [cirq.H(q)] + main_sequence + [cirq.H(q)]
        else:
            return []

    def __str__(self) -> str:
        if self._exponent == 1:
            return "Rx_d(π)"
        return f"Rx_d({self._exponent}π)"

    def __repr__(self) -> str:
        return f"rx_decomp(rads={proper_repr(self._rads)}, precision={self.precision})"

    def _qasm_(
        self, args: "cirq.QasmArgs", qubits: Tuple["cirq.Qid", ...]
    ) -> Optional[str]:
        init_string = super()._qasm_(args, qubits)
        qasm_string_list = init_string.splitlines()
        rot_qasm_string = qasm_string_list[-1]
        qubit_string = rot_qasm_string[rot_qasm_string.find("q[") : -1]
        qasm_string_list[-1] = rot_qasm_string[:1] + "z_d" + rot_qasm_string[2:]
        qasm_string_list.insert(-1, "h " + qubit_string + ";")
        tuple_string = str(
            (
                self.leading_T,
                self.gate_sequence,
                self.sequence_length,
                self.clifford_part,
            )
        )
        tuple_string = tuple_string.replace("cirq.", "")
        qasm_string_list.append("// " + tuple_string)
        qasm_string_list.append("h " + qubit_string + ";")
        return "\n".join(qasm_string_list)