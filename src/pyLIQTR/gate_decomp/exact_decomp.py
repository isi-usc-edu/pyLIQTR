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

Functions for performing exact decomposition of single qubit unitary matrices with
entries from the ring D[ω] over the Clifford+T gateset. Makes use of the Matsumoto-
Amano normal form for single qubit Clifford+T operators [1,2,3]
[1] - arXiv:0806.3834
[2] - arXiv:1312.6584
[3] - https://www.cs.umd.edu/~amchilds/qa/qa.pdf Section 3
"""

from pyLIQTR.gate_decomp.matrices import MAT_D_OMEGA, SO3
from pyLIQTR.gate_decomp.rings import Z_OMEGA
import numpy as np
from numpy.typing import ArrayLike
from pyLIQTR.gate_decomp.clifford_gates import clifford_tuple
import sys
from typing import Sequence, Tuple, TypeVar, List

A = TypeVar("A")

I2 = MAT_D_OMEGA.I()
X2 = MAT_D_OMEGA.X()
Y2 = MAT_D_OMEGA.Y()
Z2 = MAT_D_OMEGA.Z()
H2 = MAT_D_OMEGA.H()
H3 = H2.convert_to_so3()
S2 = MAT_D_OMEGA.S()
S3 = S2.convert_to_so3()
Sd2 = MAT_D_OMEGA.Sd()
Sd3 = Sd2.convert_to_so3()
T2 = MAT_D_OMEGA.T()
T3 = T2.convert_to_so3()
Td2 = MAT_D_OMEGA.Td()
Td3 = Td2.convert_to_so3()


C = np.array([1, 1, 1], dtype=np.int8)
T = np.array([2, 2, 0], dtype=np.int8)
HT = np.array([0, 2, 2], dtype=np.int8)
SHT = np.array([2, 0, 2], dtype=np.int8)

equiv_classes = (C, HT, SHT, T)
gates = (Td3 @ H3, Td3 @ H3 @ Sd3, Td3)


def exact_decomp_to_qasm(
    u: Z_OMEGA, t: Z_OMEGA, k: int, conditional=False
) -> Tuple[str, int]:
    """Return the decomposition as a list of qasm instructions

    If conditional == True, add 'if (r1 == 1)' to each instruction (used in fallback
    approximation)
    """
    qasm_instructions = [
        "s q0",
        "s q0\ns q0\ns q0",
        "h q0",
        "x q0",
        "y q0",
        "z q0",
        "t q0",
    ]
    if conditional:
        # double loop because we have to split up the "s q0\ns q0\n s q0" string
        qasm_instructions = [
            "if (r1 == 1) " + y for x in qasm_instructions for y in x.splitlines()
        ]
    gate_sequence, t_count = exact_decomp(u, t, k, qasm_instructions)
    return "\n".join(gate_sequence), t_count


def exact_decomp_to_matrix_string(u: Z_OMEGA, t: Z_OMEGA, k: int) -> Tuple[str, int]:
    """Return the gate decomposition as a string in matrix order"""
    gate_strs = ["S", "Sd", "H", "X", "Y", "Z", "T"]
    gate_sequence, t_count = exact_decomp(u, t, k, gate_strs, circuit_order=False)
    return "".join(gate_sequence), t_count


def exact_decomp(
    u: Z_OMEGA, t: Z_OMEGA, k: int, return_objs: Sequence[A], circuit_order: bool = True
) -> Tuple[List[A], int]:
    """Decompose a matrix
    >>> ((1/sqrt(2)**k) * ([[u, -t.conj()],
                            [t,  u.conj()]])

    exactly into clifford+T gates.


    Parameters
    ----------
    u: Z_OMEGA
        upper left matrix entry
    t: Z_OMEGA
        lower left matrix entry
    k: int
        exponent on the 1/sqrt(2) scaling factor
    return_objs: Sequence[A]
        A list containing the representation of the gates you would like to return.
        They need to be in the following order:
        [S gate, S dagger gate, H gate, X gate, Y gate, Z dagger gate, T gate].
        For example, if you want to return qasm strings, this would be
        >>> ["s q0\\n", "s q0\\ns q0\\ns q0", ...]
        whereas if you wanted cirq gates,
        >>> [cirq.S, cirq.S**-1, ...]
    circuit_order: bool
        Whether or not to arrange the gates in circuit order (the first gate applied
        to the qubit is first) or "matrix order" (the first gate applied to the qubit
        is last)

    Returns
    -------
    Tuple[List[A], int]
        A tuple consisting of a list of gates (of same type as the return_objs argument), and
        and the number of T gates in the circuit.
    """
    m = MAT_D_OMEGA(u, -t.conj(), t, u.conj(), k)
    s = m.convert_to_so3()
    if circuit_order:
        return decompose(s, return_objs), s.k
    else:
        return decompose(s, return_objs)[::-1], s.k


def exact_decomp_compressed(
    u: Z_OMEGA, t: Z_OMEGA, k: int, return_objs: Sequence[A]
) -> Tuple[bool, int, int, A]:
    """
    Decompose a matrix
    >>> ((1/sqrt(2)**k) * ([[u, -t.conj()],
                            [t,  u.conj()]])

    exactly into clifford+T gates.


    Every Clifford+T operator U can be represented in MA normal form (arXiv:0806.3834):

    U = B_n B_{n-1} B_{n-2} ... B_1 C_0

    where B_k ∈ {HT, SHT} for k ≠ n,  B_n ∈ {T, HT, SHT}, and C_0 is a clifford
    operator. This function returns a "compressed" representation of the MA normal
    form of a given Clifford+T operator. The compressed representation is a tuple of
    four elements:

    1. A bool indicating whether or not B_n = T
    2. An integer who's binary representation corresponds to B_n ... B_1 (if 1 is false)
    or B_{n-1} ... B_1 if (1 is true). We represent HT by 0 and SHT by 1.
    3. The length of the sequence.
    4. The clifford operator at the end of the sequence.

    Example 1:
    (T)(HT)(HT)(SHT)(SHT)(HT)(X)
    has a binary representation 00110, so this becomes the tuple
    (True, 6, 5, X)

    Example 2:
    (SHT)(SHT)(HT)(HT)(SHT)(S)
    has binary representation 11001, so this becomes the tuple
    (False, 25, 5, S)

    Note that the binary representation is in "Matrix order" - this means that when we
    want to convert the integer to a sequence of gates, we read the binary
    representation "backwards" starting from the LSB.
    """
    m = MAT_D_OMEGA(u, -t.conj(), t, u.conj(), k)
    return exact_decomp_compressed_m(m, return_objs)


def exact_decomp_compressed_m(
    m: MAT_D_OMEGA, return_objs: Sequence[A]
) -> Tuple[bool, int, int, List[A]]:
    s = m.convert_to_so3()
    sequence_length = s.k
    sequence_index = sequence_length - 1
    gate_sequence = 0
    equiv_class = get_equiv_class(s.get_parity())
    first_gate = False
    if equiv_class == 3:
        first_gate = True
        sequence_length -= 1
    else:
        gate_sequence += (equiv_class - 1) * pow(2, sequence_index)
    sequence_index -= 1
    s = gates[equiv_class - 1] @ s
    equiv_class = get_equiv_class(s.get_parity())
    while equiv_class != 0:
        assert equiv_class in [1, 2]
        gate_sequence += (equiv_class - 1) * pow(2, sequence_index)
        s = gates[equiv_class - 1] @ s
        equiv_class = get_equiv_class(s.get_parity())
        sequence_index -= 1

    i = 0
    while clifford_tuple[i][1] != s:
        i += 1
        if i > 23:
            sys.exit(
                "Final matrix did not match an element of the clifford group. Gate = ",
                s,
            )
    final_gates = clifford_tuple[i][0]
    clifford_part = []
    for j in range(len(final_gates)):
        if final_gates[j : j + 2] == "Sd":
            clifford_part.insert(0, return_objs[1])
        elif final_gates[j : j + 1] == "S":
            clifford_part.insert(0, return_objs[0])
        elif final_gates[j : j + 1] == "H":
            clifford_part.insert(0, return_objs[2])
        elif final_gates[j : j + 1] == "X":
            clifford_part.insert(0, return_objs[3])
        elif final_gates[j : j + 1] == "Y":
            clifford_part.insert(0, return_objs[4])
        elif final_gates[j : j + 1] == "Z":
            clifford_part.insert(0, return_objs[5])

    return first_gate, gate_sequence, sequence_length, clifford_part


def decompose(s: SO3, return_objs: Sequence[A]) -> List[A]:
    """Decompose a matrix into its MA normal form"""
    gate_sequence = []
    equiv_class = get_equiv_class(s.get_parity())
    syllables = get_syllables(return_objs)
    while equiv_class != 0:
        gate_sequence = syllables[equiv_class - 1] + gate_sequence
        s = gates[equiv_class - 1] @ s
        equiv_class = get_equiv_class(s.get_parity())
    i = 0
    while clifford_tuple[i][1] != s:
        i += 1
        if i > 23:
            sys.exit(
                "Final matrix did not match an element of the clifford group. Gate = ",
                s,
            )
    final_gates = clifford_tuple[i][0]
    for j in range(len(final_gates)):
        if final_gates[j : j + 2] == "Sd":
            gate_sequence.insert(0, return_objs[1])
        elif final_gates[j : j + 1] == "S":
            gate_sequence.insert(0, return_objs[0])
        elif final_gates[j : j + 1] == "H":
            gate_sequence.insert(0, return_objs[2])
        elif final_gates[j : j + 1] == "X":
            gate_sequence.insert(0, return_objs[3])
        elif final_gates[j : j + 1] == "Y":
            gate_sequence.insert(0, return_objs[4])
        elif final_gates[j : j + 1] == "Z":
            gate_sequence.insert(0, return_objs[5])

    return gate_sequence


def get_syllables(return_objs: Sequence[A]) -> Tuple[List[A]]:
    """Return the 'syllables' of MA normal form using the return objs

    MA normal form puts the gates into sequnces A1A2...AnC, where each Ai
    is one of the 'syllables' HT, SHT, or T, and then C is a clifford. This
    function returns a tuple of the syllables using the return_objs."""
    first = [return_objs[6], return_objs[2]]
    second = [return_objs[6], return_objs[2], return_objs[0]]
    third = [return_objs[6]]
    return (first, second, third)


def get_equiv_class(x: ArrayLike) -> int:
    rep: ArrayLike = x.sum(axis=1)
    for i in range(4):
        if np.array_equal(rep, equiv_classes[i]):
            return i
    sys.exit("Parity matrix did not match any equivalence class. Matrix = ", rep)


def are_equivalent(gates: str, U: MAT_D_OMEGA, verbose=True) -> bool:
    """Verify that a given gate sequence is exactly equal to the unitary U

    Parameters
    ----------
    gates: str
        A string of representing a circuit. Gates can be
        - H: Hadamard
        - S: Phase gate (√Z gate)
        - Sd: Adjoint of the phase gate
        - T: T gate (√S gate)
        - X: Pauli X
        - Y: Pauli Y
        - Z: Pauli Z
    U: MAT_D_OMEGA
        The matrix that the circuit represented by gates should be equivalent to
    verbose: bool
        When true, prints out whether the circuit and matrix are equal and the phase
        (as a power of ω = exp(iπ/4)), if any, to which they differ.

    Returns
    -------
    bool:
        True if the circuit and matrix represent the same operation (up to a global
        phase), and False if they do not.
    """
    zw = Z_OMEGA(0, 0, 1, 0)
    matm = I2
    applied_gates = ""
    for i in range(len(gates)):
        if gates[i : i + 2] == "Sd":
            matm = matm @ Sd2
            applied_gates += "Sd"
        elif gates[i] == "H":
            matm = matm @ H2
            applied_gates += "H"
        elif gates[i] == "Z":
            matm = matm @ Z2
            applied_gates += "Z"
        elif gates[i] == "X":
            matm = matm @ X2
            applied_gates += "X"
        elif gates[i] == "Y":
            matm = matm @ Y2
            applied_gates += "Y"
        elif gates[i] == "S":
            matm = matm @ S2
            applied_gates += "S"
        elif gates[i] == "T":
            matm = matm @ T2
            applied_gates += "T"

    diff = matm.k - U.k
    if diff % 2 == 0:
        U = U.mult_pow_2(diff // 2)

    if diff % 2 == 1:
        diff += 1
        U = U.mult_pow_2(diff // 2)
        matm.k += 1
        matm = matm * Z_OMEGA(0, 1, 0, 1)
        matm = matm * zw**7

    for i in range(8):
        if (matm * zw**i) == U:
            if verbose:
                print(
                    f"The circuit is equal to the matrix up to a global phase of ω^{i}"
                )
            return True
    if verbose:
        print("The circuit is not equal to the approximate unitary")
    return False
