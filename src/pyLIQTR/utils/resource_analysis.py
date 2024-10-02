"""
Copyright (c) 2024 Massachusetts Institute of Technology 
SPDX-License-Identifier: BSD-2-Clause
"""
import random
import cachetools
import numpy    as  np
import cirq     as  cirq
import qualtran as  qt
from typing import Dict, Any, Union, Optional

from warnings import warn


from   pyLIQTR.utils.circuit_decomposition import decompose_once, circuit_decompose_multi
from   pyLIQTR.gate_decomp.cirq_transforms import clifford_plus_t_direct_transform
from pyLIQTR.gate_decomp.rotation_gates import T_COUNT_CONST, T_COUNT_SLOPE, T_COUNT_STD_DEV
from pyLIQTR.scheduler.scheduler import schedule_circuit

from qualtran.cirq_interop.t_complexity_protocol import TComplexity, _get_hash,_t_complexity_from_strategies, \
         _t_complexity_for_gate_or_op,\
        _from_explicit_annotation,\
        _from_directly_countable,\
        _from_cirq_decomposition,\
        _from_iterable
from typing import Any, Hashable
from qualtran import Bloq



def _plyqt_from_bloq_build_call_graph(stc: Any) -> Optional[TComplexity]:
    # Uses the depth 1 call graph of Bloq `stc` to recursively compute the complexity.
    from qualtran.resource_counting import get_bloq_callee_counts
    from qualtran.resource_counting.generalizers import cirq_to_bloqs

    if not isinstance(stc, Bloq):
        return None
    callee_counts = get_bloq_callee_counts(bloq=stc, generalizer=cirq_to_bloqs)
    if len(callee_counts) == 0:
        return None
    ret = TComplexity()
    for bloq, n in callee_counts:
        r = pylqt_t_complexity(bloq)
        if r is None:
            return None
        ret += n * r
    return ret


@cachetools.cached(cachetools.LRUCache(128), key=_get_hash, info=True)
def _pylqt_t_complexity_for_gate_or_op(
    gate_or_op: Union[cirq.Gate, cirq.Operation, Bloq]
) -> Optional[TComplexity]:
    """
    A modified version of Qualtran's _t_complexity_for_Gate_or_op
    """
    if isinstance(gate_or_op, cirq.Operation) and gate_or_op.gate is not None:
        gate_or_op = gate_or_op.gate

    strategies = [
            from_measurement,
            _from_explicit_annotation,
            _from_directly_countable,
            _plyqt_from_bloq_build_call_graph,
            _from_cirq_decomposition,
            _from_iterable,
        ]
    return _t_complexity_from_strategies(gate_or_op, strategies)

def pylqt_t_complexity(stc: Any) -> TComplexity:
    """
    A modified version fo Qualtran's t_complexity
    Returns the TComplexity.
    Args:
        stc: an object to compute its TComplexity.
    Returns:
        The TComplexity of the given object.
    Raises:
        TypeError: if the methods fails to compute TComplexity.
    """
    if isinstance(stc, (cirq.Gate, cirq.Operation, Bloq)) and isinstance(stc, Hashable):
        ret = _pylqt_t_complexity_for_gate_or_op(stc)
    elif isinstance(stc, (cirq.Circuit)):
        ret = TComplexity()
        for moment in stc:
            for op in moment:
                thisComplexity = _pylqt_t_complexity_for_gate_or_op(op)
                if thisComplexity is None:
                    ret = None
                    stc = op
                    break
                ret += thisComplexity
    else:
        strategies = [
            from_measurement,
            _from_explicit_annotation,
            _from_directly_countable,
            _plyqt_from_bloq_build_call_graph,
            _from_cirq_decomposition,
            _from_iterable,
        ]
        ret = _t_complexity_from_strategies(stc, strategies)

    if ret is None:
        raise TypeError("couldn't compute TComplexity of:\n" f"type: {type(stc)}\n" f"value: {stc}")
    return ret

def from_measurement(stc:Any)->TComplexity:
    if 'cirq.MeasurementGate(' in (str(stc)):
        return TComplexity()
    else:
        return None

def estimate_resources(circuit_element:Any, rotation_gate_precision=1e-10, circuit_precision=None,
                       profile=False) -> Dict:
    """
    Keyword arguments:
    circuit_element -- element to estimate resources of. Can be circuit, gate, or operation.
    rotation_gate_precision -- maximum approximation error for each rotation gate decomposition
    circuit_precision -- If not None, the approximation error for each rotation gate will be bounded by `circuit_precision` divided by the number of rotation gates in `circuit_element`
    profile -- set to True to include rotation gates as a separate count rather than estimating their Clifford+T counts
    """
    try:
        resource_dict = {'LogicalQubits':cirq.num_qubits(circuit_element)}
    except:
        resource_dict={}
    t_cliff_rot_resources = pylqt_t_complexity(circuit_element)

    resource_dict["T"] = t_cliff_rot_resources.t 
    resource_dict["Clifford"] = t_cliff_rot_resources.clifford


    ## if profile is specified only take cirq_ft counting, otherwise
    ## do full Clifford+T estimation.
    ##
    if (not profile):
    # estimate gate counts based on number of rotations and precision
        if t_cliff_rot_resources.rotations:

            if circuit_precision is not None:
                Ts_from_rotations = get_T_counts_from_rotations(t_cliff_rot_resources.rotations,circuit_precision=circuit_precision)
            else:
                Ts_from_rotations = get_T_counts_from_rotations(t_cliff_rot_resources.rotations,gate_precision=rotation_gate_precision)
            
            resource_dict["T"] +=  Ts_from_rotations
            resource_dict["Clifford"] += 2*Ts_from_rotations + 1    
    else:
        ## Should probably do a better job profiling via some heuristics
        ##
        resource_dict["Rotations"] = t_cliff_rot_resources.rotations

    return resource_dict

def get_T_counts_from_rotations(num_rotation_gates,gate_precision=1e-8,circuit_precision=None,seed=0):
    # For single qubit rotation gates with a precision of ε, the T count is
    # 3.02 * log2(1/ε) + 0.77 (these numbers were obtained from decomposing 1000 random angles at varying precisions).
    if circuit_precision is not None:
        precision = (circuit_precision / num_rotation_gates)
    else:
        precision = gate_precision

    random.seed(seed)
    T_from_single_rotation = int(random.gauss(
                T_COUNT_SLOPE * np.log2(1 / precision) + T_COUNT_CONST,
                T_COUNT_STD_DEV,
            ))
    return T_from_single_rotation*num_rotation_gates

def get_qubits_by_keys(registers,keys):

    qubits = []

    for key in keys:
        qbs = registers[key].tolist()
        qubits += qbs

    return(qubits)


def count_clifford_and_rotations(circ):

    n_clifford     =  0       # number of Cliffords
    n_rotation     =  0       # number of Rotations
    n_other        =  0       # number of other gates (to reclassify)

    clifford_gates = [cirq.Y, cirq.YPowGate(exponent=-0.5),cirq.YPowGate(exponent=0.5),
                      cirq.X, cirq.XPowGate(exponent=-0.5),cirq.XPowGate(exponent=0.5),
                      cirq.Z, cirq.ZPowGate(exponent=-0.5),cirq.ZPowGate(exponent=0.5),
                      cirq.CX, cirq.CZ, cirq.SWAP, cirq.H, cirq.S]
    
    rotation_gates = [cirq.Rx, cirq.Ry, cirq.Rz]
    rotation_gate_names = ['Rx', 'Ry', 'Rz']


    for moment in circ:
        for gate_op in moment:
            if (gate_op.gate in clifford_gates):
                n_clifford     += 1
#            elif (gate_op.gate in rotation_gates):
            elif ( gate_op.gate.__str__()[0:2] in rotation_gate_names):      ### lazy - sure there is a better way
                n_rotation += 1
            else:
                n_other += 1

    return(n_clifford,n_rotation,n_other)




def count_t_gates(circ):

    ## Note that T^\dagger is given as 
    ## a ZPowGate 
    ## (e.g., as the fourth root of a Z)

    n_t     =  0       # number of T
    n_t_inv =  0       # number of T^\dagger

    for moment in circ:
        for gate_op in moment:
            if (gate_op.gate == cirq.T):
                n_t     += 1
            elif (gate_op.gate == cirq.ZPowGate(exponent=-0.25)):
                n_t_inv += 1

    return(n_t + n_t_inv)


def t_complexity_from_circuit(circ):

    n_t = count_t_gates(circ)

    n_clifford,n_rotation,n_other = count_clifford_and_rotations(circ)

    t_complexity = qt.cirq_interop.t_complexity_protocol.TComplexity(t=n_t,clifford=n_clifford,rotations=n_rotation)

    return(t_complexity)


def legacy_resource_profile(gate):
    
    circ = cirq.Circuit()
    circ.append(gate)

    circ_cops = circuit_decompose_multi(circ,2)
    #circ_cliff_t = clifford_plus_t_direct_transform(circ_cops)

    t_complexity = t_complexity_from_circuit(circ_cops)

    return(t_complexity)