from typing                                     import Any, Callable, Hashable, Iterable, Literal, Optional, overload, Protocol, Union
     
import attrs
import cachetools
import cirq
import random
import numpy as np
import warnings
     
from pyLIQTR.gate_decomp.rotation_gates         import T_COUNT_CONST, T_COUNT_SLOPE, T_COUNT_STD_DEV
from qualtran.cirq_interop.decompose_protocol   import _decompose_once_considering_known_decomposition
import qualtran

def _get_hash(self, val: Any):
        if isinstance(val, cirq.Operation) and val.gate is not None:
            val = val.gate
        return val

class GateComplexity:
    """Dataclass storing counts gates or operations.
    Gatesets: dict{"gate_name": cirq.Gateset}
    stc: circuit, gate, or operation to be decomposed."""

    def __init__(self, gatesets):
        self._gatesets = gatesets

        '''
        for gate_type in self._gatesets:
            setattr(self, gate_type, 0)

        setattr(self, 'uncounted', 0)
        self._max_qubits = 0
        '''
        self._analytic_counts = []
        
    def __add__(self, other: 'GateComplexity') -> 'GateComplexity':
        total_gc = GateComplexity(self._gatesets)
        for gate_type in self._gatesets:
            other_val = getattr(other, gate_type)
            self_val = getattr(self, gate_type)
            setattr(total_gc, gate_type, other_val+self_val)
        return total_gc


    def __str__(self) -> str:
        outputs = []
        for gate_type in self._gatesets:
            count_str = f'{gate_type}:  {getattr(self, gate_type):g}\n'
            outputs.append(count_str)
        maxq_string = f'Maximum number of qubits used: {self._max_qubits}'
        outputs.append(maxq_string)
        if getattr(self, 'uncounted') != 0:
            uncounted_str = f"Warning: not all gates in circuit have been included in gateset. {getattr(self, 'uncounted'):g} gates uncounted."
            outputs.append(uncounted_str)
        str_outputs = '\n'.join(outputs)
        return (str_outputs)

    
    def _has_gate_complexity(self, stc: Any):
        modified_gatetypes = []
        cliff_estimator = getattr(stc, "_clifford_complexity_", None)
        rot_estimator = getattr(stc, "_rotation_complexity_", None)
        for gate_type in self._gatesets:
            if gate_type == 't':
                estimator = getattr(stc, '_t_complexity_', None)
                if estimator is not None:
                    t_comp = estimator()    
                    if t_comp is not NotImplemented:
                        t_count = t_comp.t
                        setattr(self, 't', t_count)
                        modified_gatetypes.append('t')
                    
            elif gate_type == 'clifford' and (cliff_estimator is None):
                estimator = getattr(stc, '_t_complexity_', None)
                if estimator is not None:
                    t_comp = estimator()    
                    if t_comp is not NotImplemented:
                        cliff_count = t_comp.clifford
                        setattr(self, 'clifford', cliff_count)
                        modified_gatetypes.append('clifford')

            
            elif gate_type == 'rotation' and (rot_estimator is None):
                estimator = getattr(stc, '_t_complexity_', None)
                if estimator is not None:
                    t_comp = estimator()    
                    if t_comp is not NotImplemented:
                        rot_count = t_comp.rotations
                        setattr(self, 'rotation', rot_count)
                        modified_gatetypes.append('rotation')
                    
            else:                
                attr_str = f"_{gate_type}_complexity_"
                estimator = getattr(stc, attr_str, None)
                if estimator is not None:
                    result = estimator()
                    if result is not NotImplemented:
                        count = getattr(result, gate_type, None)
                        setattr(self, gate_type, count)
                        modified_gatetypes.append(gate_type)
        
        if isinstance(stc, cirq.Operation) and stc.gate is not None:
            return self._has_gate_complexity(stc.gate)

        for gate_type in modified_gatetypes:
            self._analytic_counts.append(gate_type)
        return None
         

    def _is_in_gatesets(self, stc: Any):
        """Attempts to infer the type of a gate/operation as one of defined gatesets."""
        if not isinstance(stc, (cirq.Gate, cirq.Operation)):
            return None
        
        '''
        if hasattr(stc, 'num_qubits'):
            print("qubits counted")
            nq = it.num_qubits
            if nq > self._max_qubits:
                self._max_qubits = nq
        '''
        
        if isinstance(stc, cirq.ClassicallyControlledOperation):
            stc = stc.without_classical_controls()
        
        for gate_type in self._gatesets:
            if gate_type in self._analytic_counts:
                return None
                
            if stc in self._gatesets[gate_type]:
                empty_gc = GateComplexity(self._gatesets)
                setattr(empty_gc, gate_type, 1)
                return empty_gc

        basic_cliff_set = cirq.Gateset(cirq.I, cirq.H, cirq.X, cirq.Y, cirq.Z, cirq.S,  
                                     cirq.H**-1, cirq.S**-1, cirq.X**-1, cirq.Y**-1, cirq.Z**-1,
                                     cirq.XPowGate, cirq.YPowGate, cirq.ZPowGate,
                                     cirq.SwapPowGate, unroll_circuit_op=False)
        if (cirq.num_qubits(stc) <= 2 and cirq.has_stabilizer_effect(stc)) or (stc in basic_cliff_set):
            print(stc)
            empty_gc = GateComplexity(self._gatesets)
            setattr(empty_gc, 'uncounted', 1)
            return empty_gc

        rotation_set = cirq.Gateset(cirq.Rx, cirq.Ry, cirq.Rz, unroll_circuit_op=False)
        if stc in rotation_set:      
            if 't' in self._gatesets:
                random.seed(0)
                T_from_rotation = int(random.gauss(T_COUNT_SLOPE * np.log2(1 / 1e-8) + T_COUNT_CONST, T_COUNT_STD_DEV,))
                                                
                empty_gc = GateComplexity(self._gatesets)
                setattr(empty_gc, 't', T_from_rotation)
                return empty_gc

            else:
                empty_gc = GateComplexity(self._gatesets)
                setattr(empty_gc, 'uncounted', 1)
                return empty_gc

        t_set = cirq.Gateset(cirq.T, cirq.T**-1, unroll_circuit_op=False)
        if stc in t_set:
            empty_gc = GateComplexity(self._gatesets)
            setattr(empty_gc, 'uncounted', 1)
            return empty_gc
 
        return None     

                             
    def _is_iterable(self, it: Any):
        if not isinstance(it, Iterable):
            return None

        empty_gc = GateComplexity(self._gatesets)
        for v in it:
            r = gate_complexity(v, self._gatesets)
            if r is None:
                return None
            empty_gc = empty_gc + r
        return empty_gc

    
    def _from_cirq_decomposition(self, stc: Any):
        # Decompose the object and recursively compute the complexity.
        decomposition = _decompose_once_considering_known_decomposition(stc)
        if decomposition is None:
            return None
            
        return self._is_iterable(decomposition)
        

    def _gate_complexity_from_strategies(self, stc: Any, strategies: Iterable[Callable[[Any, bool], dict]]):
        ret = None
        for strategy in strategies:
            ret= strategy(stc)
            if ret is not None:
                break
        return ret

    
    def cliff_t_cx(self, stc: Any):
        gatesets = {
            't': cirq.Gateset(cirq.T, cirq.T**-1, unroll_circuit_op=False),
            'cx': cirq.Gateset(cirq.CNOT, cirq.CNOT**-1, cirq.CXPowGate, unroll_circuit_op=False),
            'clifford': cirq.Gateset(cirq.I, cirq.H, cirq.X, cirq.Y, cirq.Z, cirq.S,  
                                     cirq.H**-1, cirq.S**-1, cirq.X**-1, cirq.Y**-1, cirq.Z**-1,
                                     cirq.XPowGate, cirq.YPowGate, cirq.ZPowGate,
                                     cirq.SwapPowGate, unroll_circuit_op=False)
        }

        return gate_complexity(stc, gatesets)

    
    def cliff_t_cx_rot(self, stc: Any):
        gatesets = {
            't': cirq.Gateset(cirq.T, cirq.T**-1, unroll_circuit_op=False),
            'cx': cirq.Gateset(cirq.CNOT, cirq.CNOT**-1, cirq.CXPowGate, unroll_circuit_op=False),
            'rotation': cirq.Gateset(cirq.Rx, cirq.Ry, cirq.Rz, unroll_circuit_op=False),
            'clifford': cirq.Gateset(cirq.I, cirq.H, cirq.X, cirq.Y, cirq.Z, cirq.S,  
                                     cirq.H**-1, cirq.S**-1, cirq.X**-1, cirq.Y**-1, cirq.Z**-1,
                                     cirq.XPowGate, cirq.YPowGate, cirq.ZPowGate,
                                     cirq.SwapPowGate, unroll_circuit_op=False)
            
        }
        return gate_complexity(stc, gatesets)

    
    def cliff_t_cx_rot_toff(self, stc: Any):
        gatesets = {
            't': cirq.Gateset(cirq.T, cirq.T**-1, unroll_circuit_op=False),
            'cx': cirq.Gateset(cirq.CNOT, cirq.CNOT**-1, cirq.CXPowGate, unroll_circuit_op=False),
            'rotation': cirq.Gateset(cirq.Rx, cirq.Ry, cirq.Rz, unroll_circuit_op=False),
            'clifford': cirq.Gateset(cirq.I, cirq.H, cirq.X, cirq.Y, cirq.Z, cirq.S,  
                                     cirq.H**-1, cirq.S**-1, cirq.X**-1, cirq.Y**-1, cirq.Z**-1,
                                     cirq.XPowGate, cirq.YPowGate, cirq.ZPowGate,
                                     cirq.SwapPowGate, unroll_circuit_op=False),
            'toffoli': cirq.Gateset(cirq.CCNOT, cirq.TOFFOLI, cirq.CCXPowGate, cirq.CCNOT**-1, cirq.TOFFOLI**-1, unroll_circuit_op=False)
        }
        return gate_complexity(stc, gatesets)

    
    #@cachetools.cached(cachetools.LRUCache(128), key=_get_hash, info=True)
    def _gate_complexity_for_gate_or_op(self, gate_or_op: Union[cirq.Gate, cirq.Operation, qualtran.Bloq]):
        strategies = [
            self._has_gate_complexity,
            self._is_in_gatesets,
            self._from_cirq_decomposition,
        ]
        return self._gate_complexity_from_strategies(gate_or_op, strategies)

    
    
def gate_complexity(stc: Any, gatesets) -> Optional[GateComplexity]:
    """Returns the TComplexity.

    Args:
        stc: an object to compute its GateComplexity.
        gatesets: dictionary of gate types: gatesets to count.
        
    Returns:
        The GateComplexity of the given object or None on failure.
    """
    
    if isinstance(stc, (cirq.Gate, cirq.Operation, qualtran.Bloq, cirq.Circuit)) and isinstance(stc, Hashable):
        ret = GateComplexity(gatesets)._gate_complexity_for_gate_or_op(stc)
    
    else:
        strategies = [
            GateComplexity(gatesets)._has_gate_complexity,
            GateComplexity(gatesets)._is_in_gatesets,
            GateComplexity(gatesets)._from_cirq_decomposition,
            GateComplexity(gatesets)._is_iterable,
        ]
        ret = GateComplexity(gatesets)._gate_complexity_from_strategies(stc, strategies)

    if ret is None:
        raise TypeError("couldn't compute GateComplexity of:\n" f"type: {type(stc)}\n" f"value: {stc}")
    return ret

    #gate_complexity.cache_clear = GateComplexity({})._gate_complexity_for_gate_or_op.cache_clear  # type: ignore[attr-defined]
    #gate_complexity.cache_info = GateComplexity({})._gate_complexity_for_gate_or_op.cache_info  # type: ignore[attr-defined]
    #gate_complexity.cache = GateComplexity({})._gate_complexity_for_gate_or_op.cache  # type: ignore[attr-defined]
