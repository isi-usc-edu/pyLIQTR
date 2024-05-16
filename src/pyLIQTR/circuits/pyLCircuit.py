"""
DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.

This material is based upon work supported by the Under Secretary of Defense for
Research and Engineering under Air Force Contract No. FA8702-15-D-0001. Any opinions,
findings, conclusions or recommendations expressed in this material are those of the
author(s) and do not necessarily reflect the views of the Under Secretary of Defense
for Research and Engineering.

© 2023 Massachusetts Institute of Technology.

The software/firmware is provided to you on an As-Is basis

Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS Part
252.227-7013 or 7014 (Feb 2014). Notwithstanding any copyright notice, U.S. Government
rights in this work are defined by DFARS 252.227-7013 or DFARS 252.227-7014 as detailed
above. Use of this work other than as specifically authorized by the U.S. Government
may violate any copyrights that exist in this work.
"""

import cirq
import random
import math
from enum import Enum

from pyLIQTR.circuits.pyLOperator import pyLOperator
from pyLIQTR.utils.circuit_decomposition import decompose_op
from pyLIQTR.gate_decomp.cirq_transforms import clifford_plus_t_direct_transform
from pyLIQTR.gate_decomp.rotation_gates import decomp_mixin, T_COUNT_CONST, T_COUNT_SLOPE, T_COUNT_STD_DEV
from pyLIQTR.utils.printing import openqasm

from qualtran import _infra, bloqs
import qualtran.cirq_interop.testing as qt_test
import qualtran.cirq_interop.t_complexity_protocol as t_complexity

# class syntax
class RESOURCE_ANALYSIS_MODE(Enum):
    Exact = 0
RESOURCE_ANALYSIS_MODE = Enum('ResourceAnalysisMode', ['Exact'])

# functional syntax
class pyLCircuit (cirq.Circuit):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._resources = {}
        self._pcft_resources = {}
        self._operator_cache = []
        self._operator_cached_results = []
        self._operator_decomp_cache = {}
        self._current_decomp_level = 0
        self._max_decomp_level = 0

    def _mycopy_(self,other):
        super().__init__(other)
    

    def save(self,mode):
        raise NotImplementedError("IMPLEMENT ME")
    
    def load(self,filepath):
        raise NotImplementedError("IMPLEMENT ME")

    @property
    def decomp_level(self):
        return self._current_decomp_level

    @property
    def resources(self):
        tmp = {}
        tmp["CliffT"] = self._resources
        tmp["BeforeCliffT"] = self._pcft_resources
        tmp["Qubits"] = len(self.all_qubits())
        return tmp
    
    def clear_resources(self):
        self._resources = {}
        self._pcft_resources = {}

    @resources.setter
    def resources(self, rsrcs):
        #print('in resource setter')
        #print(rsrcs)
        #rint(self._resources)
        for key in rsrcs:
            pcft = False
            #cleanup
            corrected_key = key
            corrected_key = corrected_key.replace("**-1.0","").replace("**-1","")
            if "PRECLIFFT_" in corrected_key:
                corrected_key = corrected_key.replace("PRECLIFFT_","")
                pcft = True

            if corrected_key.upper() == "CNOT":
                corrected_key = "CX"
            if "Rx" in corrected_key:
                corrected_key = "Rx"
            if "Ry" in corrected_key:
                corrected_key = "Ry"
            if "Rz" in corrected_key:
                corrected_key = "Rz"

            # added a new sum of rotations
            if 'Rotations' in corrected_key:
                if "Rx" in rsrcs:
                    self._resources['Rotations'] += self._resources['Rx']
                if "Ry" in rsrcs:
                    self._resources['Rotations'] += self._resources['Ry']
                if "Rz" in rsrcs:
                    self._resources['Rotations'] += self._resources['Rz']
            if not pcft:
                if corrected_key in self._resources:
                    self._resources[corrected_key] += rsrcs[key]
                else:
                    self._resources[corrected_key] = rsrcs[key] 
            else:
                if corrected_key in self._pcft_resources:
                    self._pcft_resources[corrected_key] += rsrcs[key]
                else:
                    self._pcft_resources[corrected_key] = rsrcs[key]  
        # print(self._resources)


    def get_decomp_levels(self):
        #default decomp level is always zero (aka, were a circuit of just gates)
        self._max_decomp_level = 0
        moments = self.findall_operations(self._is_pyLOperator)
        for moment in moments:
            moment = moment[0]
            for opIdx,__ in enumerate(self[moment].operations):
                if self._is_pyLOperator(self[moment].operations[opIdx]):
                    if isinstance(self[moment].operations[opIdx]._gate, cirq.ops.raw_types._InverseCompositeGate):
                        if self[moment].operations[opIdx]._gate._original.total_decomp > self._max_decomp_level:
                            self._max_decomp_level = self[moment].operations[opIdx]._gate._original.total_decomp
                    else:
                        if self[moment].operations[opIdx]._gate.total_decomp > self._max_decomp_level:
                            self._max_decomp_level = self[moment].operations[opIdx]._gate.total_decomp
        #have max level, now need to readjust all operators and update their decomp level
        #need to recall the generator
        """
        moments = self.findall_operations(self._is_pyLOperator)
        for moment in moments:
            moment = moment[0]
            for opIdx,__ in enumerate(self[moment].operations):
                if self._is_pyLOperator(self[moment].operations[opIdx]):
                    self[moment].operations[opIdx]._gate.total_decomp = self._max_decomp_level
        """
        self._current_decomp_level = self._max_decomp_level
    
    def view_decomp_levels(self):
        print("Circuit @ decomp level = {}".format(self.decomp_level))
        for x in self.all_operations():
            if self._is_pyLOperator(x):
                if isinstance(x._gate, cirq.ops.raw_types._InverseCompositeGate):
                    print("inverse({})\n\t @ Level = {}".format(str(x._gate._original),x._gate._original.total_decomp))
                else:
                    print("{}\n\t @ Level = {}".format(str(x),x._gate.total_decomp))

    def _decomp(self,op):
        #yield from decompose_op
        #return
        if str(op) in self._operator_decomp_cache:
            yield self._operator_decomp_cache[str(op)]
        else:
            self._operator_decomp_cache[str(op)] = decompose_op(op)
            yield self._operator_decomp_cache[str(op)]

    def _is_pyLOperator(self,operation):
        try:
            if isinstance(operation._gate, cirq.ops.raw_types._InverseCompositeGate):
                return isinstance(operation._gate._original,pyLOperator)
            else:
                return isinstance(operation._gate,pyLOperator)
        except:
            return False
    
    def _isRotation(self, x):
        return True if ("Rx" in str(x) or "Ry" in str(x) or "Rz" in str(x)) else False
    
    def _get_cached_count(self,g):
        return self._operator_cached_results[self._operator_cache.index(g)]
    
    def _do_resource_analysis(self,operation,mode):
        if mode == RESOURCE_ANALYSIS_MODE.Exact:
            if isinstance(operation._gate, cirq.ops.raw_types._InverseCompositeGate):
                #print("\tinverse")
                isCached =  operation._gate._original in self._operator_cache
                if not isCached:
                    #print("\t\tuncached {}".format(str(operation)))
                    rsrcs = operation._gate._original.count_exact()
                    self.resources = rsrcs
                    self._operator_cached_results.append(rsrcs)
                    #print("\t\t\t{}".format(rsrcs))
                    self._operator_cache.append(operation._gate._original)
                    #print("\t\t\t{}".format(rsrcs))
                else:
                    #print("\t\tcached {}".format(str(operation)))
                    self.resources = self._get_cached_count(operation._gate._original)
                    #print("\t\t\t{}".format(self._get_cached_count(operation._gate._original)))
            else:
                #print("\tog")
                isCached =  operation._gate in self._operator_cache
                if not isCached:
                    #print("\t\tuncached {}".format(str(operation)))
                    rsrcs = operation._gate.count_exact()
                    self.resources = rsrcs
                    self._operator_cached_results.append(rsrcs)
                    #print("\t\t\t{}".format(rsrcs))
                    self._operator_cache.append(operation._gate)
                else:
                    #print("\t\tcached {}".format(str(operation)))
                    self.resources = self._get_cached_count(operation._gate)
                    #print("\t\t\t{}".format(self._get_cached_count(operation._gate)))
        else:
            raise NotImplementedError


    def decompose(self,level = -1):
        #decompose up to the chosen_level, w
        if level == -1:
            #fully decompose circuit
            level = 0
        elif level > self.decomp_level:
            raise ValueError("Cannot 'recompose' circuit, can only further decompose")
        elif level < 0:
            raise ValueError("Cannot decompose past base level of circuit decomposition")
        
        while True:
            if self.decomp_level == level:
                break
            else:
                self._current_decomp_level -= 1
                #Making use of cirqs map_operations and unroll function, 
                #We unfortunately need to have this function redfined each time we loop through
                #since we cant seem to pass multiple arguments to it.
                def on_the_fly_decomp(op: cirq.Operation, _: int) -> cirq.OP_TREE:
                    if self._is_pyLOperator(op):
                        if isinstance(op._gate, cirq.ops.raw_types._InverseCompositeGate):
                            if op._gate._original.total_decomp >= level:
                                yield from self._decomp(op)
                            else:
                                yield op
                        else:
                            if op._gate.total_decomp >= level:
                                yield from self._decomp(op)
                            else:
                                yield op
                    else:
                        if level==0:
                            yield cirq.align_left(clifford_plus_t_direct_transform(cirq.Circuit(op),gate_precision=1e-8,use_rotation_decomp_gates=True))
                        else:
                            yield op
                self._mycopy_(cirq.map_operations_and_unroll(self,on_the_fly_decomp))
        
        
    
    def resourceAnalyze(self,rotation_gate_precision=1e-8,circuit_precision=None,mode=RESOURCE_ANALYSIS_MODE.Exact,decompose_rotations=True):
        # NOTE: Currently only estimates resources from rotation gates based on precision, ie does not generate explicit circuits and count
        self.clear_resources()
        for x in self.all_operations():
            if self._is_pyLOperator(x):
                self._do_resource_analysis(x,mode)
            elif isinstance(x._gate, _infra.gate_with_registers.GateWithRegisters):
                # Handle Complex (cirq-ft) Gate/Qubits operations
                # Hold off on calling the cirq.decompose and simply run the cirq_ft.t_complexity
                # then populate self.resources intelligently from that result
                # tmp = cirq.Circuit(x)
                # tmp = tmp._decompose_(self=tmp)
                gate_dict = {"PRECLIFFT_{}".format(str(x.gate)):1}

                t = t_complexity.t_complexity(x)

                # estimate gate counts based on number of rotations, but only if we don't want rotations
                if decompose_rotations:
                    if circuit_precision is not None:
                        Ts_from_rotations = get_T_counts_from_rotations(t.rotations,circuit_precision=circuit_precision)
                    else:
                        Ts_from_rotations = get_T_counts_from_rotations(t.rotations,gate_precision=rotation_gate_precision)
                    gate_dict["T"] = t.t + Ts_from_rotations
                    gate_dict["Clifford"] = t.clifford + 2*Ts_from_rotations + 1
                else:
                    gate_dict["Rotations"] = t.rotations
                    gate_dict["T"] = t.t
                    gate_dict["Clifford"] = t.clifford


                self.resources = gate_dict
            elif isinstance(x._gate, cirq.QuantumFourierTransformGate):
                # Need to add some specific logic for this as currently cirq doesn't have a way to calculate the resources of it's own QFT
                gate_dict = {"PRECLIFFT_{}".format(str(x.gate)):1}

                t = t_complexity.t_complexity(x)

                # estimate gate counts based on number of rotations, but only if we don't want rotations
                if decompose_rotations:
                    if circuit_precision is not None:
                        Ts_from_rotations = get_T_counts_from_rotations(t.rotations,circuit_precision=circuit_precision)
                    else:
                        Ts_from_rotations = get_T_counts_from_rotations(t.rotations,gate_precision=rotation_gate_precision)
                    gate_dict["T"] = t.t + Ts_from_rotations
                    gate_dict["Clifford"] = t.clifford + 2*Ts_from_rotations + 1
                else:
                    gate_dict["Rotations"] = t.rotations
                    gate_dict["T"] = t.t
                    gate_dict["Clifford"] = t.clifford

                self.resources = gate_dict
            elif isinstance(x._gate,decomp_mixin):
                # case for when rotation decomp gate is passed
                gate_dict = {"PRECLIFFT_{}".format(str(x.gate)):1}

                t_count = x._gate.num_T_gates

                gate_dict["T"] = t_count
                gate_dict["Clifford"] = 2*t_count + 1

                self.resources = gate_dict
            else:
                # Handle Simple Gate/Qubit operations
                tmp = cirq.Circuit(x)
                # these should be single qubit gates before going to clifford_plus_t_direct_transform()
                if circuit_precision is not None:
                    tmp = clifford_plus_t_direct_transform(tmp,circuit_precision=circuit_precision,use_random_decomp=True,use_rotation_decomp_gates=True)
                else:
                    tmp = clifford_plus_t_direct_transform(tmp,gate_precision=rotation_gate_precision,use_random_decomp=True,use_rotation_decomp_gates=True)

                gate_dict = {"PRECLIFFT_{}".format(str(x.gate)):1}
                for xnu in tmp.all_operations():
                    gateStr = str(xnu.gate)
                    if gateStr not in gate_dict:
                        gate_dict[gateStr] = 1
                    else:
                        gate_dict[gateStr] += 1
                self.resources = gate_dict

        return self.resources

    def to_openqasm(self,use_rotation_decomp_gates=False):
        yield from to_openqasm(self,use_rotation_decomp_gates=use_rotation_decomp_gates)


def get_T_counts_from_rotations(num_rotation_gates,gate_precision=1e-8,circuit_precision=None):
    
    if circuit_precision is not None:
        precision = (circuit_precision / num_rotation_gates)
    else:
        precision = gate_precision

    T_from_single_rotation = int(random.gauss(
                T_COUNT_SLOPE * math.log2(1 / precision) + T_COUNT_CONST,
                T_COUNT_STD_DEV,
            ))
    return T_from_single_rotation*num_rotation_gates
    
