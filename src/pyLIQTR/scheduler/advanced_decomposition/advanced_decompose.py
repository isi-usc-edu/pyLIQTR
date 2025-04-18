"""
Copyright (c) 2024 Massachusetts Institute of Technology 
SPDX-License-Identifier: BSD-2-Clause
"""
import cirq
from pyLIQTR.scheduler.Instruction import CirqInstruction
import pyLIQTR.scheduler.sets as sets
from pyLIQTR.utils.circuit_decomposition import generator_decompose
from pyLIQTR.utils.printing import keep
import pyLIQTR.utils.global_ancilla_manager as gam
from pyLIQTR.gate_decomp.rotation_gates import T_COUNT_CONST, T_COUNT_SLOPE, T_COUNT_STD_DEV
from pyLIQTR.utils.resource_analysis import get_T_counts_from_rotations
import copy
import random
import numpy as np
from pyLIQTR.scheduler.scheduler import schedule_circuit, decomposition_protocol
from pyLIQTR.utils.circuit_decomposition import decompose_once
import time


'''

'''
class Decompose:
    """
    This class is intended as an abstract decomposition tool. It offers the efficiency of the hierarchical/recursive decomposition 
    protocol featured in the scheduler while allowing for custom analysis in place of scheduling. This tool accepts (at minimum) a 
    circuit, decomposition level, and an analysis type. It returns a circuit decomposed to the specified level, with each operator 
    'tagged' with the desired analysis result.

    All core decomposition protocols are dictated by the Decompose class, while attributes specific to each analysis type are written 
    in a unique child class. Analysis child classes inherit most of their functionality from Decompose, but must have a few key functions defined:
    * self.analysis
    * self.convert_rotations
    * self.finalize

    Essentially, any Decompose child class should clearly define two components:
    * Resource Determination: how do we determine what resources are used by a single operator?
    * Resource Accumulation: how do we compile the resources used by all operators?

    :param cirq.Circuit input_circuit: circuit for decomposition.
    :param int decomp_level: how many levels to fully decompose BEFORE analysis. Output circuit will also be at this level.

    Optional inputs:

    :param decomposition_protocol decomposition_protocol: whether to decompose recursively or hierarchically.
    :param cirq.DecompositionContext context: qubit manager
    :param bool rotations_allowed: whether to include rotation gates in final counts or decompose them into Clifford+T.
    :param float precision: precision for rotation decomposition, if applicable.

    :returns: an instance of the Decompose() class.

    :rtype: Decompose
    """

    def __init__(self, input_circuit, decomp_level, decomposition_protocol=decomposition_protocol.recursive, recursion_level=0, context=None, rotations_allowed=False, precision=1e-10):
        self.input_circuit = input_circuit
        self.decomp_level = decomp_level
        self.decomposition_protocol = decomposition_protocol
        self.analysis_type = None
        self.output_circuit = cirq.Circuit()
        self.recursion_level = recursion_level
        self.rotations_allowed = rotations_allowed
        self.precision = precision
        self.custom_gateset = None
        self.operator_cache = {}
        self.results = None
        self.context = context
        if self.context is None:
            self.context = cirq.DecompositionContext(gam.gam)

    def recursive(self, operation: CirqInstruction):
        '''
        Recursive decomposition routine. A complex operator that has not been previously encountered will be
        decomposed recursively, one level at a time, with each sub-operator also being checked against the existing
        operator cache. The results of 'analysis' will be stored with each operator at each level.
        '''
        found = False
        for op in self.operator_cache:
            if found == False:
                if op == operation:
                    self.operator_cache[op][1] += 1
                    found = True
                    return self.operator_cache[op][0]
        if found == False:
            op_circuit = cirq.Circuit(operation._op)
            rd = self.analysis_type(op_circuit, decomp_level=1, decomposition_protocol=self.decomposition_protocol, recursion_level=self.recursion_level+1, context=self.context, rotations_allowed=self.rotations_allowed, custom_gateset=self.custom_gateset)
            rd.decompose()
            result = rd.results
            self.operator_cache[operation] = [result, 1]
            for op in rd.operator_cache:
                if op not in self.operator_cache:
                    self.operator_cache[op] = rd.operator_cache[op]
            return result

    def hierarchical(self, operation: CirqInstruction):
        '''
        Hierarchical decomposition routine. A complex operator that has not been previously encountered will be
        decomposed fully, i.e.; down to only basic gates. The results of 'analysis' will be stored with the top
        level operator.
        '''
        found = False
        for op in self.operator_cache:
            if found == False:
                if op == operation:
                    print(True)
                    self.operator_cache[op][1] += 1
                    found = True
                    return self.operator_cache[op][0]
        if found == False:
            op_circuit = cirq.Circuit(operation._op)
            op_circuit = cirq.align_left(cirq.Circuit(generator_decompose(op_circuit,\
                                               keep=keep, \
                                                on_stuck_raise = None, \
                                                context=self.context,\
                                                max_decomposition_passes=1)))

            rd = self.analysis_type(op_circuit, decomp_level=0, decomposition_protocol=self.decomposition_protocol, recursion_level=self.recursion_level+1, context=self.context, rotations_allowed=self.rotations_allowed, custom_gateset=self.custom_gateset)
            rd.decompose()
            result = rd.results
            self.operator_cache[operation] = [result, 1]
            return result

    def operation_tagger(self, operation: cirq.Operation, tag):
        tagged_operation = cirq.TaggedOperation(operation, tag)
        return tagged_operation
    
    def analysis(self, op: CirqInstruction):
        '''
        This function should be implemented in each Decompose child class individually.
        This function dictates what happens when analysis is called on a particular operator. It should take in an operator (of type Instruction), perform whatever 
        analysis is needed, and return a result (usually a dict) for the operator to be tagged with. This function should also modify/append the overall 
        results dictionary as needed.
        '''
        raise NotImplementedError()

    def decompose(self):
        '''
        This is the main function of the Decompose class. Each operator at the designated decomposition level will be
        analyzed, tagged, and appended to the output circuit.
        '''

        def convertMe(circuit, decomp_level=None):
            #A deep copy MUST be done here in order to keep qubit naming conventions consistent.
            tcontext = copy.deepcopy(self.context)
            for op in generator_decompose(circuit, keep=keep, on_stuck_raise = None, context=tcontext,max_decomposition_passes=decomp_level):
                yield op

        for op in convertMe(self.input_circuit, decomp_level=self.decomp_level):
            inst = CirqInstruction(op, self.custom_gateset)
            inst.set_complexity()
            tag = self.analysis(inst)
            if tag is not None:
                tagged_op = self.operation_tagger(inst._op, tag)
            else:
                tagged_op = inst._op
            if self.recursion_level == 0:
                self.output_circuit.append(tagged_op)

        self.finalize()
    
    def convert_rotations(self):
        '''
        This function should be implemented in each Decompose child class individually.
        This function defines how rotations are handled if the user sets rotations_allowed = False. If this has no effect 
        on the analysis results, it can be left undefined or set to Pass.
        '''
        pass
    
    def finalize(self):
        '''
        This function should be implemented in each Decompose child class individually.
        This function specifies any operations that must be done AFTER analysis of all operators has been completed. 
        If there is no need for final operations, it can be left undefined or set to Pass.
        '''
        pass


class GateCountingDecompose(Decompose):
    """
    Returns a circuit at the defined decomposition level with each operator tagged by its individual sub-gate counts (in terms of either a default set of basic gates or a user-defined set of gates) AND the overall circuit gate counts.

    :param cirq.Circuit input_circuit: circuit for decomposition.
    :param int decomp_level: how many levels to fully decompose BEFORE analysis. Output circuit will also be at this level.

    Optional inputs:

    :param decomposition_protocol decomposition_protocol: whether to decompose recursively or hierarchically.
    :param cirq.DecompositionContext context: qubit manager
    :param bool rotations_allowed: whether to include rotation gates in final counts or decompose them into Clifford+T.
    :param float precision: precision for rotation decomposition, if applicable.

    :returns: an instance of the Decompose() class.

    :rtype: Decompose
    """
    def __init__(self, input_circuit, decomp_level, decomposition_protocol=decomposition_protocol.recursive, recursion_level=0, context=None, rotations_allowed=False, precision=1e-10, custom_gateset=None):
        super().__init__(input_circuit, decomp_level, decomposition_protocol, recursion_level, context, rotations_allowed, precision)
        self.analysis_type = GateCountingDecompose
        self.results = {
                'T': 0,
                'H': 0,
                'S': 0,
                'CX': 0,
                'CZ': 0,
                'Pauli (X, Y, Z)': 0,
                'Rotation': 0
            }
        self.custom_gateset = custom_gateset
        
    def analysis(self, op: CirqInstruction):
        '''
        This function dictates what happens when analysis is called on a particular operator. It should take in an operator (of type Instruction), perform whatever 
        analysis is needed, and return a result (usually a dict) for the operator to be tagged with. This function should also modify/append the overall 
        results dictionary as needed.
        '''
        
        if op.complex:
            if self.decomposition_protocol == decomposition_protocol.recursive:
                counts = self.recursive(op)
            else:
                counts = self.hierarchical(op)
            
            for gate_type in counts:
                self.results[gate_type] += counts[gate_type]

            return counts

            # not complex:
        else:
            if isinstance(op._op, cirq.ClassicallyControlledOperation):
                operator = op._op.without_classical_controls()
            else:
                operator = op._op
            first_two = str(operator)[:2]

            added = False
            #custom gateset
            if self.custom_gateset is not None:
                for set in self.custom_gateset:
                    if added == False:
                        if first_two in set:
                            self.results[set[-1]] += 1
                            added = True
            
            else:
                for set in [sets.T, sets.H, sets.S, sets.CX, sets.CCX, sets.CZ, sets.PAULI, sets.ROT]:
                    if first_two in set:
                        self.results[set[-1]] += 1

            return None

    def convert_rotations(self):
        '''
        This function defines how rotations are handled if the user sets rotations_allowed = False. If this has no effect 
        on the analysis results, it can be left undefined or set to Pass.
        '''

        num_rotations = self.results['Rotation']
        sub_t = get_T_counts_from_rotations(num_rotations, gate_precision=self.precision)
        self.results['T'] += sub_t
        self.results['H'] += 2*sub_t + 1

        self.results.pop('Rotation')

    def finalize(self):
        '''
        This function specifies any operations that must be done AFTER analysis of all operators has been completed. 
        If there is no need for final operations, it can be left undefined or set to Pass.
        '''
        
        if self.rotations_allowed == False:
            self.convert_rotations()


class SchedulingDecompose(Decompose):
    """
    Returns a circuit at the defined decomposition level with each operator tagged by its individual scheduled time AND the overall circuit scheduling results

    :param cirq.Circuit input_circuit: circuit for decomposition.
    :param int decomp_level: how many levels to fully decompose BEFORE analysis. Output circuit will also be at this level.

    Optional inputs:

    :param decomposition_protocol decomposition_protocol: whether to decompose recursively or hierarchically.
    :param cirq.DecompositionContext context: qubit manager
    :param bool rotations_allowed: whether to include rotation gates in final counts or decompose them into Clifford+T.
    :param float precision: precision for rotation decomposition, if applicable.

    :returns: an instance of the Decompose() class.

    :rtype: Decompose
    """
    def __init__(self, input_circuit, decomp_level, decomposition_protocol=decomposition_protocol.recursive, recursion_level=0, context=None, rotations_allowed=False, precision=1e-10, custom_gateset=None, architecture_config=None):
        super().__init__(input_circuit, decomp_level, decomposition_protocol, recursion_level, context, rotations_allowed, precision)
        self.analysis_type = SchedulingDecompose
        self.custom_gateset = custom_gateset
        self.architecture_config = architecture_config
        self.qubits = []
        self.results = {
            'Scheduled Time': 0,
            'T-depth': 0
        }

    def analysis(self, op: CirqInstruction):
        found = False
        for inst in self.operator_cache:
            if found == False:
                if inst == op:
                    self.operator_cache[inst][1] += 1
                    result = self.operator_cache[inst][0]
                    found = True
            
                    for qubit in op.qubits:
                        if qubit not in self.qubits:
                            self.qubits.append(qubit)

                    return result
        
        if found == False:
            op_circuit = cirq.Circuit(op._op)
            result = schedule_circuit(op_circuit, decomposition_protocol=self.decomposition_protocol, architecture_config=self.architecture_config, context=self.context, custom_gateset=self.custom_gateset, rotation_gate_precision=self.precision)

            self.results['Scheduled Time'] += result['Total time for execution']
            self.results['T-depth'] += result['Circuit T-depth']
            for qubit in op.qubits:
                if qubit not in self.qubits:
                    self.qubits.append(qubit)

            self.operator_cache[op] = [{'Scheduled Time': result['Total time for execution']}, 1]
            tag = {'Scheduled Time': result['Total time for execution'], 'Qubits': result['Number of qubits used'], 'T-depth': result['Circuit T-depth']}
            return tag


    def finalize(self):
        self.results['Qubits'] = len(self.qubits)


class GatesetDecompose(Decompose):
    """
    Takes in a set of gate types defined by the user, returns a decomposed circuit only in terms of those gate types.

    :param cirq.Circuit input_circuit: circuit for decomposition.
    :param int decomp_level: how many levels to fully decompose BEFORE analysis. Output circuit will also be at this level.

    Optional inputs:

    :param decomposition_protocol decomposition_protocol: whether to decompose recursively or hierarchically.
    :param cirq.DecompositionContext context: qubit manager
    :param bool rotations_allowed: whether to include rotation gates in final counts or decompose them into Clifford+T.
    :param float precision: precision for rotation decomposition, if applicable.

    :returns: an instance of the Decompose() class.

    :rtype: Decompose
    """
    def __init__(self, input_circuit, decomp_level, decomposition_protocol=decomposition_protocol.recursive, recursion_level=0, context=None, rotations_allowed=False, precision=1e-10, custom_gateset=None):
        super().__init__(input_circuit, decomp_level, decomposition_protocol, recursion_level, context, rotations_allowed, precision)
        self.analysis_type = GatesetDecompose
        self.custom_gateset = custom_gateset
        self.results = cirq.Circuit()

        if sets.ROT in self.custom_gateset:
            self.rotations_allowed = True
        else:
            self.rotations_allowed = False

    def analysis(self, op: CirqInstruction):
        if op.complex:
            first_two = str(op)[:2]
            if first_two in sets.ROT:
                op.complex = False
                t_breakdown = self.convert_rotations()
                h_breakdown = 2*t_breakdown + 1
                for i in range(t_breakdown):
                    self.results.append(cirq.T.on(op.qubits[0]))
                for i in range(h_breakdown):
                    self.results.append(cirq.H.on(op.qubits[0]))
                return None
          
            if self.decomposition_protocol == decomposition_protocol.recursive:
                res = self.recursive(op)
            else:
                res = self.hierarchical(op)
            
            self.results.append(res)

            return res

            # not complex:
        else:
            if isinstance(op._op, cirq.ClassicallyControlledOperation):
                operator = op._op.without_classical_controls()
            else:
                operator = op._op
            first_two = str(operator)[:2]

            added = False
            #custom gateset
            if self.custom_gateset is not None:
                for set in self.custom_gateset:
                    if added == False:
                        if first_two in set:
                            self.results.append(op._op)
                            added = True
            
            else:
                for set in [sets.T, sets.H, sets.S, sets.CX, sets.CCX, sets.CZ, sets.PAULI, sets.ROT]:
                    if first_two in set:
                        self.results.append(op._op)

            return None
        

    def convert_rotations(self):

        random.seed(0)
        T_from_single_rotation = int(random.gauss(
                T_COUNT_SLOPE * np.log2(1 / self.precision) + T_COUNT_CONST,
                T_COUNT_STD_DEV,
            ))
        return T_from_single_rotation

    def finalize(self):
        
        if self.rotations_allowed == False:
            self.convert_rotations() 
    

    
def decompose(circuit: cirq.Circuit, analysis_method, decomposition_protocol=decomposition_protocol.recursive, decomposition_level=0, context=None, rotations_allowed=False, precision=1e-10, custom_gateset=None, return_op_cache=False):

    rd = analysis_method(circuit, decomp_level=decomposition_level, decomposition_protocol=decomposition_protocol, context=context, rotations_allowed=rotations_allowed, precision=precision, custom_gateset=custom_gateset)
    rd.decompose()
    
    if return_op_cache == True:
        for op in rd.operator_cache:
            print(f'{op._op.gate} occured {rd.operator_cache[op][1]} times and had the following cached results: {rd.operator_cache[op][0]}.')
        return (rd.output_circuit, rd.results)
    else:
        if analysis_method == GatesetDecompose:
            return rd.results
        else:
            return (rd.output_circuit, rd.results)
    
    #call recursive decomp class here

