import cirq
import numpy as np
import random
import warnings
from tqdm import tqdm
import numpy as np
import warnings
import json

from pyLIQTR.utils.circuit_decomposition import circuit_decompose_multi, generator_decompose
from pyLIQTR.gate_decomp.rotation_gates import T_COUNT_CONST, T_COUNT_SLOPE, T_COUNT_STD_DEV
from pyLIQTR.scheduler.Instruction import CirqInstruction, OQInstruction, Instruction
from pyLIQTR.scheduler.DAG import DAG
import pyLIQTR.scheduler.sets as sets
from pyLIQTR.scheduler.sets import op_to_openqasm
from pyLIQTR.scheduler.scheduler_utils import architecture, state_factory
from rustworkx.visualization import graphviz_draw
from pyLIQTR.utils.printing import keep
import pyLIQTR.utils.global_ancilla_manager as gam
from pyLIQTR.utils.circuit_decomposition import generator_decompose
from pyLIQTR.gate_decomp.cirq_transforms import _perop_clifford_plus_t_direct_transform
from pyLIQTR.gate_decomp.cirq_transforms import determine_gate_precision
import copy

from pyLIQTR.utils.resource_analysis import pylqt_t_complexity as t_complexity
from pyLIQTR.circuits.operators.AddMod import AddMod as pyLAM
    

from enum import Enum

class decomposition_protocol(Enum):
    hierarchical = 1
    recursive = 2
class Scheduler:
    """
    A class for 'scheduling' a given circuit and returning execution time and a number of other gate and circuit analysis metrics. 
    """
    
    def __init__(self, architecture_params: architecture, precision=5e-6, recursion_level=0, decomposition_protocol=decomposition_protocol.recursive, context=None, qasm_input=False, qasm_output=False):

        '''
        :param int global_time: tracks time as instructions execute and finish.
        :param dict architecture_params: an architecture class object.
        :param list ready_q: list of instructions most recently pulled from the DAG. These instructions could potentially be executed but have not yet been checked for qubit/resource availability.
        :param list execution_q: list of instructions that will be next to execute. These instructions have been checked for qubit/resource availability.
        :param list free_q: list of instructions most recently executed.
        :param int cycle: Denotes the number of 'scheduler cycles' completed: [instructions pulled from the DAG, instructions executed, instructions freed] is one cycle.
        :param int t_depth: keeps track of the number of timesteps which include a T-gate.
        :param dict t_width: a distribution of the number of T-gates occuring within a single timestep.
        :param dict t_proportion: a distribution of the proportion of T-gates to all other gates within a single timestep.
        :param dict step_tracking: the number of Clifford gates and number of T-gates in each scheduler cycle.
        :param list qubits: list of all qubits which have been encountered during scheduling. This includes ancilla generated during recursion.
        :param dict active_qubits: a distrubtion of the number of qubits being acted on within a single timestep.
        :param float precision: decomposition precision used to decompose rotation gates into T gates.
        :param dict gate_profile: a dictionary tracking each type of gate to be counted and how many times it occurs.
        :param dict op_cache: if hierarchical or recursive decomposition is being used, this dictionary caches the execution time, gate counts, qubit counts, etc. for individual operators to be used when the operator is encountered again.
        :param int recursion_level: how many levels down from the defined top level the recursive decomposition has gone. Only applicable when recursive decomposition is being used.
        :param bool rotations_allowed: if True, rotations will not be deocomposed into T-gates, but rather counted in the 'Rotation' entry of gate_profile. If False, rotations will be decomposed using rotations_to_t.
        :param cirq.DecompositionContext context: cirq decomposition context.
        '''
        self.global_time = 0
        self.arch = architecture_params
        self.arch_copy = copy.deepcopy(architecture_params)
        self.ready_q = []
        self.execution_q = []
        self.free_q = []
        self.cycle = 0
        self.t_depth = 0
        self.t_width = {}
        self.t_proportions = {}
        self.step_tracking = {}
        self.qubits = {}
        self.active_qubits = {}
        self.idle_qubits = {}
        self.idles = 0
        self.precision = precision
        self.gate_profile = {}
        self.op_cache = {}
        self.recursion_level = recursion_level
        self.decomposition_protocol = decomposition_protocol
        self.rotations_allowed = False
        self.qasm_input = qasm_input
        self.qasm_output = qasm_output
        self.context = context
        if self.context is None:
            new_gam = gam.gam
            new_gam.reset()
            self.context = cirq.DecompositionContext(new_gam)
        
        #initialize gate counting dict
        if self.arch is None:
            self.arch = architecture()
    
        for set in self.arch.gateset:
            self.gate_profile[set.name()] = 0
        if sets.ROT in self.arch.gateset:
            self.rotations_allowed = True

        if self.qasm_input == True:
            self.gate_profile['Other'] = 0
            self.gate_profile['CY'] = 0
            self.gate_profile['CS'] = 0

        self.arch.precision = self.precision
    




    def recursion_routine(self, inst:CirqInstruction):
        found = False
        for op in self.op_cache:
            if found == False:
                if inst == op:
                    setattr(inst, 'cached_schedule', self.op_cache[op][0])
                    self.op_cache[op][1] += 1
                    found = True
                    return inst.cached_schedule['Execution time']
                    
        #if inst in self.op_cache:
        #    setattr(inst, 'cached_schedule', self.op_cache[inst][0])
        #    self.op_cache[inst][1] += 1
        #    return inst.cached_schedule['Execution time']
        if found == False:
            #essentially a repeat of the schedule_circuit function below, with a few small changes for recursion
            circuit = cirq.Circuit(inst._op)

            r_DAG = DAG(max_moments=1000000)
            r_arch = copy.deepcopy(self.arch_copy)
            right_context = isinstance(self.context.qubit_manager, gam.GlobalQubitManager)
            r_Sched = Scheduler(architecture_params=r_arch, recursion_level=(self.recursion_level+1), context=self.context, decomposition_protocol=self.decomposition_protocol)
                
            def convertMe(circuit, decomp_level=None):
                #A deep copy MUST be done here in order to keep qubit naming conventions consistent.
                for op in generator_decompose(circuit, keep=keep, on_stuck_raise = None, context=self.context,max_decomposition_passes=decomp_level):
                    yield op
            #assuming op is in defined gateset, whether default or custom:
            try:
                for q in circuit.all_qubits():
                    r_Sched.qubits[q] = 1
                try:
                    for op in convertMe(circuit, decomp_level=1):
                        for q in op.qubits:
                            if q not in r_Sched.qubits:
                                r_Sched.qubits[q] = 1  
                        #we must reserve qubits if theyre ancilla, so they dont get reused and cause duplicate qid errors
                        if right_context:
                            self.context.qubit_manager.reserve_qubits(op.qubits, level=(self.recursion_level+1))
                        if r_DAG.full():
                            r_DAG.finish()
                            while r_Sched.schedule(r_DAG):
                                pass
                            global_time += r_Sched.global_time
                            r_DAG.dependency_log.clear()
                            r_DAG.dag_cycle += 1
                        r_DAG.add_dependency(CirqInstruction(op, custom_gateset=self.arch.gateset))

                    r_DAG.finish()
                    while r_Sched.schedule(r_DAG):
                        pass

                    if right_context:
                        self.context.qubit_manager.unlock_qubits(level=(self.recursion_level+1))
                    results_dict = {
                    'Execution time': r_Sched.global_time,
                    'T-depth': r_Sched.t_depth,
                    'T-width': r_Sched.t_width,
                    'T-proportion': r_Sched.t_proportions,
                    'Gates': r_Sched.gate_profile,
                    'Qubits': r_Sched.qubits,
                    'Active Qubits': r_Sched.active_qubits,
                    'Cache': r_Sched.op_cache
                    }
            
                    self.op_cache[inst] = [results_dict, 1]
                    for q in results_dict['Qubits']:
                        if q not in self.qubits:
                            self.qubits[q] = 1
                    for op in results_dict['Cache']:
                        if op not in self.op_cache:
                            self.op_cache[op] = results_dict['Cache'][op]
                    #throw all of our results for this op into our overall cache
                    setattr(inst, 'cached_schedule', results_dict)
                    return inst.cached_schedule['Execution time']
                
                except NotImplementedError:
                    ex_time = self.custom_gateset_decomposition_catchall(inst)
                    return ex_time  
            #oh no it's NOT in our gateset??? 
            except RecursionError:
                ex_time = self.custom_gateset_decomposition_catchall(inst)
                return ex_time

            

    def hierarchy_routine(self, inst:CirqInstruction):
        #check if it's in the cache already
        
        found = False
        for op in self.op_cache:
            if found == False:
                if inst == op:
                    setattr(inst, 'cached_schedule', self.op_cache[op][0])
                    self.op_cache[op][1] += 1
                    found = True
                    return inst.cached_schedule['Execution time']

        if found == False:
            #initialize circuit containing just the op
            circuit = cirq.Circuit(inst._op)

            #pull in all of the resource configuration parameters from the top-level scheduler
            #initialize DAG, scheduler, and context
            r_DAG = DAG(max_moments=1000000)
            r_arch = architecture
            r_Sched = Scheduler(architecture_params=self.arch, recursion_level=0, context=self.context, decomposition_protocol=self.decomposition_protocol)

            #same procedure as a full decomposition at the top level to get circuit in terms of basic gates
            level = 0
            prev_decomp = None
            while circuit != prev_decomp:
                prev_decomp = circuit
                circuit = circuit_decompose_multi(circuit, 1, context=self.context)
                for q in circuit.all_qubits():
                    if q not in r_Sched.qubits:
                        r_Sched.qubits[q] = 1
                level += 1

            #same procedure as schedule_circuit
            for op in circuit.all_operations():
                for q in op.qubits:
                    if q not in r_Sched.qubits:
                        r_Sched.qubits[q] = 1  
                if r_DAG.full():
                    r_DAG.finish()
                    while r_Sched.schedule(r_DAG):
                        pass
                    r_DAG.dependency_log.clear()
                    r_DAG.dag_cycle += 1
                r_DAG.add_dependency(CirqInstruction(op, custom_gateset=self.arch.gateset))

            r_DAG.finish()
            while r_Sched.schedule(r_DAG):
                pass
    
            results_dict = {
                'Execution time': r_Sched.global_time,
                'T-depth': r_Sched.t_depth,
                'T-width': r_Sched.t_width,
                'T-proportion': r_Sched.t_proportions,
                'Gates': r_Sched.gate_profile,
                'Qubits': r_Sched.qubits,
                'Active Qubits': r_Sched.active_qubits,
                'Cache': r_Sched.op_cache
            }
            
            self.op_cache[inst] = [results_dict, 1]
            for q in results_dict['Qubits']:
                if q not in self.qubits:
                    self.qubits[q] = 1
            #throw all of our results for this op into our overall cache
            setattr(inst, 'cached_schedule', results_dict)
            return inst.cached_schedule['Execution time']

            
    def custom_gateset_decomposition_catchall(self, inst:CirqInstruction):
        '''
        This function should only be accessed in the event that an error arises from recursive decomposition of a complex operator. It will determine whether the error was caused by a basic operator being left out of a user-defined gateset or an operation without a defined decomposition method. 
        In the event that the user-defined gateset is incomplete, this function will raise an error notifying them of the missing gate type.
        If the operator in question is NOT a basic operator but does not have a defined decomposition method, this function will access that operator's T-Complexity method and use those counts to add to the global time, gate counts, t-depth, etc. While this will not account for any parallelism, it will at least acknowledge the existence of these gates and their effect on the circuit. The user will be warned about this action.
        If the operator has NEITHER a decomposition method nor a defined T-Complexity, this function will cache an execution time of 0, gate counts of 0, etc since no information about makeup can be accessed. This will essentially exclude this operator from the results, but will allow the rest of the circuit to continue scheduling. The user will be warned about this action.
        '''
        
        #if it's a rotation, we assume user just wants rotations decomposed into T gates

        inst_str = str(inst)
        if inst.gateset == sets.ROT:
            inst.complex = False
            self.rotations_allowed = False
            sub_gates = self.rotation_to_t(inst, self.precision)
            cliff_time = self.arch.__getitem__('H', global_time=self.global_time)
            t_time = self.arch.__getitem__('T', global_time=self.global_time)
            total_rotation_time = (cliff_time * sub_gates[0]) + (t_time * sub_gates[1])

            return total_rotation_time
        
        #check if it's one of our common excepted bloqs
        if inst_str.startswith("Allocate") or inst_str.startswith('Free'):
            results_dict = {
                    'Execution time': 0,
                    'T-depth': 0,
                    'T-width': {},
                    'T-proportion': {},
                    'Gates': {'T': 0,
                            'H': 0,
                            'S': 0, 
                            'CX': 0,
                            'CZ': 0,
                            'Toffoli': 0,
                            'Pauli (X, Y, Z)': 0,
                            'Measurement/Reset': 0},
                    'Qubits': inst.qubits,
                    'Executions': {},
                    'Cache': {}
                }
            self.op_cache[inst] = [results_dict, 1]
            inst.cached_schedule = results_dict
            return 0
                    
        #check if it's a basic instruction that was left out of a custom gateset
        basic = False
        if inst.gateset is not None:
            basic = True
        #if it's not a basic instruction, then it either doesn't have a decomposition (not the user's fault) and/or it wasn't included in the custom gateset (user's fault)
        if basic==False:
            #if there's a custom gateset definied, we can't use our t complexity method because the custom gateset might not be in terms of t-rot-cliff. tell the user to fix it
            if self.arch.gateset != [sets.T, sets.H, sets.S, sets.CX, sets.CCX, sets.CZ, sets.PAULI, sets.MISC]:
                raise TypeError(f'{inst._op} does not have a defined decomposition and is not included in custom gateset.')
            
            if inst_str.startswith('ModAddK'):
                qbs = inst._op.qubits
                replacement_op = pyLAM(bitsize=inst._op.gate.bitsize, add_val=inst._op.gate.add_val, \
                           mod=inst._op.gate.mod,cvs=inst._op.gate.cvs).on(*qbs)
                inst._op = replacement_op
                ex_time = self.recursion_routine(inst)
                return ex_time
            #otherwise, we're gonna try to fix it
            #warnings.warn(f'{inst._op} does not have a defined decomposition- defaulting to defined T-Complexity. Scheduling results will not represent the most efficient circuit possible.')
            #basically the procedure here is grab the t-complexity (which ideally exists but we'll check for that), use those counts to sub for our gate counts and execution time, cache that result so we don't have to go through this again, and move on. 
            counts = t_complexity(inst._op)
            #oh no!!! qualtran didn't define a t_complexity for this operator??? shocker. set everything to 0, warn the user that the results aren't gonna be right, and move on.
            if counts.clifford<=0 and counts.rotations<=0 and counts.t<=0:
                #warnings.warn(f'{inst._op} does not have a defined T-Complexity. Operator will be skipped.')
                results_dict = {
                    'Execution time': 0,
                    'T-depth': 0,
                    'T-width': {},
                    'T-proportion': {},
                    'Gates': {'T': 0,
                            'H': 0,
                            'S': 0, 
                            'CX': 0,
                            'CZ': 0,
                            'Toffoli': 0,
                            'Pauli (X, Y, Z)': 0,
                            'Measurement/Reset': 0},
                    'Qubits': inst.qubits,
                    'Executions': {},
                    'Cache': {}
                }
                self.op_cache[inst] = [results_dict, 1]
                inst.cached_schedule = results_dict
                return 0

            sub_gates = self.rotation_to_t(inst, precision=self.precision)

            gate_profile = {
                'T': counts.t + (sub_gates[1] * counts.rotations),
                'H': counts.clifford + (sub_gates[0] * counts.rotations),
                'S': 0, 
                'CX': 0,
                'CZ': 0,
                'Toffoli': 0,
                'Pauli (X, Y, Z)': 0,
                'Measurement/Reset': 0
            }

            
            t_time = self.arch.__getitem__('T', global_time=self.global_time)
            cliff_time = self.arch.__getitem__('H', global_time=self.global_time)
            ex_time = (t_time * gate_profile['T']) + (cliff_time * gate_profile['H'])    

            inst.complex = True
            results_dict = {
                    'Execution time': ex_time,
                    'T-depth': gate_profile['T'],
                    'T-width': {},
                    'T-proportion': {},
                    'Gates': gate_profile,
                    'Qubits': inst.qubits,
                    'Executions': {},
                    'Cache': {}
            }

            self.op_cache[inst] = [results_dict, 1]
            inst.cached_schedule = results_dict
            return ex_time


        #if we get all the way to here, it means our op is actually a basic operator that the user forgot to include in their custom gateset. tell them to fix it
        raise TypeError(f'Custom gateset is incomplete: {inst._op} cannot be decomposed or classified as an operation in custom gateset.')
    
    def assign_execution_time(self,inst):
        """
        Given the instruction, assign an execution time for it. 
        """
        # is this a basic instruction or one that needs to be broken down?
        if inst.gateset is not None:
            time = self.arch.__getitem__(inst.gateset.name(), self.global_time)
            return time
        
        if inst.complex:
            if self.decomposition_protocol == decomposition_protocol.recursive:
                ex_time = self.recursion_routine(inst)
            else:
                ex_time = self.hierarchy_routine(inst)
            return ex_time
        
    

    def resource_available(self, inst):
        """
        Given an instruction, determine if we have resources (qubit is not being used by another operation, it doesn't violate restrictions on simultaneous T gates, etc.) available to run it. 
        """
        availability= True
        resources_needed = inst.resource_counts()

        # check each resource needed against the resources we've defined as available
        # if it's available or undefined(unlimited), return true
        for resource_type in resources_needed:
            if resource_type in self.arch.limits:
                if resource_type == 'Rotation':
                    if 'Clifford' in self.arch.limits:
                        cliff_type = 'Clifford'
                    elif 'H' in self.arch.limits:
                        cliff_type = 'H'
                    if (self.arch.limits['T'] >= resources_needed[resource_type]) and (self.arch.limits[cliff_type] >= resources_needed[resource_type]):
                        pass
                    else:
                        return False
                else:
                    if self.arch.limits[resource_type] >= resources_needed[resource_type]:
                        pass
                    else:
                        return False
            else:
                if type(resource_type) == str:
                    pass
                else:
                    if self.qubits[resource_type] >= resources_needed[resource_type]:
                        pass
                    else:
                        return False
        
        return availability

    def allocate_resources(self, inst):
        """
        Given an instruction, if the resources are available, remove them from the common pool. This essentially reserves them for the instruction's use until it has finished executing.
        """
        resources_needed = inst.resource_counts()
        for resource_type in resources_needed:
            if resource_type in self.arch.limits:
                if resource_type == 'Rotation':
                    self.arch.limits['T'] -= resources_needed[resource_type]
                    if 'Clifford' in self.arch.limits:
                        self.arch.limits['Clifford'] -= resources_needed[resource_type]
                    elif 'H' in self.arch.limits:
                        self.arch.limits['H'] -= resources_needed[resource_type]
                else:
                    self.arch.limits[resource_type] -= resources_needed[resource_type]
            else:
                if type(resource_type) == str:
                    pass
                else:
                    self.qubits[resource_type] -= resources_needed[resource_type]



    def free_resources(self, inst):
        """
        Given an instruction, determine which resources were being used by it during execution and return them to the common pool.
        """
        resources_used = inst.resource_counts()
        #for each resource used by a completed instruction, re-add it to the available pool.
        for resource_type in resources_used:
            if resource_type in self.arch.limits:
                if resource_type == 'Rotation':
                    self.arch.limits['T'] += resources_used[resource_type]
                    if 'Clifford' in self.arch.limits:
                        self.arch.limits['Clifford'] += resources_used[resource_type]
                    elif 'H' in self.arch.limits:
                        self.arch.limits['H'] += resources_used[resource_type]
                else:
                    self.arch.limits[resource_type] += resources_used[resource_type]
            else:
                if type(resource_type) == str:
                    pass
                else:
                    self.qubits[resource_type] += resources_used[resource_type]


    def populate_ready(self,dag: DAG):
        '''
        Query the DAG for free nodes (nodes with no incoming dependencies) and add them to the ready queue.
        '''
        #grab free nodes from the DAG
        free_nodes = list(dag.get_available())
        #the 'END' node denotes the last instruction from the circuit
        if free_nodes[0] == 'E':
            return False
        #make sure the instruction isn't being duplicated in the ready q
        for node in free_nodes:
            self.ready_q.append(node)
            if type(node) == cirq.MeasurementKey or type(node) == CirqInstruction or type(node) == OQInstruction:
                pass
            else:
                if node not in self.qubits:
                    self.qubits[node] = 1
        

        for node in free_nodes:
            dag._remove_dependency(node)
            dag.free_instructions.remove(node)
                

        

        return True
    

        
    def move_ready_to_execute(self, dag: DAG):
        """
        For each instruction in the ready queue, determine if the resources needed to execute it are available. If they are, allocate them and move the instruction from the ready queue to the execute queue. Otherwise, the instruction remains in the ready queue until the next cycle, when it will be checked again for resource availability.
        """

        removed_insts = []
        for inst in self.ready_q:
            # is this a qubit or an operation?
            if type(inst) != CirqInstruction and type(inst) != OQInstruction:
                # if it's a qubit, don't worry about execution time and just throw it on the execution queue
                self.execution_q.append(inst)
                removed_insts.append(inst)
            else:
                # otherwise, check if the resources are free for the op, set the execution time, and add to
                # execution queue
                if self.resource_available(inst):
                    self.allocate_resources(inst)
                    inst.set_execution_time(start_time = self.global_time,\
                                        execution_time = self.assign_execution_time(inst))
                    self.execution_q.append(inst)
                    removed_insts.append(inst)
        t = 0
        #clear ready queue
        for inst in removed_insts:
            self.ready_q.remove(inst)

    
    def free_instructions(self,dag: DAG):
        '''
        Update the global time to the next timestep where an instruction completes. For each instruction in the execution queue, determine if its execution finishes before or at this 'first_done_time'. If it is completed, add it to the free queue and free the resources it was using. Otherwise, it will remain in the execution queue.
        Perform gate counting and paralellism analysis on the instructions in the free queue.
        '''
        if len(self.execution_q) == 0:
            return False
        
        # if the shortest execution time belongs to an op, set first done time to that op's ex. time
        if getattr(self.execution_q[0], '_op', False):
            first_done_time = self.execution_q[0].finish_time
        # if it's a qubit, set first done time to 0
        else:
            first_done_time = 0

        # update global time
        self.global_time = first_done_time
        
        finished_insts = []
        # find everything that finishes before or at the same time as the first done time
        for inst in self.execution_q:
            if getattr(inst, '_op', False):
                if inst.complex and (self.recursion_level == 0):
                    pass
                finish_time = inst.finish_time
            else:
                finish_time = 0
            if finish_time <= first_done_time:
                if getattr(inst, '_op', False):
                    self.free_resources(inst)
                finished_insts.append(inst)
        
        self.gate_counting(finished_insts)
        self.parallelism_analysis(finished_insts)

        if self.qasm_output == True:
            print(f'#Operations finishing at {self.global_time}: ')
            for inst in finished_insts:
                if type(inst) == CirqInstruction:
                    qubit_idxs = []
                    for qubit in inst.qubits:
                        idx = self.qubits.index(qubit)
                        qubit_idxs.append(f'q[{idx}]')
                    if inst.gateset == sets.ROT:
                        rotation_angle = (str(inst).split('(')[1]).split(')')[0]
                    else:
                        rotation_angle = None
                    op_str = str(inst)[:2]
                    qasm_str = op_to_openqasm(op_str=op_str, qubits=qubit_idxs, angle=rotation_angle)
                    print(qasm_str)
                elif type(inst) == OQInstruction:
                    if len(inst.qubits) == 1:
                        qasm_str = f'{str(inst)}{inst.qubits[0]}'
                    elif len(inst.qubits) == 2:
                        qasm_str = f'{str(inst)}{inst.qubits[0]}, {inst.qubits[1]}'
                    else:
                        qasm_str = f'{str(inst)}{inst.qubits[0]}, {inst.qubits[1]}, {inst.qubits[2]}'
                    print(qasm_str)
    
    
    def parallelism_analysis(self, insts:list):
        '''
        For each instruction that has just executed, determine how many qubits it used and update the active_qubits distribution. Determine how many T-gates were used and update T-width and T-depth accordingly.
        '''
        # gathering parallelism data
        idle_qubits = 0
        active_qubits = 0
        for inst in insts:
            if type(inst) == CirqInstruction or type(inst) == OQInstruction:
                qubits = len(inst.qubits)
                active_qubits += qubits
                idle_qubits += (len(self.qubits) - qubits)

        if self.cycle == 0:
            pass
        elif active_qubits in self.active_qubits:
            self.active_qubits[active_qubits] += 1
        else:
            self.active_qubits[active_qubits] = 1

        if self.cycle == 0:
            pass
        elif idle_qubits in self.idle_qubits:
            self.idle_qubits[idle_qubits] += 1
        else:
            self.idle_qubits[idle_qubits] = 1

        
        t_depth = [0] 
        t_width = 0
        cliff_width = 0
        t_from_rotation_total = 0
        num_rotations = 0
        for inst in insts:
            if getattr(inst, '_op', False):
                if inst.gateset is not None:
                    if inst.gateset == sets.ROT and self.rotations_allowed == False:
                        t_width += 1
                        # if we have a restriciton on how many t-gates can execute simultaneously, add all t-gates
                        # to t-depth.
                        t_per_rot = self.rotation_to_t(inst,self.precision)
                        t_from_rotation_total += int(t_per_rot[1])
                        num_rotations += 1
                    # if op is a t-gate, add one to t-depth and t-width.
                    elif inst.gateset == sets.T:
                        t_depth.append(1)
                        t_width += 1
                    else:
                        cliff_width += 1
                # complex?
                elif getattr(inst, 'complex', False) :
                    #pull t-depth and t-width from cache
                    t_depth.append(inst.cached_schedule['T-depth'])
                    self.t_proportions.update(inst.cached_schedule['T-proportion'])
                    self.t_width.update(inst.cached_schedule['T-width'])
        
            self.execution_q.remove(inst)
        self.t_depth += np.max(t_depth)

        if num_rotations != 0:
            depth_from_rotations = t_from_rotation_total / self.arch.limits['T']
            if depth_from_rotations >= t_from_rotation_total / num_rotations:
                self.t_depth += (depth_from_rotations - np.max(t_depth))
            else:
                self.t_depth += ((t_from_rotation_total / num_rotations) - np.max(t_depth))

        if t_width in self.t_width:
            self.t_width[t_width] += 1
        else:
            self.t_width[t_width] = 1

        if (t_width + cliff_width) != 0:
            t_proportion = round(t_width/(t_width + cliff_width), 2)
        else:
            t_proportion = 0
        if t_proportion in self.t_proportions:
            self.t_proportions[t_proportion] += 1
        else:
            self.t_proportions[t_proportion] = 1
        
        self.step_tracking[self.cycle] = (cliff_width, t_width)


        
    def gate_counting(self, insts:list):
        '''
        For each instruction that has just executed, either pull its cached gate counts and add them to the total or classify it as one of the gates in the defined gateset.
        '''
        total_qubits_used = 0
        for inst in insts:
            # is it an op?
            if getattr(inst, '_op', False):
                total_qubits_used += len(inst.qubits)
                # is it a basic or complex op?
                if inst.gateset is not None:
                    op = inst._op
                    if isinstance(op, cirq.ClassicallyControlledOperation):
                        op = op.without_classical_controls()
                    first_two = str(inst)[:2]
                    # classify op into one of the basic gate sets, add to count
                    # check if rotations are allowed or need to be decomposed
                    if inst.gateset == sets.ROT:
                        if self.rotations_allowed:
                            self.gate_profile['Rotation'] += 1
                        else:
                            sub_gates = self.rotation_to_t(inst, precision=self.precision)
                            self.gate_profile['T'] += sub_gates[1]
                            try:
                                self.gate_profile['H'] += sub_gates[0]
                            except KeyError:
                                try:
                                    self.gate_profile['Clifford'] += sub_gates[0]
                                except KeyError:
                                    raise TypeError('Either a general "Clifford" category or individual Clifford-type categories must be included in this gateset in order to properly decompose rotations.')
                                
                    else:
                        # this 'added' flag is so that, in the event that an op could be classified into two of the sets in a custom gateset,
                        #ex: if CX and CLIFFORD are specified, it only gets counted for the first set listed
                        added = False
                        #custom gateset
                        for set in self.arch.gateset:
                            if added == False:
                                if set.in_gateset(first_two):
                                    self.gate_profile[set.name()] += 1
                                    added = True

                elif inst.complex:
                    # pull gate counts for complex op from its cache
                    for type in self.gate_profile:
                        self.gate_profile[type] += inst.cached_schedule['Gates'][type]

                            
                    

        num_idles = len(self.qubits) - total_qubits_used
        self.idles += num_idles
                        
                            
        
    def schedule(self,dag: DAG):
        """
        In each cycle:
            1. Query the DAG for free nodes and populate the ready queue
            2. Move instructions from the ready queue to the execute queue as appropriate
            3. Sort the execute queue by finish time
            4. Execute all instructions in the execute queue which finish before or at the current timestep
        This cycle will continue for as long as the DAG has nodes in it. Once the DAG is empty, schedule any remaining instructions in the execute queue until it is empty.
        """

        #fetch finish time for sorting
        def finish_time(inst):
            if type(inst) == CirqInstruction or type(inst) == OQInstruction:
                return inst.finish_time
            else:
                return 0

        #as long as there are instructions in the DAG, keep pulling and scheduling
        while self.populate_ready(dag):
            self.move_ready_to_execute(dag)
            self.execution_q.sort(key = finish_time)
            self.free_instructions(dag)
            self.cycle+=1

        while len(self.ready_q) != 0:
            self.move_ready_to_execute(dag)
            self.execution_q.sort(key = finish_time)
            self.free_instructions(dag)
        #once there's none left, finish scheduling anything left in the ex_q
        while len(self.execution_q) != 0:
            self.free_instructions(dag)

        self.arch.reset()

        #if the DAG has pulled all it can from the circuit, this will return False
        return self.populate_ready(dag)
    

    def rotation_to_t(self, inst, precision):
        '''
        Decompose rotations into T-gates at a given precision.
        '''
        if str(inst).startswith(('Rx_d', 'Ry_d', 'Rz_d')):
            cliff = inst._op.gate.get_gate().get_Clifford_count()
            t = inst._op.gate.get_gate().get_T_count()
            return (cliff, t)

        else:
            random.seed(0)
            t = int(random.gauss(
                    T_COUNT_SLOPE * np.log2(1 / precision) + T_COUNT_CONST,
                    T_COUNT_STD_DEV,
                ))
            cliff = (t * 2) + 1
            return (cliff, t)
        


import time

def schedule_circuit(circuit, architecture_config:architecture = None, full_profile=False, \
					decomp_level = 0, context = None, rotation_gate_precision = 5e-6, decomposition_protocol=decomposition_protocol.recursive, qasm_output=False, display_op_cache=False, instance_name=None, json_output_filename=None):
    '''
    This function is the main interface with the Scheduler, DAG, and Instruction classes. 
    
    :param cirq.Circuit circuit: a cirq.Circuit object to be scheduled. (Other circuit formats will be supported in the future.)

    :param dict architecture_config: an architecture class object.

    :param bool full_profile: bool specifying whether results should include gate counts and (if applicable) parallelism analysis. If False, only circuit execution time, T-depth, and total number of qubits used will be included in results.

    :param int decomp_level: levels of decomposition implemented before scheduling. If set to `0`, each operator will be decomposed to one/two-qubit gates only on its first occurance and cached for each future occurance. If set to `'Full'`, entire circuit will be decomposed to one/two-qubit gates before scheduling.

    :param cirq.DecompositionContext context: cirq decomposition context to be used.
    
    :param float rotation_gate_precision: precision used to decompose rotations into T-gates.

    :param decomposition_protocol decomposition_protocol: whether recursive or hierarchical decomposition should be used to handle complex operations.

    :param bool qasm_output: whether a qasm output of the scheduled circuit should be printed to screen.

    :param bool display_op_cache: whether details about cached operators and their occurrances should be printed to screen.

    :param str instance_name: if set, results dictionary will include an 'Instance name' key.

    :param str json_output_filename: if set, results will write to JSON file named according to input string.

    Returns:

    :param dict results: a dictionary including 'Total time for execution', 'Circuit T-depth', and 'Number of qubits used'. If `full_profile` == True, this dictionary will also include 'Gate profile'. If `full_profile` == True AND `decomp_level` == 'Full', 'Active qubit distribution', 'T-widths', 'T-proportion', and 'Clifford/T ratio' will be included as well.
        
    '''


    if type(circuit) == list or type(circuit) == str:

        if type(circuit) == str:
            circuit = circuit.splitlines()

        mySched = Scheduler(architecture_params=None, qasm_input=True, qasm_output=qasm_output)
        
        myDAG = DAG(max_moments=1000000)
        for op in circuit:
            inst = OQInstruction(operation=op)
            for qubit in inst.qubits:
                mySched.qubits[qubit] = 1
            myDAG.add_dependency(inst)

        myDAG.finish()
        
        while mySched.schedule(myDAG):
            pass
        
        results_dict = {
        'Total time for execution': mySched.global_time,
        'Circuit T-depth': mySched.t_depth,
        'Number of qubits used': len(mySched.qubits),
        'Gate profile': mySched.gate_profile,
        'Active qubit distribution': mySched.active_qubits,
        'T-widths': mySched.t_width,
        'T-proportion': mySched.t_proportions
        }

        return results_dict
    

    elif type(circuit) == cirq.Circuit:

        
        if context is None:
            #context = cirq.DecompositionContext(cirq.SimpleQubitManager())
            new_gam = gam.gam
            new_gam.reset()
            context = cirq.DecompositionContext(new_gam)
    
        rotation_allowed = True
        def convertMe(circuit, decomp_level=None):
            #A deep copy MUST be done here in order to keep qubit naming conventions consistent.
            tcontext = copy.deepcopy(context)
            for op in generator_decompose(circuit, keep=keep, on_stuck_raise = None, context=tcontext,max_decomposition_passes=decomp_level):
                yield op
    

        if (decomposition_protocol == decomposition_protocol.recursive) and not (isinstance(context.qubit_manager,gam.GlobalQubitManager) or isinstance(context.qubit_manager, cirq.SimpleQubitManager)):
            raise ValueError('Due to the nature of the recursive scheduling, we need additional control over the ancilla management. Therefore, you must use the global qubit manager (pyLIQTR.utils.global_ancilla_manager) or simple qubit manager (cirq.SimpleQubitManager)')
        elif isinstance(context.qubit_manager,gam.GlobalQubitManager):
            context.qubit_manager.reset()

        #initialize scheduler
        mySched = Scheduler(architecture_params=architecture_config, precision=rotation_gate_precision, decomposition_protocol=decomposition_protocol, qasm_output=qasm_output, context=context)

        #if decomp is full, we decompose BEFORE scheduling    
        parallelism_results = False
        if decomp_level == 'Full':
            level = 0
            prev_decomp = None
            """
            while circuit != prev_decomp:
                prev_decomp = circuit
                circuit = circuit_decompose_multi(circuit, 1, context=context)
                for q in circuit.all_qubits():
                    if q not in mySched.qubits:
                        mySched.qubits[q] = 1
                level += 1

            print(f"Decomposition levels to basic gates: {level}.")
            """
            circuit = cirq.Circuit([op for op in generator_decompose(circuit, keep=keep, on_stuck_raise = None, context=context)])
            for q in circuit.all_qubits():
                if q not in mySched.qubits:
                    mySched.qubits[q] = 1
            decomp_level = 0
            parallelism_results = True
    
        #initialize DAG
        myDAG = DAG(max_moments=1000000)

        global_time = 0
        #Loop through circuit
        for op in convertMe(circuit, decomp_level=decomp_level):
            for q in op.qubits:
                if q not in mySched.qubits:
                    mySched.qubits[q] = 1
            if myDAG.full():
                myDAG.finish()
                while mySched.schedule(myDAG):
                    pass
                global_time += mySched.global_time
                myDAG.dependency_log.clear()
                myDAG.dag_cycle += 1
            myDAG.add_dependency(CirqInstruction(op))
    
        myDAG.finish()
        while mySched.schedule(myDAG):
            pass
        
        #If the DAG fills and resets, it resets the scheduler's global time to 0. idk why. but we record it locally just in case
        global_time += mySched.global_time
    
        results_dict = {
            'Total time for execution': global_time,
            'Circuit T-depth': int(mySched.t_depth),
            'Number of qubits used': len(mySched.qubits),
            'Gate profile': mySched.gate_profile
        }

        if full_profile:
            #parallelism results will only be reported if decomp_level='Full'. otherwise they get pretty mangled by the recursion
            if parallelism_results: 
                results_dict['Active qubit distribution'] = mySched.active_qubits
                results_dict['T-widths'] = mySched.t_width
                results_dict['T-proportion'] = mySched.t_proportions
                results_dict['Clifford/T ratios'] = mySched.step_tracking
            if decomposition_protocol == decomposition_protocol.hierarchical:
                results_dict['T-widths'] = mySched.t_width
                results_dict['T-proportion'] = mySched.t_proportions

        if instance_name is not None:
            results_dict['Instance name'] = instance_name

        if display_op_cache:
            for op in mySched.op_cache:
                op_str = str(op._op.gate)
                print(f'{op._op.gate} occurred {mySched.op_cache[op][1]} times and had the following cached results: Execution time: {mySched.op_cache[op][0]["Execution time"]}, T-depth: {mySched.op_cache[op][0]["T-depth"]}, Gates: {mySched.op_cache[op][0]["Gates"]}, Qubits: {len(mySched.op_cache[op][0]["Qubits"])}.')

        if json_output_filename is not None:
            if instance_name is None:
                results_dict['Instance name'] = json_output_filename
            json_dict = results_dict.copy()

            if architecture_config is None:
                json_dict['Input parameters'] = {
                    'Gate times': {
                        'T': str([state_factory(production_time=100, consumption_time=100, production_limit=10, success_prob=0.7)]),
                        'H': 100,
                        'S': 100,
                        'CX': 200,
                        'CZ': 200,
                        'Toffoli': str([state_factory(production_time=200, consumption_time=300, production_limit=10, success_prob=0.7)]),
                        'Pauli (X, Y, Z)': 100,
                        'Measurement/Reset': 100 
                    },
                    'Max simultaneous T-gates': 20,
                    'Factory Configurations': {
                        'T': {
                            'Production time': 100,
                            'Consumption time': 100,
                            'Production limit': 10,
                            'Success probability': 0.7
                        },
                        'Toffoli': {
                            'Production time': 200,
                            'Consumption time': 300,
                            'Production limit': 10,
                            'Success probability': 0.7
                        }
                    }
                }
            else:
                json_dict['Input parameters'] = {
                    'Gate times': {},
                    'Max simultaneous T-gates': architecture_config.limits['T'],
                    'Factory Configurations': {}
                }
                

                for gate_type in architecture_config.standard_times:
                    json_dict['Input parameters']['Gate times'][gate_type] = str(architecture_config.standard_times[gate_type])
                    if type(architecture_config.standard_times[gate_type]) == state_factory:
                        factory = architecture_config.standard_times[gate_type]
                        factory_params = {
                            'Production time': factory.production_time,
                            'Consumption time': factory.consumption_time,
                            'Production limit': factory.production_limit,
                            'Success probability': factory.success_prob
                        }
                        json_dict['Input parameters']['Factory Configurations'][gate_type] = factory_params
                    elif type(architecture_config.standard_times[gate_type]) == list:
                        factory_list = architecture_config.standard_times[gate_type]
                        factory_params = {}
                        factory_idx = 0
                        for factory in factory_list:
                            individual_factory_params = {
                                'Production time': factory.production_time,
                                'Consumption time': factory.consumption_time,
                                'Production limit': factory.production_limit,
                                'Success probability': factory.success_prob
                            }
                            factory_params[f'Factory {factory_idx} parameters'] = individual_factory_params
                            factory_idx += 1
                        json_dict['Input parameters']['Factory Configurations'][gate_type] = factory_params



            with open(f"{json_output_filename}.json", "w") as outfile:
                json.dump(json_dict, outfile, indent=4)

        return results_dict
    

                

