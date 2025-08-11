#Scheduler utils
import cirq
import random
import math
import rustworkx as rx
import numpy as np
from pyLIQTR.scheduler.DAG import DAG
from pyLIQTR.scheduler.Instruction import CirqInstruction, OQInstruction, DependencyEdge
import matplotlib.pyplot as plt
from pyLIQTR.gate_decomp.rotation_gates import T_COUNT_CONST, T_COUNT_SLOPE, T_COUNT_STD_DEV
import pyLIQTR.scheduler.sets as sets


def dag_creation(circuit):
    dag = DAG(max_moments=100)
    for op in circuit.all_operations():
        inst = CirqInstruction(op)
        dag.add_dependency(inst)
    dag.finish()
    return dag

def get_instruction2node_map(graph,ops,qubits):
    graph.clear()
    ops_to_nodes = {}
    for op in ops:
        ops_to_nodes[op] = graph.add_node(CirqInstruction(op))
    
    qubits_to_nodes = {}
    for q in qubits:
        qubits_to_nodes[q] = graph.add_node(q)

    return ops_to_nodes, qubits_to_nodes

def manual_link(graph,qubit_map, inst_map,\
                start,end,\
                qubit,type):
    if isinstance(start,list) and not isinstance(end,list):
        for s in start:
            manual_link(graph,qubit_map, inst_map,\
                        s, end, 
                        qubit,type)
    elif isinstance(end,list) and not isinstance(start,list):
        for e in end:
            manual_link(graph, qubit_map, inst_map,\
                        start, e, 
                        qubit,type)
    else:
        assert(not isinstance(start,list))
        assert(not isinstance(end,list))

        if isinstance(start,cirq.ops.gate_operation.GateOperation):
            start_node = inst_map[start]
        else:
            start_node = qubit_map[start]

        if isinstance(end,cirq.ops.gate_operation.GateOperation):
            end_node = inst_map[end]
        else:
            end_node = qubit_map[end]

        graph.add_edge(start_node,end_node,DependencyEdge(qubit,type))

def plot_results(results: dict):
    """
    Plot parallelism data from scheduler output. Currently, values plotted are:
        * Active qubit distribution: how many qubits are used simultaneously vs. the # of occurances of each value.
        * T-widths: how many T-gates occur in a single "moment" vs. the # of occurances of each value.
        * T-proportion: within a single moment, the proportion of T-gates out of all gates executed in that moment vs. the # of occurances of each value.
        * Clifford/T ratios: for each scheduler timestep, a ratio of T-gates to non-T gates. Each timestep is displayed as a split bar, with non-T gates on the bottom and T-gates on the top.
    """

    if ('Active qubit distribution' not in results) or ('T-widths' not in results) or ('T-proportion' not in results) or ('Clifford/T ratios' not in results):
        raise TypeError("In order to use plot_results, scheduler run must have decomposition_level='Full' and full_profile=True.")
    
    aq = []
    aq_vals = []
    for active_qubits in results['Active qubit distribution']:
        val = results['Active qubit distribution'][active_qubits]
        aq.append(active_qubits)
        aq_vals.append(val)

    tw = []
    tw_vals = []
    for width in results['T-widths']:
        val = results['T-widths'][width]
        tw.append(width)
        tw_vals.append(val)

    tp = []
    tp_vals = []
    for proportion in results['T-proportion']:
        val = results['T-proportion'][proportion]
        tp.append(proportion)
        tp_vals.append(val)

    cliff = []
    t = []
    steps = []
    for step in results['Clifford/T ratios']:
        cliff_val = results['Clifford/T ratios'][step][0]
        t_val = results['Clifford/T ratios'][step][0]
        cliff.append(cliff_val)
        t.append(t_val)
        steps.append(step)
    
    counts = {
        'Clifford': cliff,
        'T': t
    }


    fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(15, 10))

    axs[0,0].bar(aq, aq_vals)
    axs[0,0].set_ylabel("Number of Occurances")
    axs[0,0].set_xlabel("Number of Active Qubits")
    axs[0,0].set_yscale('log')


    axs[0,1].bar(tw, tw_vals)
    axs[0,1].set_ylabel("Number of Occurances")
    axs[0,1].set_xlabel("T-width")
    axs[0,1].set_yscale('log')

    axs[1,0].bar(tp, tp_vals, width=0.05)
    axs[1,0].set_ylabel("Number of Occurances")
    axs[1,0].set_xlabel("T-proportion")
    axs[1,0].set_yscale('log')

    bottom = np.zeros(len(steps))
    for gate_type, count in counts.items():
        if gate_type == 'Clifford':
            color = 'r'
        elif gate_type == 'T':
            color = 'c'
        axs[1,1].bar(steps, count, width=1, label=gate_type, color=color, bottom=bottom)
        bottom += count
    
    axs[1,1].set_ylabel("Gate Count")
    axs[1,1].set_xlabel("Scheduler Cycle")
    axs[1,1].set_ylim(0,max(cliff))
    axs[1,1].legend()



class architecture(dict):
    '''
    A class for storing architectural information about a given scheduler run. Currently, this includes state factories and gate execution timings.
    '''

    def __init__(self, user_input_timings:dict = None, overall_t_limit = None):
        """
        :param list factories: tracks all initialized factories for the current run.
        :param dict standard_times: dictionary of each type of gate to be tracked and its respective execution time or assigned state factory.
        :param list gateset: list of sets.(type) objects corresponding to each type of gate being tracked.
        :param dict limits: dictionary of every gate type which is a limited resource for the scheduler.
        """

        self.factories = []

        if user_input_timings is not None:
            self.standard_times = {}
            self.gateset = []
            for gate_type in user_input_timings:
                self.gateset.append(gate_type)
                name_key = gate_type.name()
                self.standard_times[name_key] = user_input_timings[gate_type]
                
        else:
            self.standard_times = {
                'T': [state_factory(production_time=100, consumption_time=100, production_limit=10, success_prob=1)],
                'H': 100,
                'S': 100,
                'CX': 200,
                'CZ': 200,
                'Toffoli': [state_factory(production_time=200, consumption_time=300, production_limit=10, success_prob=1)],
                'Pauli (X, Y, Z)': 100,
                'Measurement/Reset': 0 
            }
            self.gateset = [sets.T, sets.H, sets.S, sets.CX, sets.CCX, sets.CZ, sets.PAULI, sets.MISC]

        self.limits = {}
        if overall_t_limit is not None:
            self.limits['T'] = overall_t_limit
        else:
            for gate_type in self.standard_times:
                if type(self.standard_times[gate_type]) == list:
                    self.limits[gate_type] = 0
                    for factory in self.standard_times[gate_type]:
                        if factory in self.factories:
                            raise TypeError("The same state_factory object has been assigned multiple times in your timing dictionary. Please initialize separate state_factory objects for each assignment.")
                        self.factories.append(factory)
                        self.limits[gate_type] += factory.production_limit
                elif type(self.standard_times[gate_type]) == state_factory:
                    if self.standard_times[gate_type] in self.factories:
                            raise TypeError("The same state_factory object has been assigned multiple times in your timing dictionary. Please initialize separate state_factory objects for each assignment.")
                    self.factories.append(self.standard_times[gate_type])
                    self.limits[gate_type] = self.standard_times[gate_type].production_limit

            if 'T' not in self.limits:
                self.limits['T'] = 1e8
            if 'Clifford' not in self.limits and 'H' not in self.limits:
                self.limits['Clifford'] = 1e8
            self.limits['Rotation'] = 0

        self.precision = None
    
    def __getitem__(self, key, global_time):
        """
        Returns an integer execution time given a type of gate based on user or default timing inputs.
        """

        if key in self.standard_times:
            if type(self.standard_times[key]) == int:
                return self.standard_times[key]
            elif type(self.standard_times[key]) == list or type(self.standard_times[key]) == state_factory:
                return self.get_factory_time(self.standard_times[key], global_time=global_time)

        elif key == 'Rotation':
            t_count = self.rotation_to_t(precision=self.precision)
            h_count = (2 * t_count) + 1
            total_t_time = 0
            total_h_time = 0
            for i in range(t_count):
                t_time = self.__getitem__('T', global_time=global_time)
                total_t_time += t_time
            if 'H' in self.standard_times:
                for i in range(h_count):
                    h_time = self.__getitem__('H', global_time=global_time)
                    total_h_time += h_time
            elif 'Clifford' in self.standard_times:
                for i in range(h_count):
                    h_time = self.__getitem__('Clifford', global_time=global_time)
                    total_h_time += h_time
            
            total_time = total_h_time + total_t_time
            return total_time

        else:
            for gateset in self.gateset:
                if gateset.subsets is not None:
                    for subset in gateset.subsets:
                        if key == subset.name():
                            if type(self.standard_times[gateset.name()]) == int:
                                return self.standard_times[gateset.name()]
                            elif type(self.standard_times[gateset.name()]) == list or type(self.standard_times[gateset.name()]) == state_factory:
                                return self.get_factory_time(self.standard_times[gateset.name()], global_time=global_time)
                    
                        
    def get_factory_time(self, factory_or_list, global_time):
        """
        If a gate type's time assignment is either a state factory or a list of state factories, find the fastest available state from available factories and return the time it takes to generate it.
        """
        if type(factory_or_list) == state_factory:
            total_time = factory_or_list.get_total_time(global_time=global_time)
        elif type(factory_or_list) == list:
            for factory in factory_or_list:
                if factory.availability():
                    total_time = factory.get_total_time(global_time=global_time)
                    return total_time
            potential_times = []
            for factory in factory_or_list:
                time = factory.get_total_time(global_time=global_time, report_only=True)
                potential_times.append(time)
            min_factory_idx = potential_times.index(min(potential_times))
            total_time = factory_or_list[min_factory_idx].get_total_time(global_time=global_time)
        return total_time
        
        
    def rotation_to_t(self, precision=1e-10):
        '''
        Decompose rotations into T-gates at a given precision.
        '''
        random.seed(0)
        T_from_single_rotation = int(random.gauss(
                T_COUNT_SLOPE * np.log2(1 / precision) + T_COUNT_CONST,
                T_COUNT_STD_DEV,
            ))
        return T_from_single_rotation
    
    def reset(self):
        for factory in self.factories:
            setattr(factory, 'last_query_time', 0)
            setattr(factory, 'pool', 0)
    

class state_factory():
    """
    A class representing the production of states by a magic state factory. Not gate-type specific.
    """

    def __init__(self, production_time:int, consumption_time:int, production_limit:int, success_prob:float):
        """
        :param int production_time: the amount of time it takes for the factory to produce a single state.
        :param int consumption_time: the amount of time it takes for a single state to be consumed after production.
        :param int production_limit: the maximum amount of completed states the factory can hold in its "pool" before it must stop production and wait for states to be consumed.
        :param float success_prob: the probability (expressed as a decimal) that the factory produces a successful state on each attempt.
        :param int last_query_time: the last time (in terms of scheduler global time) that a state was used from the factory.
        :param int pool: how many states are currently available in the factory's "pool".
        """
        self.production_time = production_time
        self.consumption_time = consumption_time
        self.production_limit = production_limit
        self.success_prob = success_prob
        self.last_query_time = 0
        self.pool = 0

    def __str__(self):
        return 'Magic State Factory'

    def availability(self):
        """
        Return a bool corresponding to whether or not there is a state currently available for consumption, without having to produce more.
        """
        if self.pool > 0:
            return True
        else:
            return False
        
    def get_total_time(self, global_time, report_only=False):
        """
        Return the total time it takes to produce and consume a state, given the current pool level and global time. Determines how many states have been produced since the last call to the factory.
        """
        elapsed_time = global_time - self.last_query_time
        self.last_query_time = global_time
        #if the pool is at the limit, nothing has been produced since last query. take one from the pool, production time is 0.
        if self.pool == self.production_limit:
            if not report_only:
                self.pool -= 1
            return self.consumption_time
        #pool is not at limit:
        else:
            #first, figure out how many states have been produced since last query and update the pool total.
            states_produced = math.floor((1/self.production_time) * elapsed_time * self.success_prob)
            #if the pool is still under the limit, add all states produced to the total.
            if self.production_limit - self.pool >= states_produced:
                self.pool += states_produced
            #otherwise, fill to the limit.
            else:
                self.pool = self.production_limit
            #if no states have been produced in the elapsed time since last query (or elapsed time=0), and the last query brought the pool down to 0, then the pool will still be at 0 after the above update:
            if self.pool == 0:
                #return the amount of time it takes to successfully produce a single state. since it is consumed right away, the pool will return to 0, no need to update the total.
                successful_state = False
                running_production_time = 0
                while successful_state == False:
                    successful_state = random.random() < self.success_prob
                    running_production_time += self.production_time
                return (running_production_time + self.consumption_time)
            #if any states have been produced in the elapsed time, then the pool will be either partially full or at the limit:
            else:
                #either way, we take one from the pool and return production time of 0.
                if not report_only:
                    self.pool -= 1
                return self.consumption_time


        


    
