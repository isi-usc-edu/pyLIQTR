import cirq
import pyLIQTR.scheduler.sets as sets
from qualtran.resource_counting._generalization import GeneralizerT
from typing import cast

class Instruction:
    """
    Stores operation, qubits it acts on, the index of the associated node in a DAG, its assigned execution time, and its status with regard to a Scheduler (ready, executing, or complete).
    """
    def __init__(self, cirq_operation: cirq.Operation, custom_gateset=None):
        self._op = cirq_operation
        self.start_time = 0
        self.finish_time = 0
        self.status = None
        self.index = None
        self.qubits = self._op.qubits
        self.complex = False
        self.cached_schedule = None
        self.ready = False 
        self.custom_gateset = custom_gateset
    

    def __str__(self):
        '''
        Return a str of the operation (does not include qubits the operation acts on).
        '''
        if str(self._op).startswith('bloq'):
            return str(self._op)[5:]
        return self._op.__str__()
    
    def __eq__(self, other, generalizer: GeneralizerT = cast(GeneralizerT, lambda b: b)):
        '''
        Determine equivalency of two operations. If the operation is the same, instructions should return equivalently for data caching purposes, even if they are two distinct Instruction objects.
        '''
        if not getattr(other, 'start_time', False):
            return False
        
        return (generalizer(self._op) == generalizer(other._op))

        
        
    
    def __hash__(self) -> int:
        return hash((len(self.qubits), self._op.gate.__hash__))
    
    
    def resource_counts(self):
        '''
        Return a dictionary of the form {resourceX: # required, resourceY: # required, ...} with, at minimum, an entry for each qubit the instruction acts on. Resources could also include how many of a particular type of gate are required for the instruction tobe performed.
        '''
        resources = {}
        for qubit in self.qubits:
            resources[qubit] = 1

        first_two = self.__str__()[:2]
        if first_two in sets.T:
            resources['T'] = 1

        return resources

    def get_data_dependencies(self):
        """
        Return a dictionary of the form {qX: dependency type, qY: dependency type, ...} with an entry for each qubit the instruction acts on.
        """

        dependencies = {}
        first_two = self.__str__()[:2]
        if self.custom_gateset is not None:
            for set in self.custom_gateset:
                if first_two in set:
                    for q in self.qubits:
                        dependencies[q] = 'All'
            if len(dependencies) == 0:
                for q in self.qubits: 
                    dependencies[q] = 'All'
                self.complex = True
            return dependencies


        if first_two in sets.CX:
            ctl_q = self.qubits[0]
            tgt_q = self.qubits[1]
            dependencies[ctl_q] = 'Z'
            dependencies[tgt_q] = 'X'
        elif first_two in sets.CZ:
            ctl_q = self.qubits[0]
            tgt_q = self.qubits[1]
            dependencies[ctl_q] = 'Z'
            dependencies[tgt_q] = 'Z'
        elif first_two in sets.X:
            qubit = self.qubits[0]
            dependencies[qubit] = 'X'
        elif first_two in sets.Z:
            qubit = self.qubits[0]
            dependencies[qubit] = 'Z'
        elif first_two in sets.ALL:
            qubit = self.qubits[0]
            dependencies[qubit] = 'All'
        else:
            qubits = self.qubits
            for q in qubits:
                dependencies[q] = 'All'
            self.complex = True

        return dependencies
    

    def set_execution_time(self,start_time,execution_time):
        self.start_time = start_time
        self.finish_time = self.start_time+execution_time

class DependencyEdge:
    '''
    Simple class to store information attached to a DAG edge representing a data dependency.

    :param cirq.Qubit qubit: qubit that the dependency is on
    :param str type: dependency type ('Z', 'X', 'All', etc.)
    '''
    def __init__(self, qubit, type:str):
        self.qubit = qubit
        self.type = type
        self.index = None

    def __repr__(self):
        qubit_str = str(self.qubit)
        return f"({qubit_str}, {self.type})"

