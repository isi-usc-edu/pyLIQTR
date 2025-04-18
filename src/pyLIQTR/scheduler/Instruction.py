import cirq
import pyLIQTR.scheduler.sets as sets
from qualtran.resource_counting._generalization import GeneralizerT
from typing import cast

class Instruction:
    """
    Stores operation, qubits it acts on, the index of the associated node in a DAG, its assigned execution time, and its status with regard to a Scheduler (ready, executing, or complete).
    """
    def __init__(self):
        self._op = None
        self.start_time = 0
        self.finish_time = 0
        self.status = None
        self.index = None
        self.qubits = None
        self.complex = False
        self.cached_schedule = None
        self.free = False 
        self.custom_gateset = None
    
    
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
        if isinstance(self._op, cirq.ClassicallyControlledOperation):
            for cbit in self._op.classical_controls:
                dependencies[cirq.MeasurementKey(f'{cbit}')] = 'All'
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
        elif first_two in sets.CCX:
            ctl_q0 = self.qubits[0]
            ctl_q1 = self.qubits[1]
            tgt_q = self.qubits[2]
            dependencies[ctl_q0] = 'Z'
            dependencies[ctl_q1] = 'Z'
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
        elif first_two in sets.ALL or first_two in sets.ALL_CIRQ:
            if first_two == 'ci':
                cbit = cirq.MeasurementKey(f'{self.qubits[0]}')
                qubit = self.qubits[0]
                dependencies[qubit] = 'All'
                dependencies[cbit] = 'All'
            else:
                for qubit in self.qubits:
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


class CirqInstruction(Instruction):

    def __init__(self, operation: cirq.Operation, custom_gateset=None):
        super().__init__()
        self._op = operation
        self.qubits = self._op.qubits
        self.custom_gateset = custom_gateset

    def __str__(self):
        if str(self._op).startswith('bloq'):
            return str(self._op)[5:]
        return self._op.__str__()
    
    def __eq__(self, other):

        if not type(other) == CirqInstruction:
            return False
        if (self._op.gate == other._op.gate) and (len(self.qubits) == len(other.qubits)):
            if getattr(self._op.gate, 'adjoint', False) and getattr(other._op.gate, 'adjoint', False):
                if self._op.gate.adjoint() == other._op.gate.adjoint():
                    return True
                else:
                    return False
            else:
                return True           
        else:
            return False
    
    def __hash__(self):
        return hash((len(self.qubits), hash(self._op.gate)))
    
    def get_clifford_t(self):
        if 'Ry_d' in str(self._op.gate):
            return self._op.gate.t_complexity().t, self._op.gate.t_complexity().clifford
        else:
            return None, None

    def set_complexity(self):

        first_two = self.__str__()[:2]
        if self.custom_gateset is not None:
            found = False
            for set in self.custom_gateset:
                if first_two in set:
                    self.complex = False
                    found = True
            if found == False:
                self.complex = True

        else:
            if first_two in sets.CX:
                self.complex = False
            elif first_two in sets.CCX:
                self.complex = False
            elif first_two in sets.CZ:
                self.complex = False
            elif first_two in sets.X_CIRQ:
                self.complex = False
            elif first_two in sets.Z_CIRQ:
                self.complex = False
            elif first_two in sets.ALL_CIRQ:
                self.complex = False
            else:
                self.complex = True
    
class OQInstruction(Instruction):

    def __init__(self, operation: str):
        super().__init__()
        self._op = operation.split()[0]
        self.qubits = [q.replace(',', '') for q in operation.split()[1:]]
        self.custom_gateset = None

    def __str__(self):
        return (self._op + ' ')
    
    def __eq__(self, other):

        if not getattr(other, 'start_time', False):
            return False
        
        return (self._op == other._op)
    
    def __hash__(self):
        return hash((len(self.qubits), hash(self._op)))
    
    def get_data_dependencies(self):
        dep = super().get_data_dependencies()
        self.complex = False
        return dep
    
    
