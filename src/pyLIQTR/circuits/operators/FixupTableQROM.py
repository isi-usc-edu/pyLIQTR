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
import numpy as np
from typing import Tuple, Union
from numpy.typing import NDArray
from functools import cached_property
from cirq.value import Condition
from qualtran.bloqs.unary_iteration_bloq import UnaryIterationGate
from qualtran._infra.registers import Signature, Register, SelectionRegister

class FixupTableQROM(UnaryIterationGate):
    '''
    This gate outputs the bits for the fixup table F with the jth entry given by 

        Fj = 00 (2j not in S) and (2j+1 not in S)
             01 (2j in S)     and (2j+1 not in S)
             10 (2j not in S) and (2j+1 in S)
             11 (2j in S)     and (2j+1 in S)

    where S is the set of original selection register states that need to have their amplitude negated. The first bit is output on |q> and the second on |u>, where |q> is the qubit representing the least significant bit from the original selection register and |u> is an ancilla initialized to be in the |1> state. The selection register for this fixup table is the original selection register from the data lookup minus |q>. See Ref [1] Appendix C, Equation 68 for more details.

    Args:
        - data_to_uncompute: Array of classical data to be uncomputed. data_to_uncompute[l] will be uncomputed when the orgiinal selection register stores index l.
        - measurement_key: The string used as the key to store the x basis measurement of the data register
        - controlled: If True a controlled version of the gate is constructed.

    References:
    [1] [Qubitization of Arbitrary Basis Quantum Chemistry Leveraging Sparsity and Low Rank Factorization](https://arxiv.org/abs/1902.02134)
    '''

    def __init__(self, data_to_uncompute:NDArray, measurement_key:str, max_uncompute_bits:int, controlled = False):
        self.data_to_uncompute = data_to_uncompute
        self.measurement_key = measurement_key
        self.controlled = controlled
        self.nSelect = (len(data_to_uncompute)-1).bit_length() - 1

        self.max_uncompute_bits = max_uncompute_bits

    @cached_property
    def control_registers(self) -> Tuple[Register]:
        register = () if not self.controlled else (Register('control', 1),)
        return register

    @cached_property
    def selection_registers(self) -> Tuple[SelectionRegister]:
        iteration_length = int(np.ceil(len(self.data_to_uncompute)/2))
        return (SelectionRegister('selection',self.nSelect,iteration_length=iteration_length),)

    @cached_property
    def target_registers(self) -> Tuple[Register]:
        return (Register('u',1),Register('q',1))
    
    @cached_property
    def signature(self) -> Signature:
        return Signature([*self.control_registers, *self.selection_registers,*self.target_registers])

    def nth_operation(
        self, context: cirq.DecompositionContext, control: cirq.Qid, **kwargs
    ) -> cirq.OP_TREE:
        selection_idx = kwargs['selection']
        u_ancilla = kwargs['u']
        q_bit = kwargs['q']
        if not self.controlled:
            yield cirq.CX(control,*u_ancilla).with_classical_controls(DataAndKeyCondition(key=self.measurement_key,data=self.data_to_uncompute[2*selection_idx],\
                                                                                          max_meas_bits=self.max_uncompute_bits))
            try:
                yield cirq.CX(control,*q_bit).with_classical_controls(DataAndKeyCondition(key=self.measurement_key,data=self.data_to_uncompute[2*selection_idx+1],\
                                                                                           max_meas_bits=self.max_uncompute_bits))
            except IndexError:
                pass
        else:
            raise NotImplementedError

    def _circuit_diagram_info_(self, _) -> cirq.CircuitDiagramInfo:
        if self.controlled:
            wire_symbols = ["@"]
        else:
            wire_symbols = []
        wire_symbols += ["In"] * self.nSelect
        wire_symbols += ["FixupTableQROM"]*2
        return cirq.CircuitDiagramInfo(wire_symbols=wire_symbols)

class DataAndKeyCondition(Condition):
    '''
    A classical control condition based on a measurement key and classical data.

    This condition resolves to True iff the number of ones in the binary representation of a bitwise AND between the classical data and the measurement result is odd. This condition is used to determine if the amplitude of a state needs to be negated in the phase fixup procedure.

    Args:
        - key: measurement key
        - data: classical data value. Can be provided as an integer or a bit array of the binary representation.
    '''

    def __init__(self,key:cirq.MeasurementKey,data:Union[int,NDArray],index:int=-1,max_meas_bits=None):
        self.key = key
        self.data = data
        self.index=index
        self.max_meas_bits = int(max_meas_bits)

    @property
    def keys(self):
        return (self.key,)

    def replace_key(self, current: 'cirq.MeasurementKey', replacement: 'cirq.MeasurementKey'):
        return DataAndKeyCondition(replacement,self.data) if self.key == current else self

    def resolve(self, classical_data: 'cirq.ClassicalDataStoreReader') -> bool:
        if self.key not in classical_data.keys():
            raise ValueError(f'Measurement key {self.key} missing when testing classical control')

        if isinstance(self.data,np.ndarray):

            measurement_digits = classical_data.get_digits(self.key,self.index) # measured values of qubits

            num_ones = sum(measurement_digits & self.data) # number of ones in result of bitwise AND
        
        else:

            measurement_val = classical_data.get_int(self.key,self.index) # integer rep of binary measurement result
            
            num_ones = (measurement_val & self.data).bit_count() # number of ones in result of bitwise AND

        return num_ones%2 == 1 # true if num_ones is odd

    def get_condition_format_string(self):
        """
        Used for QASM export, basically remaps _repr_ to act on something more akin to qasm classical measurement
        """
        if isinstance(self.data,np.ndarray):
            return "(({} & {}).bit_count() % 2 == 1)".format("{}","".join([str(x) for x in self.data]))
        else:
            data_binary_string = format(self.data, '#0{}b'.format(2+self.max_meas_bits))[2:]
            return "(({} & {}).bit_count() % 2 == 1)".format("{}",data_binary_string)

    def __repr__(self):
        return f'pyLIQTR.DataAndKeyCondition(({self.key} & {self.data}).bit_count() % 2 == 1)'

    @property
    def qasm(self):
        raise ValueError('QASM is defined only for SympyConditions of type key == constant. Try using pyLIQTR.utils.printing.openqasm')