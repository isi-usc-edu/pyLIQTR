"""
Copyright (c) 2024 Massachusetts Institute of Technology 
SPDX-License-Identifier: BSD-2-Clause
"""
import  numpy                  as  np
import  cirq                   as  cirq
import  qualtran               as  qt
import  qualtran.cirq_interop.testing  as  qt_test
 
from    pyLIQTR.utils.pauli_string_manip  import  convert_to_dense_pauli_string

import abc
from   typing import Tuple
from pyLIQTR.utils.resource_analysis import pylqt_t_complexity as t_complexity




class BlockEncoding(qt._infra.gate_with_registers.GateWithRegisters):
    """This class creates an instance of the Qualtran BlockEncoding, which is a container for gates.

    :param ProblemInstance ProblemInstance: an instance of the ProblemInstance class
    :param Any control_val: a variable that will get passed to the BlockEncoding and used as a control value

    :returns: an instance of the BlockEncoding() class

    :rtype: BlockEncoding
    """    
    def __init__(self,ProblemInstance,control_val=None,**kwargs):
        self.PI                 =  ProblemInstance

        self._encoding_type     =  None

        self._controlled        =  False
        self._control_val       =  control_val

        if (self._control_val is not None):
            self._controlled    =  True


    @property
    @abc.abstractmethod
    def control_registers(self) -> Tuple[qt._infra.registers.Register, ...]:
        """returns the list of control registers from the circuit."""
        ...

    @property
    @abc.abstractmethod
    def selection_registers(self) -> Tuple[qt._infra.registers.Register, ...]:
        ...

    @property
    @abc.abstractmethod
    def target_registers(self) -> Tuple[qt._infra.registers.Register, ...]:
        ...


    @property
    @abc.abstractmethod
    def signature(self) -> qt._infra.registers.Signature:
        ...


    @property
    def alpha(self):
        return (self.PI.alpha)


    @property
    def all_qubits(self):
        helper = qt_test.GateHelper(self)
        return(helper.all_qubits)


    @property
    def circuit(self):
        helper = qt_test.GateHelper(self)
        return(helper.circuit)  





class BlockEncoding_select_prepare(BlockEncoding):


    def __init__(self, ProblemInstance, select_oracle=None, prepare_oracle=None, 
                    do_prepare=True, do_prepare_inverse=True, invert_select=False,**kwargs):

        self._do_prepare          =  do_prepare
        self._do_prepare_inverse  =  do_prepare_inverse

        self._select_gate         =  select_oracle
        self._prepare_gate        =  prepare_oracle

        self._invert_select       =  invert_select

        super().__init__(ProblemInstance,**kwargs)


    @property
    def selection_registers(self):
        return(self._select_gate.selection_registers)


    @property
    def control_registers(self):
        return(self._select_gate.control_registers)


    @property
    def target_registers(self):
        return(self._select_gate.target_registers)


    @property
    def junk_registers(self):
        other_regs = [*self.selection_registers,*self.control_registers,*self.target_registers]
        junk_regs = ()
        for reg in [*self._prepare_gate.signature]:
            if reg not in other_regs:
                junk_regs += (reg,)
        return(junk_regs)


    @property
    def signature(self):
        return qt._infra.registers.Signature(
            [*self.control_registers, *self.selection_registers, 
             *self.target_registers, *self.junk_registers] )



    def _t_complexity_(self) -> qt.cirq_interop.t_complexity_protocol.TComplexity:

        prepare_cost  =  t_complexity(self._prepare_gate)
        select_cost   =  t_complexity(self._select_gate)

        if (self._do_prepare and self._do_prepare_inverse):
            total_cost = 2*prepare_cost + select_cost
        else:
            total_cost = prepare_cost + select_cost

        return (total_cost)



    def decompose_from_registers(self, context, **quregs):

        if ( self._do_prepare ):
            yield self._prepare_gate.on_registers(**quregs)

        if self._invert_select:
            yield  cirq.inverse(self._select_gate.on_registers(**quregs))
        else:
            yield  self._select_gate.on_registers(**quregs)

        if ( self._do_prepare_inverse):
            yield cirq.inverse(self._prepare_gate.on_registers(**quregs))

