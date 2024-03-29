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
import  numpy                  as  np
import  cirq                   as  cirq
import  qualtran               as  qt
import  qualtran.cirq_interop.testing  as  qt_test
 
from    pyLIQTR.utils.pauli_string_manip  import  convert_to_dense_pauli_string

import abc
from   typing import Tuple




class BlockEncoding(qt._infra.gate_with_registers.GateWithRegisters):
    
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
        ...

    @property
    @abc.abstractmethod
    def selection_registers(self) -> Tuple[qt._infra.registers.SelectionRegister, ...]:
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
        if hasattr(self._prepare_gate, 'junk_registers'):
            return(self._prepare_gate.junk_registers)
        else:
            return( () )


    @property
    def signature(self):
        return qt._infra.registers.Signature(
            [*self.control_registers, *self.selection_registers, 
             *self.target_registers, *self.junk_registers] )
 #       return( self._get_common_signature( [self._select_gate,self._prepare_gate] ) )



    def _t_complexity_(self) -> qt.cirq_interop.t_complexity_protocol.TComplexity:

        prepare_cost  =  qt.cirq_interop.t_complexity_protocol.t_complexity(self._prepare_gate)
        select_cost   =  qt.cirq_interop.t_complexity_protocol.t_complexity(self._select_gate)

        if (self._do_prepare and self._do_prepare_inverse):
            total_cost = 2*prepare_cost + select_cost
        else:
            total_cost = prepare_cost + select_cost

        return (total_cost)



    def decompose_from_registers(self, context, **quregs):

#        quregs         =   cirq_test.GateHelper(self).quregs

        if ( self._do_prepare ):
            yield self._prepare_gate.on_registers(**quregs)

        if self._invert_select:
            yield  cirq.inverse(self._select_gate.on_registers(**quregs))
        else:
            yield  self._select_gate.on_registers(**quregs)

        if ( self._do_prepare_inverse):
            yield cirq.inverse(self._prepare_gate.on_registers(**quregs))



