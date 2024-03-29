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

################################################################################
###                                                                          ###
###           QSVT  -  Circuits and methods for QSP, QET, and QSVT           ###
###                                                                          ###
################################################################################
from typing import Any, DefaultDict, Iterable, Optional, Tuple, TypeVar, Union
import  numpy                  as  np
import  cirq                   as  cirq
import  qualtran               as  qt
import  qualtran.cirq_interop.testing  as  qt_test
from cirq import QasmArgs

from  pyLIQTR.phase_factors.phase_factors   import  PhaseFactors
from  pyLIQTR.qubitization.qubitized_gates  import  QubitizedReflection, QubitizedRotation
###
### Need to add phase regsiter attribute (like .selection_registers)
###

TDefault = TypeVar('TDefault')
RaiseTypeErrorIfNotProvided: Any = ([],)



class QSVT_abstract(qt._infra.gate_with_registers.GateWithRegisters):
    def __init__( self, 
                  block_encoding, 
                  phis, 
                #   instance=None, 
                  phase_convention='R',
                #   control_val=None,
                  multi_control_val=1,
                  sequence_name=None ):

        self._sequence_name           =  sequence_name

        if isinstance(phis,PhaseFactors):
            self._phis                    =  phis.phis
        else:
            self._phis                    =  phis

 #      self._phis                    =  phis
        self._n_phis                  =  len(self._phis)
        self._phase_convention        =  phase_convention

        self._multi_control_val       =  multi_control_val
        self._multi_target_val        =  1

        ## Set rotataion and reflection / projection gates based on 
        ## the phase angle convention.
        ##
        ##
        if (self._phase_convention == 'R'):
            self._rotation_gate       =  cirq.Rz
            self._multi_target_gate   =  cirq.X
        elif (self._phase_convention == 'Wx'):
            self._rotation_gate       =  cirq.Rz
            self._multi_target_gate   =  cirq.X
        elif (self._phase_convention == 'Wz'):
            self._rotation_gate       =  cirq.Rx
            self._multi_target_gate   =  cirq.Z


        ## Calculate the number of controls on the the qubitized rotation gate
        ## based on the number of qubits in the selection register.
        ##

        self._n_rotation_controls  =  self.num_qubits_in_registers(self.selection_registers)

    def num_qubits_in_registers(self,registers):
        num = 0
        for reg in registers:
            num += reg.bitsize
        return (num)      
    
    @property
    def all_qubits(self):
        helper = qt_test.GateHelper(self)
        return(helper.all_qubits)

    @property
    def circuit(self):
        helper = qt_test.GateHelper(self)
        return(helper.circuit)  
    
    @property
    def selection_registers(self):
        return(self._block_encoding.selection_registers)

    @property
    def control_registers(self):
        return(self._block_encoding.control_registers)

    @property
    def target_registers(self):
        return(self._block_encoding.target_registers)

    @property
    def junk_registers(self):
        return(self._block_encoding.junk_registers)

    @property
    def phase_registers(self):
        return( [qt._infra.registers.Register('phase',1)] )

    # @property
    # def phase_registers(self):
    #     return(self.junk_registers)

    @property
    def signature(self):

    #    phase_reg =  (cirq_ft.Register('phase',1),)

        if self._controlled:
            ctl_reg =  (qt._infra.registers.Register('control',1),)
        else:
            ctl_reg = ()

        return qt._infra.registers.Signature( [ *ctl_reg, *self.selection_registers, 
              *self.target_registers, *self.junk_registers, *self.phase_registers] )
    
#            [ *phase_reg, *ctl_reg, *self.selection_registers, 
#              *self.target_registers, *self.junk_registers, *self.phase_registers] )

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs) -> cirq.CircuitDiagramInfo:
            
##
##  Add phase sequence tag.
##

        title = 'QSVT'
        if (self._sequence_name is not None):
            title +=  (  '('+str(self._sequence_name)+')' )

        wire_symbols  = [title]
#        wire_symbols  += ['#']*(self.num_qubits()-1)
        wire_symbols  += ['QSVT']*(self.num_qubits()-1)

        return cirq.CircuitDiagramInfo(wire_symbols=wire_symbols)


class QSVT_real_polynomial(QSVT_abstract):
    def __init__( self, block_encoding, phis, instance=None, control_val=None,  **kwargs ):


        self._control_val             =  control_val
        self._controlled              =  False

        if (self._control_val is not None):
            self._controlled = True

        ## If uninstantiated block encoding and problem instance are supplied, 
        ## go ahead and build the encoding.  Otherwise check the exiting encoding to 
        ## ensure compatibilty with the sequence control conventions.
        ##
        if (instance is not None):

            self._block_encoding      =  block_encoding(instance,control_val=control_val)

        else:

            self._block_encoding          =  block_encoding

            if (self._block_encoding._controlled != self._controlled):
                raise Exception("Block encoding and QSVT sequence must share a common control strategy.")
            if (self._block_encoding._control_val != self._control_val):
                raise Exception("Block encoding and QSVT squence common control condition.")         


        super().__init__(block_encoding, phis, **kwargs)

    def decompose_from_registers(self, context, **quregs):

        selection_qubits = []
        junk_qubits      = []

        kw_block_regs     = { 'target'     :  quregs['target'] } 
        kw_rotation_regs  = { 'target'     :  quregs['phase']  } 

        for reg in self.selection_registers:
            selection_qubits += quregs[ reg.name ].tolist()
            kw_block_regs[reg.name]     =  quregs[ reg.name ]

        ## Need since we allow these to be entangled with preparation
        ## registers (do some for qubitized walk.)
        ##
        for reg in self.junk_registers:
            junk_qubits += quregs[ reg.name ].tolist()
            kw_block_regs[reg.name]     =  quregs[ reg.name ]
            kw_rotation_regs[reg.name]  =  quregs[ reg.name ]

        kw_rotation_regs['control']  =  selection_qubits + junk_qubits

        reg_phase    =  quregs['phase']

        if self._controlled:
            kw_block_regs['control']              =  quregs['control']
            kw_rotation_regs['rotation_control']  =  quregs['control']

        n_rotation_controls = len(selection_qubits) + len(junk_qubits) 


        kw_rotation_args  = { 'control_val'         :   self._control_val,
                              'rotation_gate'       :   self._rotation_gate,
                              'multi_control_val'   :   self._multi_control_val,
                              'multi_target_gate'   :   self._multi_target_gate  }


        yield cirq.H.on(reg_phase[0])
        yield QubitizedRotation( n_rotation_controls,
                                 rads=self._phis[0], **kw_rotation_args).\
                                    on_registers(**kw_rotation_regs)  
        yield cirq.Z.on(reg_phase[0])

        for phi in self._phis[1:-1]:
            yield self._block_encoding.on_registers(**kw_block_regs)
            yield QubitizedRotation( n_rotation_controls, rads=phi, **kw_rotation_args).\
                                        on_registers(**kw_rotation_regs)  
            yield cirq.Z.on(reg_phase[0])


        yield self._block_encoding.on_registers(**kw_block_regs)
        yield QubitizedRotation( n_rotation_controls,rads=self._phis[-1], **kw_rotation_args).\
                                    on_registers(**kw_rotation_regs) 
        yield cirq.H.on(reg_phase[0])

        


    def _t_complexity_(self) -> qt.cirq_interop.t_complexity_protocol.TComplexity:

        n_reflect_controls  =  0

        for reg in self.selection_registers:
            n_reflect_controls += reg.bitsize

        for reg in self.junk_registers:
            n_reflect_controls += reg.bitsize


        encoding_cost   =  qt.cirq_interop.t_complexity_protocol.t_complexity(self._block_encoding)
        rotation_cost   =  qt.cirq_interop.t_complexity_protocol.t_complexity( QubitizedRotation( n_reflect_controls, control_val=self._control_val ))
        clifford_cost   =  qt.cirq_interop.t_complexity_protocol.TComplexity(clifford=(2 + 2*(self._n_phis-1)))

        return ( rotation_cost*(self._n_phis) + encoding_cost*(self._n_phis-1) + clifford_cost)

    #def _qasm_(self, args: 'cirq.QasmArgs') -> str:
    def _qasm_(
        self,
        *,
        args: Optional[QasmArgs] = None,
        qubits: Optional[Iterable['cirq.Qid']] = None,
        default: DefaultDict = RaiseTypeErrorIfNotProvided,
    ) -> Union[str, TDefault]:
        # args.validate_version('2.0')
        # # build up all the qubits, minus the ancilla ones for now
        # allQ = [
        #     *self.selection_registers,
        #     *self.control_registers,
        #     *self.target_registers,
        #     *self.phase_registers
        #     ]

        # ops = self.decompose_from_registers(allQ)
        # return QasmOutput(
        #     operations=self._block_encoding,
        #     qubits=allQ,
        #     # header=None,
        #     # precision=10,
        #     # version='2.0',
        # )
        raise NotImplementedError













class QSVT_real_polynomial_sum(qt._infra.gate_with_registers.GateWithRegisters):
    
    def __init__( self, 
                  block_encoding, 
                  phis, 
                  instance=None, 
                  ctl_s_gate=False, 
                  ctl_z_gate=False, 
                  ctl_h_gate=True,  
                  **kwargs ):

        ## Combo doesn't have an exernal control (for now)
        ##
        self._controlled              =  False

        self._control_val_0           =  0
        self._control_val_1           =  1


        ## Note the flip in indexing vs. labels: 
        ##   We condition the first argument on |1> and 
        ##   the second on |0>
        ##
        if isinstance(phis[0],PhaseFactors):
            self._phis_0     =   phis[1].phis
            self._phis_1     =   phis[0].phis
            name_0           =   phis[1].name
            name_1           =   phis[0].name

        else:
            self._phis_0     =  phis[1]
            self._phis_1     =  phis[0]
            name_0           =  "{phis}_0"
            name_1           =  "{phis}_1"

        # self._phis_0                  =  phis[0]
        # self._phis_1                  =  phis[1]

  #      self.n_phis                   =  [len(self._phis_0),len(self._phis_1)]

        self._ctl_s_gate              =  ctl_s_gate
        self._ctl_z_gate              =  ctl_z_gate
        self._ctl_h_gate              =  ctl_h_gate

        self.qsvt_gate_0   =   QSVT_real_polynomial( block_encoding, 
                                                     phis=self._phis_0,
                                                     instance=instance,
                                                     sequence_name=name_0,
                                                     control_val=self._control_val_0  )
        
        self.qsvt_gate_1   =   QSVT_real_polynomial( block_encoding, 
                                                     phis=self._phis_1,
                                                     instance=instance,
                                                     sequence_name=name_1,
                                                     control_val=self._control_val_1  )

        super().__init__(**kwargs)


    @property
    def all_qubits(self):
        helper = qt_test.GateHelper(self)
        return(helper.all_qubits)


    @property
    def circuit(self):
        helper = qt_test.GateHelper(self)
        return(helper.circuit)  


    @property
    def n_phis(self):
        return( (len(self._phis_0),len(self._phis_1)) )


    @property
    def selection_registers(self):
        return(self.qsvt_gate_0.selection_registers)


    @property
    def control_registers(self):
        return(self.qsvt_gate_0.control_registers)


    @property
    def target_registers(self):
        return(self.qsvt_gate_0.target_registers)


    @property
    def junk_registers(self):
        return(self.qsvt_gate_0.junk_registers)


    @property
    def phase_registers(self):
        return(self.qsvt_gate_0.phase)


    @property
    def signature(self):

        phase_reg =  (qt._infra.registers.Register('phase',1),)

        if self._controlled:
            ctl_reg =  (qt._infra.registers.Register('control',1),)
        else:
            ctl_reg = ()

        return qt._infra.registers.Signature(
            [ *phase_reg, *self.control_registers,
              *self.target_registers,  *self.selection_registers, 
              *self.junk_registers] )


    def decompose_from_registers(self, context, **quregs):


        kw_qsvt_regs      = { 'control'     :   quregs['control'],
                              'phase'       :   quregs['phase'],
#                              'selection'   :   quregs['selection'],
                              'target'      :   quregs['target']  }



        for reg in self.selection_registers:
 #           selection_qubits += quregs[ reg.name ].tolist()
            kw_qsvt_regs[reg.name] = quregs[ reg.name ]

        for reg in self.junk_registers:
            kw_qsvt_regs[reg.name] = quregs[ reg.name ]


        if self._ctl_h_gate:
            yield cirq.H.on(quregs['control'][0])

        yield self.qsvt_gate_1.on_registers(**kw_qsvt_regs)  
        
        if self._ctl_z_gate:
            yield cirq.Z.on(quregs['control'][0])
        if self._ctl_s_gate:
            yield cirq.S.on(quregs['control'][0])

        yield self.qsvt_gate_0.on_registers(**kw_qsvt_regs)  
        
        if self._ctl_h_gate:
            yield cirq.H.on(quregs['control'][0])
        

    def _t_complexity_(self) -> qt.cirq_interop.t_complexity_protocol.TComplexity:

        n_reflect_controls  =  0

        encoding_cost_0   =  qt.cirq_interop.t_complexity_protocol.t_complexity(self.qsvt_gate_0)
        encoding_cost_1   =  qt.cirq_interop.t_complexity_protocol.t_complexity(self.qsvt_gate_1)

        cliff  =  2*self._ctl_h_gate + self._ctl_s_gate + self._ctl_z_gate

        clifford_cost   =  qt.cirq_interop.t_complexity_protocol.TComplexity(clifford=cliff)

        return ( encoding_cost_0 + encoding_cost_1 + clifford_cost)






  





class QSP_fourier_response(QSVT_abstract):
    

    def __init__( self, block_encoding, phis, instance=None, **kwargs ):

        self._controlled              =  False

        self._control_val             =  None

        self._invert_later            =  False

        if (instance is not None):
            self._block_encoding_prep      =  block_encoding(instance,do_prepare_inverse=False,control_val=1)
            self._block_encoding_prep_inv  =  block_encoding(instance,do_prepare=False,invert_select=True,control_val=1)
            self._block_encoding           =  self._block_encoding_prep
        else:

            if (block_encoding._do_prepare_inverse != False):
                raise Exception("Block encoding must only contain a prepare and select.")
            if (block_encoding._controlled != False):
                raise Exception("Block encoding be controlled and the control conditioned on |1>.")

            self._block_encoding            =  block_encoding
            self._block_encoding_prep       =  self._block_encoding
            self._block_encoding_prep_inv   =  self._block_encoding
            self._invert_later              =  True
        
        super().__init__(self._block_encoding, phis, phase_convention='Wz', **kwargs)
            

    def decompose_from_registers(self, context, **quregs):

        selection_qubits = []
        junk_qubits      = []

        kw_block_regs     = { 'target'     :  quregs['target'] } 
        kw_rotation_regs  = { 'target'     :  quregs['phase']  } 

        for reg in self.selection_registers:
            selection_qubits += quregs[ reg.name ].tolist()
            kw_block_regs[reg.name]     =  quregs[ reg.name ]

        for reg in self.junk_registers:
            junk_qubits += quregs[ reg.name ].tolist()
            kw_block_regs[reg.name]     =  quregs[ reg.name ]
            kw_rotation_regs[reg.name]  =  quregs[ reg.name ]

        kw_rotation_regs['control']  =  selection_qubits + junk_qubits

        reg_phase    =  quregs['phase']


        n_rotation_controls = len(selection_qubits) + len(junk_qubits) 


        kw_rotation_args  = { 'control_val'         :   self._control_val,
                              'rotation_gate'       :   self._rotation_gate,
                              'multi_control_val'   :   self._multi_control_val,
                              'multi_target_gate'   :   self._multi_target_gate  }

        kw_block_regs['control']  =  reg_phase

        def encoding_plus_rotation(phi,kw_block_regs):

            yield self._block_encoding_prep.on_registers(**kw_block_regs)
            yield cirq.Ry(rads=phi).on(reg_phase[0])
            if self._invert_later:
                yield cirq.inverse(self._block_encoding_prep_inv.on_registers(**kw_block_regs))
            else:
                yield self._block_encoding_prep_inv.on_registers(**kw_block_regs)


        yield cirq.Rx(rads=self._phis[0]).on(reg_phase[0])


        for n in range(1,int((self._n_phis-1)/2)):
            yield encoding_plus_rotation(self._phis[2*n - 1],kw_block_regs)
            yield QubitizedRotation( n_rotation_controls,
                                     rads=self._phis[2*n], **kw_rotation_args).\
                                     on_registers(**kw_rotation_regs)  
        

        yield encoding_plus_rotation(self._phis[-2],kw_block_regs)
        yield cirq.Rx(rads=self._phis[-1]).on(reg_phase[0])



    def _t_complexity_(self) -> qt.cirq_interop.t_complexity_protocol.TComplexity:

        n_reflect_controls  =  0

        for reg in self.selection_registers:
            n_reflect_controls += reg.bitsize

        for reg in self.junk_registers:
            n_reflect_controls += reg.bitsize



        encoding_cost   =  qt.cirq_interop.t_complexity_protocol.t_complexity(self._block_encoding)
        rotation_cost   =  qt.cirq_interop.t_complexity_protocol.t_complexity( QubitizedRotation( n_reflect_controls))
        clifford_cost   =  qt.cirq_interop.t_complexity_protocol.TComplexity(clifford=(2 + 2*(self._n_phis-1)))

        return ( rotation_cost*(self._n_phis) + encoding_cost*(self._n_phis-1) + clifford_cost)





