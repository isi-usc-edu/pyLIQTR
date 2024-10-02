"""
Copyright (c) 2024 Massachusetts Institute of Technology 
SPDX-License-Identifier: BSD-2-Clause
"""

################################################################################
###                                                                          ###
###          QUBITIZED_GATES  -  Qubitized quantum circuit components        ###
###                                                                          ###
################################################################################


import  numpy                  as  np
import  cirq                   as  cirq 
import  qualtran               as  qt
import  qualtran.cirq_interop.testing  as  qt_test
from pyLIQTR.utils.resource_analysis import pylqt_t_complexity as t_complexity

from qualtran import Register, QBit, Signature, GateWithRegisters



class QubitizedReflection(qt._infra.gate_with_registers.GateWithRegisters):

    def __init__(self,n_controls,target_gate=cirq.Z,multi_control_val=1,control_val=1):

        self._n_controls         =  n_controls
        self._target_gate        =  target_gate
        self._multi_control_val  =  multi_control_val
        self._control_val        =  control_val

        return


    @property
    def all_qubits(self):
        helper = qt_test.GateHelper(self)
        return(helper.all_qubits)


    @property
    def circuit(self):
        helper = qt_test.GateHelper(self)
        return(helper.circuit)   


    @property
    def signature(self):
        return qt._infra.registers.Signature.build( controls=self._n_controls,
                                        target=1 )
    


    def decompose_from_registers(self, context, **quregs):

        controls  =  quregs['controls']
        target    =  quregs['target']

        if not self._multi_control_val:
            for qb in controls:
                yield cirq.X.on(qb) 

        if not self._control_val:
            yield cirq.X.on(target[0])


        yield qt.bloqs.mcmt.MultiControlPauli( [1] * len(controls), 
                                               target_gate=self._target_gate).\
                                                on_registers(controls=[[bit] for bit in controls], target=target)
        
        if not self._control_val:
            yield cirq.X.on(target[0])
            
        if not self._multi_control_val:
            for qb in controls:
                yield cirq.X.on(qb) 



    def _t_complexity_(self) -> qt.cirq_interop.t_complexity_protocol.TComplexity:
        multi_cost =  t_complexity( qt.bloqs.mcmt.MultiControlPauli([0] * self._n_controls, 
                                            target_gate=self._target_gate))
    
        if (not self._control_val):
            clifford_cost = qt.cirq_interop.t_complexity_protocol.TComplexity(clifford=self._n_controls)
        else:
            clifford_cost = qt.cirq_interop.t_complexity_protocol.TComplexity()
        
        return (multi_cost + 2*clifford_cost)



    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs) -> cirq.CircuitDiagramInfo:
            
        if (self._target_gate == cirq.X):
            tgt_symb = 'X'
            ctl_symb = '@'
        else:
            tgt_symb = '@'
            ctl_symb = 'R'

        wire_symbols  = [ctl_symb if self._multi_control_val else ctl_symb+'(0)'] * self._n_controls
        wire_symbols += [tgt_symb if self._control_val else tgt_symb+'(0)'] 

        return cirq.CircuitDiagramInfo(wire_symbols=wire_symbols)









class QubitizedRotation(qt._infra.gate_with_registers.GateWithRegisters):
    

    def __init__(self, n_controls, 
                       rads=np.pi/2,
                       multi_target_gate=cirq.X,
                       rotation_gate=cirq.Rz,
                       multi_control_val=1,
                       multi_target_val=1,
                       rotation_control_val=1,
                       control_val=None,
                       rotation_control_qubit=None   ):

        self._rads                 =  rads                          # Rotation angle

        self._control_val          =  control_val                   # Flag to control rotation angle
        self._multi_target_val     =  multi_target_val           
        self._rotation_control_val =  rotation_control_val
        self._n_controls           =  n_controls                    # Number of controls
        self._multi_target_gate    =  multi_target_gate                   # Target gate for multi-controlled Pauli ops.
        self._rotation_gate        =  rotation_gate                 # Type of rotation gate
        self._multi_control_val    =  multi_control_val              # Condition value for multi-controlled Pauli
        self._rotation_control_qubit    =  rotation_control_qubit   # Additional qubit used to control rotation gate


        self._controlled = False
        if (self._control_val is not None):
            self._controlled = True

        return


    @property
    def all_qubits(self):
        helper = qt_test.GateHelper(self)
        return(helper.all_qubits)


    @property
    def circuit(self):
        helper = qt_test.GateHelper(self)
        return(helper.circuit)  


    @property
    def signature(self):

        if self._controlled:
            return qt._infra.registers.Signature.build( control=self._n_controls,
                                            target=1, rotation_control=1 )
        else:
            return qt._infra.registers.Signature.build( control=self._n_controls,
                                            target=1 )
    


    def decompose_from_registers(self, context, **quregs):

        controls  =  quregs['control']
        targets   =  quregs['target']

        # if self._controlled:
        #     qb_rot_ctl = quregs['rotation_control']
        

        if (not self._multi_control_val):
            for control in controls:
                yield cirq.X(control)


        yield QubitizedReflection( len(controls),
                                   target_gate=self._multi_target_gate,
                                   control_val=self._multi_target_val,
                                   multi_control_val=self._multi_control_val).\
                                    on_registers(controls=quregs['control'],target=quregs['target'])      
        
        
        if self._controlled:
        
            qb_rot_ctl = quregs['rotation_control']


            if not self._control_val:     
                yield cirq.X.on(qb_rot_ctl[0])

            yield self._rotation_gate(rads=self._rads).on(targets[0]).controlled_by(qb_rot_ctl[0])

            if not self._control_val:     
                yield cirq.X.on(qb_rot_ctl[0])

        else:
            yield self._rotation_gate(rads=self._rads).on(targets[0])

        yield QubitizedReflection( len(controls),
                                   target_gate=self._multi_target_gate,
                                   control_val=self._multi_target_val,
                                   multi_control_val=self._multi_control_val).\
                                   on_registers(controls=quregs['control'], target=quregs['target'])  
        
                            
        if (not self._multi_control_val):
            for control in controls:
                yield cirq.X(control)
                
                
                
    def _t_complexity_(self) -> qt.cirq_interop.t_complexity_protocol.TComplexity:

        multi_cost =  t_complexity( QubitizedReflection(self._n_controls) )
        rotation_cost = qt.cirq_interop.t_complexity_protocol.TComplexity(rotations=1)

        ##
        ##  Fix this to properly handle controlled rotation
        ##

        if (not self._multi_control_val):
            clifford_cost = qt.cirq_interop.t_complexity_protocol.TComplexity(clifford=self._n_controls)
        else:
            clifford_cost = qt.cirq_interop.t_complexity_protocol.TComplexity()
        
        return (2*multi_cost + rotation_cost + 2*clifford_cost)
    


    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs) -> cirq.CircuitDiagramInfo:

        ctl_symb = '@'
        if (self._rotation_gate == cirq.Rz):
            tgt_conv = "z"
        else:
            tgt_conv = "x"

        tgt_symb      = 'R'+tgt_conv+'('+"{:2.2f}".format(self._rads/np.pi)+'π)'

        wire_symbols  = [ctl_symb if self._multi_control_val else ctl_symb+'(0)'] * self._n_controls 

        if (self._controlled):
            wire_symbols +=  [ctl_symb if self._control_val else ctl_symb+'(0)']

        wire_symbols += [tgt_symb ]         

        return cirq.CircuitDiagramInfo(wire_symbols=wire_symbols)










class QubitizedWalkOperator(qt._infra.gate_with_registers.GateWithRegisters):
    
    def __init__( self, block_encoding, multi_control_val=1, control_val=1, instance=None  ):

        self._control_val             =  control_val
        self._multi_control_val       =  multi_control_val

        # self._block_encoding          =  block_encoding
        # self._controlled_encoding     =  block_encoding._controlled

        if (instance is not None):
            self._block_encoding            =  block_encoding(instance,control_val=self._control_val)
        else:
            self._block_encoding            =  block_encoding


        self._controlled_encoding     =  self._block_encoding._controlled


        self._multi_target_gate       =  cirq.Z

        self._n_selection  =  0
        for reg in self.selection_registers:
            self._n_selection += reg.bitsize

        self._n_targets  =  0
        for reg in self.target_registers:
            self._n_targets += reg.bitsize

        self._n_junk  =  0
        for reg in self.junk_registers:
            self._n_junk += reg.bitsize

        return


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
        other_regs = [*self.selection_registers,*self.control_registers,*self.target_registers]
        junk_regs = ()
        for reg in [*self._block_encoding.signature]:
            if reg not in other_regs:
                junk_regs += (reg,)
        return(junk_regs)


    @property
    def signature(self):

        if not self._controlled_encoding:
            ctl_reg = (qt._infra.registers.Register('control', QBit()),)
        else:
            ctl_reg = self.control_registers

        return qt._infra.registers.Signature(
            [ *ctl_reg, *self.selection_registers, 
              *self.target_registers, *self.junk_registers] )


    def decompose_from_registers(self, context, **quregs):

        selection_qubits = []

        kw_blockregs = {} 

        for reg in self.target_registers:
            kw_blockregs[reg.name] = quregs[ reg.name ]

        for reg in self.selection_registers:
            selection_qubits += quregs[ reg.name ].tolist()
            kw_blockregs[reg.name] = quregs[ reg.name ]

        for reg in self.junk_registers:
            kw_blockregs[reg.name] = quregs[ reg.name ]
        
        if self._block_encoding._controlled:
            kw_blockregs['control']  =  quregs['control']

        yield self._block_encoding.on_registers(**kw_blockregs)

        yield QubitizedReflection( len(selection_qubits),
                                   target_gate=self._multi_target_gate,
                                   control_val=self._control_val,
                                   multi_control_val=self._multi_control_val).\
                                    on_registers(controls=selection_qubits,target=quregs['control'])      
        


    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs) -> cirq.CircuitDiagramInfo:

        wire_symbols  = ['@' if self._control_val else '@(0)'] 
        wire_symbols += ['W'] * (self._n_selection + self._n_targets + self._n_junk)

        return cirq.CircuitDiagramInfo(wire_symbols=wire_symbols)



    def _t_complexity_(self) -> qt.cirq_interop.t_complexity_protocol.TComplexity:

        encoding_cost  =  t_complexity(self._block_encoding)
        reflect_cost   =  t_complexity( QubitizedReflection( self._n_selection ))
        
        return (reflect_cost + encoding_cost)









