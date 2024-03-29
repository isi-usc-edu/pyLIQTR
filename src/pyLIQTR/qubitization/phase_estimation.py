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

import numpy    as  np
import cirq     as  cirq
import qualtran as  qt
import qualtran.cirq_interop.testing as qt_test


from pyLIQTR.qubitization.qubitized_gates   import   QubitizedReflection, QubitizedRotation, QubitizedWalkOperator


from pyLIQTR.BlockEncodings.BlockEncoding   import   BlockEncoding
from pyLIQTR.BlockEncodings.getEncoding     import   getEncoding
from pyLIQTR.BlockEncodings                 import   VALID_ENCODINGS







class QubitizedPhaseEstimation(qt._infra.gate_with_registers.GateWithRegisters):
    

    ##
    ## Check circuit conventions to see if we can reuse the
    ## phase qubit for reflections between iterations or if 
    ## we need independent values.
    ##
    

    def __init__( self, block_encoding, instance=None, prec=4, eps=None, multi_control_val=1 ):


        if (instance is not None):

            self._block_encoding          =  block_encoding(instance)
            self._block_encoding_ctl      =  block_encoding(instance,control_val=1)

        else:

            self._block_encoding          =  block_encoding
            self._block_encoding_ctl      =  getEncoding( instance=self._block_encoding.PI,
                                                          encoding=self._block_encoding._encoding_type, 
                                                          control_val=1 )

        alpha = self._block_encoding.alpha

        if (eps is not None):
            self._prec  =  int(np.ceil( np.log2( (np.sqrt(2.0)*np.pi*alpha) / (2*eps) ) ))         
        else:
            self._prec  =  prec

        self._multi_pauli_gate        =  cirq.Z
        self._multi_control_value     =  multi_control_val


        self._n_selection  =  0
        for reg in self.selection_registers:
            self._n_selection += reg.bitsize

        return



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
    def all_qubits(self):
        helper = qt_test.GateHelper(self)
        return(helper.all_qubits)
    
    @property
    def circuit(self):
        helper = qt_test.GateHelper(self)
        return(helper.circuit)   



    @property
    def signature(self):

        registers = []

        r_prec = qt._infra.registers.Register('prec',  self._prec + 1 )
        registers.append(r_prec)

        # r_phase = cirq_ft.Register('phase', self._prec-1  )
        # registers.append(r_phase)

        sig    =  qt._infra.registers.Signature( [ *registers,  
                                       *self.selection_registers, 
                                       *self.target_registers, 
                                       *self.junk_registers] )    

        return sig
    



    def decompose_from_registers(self, context, **quregs):

        selection_qubits = []

        # for reg in self.selection_registers:
        #     selection_qubits += quregs[ reg.name ].tolist()

        # kw_blockregs = { 'selection'  :  selection_qubits,
        #                  'target'     :  quregs['target'] }

        kw_blockregs = {} 
        
        for reg in self.target_registers:
            kw_blockregs[reg.name] = quregs[ reg.name ]

        for reg in self.selection_registers:
            selection_qubits += quregs[ reg.name ].tolist()
            kw_blockregs[reg.name] = quregs[ reg.name ]

        for reg in self.junk_registers:
            kw_blockregs[reg.name] = quregs[ reg.name ]

        qbs_prec   =  quregs['prec']
 #       qbs_phase  =  quregs['phase']

        qm = context.qubit_manager
        reg_phase = qm.qalloc(self._prec)  

        yield QubitizedWalkOperator(self._block_encoding_ctl,control_val=1).\
                    on_registers(**kw_blockregs,control=qbs_prec[0])
                                                                                 
        for n in range(1,self._prec+1):

            # prec_idx = self._prec-n-1

            yield QubitizedReflection( len(selection_qubits),
                                       control_val=0,
                                       multi_control_val=self._multi_control_value).\
                                       on_registers(controls=selection_qubits,target=qbs_prec[n])  
            
            for _ in range(0,int(2**(n - 1)) ):
                yield QubitizedWalkOperator(self._block_encoding_ctl,control_val=1).\
                        on_registers(**kw_blockregs,control=reg_phase[n-1])
                
            yield QubitizedReflection( len(selection_qubits),
                                       control_val=0,
                                       multi_control_val=self._multi_control_value).\
                                       on_registers(controls=selection_qubits,target=qbs_prec[n])    



    def _t_complexity_(self) -> qt.cirq_interop.t_complexity_protocol.TComplexity:

        walk_cost        =  qt.cirq_interop.t_complexity_protocol.t_complexity( QubitizedWalkOperator( self._block_encoding ))
        walk_cost_total  =  qt.cirq_interop.t_complexity_protocol.TComplexity()

        for n in range(1,self._prec+1):
            walk_cost_total  +=  walk_cost*int(2**(n - 1))

        walk_cost_total  +=  walk_cost

        reflection_cost   =  qt.cirq_interop.t_complexity_protocol.t_complexity( QubitizedReflection( self._n_selection,control_val=0 ))

        return ( walk_cost_total + 2*(self._prec)*reflection_cost)








