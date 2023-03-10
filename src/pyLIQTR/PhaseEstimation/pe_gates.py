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

from openfermion.circuits             import trotterize_exp_qubop_to_qasm

from openfermion                      import jordan_wigner
from cirq.contrib                     import qasm_import
from pyLIQTR.utils.utils              import open_fermion_to_qasm


class S_k(cirq.Gate):
    """
    A gate for rotating away previously measured bits in a phase that is being estimated.

    Parameters:
        qubit : the cirq qubit that the gate is acting on

        prev_bits : A list containing only 1's or 0's for the previously measured bits

        verbose : A boolean flag. Set to True to get debug info output to screen

    Returns:
        N/A : Does stick the gate in the circuit where you want it.

    """
    def __init__(self, qubit, prev_bits, verbose=False):# , control_key=''):
        self.qubit     = qubit
        self.prev_bits = prev_bits
        self.verbose   = verbose
        # self.control_key = control_key

    def _num_qubits_(self) -> int:
        return 1

    def _circuit_diagram_info_(self,args):
        return 'S_k'

    ### Testing
    def _decompose_(self, qubits):
        sum_val  = 0
        if not self.prev_bits:
            sum_val = 0
            if self.verbose:
                print('-'*40)
                print(f'>> No previous bits, ==> sum_val = {sum_val}')
        else:
            for idx_, bit_ in enumerate(self.prev_bits):#[::-1]):
                sum_val += bit_/2**(idx_+2)

        yield cirq.PhasedXZGate(
            x_exponent=4, z_exponent=-2*sum_val, axis_phase_exponent=0).on(self.qubit)

### Not currently used, should be updated in the future.
class Trotter_Unitary(cirq.Gate):

    def __init__(self, tgt_qubit, reg_qubits, prec_order, operator_power, **kwargs):
        
        # Qubit stuff
        self.tgt_qubit    = tgt_qubit
        self.reg_qubits   = reg_qubits
        self.prec_order   = prec_order
        
        # Trotter param stuff
        self.trotter_args = kwargs['kwargs']
        self.ev_time      = 1
        self.trot_num     = 1
        self.trot_ord     = 1

        # Get these from the kwargs dictionary
        if 'ev_time' in self.trotter_args.keys():
            self.ev_time = self.trotter_args['ev_time']

        if 'trot_num' in self.trotter_args.keys():
            self.trot_num = self.trotter_args['trot_num']

        if 'trot_ord' in self.trotter_args.keys():
            self.trot_ord = self.trotter_args['trot_ord']

        # apply jordan-wigner transform
        self.jw_ham = jordan_wigner(self.trotter_args['mol_ham'])

        # Else:
        self.operator_power = operator_power

        # A circuit:
        # self.build_circuit = cirq.Circuit()


    def _num_qubits_(self) -> int:
        return (1 + len(self.reg_qubits))

    def _decompose_(self, qubits):    
    
        # Note: need factor of 2 bc of way OF trotterizes...
        # Note: Since we're doing this in a for loop below, the jw_ham**2**1 is always **2**1, bc we're putting them next to eachother.
        jw_ham_trotterized_generator = trotterize_exp_qubop_to_qasm(self.jw_ham*2**(1)*self.operator_power, evolution_time=self.ev_time, trotter_number=self.trot_num, trotter_order=self.trot_ord)
        jw_ham_trotterized_circuit   = open_fermion_to_qasm(self.num_qubits(), jw_ham_trotterized_generator, reg_name='reg') 
        tmp_circuit                  = qasm_import.circuit_from_qasm(jw_ham_trotterized_circuit)

        # Apply control operations on the z-rotations:
        for _ in range(2**self.prec_order):
            for moment in tmp_circuit:
                for op in moment:
                    if 'Rz' in str(op):
                        angle = float(str(op.gate).split('(')[-1].split('π')[0])
                        angle *= np.pi             
                        new_op = cirq.ControlledGate(cirq.Rz(rads=angle)).on(self.tgt_qubit,op.qubits[0])
                        # print('-----------')
                        # print(new_op)
                        # print(type(new_op))
                        yield(new_op)
                    else:
                        yield(op)
        
        ### ---> Old, was here, not tested <---
        # jw_ham_trotterized_generator = trotterize_exp_qubop_to_qasm(self.jw_ham*2**(self.prec_order+1), evolution_time=self.ev_time, trotter_number=self.trot_num, trotter_order=self.trot_ord)
        # jw_ham_trotterized_circuit   = open_fermion_to_qasm(self.num_qubits, jw_ham_trotterized_generator, reg_name='reg') 
        # tmp_circuit                  = qasm_import.circuit_from_qasm(jw_ham_trotterized_circuit)
        # 
        # Apply control operations on the z-rotations:
        # for moment in tmp_circuit:
        #     for op in moment:
        #         if 'Rz' in str(op):
        #             angle = float(str(op.gate).split('(')[-1].split('π')[0])
        #     
        #             angle *= np.pi # TODO: Check this factor of 2
        #             # angle *= 2**(precision_order)
        #     
        #             new_op = cirq.ControlledGate(cirq.Rz(rads=angle)).on(self.precision_qubit,op.qubits[0])
        #             yield (new_op)
        #         else:
        #             yield (op)
        ### ---> End of Old, was here, not tested <---
        

    def _circuit_diagram_info_(self, args):
        return [f"c-U**2**{self.prec_order+1}"] * self.num_qubits()

class PhaseOffset(cirq.Gate):
    """
    A gate for adding a phase offset to the |1> portion of a Hadamard|0>'ed state. This is for 
    GSE, but is implemented in PE bc of how the circuit is built

    Parameters:
        qubit : the cirq qubit that the gate is acting on

        phase_offset : the phase offset to be applied

        verbose : A boolean flag. Set to True to get debug info output to screen

    Returns:
        N/A : Does stick the gate in the circuit where you want it.

    """
    def __init__(self, qubit, phase_offset, verbose=False):
        self.qubit        = qubit
        self.phase_offset = phase_offset
        self.verbose      = verbose

    def _num_qubits_(self) -> int:
        return 1

    def _circuit_diagram_info_(self,args):
        return f'PhaseOffset {self.phase_offset:0.2f}'

    def _decompose_(self, qubits):

        yield cirq.PhasedXZGate(
            x_exponent=4, z_exponent=-1*self.phase_offset/np.pi, axis_phase_exponent=0).on(self.qubit)

    ###
    ### Not used bc doesnt decompose well into clifford+t
    ###
    # def _unitary_(self):
         
    #     exp_val = np.exp(-1j*self.phase_offset)
    #     if self.verbose:
    #         print('-'*40)
    #         print(f'>> Phase Offset= {self.phase_offset}')

    #     return np.array([
    #         [1.0,   0.0  ],
    #         [0.0, exp_val],
    #     ])
