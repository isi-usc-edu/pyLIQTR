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
import tqdm
import numpy  as np
import pandas as pd

from openfermion.circuits             import trotterize_exp_qubop_to_qasm

from openfermion                      import jordan_wigner
from cirq.contrib                     import qasm_import
from pyLIQTR.PhaseEstimation.pe_gates import S_k, PhaseOffset, Trotter_Unitary
from pyLIQTR.utils.utils              import open_fermion_to_qasm

class PhaseEstimation:
    """
    A class for doing phase estimation.

    Attributes:
        precision_order: The bit-precision to which you want to do the phase estimation

        init_state: A python list containing the state occupancy for the initial state

        phase_estimate: The calculated phase estimate.

        pe_circuit: The phase estimation cirq circuit

        phase_offset: A float that represents a phase offset. Used for GSE, and initialized to 0

        n_qubits: (int) = len(init_state) + 1

        bit_list: The python list of measured bits

        unitary_kw_args: Keyword arguments in a dict that describe informatoin about the unitary

        regsiter_qubits: A python list of the register (psi state) qubits

        precision_qubit: The cirq qubit that is used for measuring bits

        operator_power: To raise the entire U**2**j to another power. 
    
    Methods:
        generate

    """
    def __init__(
        self,
        precision_order:int,
        init_state:list,
        phase_offset:float=0,
        include_classical_bits:bool=True,
        **kwargs):
        """
        A function to initialize an instance of the PhaseEstimation class

        Parameters:
            precision_order : the bit-precision to which you want to do the phase estimation

            init_state: A python list containing the state occupancy for the initial state

            phase_offset: A float that represents a phase offset. Used for GSE, and initialized to 0.

            kwargs: A dictionary containing the necessary variables for your unitary. Leave empty to do T-Gate test

        Returns:
            generate_qubits()

            estimate_phase()

            add_trotterized_unitary(precision_order, **kwargs)

            generate_circuit_element(precision_order:int, add_to_pe_circuit:bool=True)

            generate_circuit()

            generate_test_unitary()

            analyze_circuit()
        """
        self.precision_order = precision_order
        self.init_state      = init_state
        self.phase_estimate  = 0
        self.pe_circuit      = cirq.Circuit()
        self.phase_offset    = phase_offset
        self.state_initialized = False
        self.include_classical_bits = include_classical_bits

        self.n_qubits        = len(init_state) + 1
        self._bit_list       = []

        if kwargs:
            self.unitary_kw_args = kwargs['kwargs']
        else:
            self.unitary_kw_args = None

        self.generate_qubits()

        self._operator_power   = 1

    
    @property
    def register_qubits(self):
        # This is a list of qubits
        return self._reg_qubits

    @property
    def precision_qubit(self):
        # This is a single qubit
        return self._prec_qubit

    @property
    def bit_list(self):
        return self._bit_list

    @property
    def operator_power(self):
        return self._operator_power

    @operator_power.setter
    def operator_power(self, new_val):
        self._operator_power = new_val

    def generate_qubits(self,):
        """
        A function to generate the qubits for the PE circuit.

        Parameters: 
            None
        
        Returns:
            None
        """
        self._reg_qubits = cirq.NamedQubit.range(len(self.init_state), prefix='reg_')
        self._prec_qubit = cirq.NamedQubit.range(
            len(self.init_state),
            len(self.init_state)+1,
            prefix='prec_')[0]


    def estimate_phase(self):
        """
        A function to calculate the estimated phase from the measured bits
        Note: self.phase_estimate is set to 0 at the start of this function

        Parameters:
            None
        
        Returns:
            self.phase_estimate: The phase estimate, a number between 0,1
        """
        # TODO: Get agreement in ascending/descending order of mzrmnts
        self.phase_estimate = 0
        for idx_, bit_ in enumerate(self.bit_list[::-1]):
            self.phase_estimate += bit_/(2**(idx_+1))

        return self.phase_estimate   
    
    def add_trotterized_unitary(self,precision_order, **kwargs):

        trotter_args = kwargs['kwargs']
        ev_time      = 1
        trot_num     = 1
        trot_ord     = 1

        if 'ev_time' in trotter_args.keys():
            ev_time = trotter_args['ev_time']

        if 'trot_num' in trotter_args.keys():
            trot_num = trotter_args['trot_num']

        if 'trot_ord' in trotter_args.keys():
            trot_ord = trotter_args['trot_ord']

        # apply jordan-wigner transform
        jw_ham = jordan_wigner(trotter_args['mol_ham'])

        # Define a build circuit:
        build_circuit = cirq.Circuit()
    
        # Note: need factor of 2 bc of way OF trotterizes...
        jw_ham_trotterized_generator = trotterize_exp_qubop_to_qasm(jw_ham*2**(precision_order+1)*self.operator_power, evolution_time=ev_time, trotter_number=trot_num, trotter_order=trot_ord)
        jw_ham_trotterized_circuit   = open_fermion_to_qasm(self.n_qubits, jw_ham_trotterized_generator, reg_name='reg') 
        tmp_circuit                  = qasm_import.circuit_from_qasm(jw_ham_trotterized_circuit)

        # Apply control operations on the z-rotations:
        #for ii in range(precision_order):
        for moment in tmp_circuit:
            for op in moment:
                if 'Rz' in str(op):
                    angle = float(str(op.gate).split('(')[-1].split('π')[0])
            
                    angle *= np.pi # TODO: Check this factor of 2
                    # angle *= 2**(precision_order)
            
                    new_op = cirq.ControlledGate(cirq.Rz(rads=angle)).on(self.precision_qubit,op.qubits[0])
                    build_circuit.append(new_op)
                else:
                    build_circuit.append(op)


        return build_circuit

    def generate_circuit(self):
        """
        A function to generate a full PE circuit. Note, uses ClassicallyControlledOperation from cirq.

        Parameters:
            None

        Returns:
            None
        """
        build_circuit           = cirq.Circuit()
        self.state_initialized  = False

        measurement_keys = []
        bit_stand_in     = []

        

        for ii in range(self.precision_order)[::-1]:
            # print(f'> Prec Order = {ii}')
            measurement_keys.append(f'measure_{ii}')
            
            build_circuit.append(cirq.H(self.precision_qubit))

            # Insert the phase offset. Defaults to I matrix.
            # TODO: Come back to this and verify the calculation is correct...
            # Looks like sign and power are all correct.
            # Note: I think we need this to have no negative to counteract the negative in the trotterization?
            build_circuit.append(
                PhaseOffset(
                    qubit=self.precision_qubit,
                    phase_offset=-self.phase_offset*(2**(ii))*self.operator_power).on(*([self.precision_qubit])))

            # Prep the state register, only on the first go:
            if not self.state_initialized:
                for idx_, occ_ in enumerate(self.init_state):
                    if occ_==1:
                        build_circuit.append(cirq.X(self.register_qubits[idx_]))
                    else:
                        build_circuit.append(cirq.I(self.register_qubits[idx_]))
                self.state_initialized = True  

            ###
            ### Add the unitary:
            ### 
            if not self.unitary_kw_args:
                # T-Gate test
                for idx in range(2**ii):
                    build_circuit += self.generate_test_unitary()
            else:
                if 'trotterize' in self.unitary_kw_args.keys():
            
                    build_circuit.append(Trotter_Unitary(
                        tgt_qubit=self.precision_qubit,
                        reg_qubits=self.register_qubits,
                        prec_order=ii,
                        operator_power=self.operator_power,
                        kwargs=self.unitary_kw_args).on(*([self.precision_qubit]+self.register_qubits)))

                    # --> Old way/Known good <--
                    # for ii_ in range(2**ii):
                    #     build_circuit += self.add_trotterized_unitary(
                    #     precision_order=0,
                    #     kwargs=self.unitary_kw_args)
            
                elif 'unitary_matrix_circuit' in self.unitary_kw_args.keys():
                    for idx in range(2**ii):
                        build_circuit += self.unitary_kw_args['unitary_matrix_circuit']

                else:
                    print('-'*50)
                    print(self.unitary_kw_args)
                    print('> Only support Trotterization')
                    print('> Inputing Test T-Gates instead')
                    for idx in range(2**ii):
                        build_circuit += self.generate_test_unitary()

            ###
            ### Rotate away the already measured bits
            ###        
            if bit_stand_in:
                for bsi_idx in range(len(bit_stand_in)):
                    tmp_bit_list = bit_stand_in[bsi_idx:]
                    
                    if self.include_classical_bits:
                        build_circuit.append(
                            S_k(
                                qubit=self.precision_qubit, 
                                prev_bits=tmp_bit_list
                            ).on(*[self.precision_qubit]).with_classical_controls(measurement_keys[bsi_idx])
                        )
                    else:
                        build_circuit.append(
                            S_k(
                                qubit=self.precision_qubit, 
                                prev_bits=tmp_bit_list
                            ).on(*[self.precision_qubit])
                        )
                bit_stand_in.insert(0,0)
            else:
                bit_stand_in = [1]

            # Hadamard the precisoin qubit to get it in correct basis:
            build_circuit.append(cirq.H(self.precision_qubit))

            # Measure the precision qubit:
            build_circuit.append(cirq.measure(self.precision_qubit, key=measurement_keys[-1]))

            # If needed, not the measurement bit
            if ii>0:
                if self.include_classical_bits:
                    build_circuit.append(cirq.X(self.precision_qubit).with_classical_controls(measurement_keys[-1]))  
                else:
                    build_circuit.append(cirq.X(self.precision_qubit))  
          
            ### That's the end, now go back to the beginning and do it again.
        
        self.pe_circuit = build_circuit

    def generate_test_unitary(self):
        """
        A function to generate a T-gate Unitary gate
        
        Parameters:
            None
        
        Returns:
            test_unitary : a c-T gate acting on the precision_qubit and the register_qubits[0]

        """
        test_unitary = cirq.Circuit()
        test_unitary.append(
            cirq.ControlledGate(cirq.T).on(self.precision_qubit, self.register_qubits[0]))

        return test_unitary