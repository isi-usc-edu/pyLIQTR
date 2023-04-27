"""
DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.

This material is based upon work supported by the Under Secretary of Defense for
Research and Engineering under Air Force Contract No. FA8702-15-D-0001. Any opinions,
findings, conclusions or recommendations expressed in this material are those of the
author(s) and do not necessarily reflect the views of the Under Secretary of Defense
for Research and Engineering.

© 2022 Massachusetts Institute of Technology.

The software/firmware is provided to you on an As-Is basis

Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS Part
252.227-7013 or 7014 (Feb 2014). Notwithstanding any copyright notice, U.S. Government
rights in this work are defined by DFARS 252.227-7013 or DFARS 252.227-7014 as detailed
above. Use of this work other than as specifically authorized by the U.S. Government
may violate any copyrights that exist in this work.
"""

from logging import raiseExceptions
from pyLIQTR.QSP.qsp_cirq_gates import SelectV, Reflect, UnitaryBlockEncode, SzegedyWalkOperator
import pyLIQTR.QSP.Hamiltonian as Hamiltonian

import cirq
import numpy as np
from typing import List, Tuple


class QSPBase:
    """
    A QSP Base Class.

    Attributes:
    -----------
    phis : List[float]
        An N-list of phi rotation angles in [radians?]

    hamiltonian : List[Tuple(str, float)]
        An N-list of 2-tuple, with first element being the Pauli term 
        and the second element the associated constant
    
    target_size : int
        Number of spins in the spin-chain (?)

    Methods:
    --------
    get_number_of_control_qubits() -> int
        Returns the required number of control qubits.

    add_phased_iterate(circuit, phiA, phiB, invert=0)
        Add selectV then reflection, or vice versa, to circuit.

    circuit()
        Generate highest-level QSP circuit based on input Phis, Hamiltonian, and Target Size.
    """
    def __init__(self, 
                 phis: List[float], 
                 hamiltonian: Hamiltonian,
                 target_size: int,
                 singleElement: bool=False):
        """
        Initializes QSP Base class and generates target, control, and phase qubits.

        Parameters:
            phis : List[float]
                An N-list of phi rotation angles in [radians?]

            hamiltonian : List[Tuple(str, float)]
                An N-list of 2-tuple, with first element being the Pauli term 
                and the second element the associated constant
        
            target_size : int
                Number of spins in the spin-chain (?)

            singleElement: bool
                Set to True to get a circuit with single selectV/reflect pair
                (useful for circuit analysis)

        Returns:
            None
        """
        self.phis = np.array(phis,dtype=np.double)
        self.hamiltonian = hamiltonian
        if self.hamiltonian.is_lcu:
            self.hamiltonian.adjust_hamiltonian()
        self.__target = self.generate_target_qubits(target_size)
        self.__phase = self.generate_phase_qubit()
        self.__control = self.generate_control_qubits()
        self.__ancilla = None
        self.__singleElement = singleElement
        
    def get_number_of_control_qubits(self) -> int:
        #hamiltonian should be a list of tuples [(),(),...()]
        #so the required number of control qubtis is:
        if self.hamiltonian.is_lcu:
            return self.hamiltonian.loglen
        elif self.hamiltonian.is_fermionic:
            # N is the total system size (functions + spins)
            # the control contains (p,alpha), (q,beta), and (U,V,theta)
            N = self.hamiltonian.problem_size
            nval = int(np.ceil(np.log2(N)))
            return 2*nval + 3

    def add_phased_iterate(self, circuit, phiA, phiB,invert=0):
        """
        Adds selectV -> reflection (or vice versa) to cirq circuit.

        Parameters:
            circuit : no type, could be cirq.circuit
                The current working circuit
            
            phiA : float
                An input phi angle
            
            phiB : float
                An input phi angle

            invert = 0 : bool
                a flag for deciding if inverting the circuit elements

        Returns:
            circuit : no type, could be cirq.circuit
                The updated working circuit.
        """
        if not invert:
            circuit = self.add_select_v(circuit, phiA)
            circuit = self.add_reflection(circuit, phiB)
        else:
            circuit = self.add_reflection(circuit, phiA)
            circuit = self.add_select_v(circuit, phiB)
            
        return circuit

    def circuit(self):
        """
        Generates the high-level circuit of the QSP-ed Hamiltonian
        This generates a circuit for the complete dynamic evolution using the prepare and select Oracles

        Parameters:
            None (Note, all inputs are class variables that come from class initialization)
            TODO: Make this a static method? (not sure if using correct pythonic term here...)

        Returns:
            circuit : no type, could be cirq.circuit
                The generated high-level circuit
        """
        circuit = self.initialize_circuit()
        phis = self.phis
        if not(len(phis) % 2):
            raise ValueError('Phi is not odd!')
        
        phi0 = phis[0]
        phis = phis[1:]
        phiLen = int(len(phis)/2)
        phiLo = phis[0:phiLen-1]
        phiMid = phis[phiLen-1]
        phiHi = phis[phiLen:-1]
        phiN = phis[-1]

        # TODO: Can this be done more compactly/intuitively?
        circuit = self.add_correction(circuit,beginning=True)
        circuit = self.add_phase_rotation(circuit, phi0, rot_type='X')
        #phased iterates
        if not self.__singleElement:
            for idx in range(0,len(phiLo)-1,2):
                circuit = self.add_phased_iterate(circuit, phiLo[idx],phiLo[idx+1],invert=0)
        #selectV
            circuit = self.add_select_v(circuit,phiLo[-1])

        #reflection
        circuit = self.add_reflection(circuit, phiMid)
        #selectV
        circuit = self.add_select_v(circuit, phiHi[0])
        #phased iterates
        if not self.__singleElement:
            for idx in range(1,len(phiHi),2):
                circuit = self.add_phased_iterate(circuit,phiHi[idx],phiHi[idx+1],invert=1)

        circuit = self.add_phase_rotation(circuit, phiN, rot_type='X')
        circuit = self.add_correction(circuit,beginning=False)
        circuit = self.clear_register(circuit)
        return circuit

    def circuit_unitary_operator(self, controlled=True):
        """
        Generates the high-level circuit for a Unitary that provides a
        block encoding of the specified Hamiltonian
        Note: this represents a single operator to be used for applications
        such as GSEE etc.

        Parameters:
            None (Note, all inputs are class variables that come from class initialization)

        Returns:
            circuit : no type, could be cirq.circuit
                The generated high-level circuit
        """

        circuit = self.initialize_circuit()

        circuit = self.add_unitary_block_encode(circuit, controlled)

        return circuit
    
    def circuit_walk_operator(self):
        """
        Generates the high-level circuit for a Unitary that provides a
        block encoding of the Szegedy walk operator of the specified Hamiltonian
        Note: this represents a single operator to be used for applications
        such as GSEE etc.

        Parameters:
            None (Note, all inputs are class variables that come from class initialization)

        Returns:
            circuit : no type, could be cirq.circuit
                The generated high-level circuit
        """

        circuit = self.initialize_circuit()

        circuit = self.add_walk_block_encode(circuit)

        return circuit
    
    @property
    def phase(self):
        return self.__phase[0]
    
    @phase.setter
    def phase(self, new_val):
        self.__phase = new_val
    
    @property
    def control(self):
        return self.__control

    @control.setter
    def control(self, new_val):
        self.__control = new_val
    
    @property
    def target(self):
        return self.__target

    @target.setter
    def target(self, new_val):
        self.__target = new_val

    @property
    def n_qubits(self):
        return 1+len(self.target)+len(self.control) #phase + target + control, should we include ancilla counts??

    @property
    def n_ancilla(self):
        if self.ancilla is None:
            return 0
        else:
            return len(self.ancilla)

    @property
    def ancilla(self):
        return self.__ancilla

    @ancilla.setter
    def ancilla(self, new_val):
        '''
        This function generates new ancilla qubits to use, as necessary for a given implementation of a subcircuit
        '''
        if not isinstance(new_val, int):
            raise ValueError('Input to QSP.ancilla is expected to be an integer.')
        else:
            if self.n_ancilla == 0:
                self.__ancilla = self.generate_ancilla_qubits(new_val)
            else:
                if self.n_ancilla < new_val:
                    tmp_new_ancilla = self.generate_ancilla_qubits(new_val - len(self.ancilla))
                    self.__ancilla.extend(tmp_new_ancilla)

    def __str__(self):
        ph=[self.phase]
        return f'''QSP\n\tTarget:  {self.target}\n\tPhase:   {ph}\n\tControl: {self.control}\n\tAncilla: {self.ancilla}\n\n\tAngles: {self.phis}\n\tHamiltonian {self.hamiltonian}'''
    #~~ABSTRACT METHODS~~
    def initialize_circuit(self):
        return None

    def generate_target_qubits(self, target_size):
        return None

    def generate_control_qubits(self):
        return None

    def generate_phase_qubit(self):
        return None

    def generate_ancilla_qubits(self,num2generate):
        return None

    def add_phase_rotation(self, circuit, angle, rot_type):
        return circuit

    def add_select_v(self, circuit, angle):
        return circuit

    def add_unitary_block_encode(self, circuit, controlled):
        return circuit
    
    def add_walk_block_encode(self, circuit, controlled):
        return circuit
    
    def add_reflection(self, circuit, angle):
        return circuit

    def add_correction(self,circuit,type=None):
        return None
    
    def add_correction(self,circuit,type=None):
        return circuit

    def clear_register(self,circuit):
        return circuit
    

#this just implements QSP via cirq
class QSP(QSPBase):
    """
    A QSP implementation with cirq.

    Inherits from, and populates some abstract methods from, QSPBase 

    Attributes:
    -----------
    phis : List[float]
        An N-list of phi rotation angles in [radians?]

    hamiltonian : List[Tuple(str, float)]
        An N-list of 2-tuple, with first element being the Pauli term 
        and the second element the associated constant
    
    target_size : int
        Number of spins in the spin-chain (?)

    Methods:
    --------
    initialize_circuit() -> cirq.Circuit
        Generates empty cirq.Circuit.
    
    generate_target_qubits() -> List[cirq.LineQubit]
        Generates the target quibits for the circuit.
    
    generate_control_qubits() -> List[cirq.LineQubit]
        Generates the control qubits for the circuit.

    generate_phase_qubit() -> List[cirq.LineQubit]
        Generates phase qubit(s) for circuit.

    add_phase_rotation(circuit, angle, rot_type) -> cirq.circuit
        Adds an X or Y rotation gate to circuit.

    add_select_v(circuit, angle) -> cirq.circuit
        Adds a SelectV gate to the circuit.

    add_reflection(circuit, angle) -> cirq.circuit
        Adds a Reflection gate to the circuit.
    """

    def __init__(self, 
                 phis: List[float], 
                 hamiltonian: List[Tuple[str, float]],
                 target_size: int,
                 singleElement: bool=False):
        """
        Initializes QSP class and generates target, control, and phase qubits.

        Parameters:
            phis : List[float]
                An N-list of phi rotation angles in [radians?]

            hamiltonian : List[Tuple(str, float)]
                An N-list of 2-tuple, with first element being the Pauli term 
                and the second element the associated constant
        
            target_size : int
                Number of spins in the spin-chain (?)

        Returns:
            None
        """
        super(QSP, self).__init__(phis, hamiltonian, target_size, singleElement)
        n_ = len(self.control) + 1 #always only 1 phase qubit
        n_anc = n_ - 3
        if n_anc > 0:
            self.ancilla = n_anc
        self.ancilla =  self.get_number_of_control_qubits()
    
    def initialize_circuit(self):
        """ Generates empty cirq.Circuit. """
        return cirq.Circuit()

    def generate_target_qubits(self, target_size):
        """Generates the target quibits for the circuit."""
        # Note: can/should we make target_size a class variable?
        return cirq.LineQubit.range(target_size)
        
    
    def generate_control_qubits(self):
        """Generates the control qubits for the circuit."""
        return cirq.NamedQubit.range(len(self.target)+1,\
                    len(self.target)+1+self.get_number_of_control_qubits(),
                    prefix="ctl_q")

    def generate_ancilla_qubits(self,new_val):        
        return cirq.NamedQubit.range(self.n_qubits + self.n_ancilla, \
            self.n_qubits+self.n_ancilla+new_val, prefix='z_anc_q')

    def generate_phase_qubit(self):
        """Generates phase qubit(s) for circuit."""
        return cirq.NamedQubit.range(len(self.target),\
                    len(self.target)+1, prefix="phs_q")
    
    def add_phase_rotation(self, circuit, angle: float, rot_type: str):
        """
        Adds an X or Y rotation gate to circuit.
        
        Parameters:
            circuit : cirq.circuit
                The current working circuit.

            angle : float
                The angle for the rotation gate

            rot_type : str
                Note: I renamed bc 'type' is overloaded
                The type of rotation gate, either 'X' or 'Y'
        
        Returns:
            circuit : cirq.circuit
                The updated current working circuit.
        """
        angle = -angle/2
        if rot_type == 'X':
            circuit.append(cirq.Rx(rads=2*angle).on(self.phase))
        elif rot_type == 'Y':
            circuit.append(cirq.Ry(rads=2*angle).on(self.phase))
        else:
            raise ValueError('unsupported phase rotation gate')
        
        return circuit
    
    def add_select_v(self, circuit, angle: float):
        """
        Adds a SelectV gate to the circuit.
        
        Parameters:
            angle : float
                The angle for the *thing*
        
        Returns:
            cirq.Gate which implements the select v operation
        """
        #See Appdx. G.4 from Childs et al in https://arxiv.org/pdf/1711.10980.pdf 
        self.ancilla =  self.get_number_of_control_qubits()
        circuit.append(SelectV(self.hamiltonian, angle, self.phase, self.target, self.control,\
                    self.ancilla).\
                on(*(self.target+[self.phase]+self.control+self.ancilla)))
        return circuit
    
    def add_unitary_block_encode(self, circuit, controlled):
        """
        Adds a Unitary Block Encoding to the circuit
        
        Parameters:
             None

        Returns:
            cirq.Gate which implements a unitary block encoding
        """

        self.ancilla =  self.get_number_of_control_qubits()
        if controlled:
            circuit.append(UnitaryBlockEncode(self.hamiltonian, self.phase, self.target, self.control, self.ancilla).\
                    on(*(self.target+[self.phase]+self.control+self.ancilla)))
        else:
            circuit.append(UnitaryBlockEncode(self.hamiltonian, None, self.target, self.control, self.ancilla).\
                    on(*(self.target+self.control+self.ancilla)))
                
        return circuit

    def add_walk_block_encode(self, circuit):
        """
        Adds a Szegedy Walk operator circuit
        
        Parameters:
             None

        Returns:
            cirq.Gate which implements a Szegedy Walk operator
        """

        #self.ancilla =  self.get_number_of_control_qubits()
        N = self.hamiltonian.problem_size
        npb = int(np.ceil(np.log2(N)))
        self.ancilla = npb + 6
        circuit.append(SzegedyWalkOperator(self.hamiltonian, self.phase, self.target, self.control, self.ancilla).\
                        on(*(self.target+[self.phase]+self.control+self.ancilla)))
                
        return circuit
        
    def add_reflection(self, circuit, angle: float):
        """
        Adds a Reflection gate to the circuit.
        
        Parameters:
            circuit : cirq.circuit
                The current working circuit.

            angle : float
                The angle for the *thing*
        
        Returns:
            circuit : cirq.circuit
                The updated current working circuit.
        """
        # The reflections rely on the implementation of a multiply controlled
        # NOT gate which is implemented over the Clifford+T gate set
        # following the technique described in https://arxiv.org/abs/1508.03273.
        # See page 6 corollary 2 for 2n-3 total qubits formula
        n_ = len(self.control) + 1 #always only 1 phase qubit
        n_anc = n_ - 3
        circuit.append(\
            Reflect(angle, self.phase, self.control, self.ancilla[:n_anc]).\
                on(*([self.phase]+self.control + self.ancilla[:n_anc]))\
            )
        return circuit

    def add_correction(self,circuit,beginning, type=None):
        subCircuit = cirq.align_left(cirq.Circuit([cirq.X.on(q) for q in self.control]))
        if beginning:
            subCircuit = cirq.align_left(cirq.Circuit(\
                                [cirq.ResetChannel().on(q) \
                                 for q in self.control+[self.phase]+self.ancilla])
                                +subCircuit)
            subCircuit.append(circuit)
            return subCircuit
        else:
            circuit.append(subCircuit)
            return circuit
    
    def get_msb_qubit_ordering(self):
        ordering = self.target + self.control + [self.phase] + self.ancilla
        return ordering[::-1]
    
    def clear_register(self,circuit):
        circuit.append(cirq.ResetChannel().on(self.phase))
        for q in self.control:
            circuit.append(cirq.ResetChannel().on(q))
        for q in self.ancilla:
            circuit.append(cirq.ResetChannel().on(q))
        return circuit

"""    
class FermionBlockEncode:
    def __init__(self, hamiltonian: List[Tuple[str, float]], target_size: int):
        self.hamiltonian = hamiltonian
        control_size = self.get_number_of_control_qubits()
        
        self.__target = cirq.LineQubit.range(target_size)
        self.__phase = cirq.NamedQubit.range(target_size, target_size+1, prefix="phs_q")
        self.__control = cirq.NamedQubit.range(target_size+1, target_size+control_size, prefix="ctl_q")
        self.__ancilla = None
        
    def __str__(self):
        ph=[self.phase]
        return f'''FermionBE\n\tTarget:  {self.target}\n\tPhase:   {ph}\n\tControl: {self.control}\n\tAncilla: {self.ancilla}\n\n\tAngles: {self.phis}\n\tHamiltonian {self.hamiltonian}'''
    
    def initialize_circuit(self):
        return cirq.Circuit()

    @property
    def n_ancilla(self):
        if self.ancilla is None:
            return 0
        else:
            return len(self.ancilla)

    @property
    def n_qubits(self):
        return 1+len(self.target)+len(self.control) #phase + target + control, should we include ancilla counts??

    @property
    def n_ancilla(self):
        if self.ancilla is None:
            return 0
        else:
            return len(self.ancilla)
    @property
    def ancilla(self):
        return self.__ancilla

    def generate_ancilla_qubits(self,new_val):        
        return cirq.NamedQubit.range(self.n_qubits + self.n_ancilla, \
            self.n_qubits+self.n_ancilla+new_val, prefix='z_anc_q')

    @ancilla.setter
    def ancilla(self, new_val):
        '''
        This function generates new ancilla qubits to use, as necessary for a given implementation of a subcircuit
        '''
        if not isinstance(new_val, int):
            raise ValueError('Input to QSP.ancilla is expected to be an integer.')
        else:
            if self.n_ancilla == 0:
                self.__ancilla = self.generate_ancilla_qubits(new_val)
            else:
                if self.n_ancilla < new_val:
                    tmp_new_ancilla = self.generate_ancilla_qubits(new_val - len(self.ancilla))
                    self.__ancilla.extend(tmp_new_ancilla)
                    
    @property
    def phase(self):
        return self.__phase[0]
    
    @phase.setter
    def phase(self, new_val):
        self.__phase = new_val
    
    @property
    def control(self):
        return self.__control

    @control.setter
    def control(self, new_val):
        self.__control = new_val
    
    @property
    def target(self):
        return self.__target

    @target.setter
    def target(self, new_val):
        self.__target = new_val

    def get_number_of_control_qubits(self) -> int:
        #hamiltonian should be a list of tuples [(),(),...()]
        #so the required number of control qubtis is:
        return self.hamiltonian.loglen

    def circuit(self):
        circuit = self.initialize_circuit()

        self.ancilla = self.get_number_of_control_qubits()
        circuit.append(FermionEncode(self.hamiltonian, self.phase, self.target, self.control, self.ancilla).on(*(self.target+[self.phase]+self.control+self.ancilla)))

        return circuit
"""    
