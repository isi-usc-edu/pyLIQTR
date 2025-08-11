"""
Copyright (c) 2024 Massachusetts Institute of Technology 
SPDX-License-Identifier: BSD-2-Clause
"""
import platform
import numpy       as np
from typing import List, Tuple
from cirq   import LineQubit
import cirq
#from pyLIQTR.
from pyLIQTR.circuits.pyLCircuit import get_T_counts_from_rotations
from io import StringIO
import struct

from importlib.metadata import version
def version_newer(pkg_name:str, check_version:str):
    def get_maj_min_rev(vrs):
        maj = int(vrs.split('.')[0])
        min = int(vrs.split('.')[1])
        rev = int(vrs.split('.')[2])
        return maj, min, rev
    
    mj,mn,rv = get_maj_min_rev(version(pkg_name))
    chk_mj, chk_mn, chk_rv = get_maj_min_rev(check_version)
    if mj > chk_mj:
        return True
    elif mj == chk_mj:
        if mn > chk_mn:
            return True
        elif mn == chk_mn:
            return rv > chk_rv
        else:
            return False
    else:
        return False

qasm_convert_one_qubit_gates = {
    'H': 'h',
    'X': 'x'
}

qasm_convert_rotation_gates = {
    'Rx': 'rx',
    'Ry': 'ry',
    'Rz': 'rz',
}

qasm_convert_two_qubit_gates = {
    'CNOT': 'cx'
}

def open_fermion_to_qasm(n_qubits:int, ofq_str, reg_name:str='reg', include_heading:bool=True, decompose_rotations:bool=False):
    """
    A function for converting the openfermion qasm to OpenQASM 2.0

    Parameters:
        n_qubits : the number of qubits in the circuit

        ofq_str : trotterized qasm string generator (output of trotterise_exp_qubop_to_qasm)

        reg_name : The prefix name for the register qubits

        include_heading: Boolean, if you want to include the heading in the file

        decompose_rotations: Boolean, if you want to decompose rotation gates to Clifford+T

    Returns:
        a string containing the OpenQASM 2.0 circuit
    """
    strio_out = StringIO()
    if include_heading:
        strio_out.write('// Generated from Cirq, Openfermion, and MIT LL\n\n')
        strio_out.write('OPENQASM 2.0;\n')
        strio_out.write('include \"qelib1.inc\";\n\n')
        strio_out.write(f'qreg {reg_name}[{n_qubits}];\n\n')

    for moment_str in ofq_str:

        moment_split = moment_str.split(' ')

        # split the string and grab the gate:
        gate = moment_split[0]

        if gate in qasm_convert_one_qubit_gates:

            qubit_id = int(moment_split[-1])
            strio_out.write(f'{qasm_convert_one_qubit_gates[gate]} {reg_name}[{qubit_id}];\n')

        elif gate in qasm_convert_two_qubit_gates:

            qubit_ids = [int(x) for x in moment_split[1:3]]
            strio_out.write(f'{qasm_convert_two_qubit_gates[gate]} {reg_name}[{qubit_ids[0]}],{reg_name}[{qubit_ids[1]}];\n')

        elif gate in qasm_convert_rotation_gates:

            rotation = float(moment_split[1])/np.pi
            qubit_id = int(moment_split[-1])
            if decompose_rotations:
                strio_out.write(f'{get_T_counts_from_rotations(gate)} {reg_name}[{qubit_id}];\n')
            else:
                strio_out.write(f'{qasm_convert_rotation_gates[gate]}(pi*{rotation}) {reg_name}[{qubit_id}];\n')
        else:
            print(f'> Gate = {gate} not in gate tables')

    return strio_out.getvalue()

def count_T_gates(circuit):
    '''
    For counting the number of T-Gates in a circuit

    Parameters:
     - circuit: The circuit to count T-Gates in

    Returns:
     - T_gate_counter: the number of T-Gates in the circuit

    '''
    T_gate_counter = 0

    for moment in circuit:
        for op in moment:
            #add a check if it is a rx/y/z_d decomp
            if (str(op).lower().startswith(("rx_d","ry_d","rz_d"))):
                count = 0
                for dop in cirq.decompose(op):
                    count+=str(dop).lower().startswith("t")
                T_gate_counter += count
            elif (str(op).startswith('T')):
                T_gate_counter += 1

    return (T_gate_counter)

def getLineQubitIndexMap(qubit_line:List[LineQubit], name:str) -> List[tuple]:
    """
    Helper function for mapping qubits to their register names.

    Parameters:
        qubit_line: List of cirq.LineQubit's

        name: A string describing the register's name.

    Returns:
        out_map: A list of tuples mapping the LineQubit index to its register name.
    """
    out_map = []
    for ii in range(len(qubit_line)):
        tmp = (ii, name)
        out_map.append(tmp)
    return out_map

def getQubitFromMap(tuple_in:tuple, ctl_q:List[LineQubit], tgt_q:List[LineQubit], anc_q:List[LineQubit]):
    """
    Helper function for getting a specific qubit from a list of 
    LineQubits using the map generated using the getLineQubitIndexMap
    function.
    
    Parameters:
        tuple_in: A tuple of the form (idx, reg_name)

        ctl_q: The list of LineQubits representing the control register
        
        tgt_q: The list of LineQubits representing the target register

        anc_q: The list of LineQubits representing the ancilla register

    Returns:
        _: The LineQubit corresponding to the index idx from the tuple_in
    """


    if tuple_in[1] == 'ctl':
        return ctl_q[tuple_in[0]]
    elif tuple_in[1] == 'tgt':
        return tgt_q[tuple_in[0]]
    elif tuple_in[1] == 'anc':
        return anc_q[tuple_in[0]]
    else:
        raise RuntimeError('Function getQubitFromMap malfunctioning...')
    
def isWindows() -> bool:
    """will determine if the OS we are running on is Windows or not.
    
    Return:
        True if OS is Windows
        False otherwise
    """
    return platform.system() == 'Windows'

def isMac() -> bool:
    """will determine if the OS we are running on is Mac or not.
    
    Return:
        True if OS is Mac
        False otherwise
    """
    return platform.platform().startswith('mac')

def isLinux() -> bool:
    """will determine if the OS we are running on is Linux or not.
    
    Return:
        True if OS is Linux
        False otherwise
    """
    return platform.system() == 'Linux'

def float_array_to_fixed_width_int(float_array, width_bits=32):
    """Converts an array of floats to a fixed-width integer."""

    # Pack the float into a binary representation
    packed = struct.pack('%sf' % len(float_array), *float_array)

    # Unpack the binary representation as an integer
    integer = int.from_bytes(packed, byteorder='big', signed=False)

    # Mask the integer to the desired width
    mask = (1 << width_bits) - 1
    return integer & mask