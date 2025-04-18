import cirq

T = ['T(', 'T*', 't ', 'td', 'T']
ROT = ['Rx', 'rx', 'Ry', 'ry', 'Rz', 'rz', 'Rotation']
H = ['H(', 'H*', 'h ', 'H']
S = ['S(', 'S*', 's ', 'sd', 'S']
CX = ['CN', 'cx', 'CX']
CY = ['cy', 'CY']
CS = ['cs', 'CS']

CCX = ['TO', 'cc', 'CC', 'To', 'Toffoli']
CZ = ['cz', 'CZ']
PAULI = ['X(', 'X*', 'XG', 'XP', 'x ', 'Y(', 'Y*', 'YP', 'YG', 'y ', 'Z(', 'Z*', 'ZP', 'ZG', 'z ', 'Pauli (X, Y, Z)']

X = ['X(', 'X*', 'XP', 'XG', 'x ', 'Rx', 'rx']
X_CIRQ = ['X(', 'X*', 'XP', 'XG', 'Rx']
Z = ['Z(', 'Z*', 'z ', 'ZG', 'Rz', 'rz']
Z_CIRQ = ['Z(', 'Z*', 'ZP', 'ZG', 'Rz']
ALL = ['H(', 'H*', 'h ', 'T(', 'T*', 'td', 't ', 'S(', 'S*', "s ", 'sd', 'Y(', 'Y*', 'YG', "y ", 'Ry', 'ry', 'ci', 're', 'no', "r ", 'rd', 'rk', "q ", 'qd']
ALL_CIRQ = ['H(', 'H*', 'T(', 'T*', 'S(', 'S*', 'Y(', 'Y*', 'YP', 'YG', 'Ry', 'ci', 're']

CLIFFORD = ['H(', 'H*', 'S(', 'S*', 'CN', 'CX', 'CZ', 'X(', 'X*', 'XG', 'Y(', 'Y*', 'YG', 'Z(', 'Z*', 'ZG', 'Clifford']
MISC = ['ci', 're', 'Miscellaneous']
QASMMISC = ['no', 'r', 'rd', 'rk', 'q', 'qd', 'tr', 'sw', 'mo', 'in', 'rm', 'mx', 'mz']

def op_to_openqasm(op_str: str, qubits: list, angle: float = None):
    if op_str in T:
        if op_str == 'T(':
            qasm_str = f't {qubits[0]}'
        elif op_str == 'T*':
            qasm_str = f'tdg {qubits[0]}'
    elif op_str in ROT:
        if op_str == 'Rx':
            qasm_str = f'rx({angle}) {qubits[0]}'
        elif op_str == 'Ry':
            qasm_str = f'ry({angle}) {qubits[0]}'
        elif op_str == 'Rz':
            qasm_str = f'rz({angle}) {qubits[0]}'
    elif op_str in H:
        qasm_str = f'h {qubits[0]}'
    elif op_str in S:
        if op_str == 'S(':
            qasm_str = f's {qubits[0]}'
        elif op_str == 'S*':
            qasm_str = f'sdg {qubits[0]}'
    elif op_str in PAULI:
        if op_str in ['X(', 'X*', 'XG']:
            qasm_str = f'x {qubits[0]}'
        elif op_str in ['Y(', 'Y*', 'YG']:
            qasm_str = f'y {qubits[0]}'
        elif op_str in ['Z(', 'Z*', 'ZG']:
            qasm_str = f'z {qubits[0]}'
    elif op_str in CX:
        qasm_str = f'cx {qubits[0]}, {qubits[1]}'
    elif op_str in CZ:
        qasm_str = f'cz {qubits[0]}, {qubits[1]}'
    elif op_str in CCX:
        qasm_str = f'ccx {qubits[0]}, {qubits[1]}, {qubits[2]}'
    elif op_str in MISC:
        qasm_str = ''
    else:
        raise ValueError(f'{op_str} not found; for qasm output ALL operations must be simple, defined gates!')
    return qasm_str
    
    

