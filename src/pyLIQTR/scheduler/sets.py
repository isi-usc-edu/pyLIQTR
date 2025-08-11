import cirq

class gateset:
    def __init__(self, possible_keys: list, gateset_name: str, subsets: list = None):
        self.possible_keys = possible_keys
        self.gateset_name = gateset_name
        self.subsets = subsets

    def in_gateset(self, gate_instance):
        if gate_instance in self.possible_keys:
            return True
        else:
            return False
        
    def name(self):
        return self.gateset_name


T = gateset(['T(', 'T*', 't ', 'td'], 'T')
ROT = gateset(['Rx', 'rx', 'Ry', 'ry', 'Rz', 'rz'], 'Rotation')
H = gateset(['H(', 'H*', 'h '], 'H')
S = gateset(['S(', 'S*', 's ', 'sd'], 'S')
CX = gateset(['CN', 'cx', 'CX'], 'CX')
CY = gateset(['cy', 'CY'], 'CY')
CS = gateset(['cs', 'CS'], 'CS')

CCX = gateset(['TO', 'cc', 'CC', 'To'], 'Toffoli')
CZ = gateset(['cz', 'CZ'], 'CZ')

X = gateset(['X(', 'X*', 'XP', 'XG', 'x ', 'Rx', 'rx'], 'X')
X_CIRQ = gateset(['X(', 'X*', 'XP', 'XG', 'Rx'], 'X Cirq')
Z = gateset(['Z(', 'Z*', 'z ', 'ZG', 'Rz', 'rz'], 'Z')
Z_CIRQ = gateset(['Z(', 'Z*', 'ZP', 'ZG', 'Rz'], 'Z Cirq')
ALL = gateset(['H(', 'H*', 'h ', 'T(', 'T*', 'td', 't ', 'S(', 'S*', "s ", 'sd', 'Y(', 'Y*', 'YG', "y ", 'Ry', 'ry', 'ci', 're', 'no', "r ", 'rd', 'rk', "q ", 'qd'], 'All')
ALL_CIRQ = gateset(['H(', 'H*', 'T(', 'T*', 'S(', 'S*', 'Y(', 'Y*', 'YP', 'YG', 'Ry', 'ci', 're'], 'All Cirq')

PAULI = gateset(['X(', 'X*', 'XG', 'XP', 'x ', 'Y(', 'Y*', 'YP', 'YG', 'y ', 'Z(', 'Z*', 'ZP', 'ZG', 'z '], 'Pauli (X, Y, Z)', subsets=[X, Z, X_CIRQ, Z_CIRQ])
CLIFFORD = gateset(['H(', 'H*', 'S(', 'S*', 'CN', 'CX', 'CZ', 'X(', 'X*', 'XG', 'Y(', 'Y*', 'YG', 'Z(', 'Z*', 'ZG'], 'Clifford', subsets=[H, S, CX, CZ, X, Z, X_CIRQ, Z_CIRQ, PAULI])
MISC = gateset(['ci', 're'], 'Measurement/Reset')
QASMMISC = gateset(['no', 'r', 'rd', 'rk', 'q', 'qd', 'tr', 'sw', 'mo', 'in', 'rm', 'mx', 'mz'], 'QASM Misc')

def op_to_openqasm(op_str: str, qubits: list, angle: float = None):
    if T.in_gateset(op_str):
        if op_str == 'T(':
            qasm_str = f't {qubits[0]}'
        elif op_str == 'T*':
            qasm_str = f'tdg {qubits[0]}'
    elif ROT.in_gateset(op_str):
        if op_str == 'Rx':
            qasm_str = f'rx({angle}) {qubits[0]}'
        elif op_str == 'Ry':
            qasm_str = f'ry({angle}) {qubits[0]}'
        elif op_str == 'Rz':
            qasm_str = f'rz({angle}) {qubits[0]}'
    elif H.in_gateset(op_str):
        qasm_str = f'h {qubits[0]}'
    elif S.in_gateset(op_str):
        if op_str == 'S(':
            qasm_str = f's {qubits[0]}'
        elif op_str == 'S*':
            qasm_str = f'sdg {qubits[0]}'
    elif PAULI.in_gateset(op_str):
        if op_str in ['X(', 'X*', 'XG']:
            qasm_str = f'x {qubits[0]}'
        elif op_str in ['Y(', 'Y*', 'YG']:
            qasm_str = f'y {qubits[0]}'
        elif op_str in ['Z(', 'Z*', 'ZG']:
            qasm_str = f'z {qubits[0]}'
    elif CX.in_gateset(op_str):
        qasm_str = f'cx {qubits[0]}, {qubits[1]}'
    elif CZ.in_gateset(op_str):
        qasm_str = f'cz {qubits[0]}, {qubits[1]}'
    elif CCX.in_gateset(op_str):
        qasm_str = f'ccx {qubits[0]}, {qubits[1]}, {qubits[2]}'
    elif MISC.in_gateset(op_str):
        qasm_str = ''
    else:
        raise ValueError(f'{op_str} not found; for qasm output ALL operations must be simple, defined gates!')
    return qasm_str
    
    

