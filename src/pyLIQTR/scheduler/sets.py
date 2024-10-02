

T = ['T(', 'T*', 'T']
ROT = ['Rx', 'Ry', 'Rz', 'Rotation']
H = ['H(', 'H*', 'H']
S = ['S(', 'S*', 'S']
CX = ['CN', 'CX']
CCX = ['TO', 'Toffoli']
CZ = ['CZ']
PAULI = ['X(', 'X*', 'XG', 'Y(', 'Y*', 'YG', 'Z(', 'Z*', 'ZG', 'Pauli (X, Y, Z)']

X = ['X(', 'X*', 'XG', 'Rx']
Z = ['Z(', 'Z*', 'ZG', 'Rz']
ALL = ['H(', 'H*', 'T(', 'T*', 'S(', 'S*', 'Y(', 'Y*', 'YG', 'Ry', 'ci', 're']

CLIFFORD = ['H(', 'H*', 'S(', 'S*', 'CN', 'CX', 'CZ', 'X(', 'X*', 'XG', 'Y(', 'Y*', 'YG', 'Z(', 'Z*', 'ZG', 'Clifford']
MISC = ['ci', 're', 'Miscellaneous']

#specify gates to be returned - full profile, full profile (no rotations), clifford+t, etc.