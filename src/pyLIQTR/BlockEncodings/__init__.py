from enum import Enum

# class syntax
class VALID_ENCODINGS(Enum):
    PauliLCU = 1
    LinearT = 2
    Fermionic=3
    FermiHubbardSquare=4
    DoubleFactorized = 5
    CarlemanLinearization = 6