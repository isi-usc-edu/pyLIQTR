# pyLIQTR Features

Included below are the exhaustive list of features contained within pyLIQTR. This Table provides quick links to some of the new features.

1. [OpenQASM](src/pyLIQTR/utils/printing.py#openqasm)
2. [Heisenberg](#heisenberg)
3. [Fermi-Hubbard](#fermi-hubbard)
4. [Dual Plane Wave](#dpw)
5. [Gaussian Orbitals](#gto)
6. [Arbitrary Spin](#arbitrary-spin-lattice)
8. [Ground State Estimation](#gse)
9. [Quantum Signal Processing](#qsp)
10. [Quantum Singular Value Transformation](#qsvt)

## I. Quantum Signal Processing

### Phase Factor Generation

Phase-Factor Generation Algorithms:

- Integrated phase factor framework
  - Description: Integrated framework for phase factor determination.
  - Source: [phase_factors](src/pyLIQTR/phase_factors/)
  - Notebooks: [qsvt_and_qsp.ipynb](Examples/Algorithms_and_Infrastructure/qsvt_and_qsp.ipynb)
- Optimizer based (R-convention)
  - Description: Optimizer-based generation of phase angles for QSP/QET/QSVT.
  - Source: [angler.py](src/pyLIQTR/phase_factors/angler.py)
  - Notebooks: [AngleGeneration](Examples/AngleGeneration/)
- Function fitting / Analytical expansions (for optimizer)
  - Description: Approximation of functions in a Chebyshev polynomial basis.
  - Source: [expander.py](src/pyLIQTR/phase_factors/optimization/expander.py)
                [fitter.py](src/pyLIQTR/phase_factors/optimization/fitter.py)
  - Notebooks: [AngleGeneration](Examples/AngleGeneration/)


### QSP/QSVT Sequences

- QSP/QET/QSVT Circuits <a id="qsvt"></a>
  - Description: Circuits implementing Quantum Signal Processing (QSP),  Quantum Eigenvalue Transform (QET) and the Quantum Singular Value Transform (QSVT)--based algorithms.
  - Source: [qsvt.py](src/pyLIQTR/qubitization/qsvt.py)
  - Classes:
    - Parent class for QSVT: `QSVT_Abstract()`
    - QET/QSVT with real polynomials: `QSVT_real_polynomial(QSVT_abstract)`
    - QET/QSVT with "sum" of real polynomials: `QSVT_real_polynomial_sum(cirq_ft.GateWithRegisters)`
    - Legacy Fourier-response sequence: `QSP_fourier_response(QSVT_abstract)`

- Quantum dynamics via QET/QSVT <a id="qsvt_dynamics"></a>
  - Description: Functions to facilitate quantum dynamics simulations via QSP/QET/QSVT.
  - Source: [qsvt_dynamics.py](src/pyLIQTR/qubitization/qsvt_dynamics.py)
  - Functions:
    - Automated construction of QET/QSVT simulation circuits: `qsvt_dynamics()`
    - Convenient determination of phase angles: `simulation_phases()`


## II. Eigenvalue Estimation

### Qubitized phase estimation

- Qubitized Phase Estimation Circuits <a id="qsvt"></a>
  - Description: Implementation of phase estimation / ground-state energy estimation via methods from Phys. Rev. X <b>8</b>, 041015.
  - Source: [phase_estimation.py](src/pyLIQTR/qubitization/phase_estimation.py)
  - Classes:
    - Phase-estimation circuit: `QubitizedPhaseEstimation(cirq_ft.GateWithRegisters)`

## III. Qubitized Components

### Qubitized primitives

- Qubitized Reflection / Projection Operator
  - Source: [qubitized_gates.py](src/pyLIQTR/qubitization/qubitized_gates.py)
  - Classes:
    - `QubitizedReflection(cirq_ft.GateWithRegisters)`
- Qubitized Rotation Operator
  - Source: [qubitized_gates.py](src/pyLIQTR/qubitization/qubitized_gates.py)
  - Classes:
    - `QubitizedRotation(cirq_ft.GateWithRegisters)`
- Qubitized Walk Operator
  - Source: [qubitized_gates.py](src/pyLIQTR/qubitization/qubitized_gates.py)
  - Classes:
    - `QubitizedWalkOperator(cirq_ft.GateWithRegisters)`

### Select & Prepare Oracles

Oracles:

- Pauli-String LCU (QROM-based): Select
  - Description:
  - Source: [select_oracle_pauli_lcu.py](src/pyLIQTR/circuits/operators/select_oracle_pauli_lcu.py)
  - Classes:
    - Pauli-string LCU select; QROM-implementation which can be (optionally) conditioned on a control qubit. The optional control is required for the new QSVT and block-encoding methods.
- Pauli-String LCU (QROM-based): Prepare
  - Description:
  - Source: [prepare_oracle_pauli_lcu.py](src/pyLIQTR/circuits/operators/prepare_oracle_pauli_lcu.py)
  - Classes:
    - Pauli-string LCU prepare.</b>
- Fermionic Encoding: Select
  - Description: Select oracle for molecular electronic structure Hamiltonians expressed in a basis that diagonalizes the Coulomb potential.
  - Source: [select_LinearT](src/pyLIQTR/circuits/operators/select_LinearT.py)
  - Classes:
    - `FermionicSelect_LinearT(cirq_ft.GateWithRegisters)`
    - Select from [Linear T paper](https://arxiv.org/abs/1805.03662) Section IV.A, Fig. 14
- Fermionic Encoding: Prepare
  - Description: Prepare oracle for molecular electronic structure Hamiltonians expressed in a basis that diagonalizes the Coulomb potential.
  - Source: [prepare_LinearT](src/pyLIQTR/circuits/operators/prepare_LinearT.py)
  - Classes:
    - `FermionicPrepare_LinearT(cirq_ft.GateWithRegisters)`
    - Prepare from [Linear T paper](https://arxiv.org/abs/1805.03662) Section IV.B, Fig. 16
    - `Subprepare_LinearT(cirq_ft.GateWithRegisters)`
    - Subprepare from [Linear T paper](https://arxiv.org/abs/1805.03662) Section IV.B, Fig. 15

### Block Encodings

Abstract Encoding Classes:

- Abstract Block Encodings
  - Description: Abstrct template for block encodings.
  - Source: [BlockEncoding.py](src/pyLIQTR/BlockEncodings/BlockEncoding.py)
  - Classes:
    - `BlockEncoding(cirq_ft.GateWithRegisters)`
- Abstract Select-Prepare Encodings
  - Description: Abstract class for block encodings via Select and Prepare oracles
  - Source: [BlockEncoding.py](src/pyLIQTR/BlockEncodings/BlockEncoding.py)
  - Classes:
    - `BlockEncoding_select_prepare(BlockEncoding)`

Specialized Encodings:

- Block encoding: Fermi-Hubbard
  - Description: Optimized block encodings for the single-orbital (band) Fermi-Hubbard model on a 2d, periodic square lattice (following Phys. Rev. X <b>8</b>, 041015)
  - Source: [fermi_hubbard_square_encoding.py](src/pyLIQTR/BlockEncodings/fermi_hubbard_square_encoding.py)
  - Classes:
    - `fermi_hubbard_square_encoding(BlockEncoding_select_prepare)`
- Block encoding: Fermionic_LinearT
  - Description: Implements encoding from section IV of [Linear T paper](https://arxiv.org/pdf/1805.03662.pdf)
  - Source: [LinearT.py](src/pyLIQTR/BlockEncodings/LinearT.py)
  - Classes:
    - `Fermionic_LinearT(BlockEncoding)`
- Block encoding: Generic Linear Combination of Unitaries (LCU)
  - Description: Generic block encoding for a linear combination of unitary operators, as expressed in terms of Pauli strings.
  - Source: [PauliStringLCU.py](src/pyLIQTR/BlockEncodings/PauliStringLCU.py)
  - Classes:
    - `PauliStringLCU(BlockEncoding_select_prepare)`

Helper Functions:

- Encoding Handler
  - Description: Handler function for specalized block encodings (in terms of problem instances)
  - Source: [getEncoding.py](src/pyLIQTR/BlockEncodings/getEncoding.py)
  - Functions:
    - `getEncoding()`

## IV. PEST: Generating Electronic Structure Hamiltonians

Plane-wave basis (PW)
Dual-plane wave basis (DPW)
Tools to manipulate atomic coordinates

## V. Problem Instances

Problem Specification:

- ProblemInstance Class
  - Description: Class used to specify problem instances, for use with block encodings
  - Source: [ProblemInstance.py](src/pyLIQTR/ProblemInstances/ProblemInstance.py)
  - Classes:
    - `ProblemInstance()`

Application Problems:

- Fermionic Model Hamiltonians
  - Description: Model Hamiltonians for fermionic systems
  - Source: [fermionic_models.py](src/pyLIQTR/ProblemInstances/fermionic_models.py)
  - Classes:
    - Fermi-Hubbard Model: `FermiHubbard(LatticeInstance)`
- Generic Lattice Hamiltonians
  - Description: Flexible problem instance for arbitrary models on lattices (currently restricted to spin models).
  - Source: [LatticeInstance.py](src/pyLIQTR/ProblemInstances/LatticeInstance.py)
  - Classes:
    - `LatticeInstance(ProblemInstance)`
- Spin Hamiltonians
  - Description: Named spin models for lattice systems.
  - Source: [spin_models.py](src/pyLIQTR/ProblemInstances/spin_models.py)
  - Classes:
    - Heisenberg Model: `Heisenberg(LatticeInstance)`
    - Transverse-Field Ising Model: `Transverse_Field_Ising(Heisenberg)`
- Chemical Hamiltonians (Local-basis electronic structure)
  - Description: Electronic structure for Hamiltonians defined with a local basis set (Slatter/Gaussian-Type Orbitals).
  - Source: [ChemicalHamiltonian.py](src/pyLIQTR/ProblemInstances/ChemicalHamiltonian.py)
  - Classes:
    - `ChemicalHamiltonian(ProblemInstance)`
- Material Hamiltonians (Nonlocal-basis electronic structure)
  - Description: Electronic structure for Hamiltonians defined with a nonlocal basis set (Plane-wave/Dual plane-wave bases).
  - Source: [ElectronicStructure.py](src/pyLIQTR/ProblemInstances/ElectronicStructure.py)
  - Classes:
    - `ElectronicStructure(ProblemInstance)`

## VI. Tutorials

Application Problems:

- Electronic Structure: Local basis (small-molecules in a STO/GTO basis) <a id="gto"></a>
  - Description: Resource estimation for electronic structure problems that are defined in a local basis.  Applies to most calculations with isolated (i.e., non-periodic) small-molecules
  - Notebooks: [LocalChemistry](Examples/ApplicationInstances/LocalChemistry/)
- Electronic Structure: Nonlocal basis (materials in a periodic dual plane-wave basis) <a id="dpw"></a>
  - Description: Resource estimation for ground state energy estimation (GSEE) of periodic electronic structure systems with Hamiltonians expressed in the dual plan wave (DPW) basis. Main example used is a magnesium slab.
  - Notebooks: [PeriodicChemistry](Examples/ApplicationInstances/PeriodicChemistry/)
- Fermi-Hubbard Model <a id="fermi-hubbard"></a>
  - Description: Resource estimation for the Fermi-Hubbard model.  Includes demonstrations with isolated block encodings, qubitized quantum dynamics, and qubitized phase estimation.
  - Notebooks: [FermiHubbard](Examples/ApplicationInstances/FermiHubbard/)
- General Lattice Models <a id="arbitrary-spin-lattice"></a>
  - Description: Block encodings and circuits for general spin models on arbitrary lattices.  Demonstrated briefly in the cited notebooks.
  - Notebooks: [Algorithms_and_Infrastructure](Examples/Algorithms_and_Infrastructure/)
- Heisenberg Model <a id="heisenberg"></a>
  - Description: Resource estimation for the Heisenberg model.  Includes demonstrations with isolated block encodings, qubitized quantum dynamics, and qubitized phase estimation.
  - Notebooks: [Heisenberg](Examples/ApplicationInstances/Heisenberg/)

General Infrastructure:

- Detailed Discussions (broader capabilities)  <a id="gto"></a>
  - Description: Notebooks that demonstrate pyLIQTRs general capabities for block encodings, problem instances, and qubitized circuits.  Also demonstrates how to define general spin/fermionic Hamiltonians on arbitrary lattices in any dimensionality.
  - Notebooks: [Algorithms_and_Infrastructure](Examples/Algorithms_and_Infrastructure/)


## VII. Classical Simulation Methods (pyLIQTR.clam)

Integration of classical ODE’s 
Conversion of spin / fermionic Hamiltonians for classical numerics
Simulation of quantum dynamics
Differential equations
  - Description: Linearization methods (Carleman) and definition of different nonlinear differential equations.  Includes: (i) van der Pol oscillator, Lotka-Volterra system; (2) Viscous Burger’s equation; (3) Nonlinear Vlasov equation w/ scalar collision kernel
  - Source: [linearization_tools](src/pyLIQTR/clam/linearization_tools)
  - Notebooks: [linearization_tools.ipynb](src/pyLIQTR/clam/linearization_tools/linearization_tools.ipynb)

## VIII. General Utilities / Operations (pyLIQTR.clam)

Multilinear algebra
Pauli / Fermionic operators (classical methods)
Periodic system preparation
