.. currentmodule:: src

pyLIQTR API Reference
=====================

Block Encodings
'''''''''''''''
The different types of BlockEncodings that are currently supported within pyLIQTR

.. autosummary::
   :toctree: generated/

   pyLIQTR.BlockEncodings.getEncoding
   pyLIQTR.BlockEncodings.BlockEncoding
   pyLIQTR.BlockEncodings.CarlemanLinearization
   pyLIQTR.BlockEncodings.DoubleFactorized
   pyLIQTR.BlockEncodings.fermi_hubbard_square_encoding
   pyLIQTR.BlockEncodings.FirstQuantized
   pyLIQTR.BlockEncodings.LinearT
   pyLIQTR.BlockEncodings.PauliStringLCU

Circuit Operators
'''''''''''''''''
Classes that define different circuit operations

.. autosummary::
   :toctree: generated/

   pyLIQTR.circuits.operators.cyclic_shift
   pyLIQTR.circuits.operators.diagonal_bitstring
   pyLIQTR.circuits.operators.BinaryToUnary
   pyLIQTR.circuits.operators.BitwiseZRotation
   pyLIQTR.circuits.operators.ControlledUniformSuperposition
   pyLIQTR.circuits.operators.DF_InnerPrepare
   pyLIQTR.circuits.operators.DF_OuterPrepare
   pyLIQTR.circuits.operators.FixupTableQROM
   pyLIQTR.circuits.operators.FlaggedPrepareUniformSuperposition
   pyLIQTR.circuits.operators.PhaseGradientRotation
   pyLIQTR.circuits.operators.prepare_LinearT
   pyLIQTR.circuits.operators.ProductPauliExponentials
   pyLIQTR.circuits.operators.QROMwithMeasurementUncompute
   pyLIQTR.circuits.operators.RotationsQROM
   pyLIQTR.circuits.operators.select_LinearT
   pyLIQTR.circuits.operators.translation

CLAM
''''
Classical Simulation Methods

.. autosummary::
   :toctree: generated/

   pyLIQTR.clam.operator_strings
   pyLIQTR.clam.operator_lattice
   pyLIQTR.clam.lattice_definitions

   
Phase Factors
'''''''''''''''

.. autosummary::
   :toctree: generated/

   pyLIQTR.phase_factors.angler

Phase Estimation
''''''''''''''''

.. autosummary::
   :toctree: generated/

   pyLIQTR.PhaseEstimation.pe
   pyLIQTR.PhaseEstimation.pe_gates
   pyLIQTR.PhaseEstimation.pe_sim

Problem Instances
'''''''''''''''''
Classes that are used by specific Quantum Algorithms

.. autosummary::
   :toctree: generated/

   pyLIQTR.ProblemInstances.ProblemInstance
   pyLIQTR.ProblemInstances.getInstance
   pyLIQTR.ProblemInstances.ChemicalHamiltonian
   pyLIQTR.ProblemInstances.ElectronicStructure
   pyLIQTR.ProblemInstances.fermionic_models
   pyLIQTR.ProblemInstances.LatticeInstance
   pyLIQTR.ProblemInstances.MatrixInstance
   pyLIQTR.ProblemInstances.NonlinearODE
   pyLIQTR.ProblemInstances.spin_models

Gate Decomposition
''''''''''''''''''

.. autosummary::
   :toctree: generated/

   pyLIQTR.gate_decomp.cirq_transforms
   pyLIQTR.scheduler.advanced_decomposition.advanced_decompose

Qubitization
''''''''''''

.. autosummary::
   :toctree: generated/

   pyLIQTR.qubitization.phase_estimation
   pyLIQTR.qubitization.qsvt_dynamics
   pyLIQTR.qubitization.qsvt
   pyLIQTR.qubitization.qubitized_gates

Scheduler
'''''''''

.. autosummary::
   :toctree: generated/

   pyLIQTR.scheduler.scheduler
   pyLIQTR.scheduler.Instruction
   pyLIQTR.scheduler.sets

Utility Methods
'''''''''''''''
A set of methods to provide functional support to help with things such as generating QASM strings or managing ancillas. 
Plus a lot of other functionality that doesn't fit well anywhere else.

.. autosummary::
   :toctree: generated/

   pyLIQTR.utils.circuit_decomposition
   pyLIQTR.utils.printing
   pyLIQTR.utils.resource_analysis