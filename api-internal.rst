.. currentmodule:: src

Internal API reference, for users customizing behavior
======================================================
If you are contributing to the source code, not just using it, these are the classes that you should fix/extend/add to

Circuit Operators
'''''''''''''''''
These are gate operations

.. autosummary::
   :toctree: generated/

   pyLIQTR.circuits.operators.AddMod
   pyLIQTR.circuits.operators.cascading_translation
   pyLIQTR.circuits.operators.DF_RotationsBlock
   pyLIQTR.circuits.operators.ExampleFMatrices
   pyLIQTR.circuits.operators.hamiltonian_encodings
   pyLIQTR.circuits.operators.multiCZ
   pyLIQTR.circuits.operators.prepare_oracle_pauli_lcu
   pyLIQTR.circuits.operators.prepare
   pyLIQTR.circuits.operators.reflect
   pyLIQTR.circuits.operators.select_oracle_pauli_lcu
   pyLIQTR.circuits.operators.select_prepare_pauli
   pyLIQTR.circuits.operators.selectV
   pyLIQTR.circuits.operators.selectVutils
   pyLIQTR.circuits.operators.swap_network    

CLAM
''''
Classical Simulation Methods

.. autosummary::
   :toctree: generated/

   pyLIQTR.clam.cell_complex
   pyLIQTR.clam.geometry
   pyLIQTR.clam.multilinear_algebra
   pyLIQTR.clam.operator_complex
   pyLIQTR.clam.operator_sims
   pyLIQTR.clam.operators
   pyLIQTR.clam.physical_data
   pyLIQTR.clam.simqsp
   pyLIQTR.clam.tensor_methods
   pyLIQTR.clam.time_dynamics

Gate Decomposition
''''''''''''''''''

.. autosummary::
   :toctree: generated/

   pyLIQTR.gate_decomp.clifford_gates
   pyLIQTR.gate_decomp.ellipse
   pyLIQTR.gate_decomp.exact_decomp
   pyLIQTR.gate_decomp.gate_approximation
   pyLIQTR.gate_decomp.grid_operator
   pyLIQTR.gate_decomp.matrices
   pyLIQTR.gate_decomp.point_enumeration
   pyLIQTR.gate_decomp.rings
   pyLIQTR.gate_decomp.rotation_gates
   pyLIQTR.gate_decomp.solve_diophantine

Phase Factors
'''''''''''''''

.. autosummary::
   :toctree: generated/

   pyLIQTR.phase_factors.conversions
   pyLIQTR.phase_factors.named_factors
   pyLIQTR.phase_factors.phase_factors
   pyLIQTR.phase_factors.simulation

Scheduler
'''''''''

.. autosummary::
   :toctree: generated/

   pyLIQTR.scheduler.DAG
   pyLIQTR.scheduler.scheduler_utils

Utility Methods
'''''''''''''''
A set of methods to provide functional support to help with things such as generating QASM strings or managing ancillas. 
Plus a lot of other functionality that doesn't fit well anywhere else.

.. autosummary::
   :toctree: generated/

   pyLIQTR.utils.get_hdf5
   pyLIQTR.utils.global_ancilla_manager
   pyLIQTR.utils.Hamiltonian
   pyLIQTR.utils.pauli_string_manip
   pyLIQTR.utils.plot_helpers
   pyLIQTR.utils.qsp_helpers
   pyLIQTR.utils.resource_analysis
   pyLIQTR.utils.utils
