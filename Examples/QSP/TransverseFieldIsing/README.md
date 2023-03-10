# Transverse Field Ising
This directory contains five jupyter notebooks that demonstrate how to simulate systems described by the Transverse Field Ising Model (TFIM) using both classical methods and using the Quantum Signal Processing (QSP) algorithm. Additionally, the notebooks outline how to transform the generated circuits for resource analyses, as well as how to export the generated quantum circuits to the _OpenQASM 2.0_ format.

## Notebook Summaries
### Tutorial_1(a|b)_Transverse_Field_Ising_Model.ipynb
This notebook introduces the TFIM and shows a classical method both using classical analytical methods (Tutorial_1a) and through QSP (Tutorial_1b).

### Tutorial_2_QSP_Simulation.ipynb
This notebook contains a step-by-step demonstration for generating a quantum circuit using QSP for the TFIM described in _Tutorial_1a_. After the first circuit is generated and displayed for the initial timestep, subsequent circuits are generated for future timesteps to time-evolve the system. A simulation of the quantum circuits is then performed, and the results are validated against the classical method. This effectively demonstrates how one would implement the quantum algorithm from _Tutorial_1b_ as a circuit.

### Tutorial_3_Generate_Circuit.ipynb
This notebook uses pyLIQTR to generate a quantum circuit for the TFIM. The notebook then demonstrates how to output the generated `cirq` circuit to _OpenQASM 2.0_.

### Tutorial_4_Circuit_Scaling.ipynb
This notebook contains a series of cells focused on scaling the QSP generated circuit to larger problems by scaling the TFIM to include more spins.
