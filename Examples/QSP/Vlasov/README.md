# The Vlasov Equation  
This directory contains five jupyter notebooks that demonstrate how to simulate systems described by the Vlasov equation using both classical methods and using the Quantum Signal Processing (QSP) algorithm. Additionally, the notebooks outline how to transform the generated circuits for resource analyses, as well as how to export the generated quantum circuits to the _OpenQASM 2.0_ format.

## Notebook Summaries
### Tutorial_1a_Vlasov_Evolution.ipynb
This notebook introduces the Vlasov equation, including a discussion on some of the limiting cases. It goes on to discuss ways of solving the Vlasov-Maxwell system, both using classical analytical methods and quantum algorithms. An example problem is then solved using two classical techniques.

### Tutorial_1b_QSP_Demonstration-Operator_Level.ipynb
This focuses on implementing QSP at an operator level (as opposed to a gate level). Example problems are solved to demonstrate the implementation of the QSP algorithm. Note, no quantum circuits are constructed.

### Tutorial_2_QSP_Simulation.ipynb
This notebook contains a step-by-step demonstration for generating a quantum circuit using QSP for the Vlasov problem formulated in _Tutorial_1a_Vlasov_Evolution.ipynb_. After the first circuit is generated for the initial timestep, subsequent circuits are generated for future timesteps to time-evolve the system. A simulation of the quantum circuits is then performed, and the results are compared to those determined using the classical methods in _Tutorial_1a_Vlasov_Evolution.ipynb_. This effectively demonstrates how one would implement the quantum algorithm from _Tutorial_1b_ as a circuit.

### Tutorial_3_Generate_Circuit.ipynb
This notebook uses pyLIQTR to generate a quantum circuit for a single time-step of the Vlasov problem formulated in _Tutorial_1a_Vlasov_Evolution.ipynb_. The notebook then demonstrates how to output the generated `cirq` circuit to _OpenQASM 2.0_.

### Tutorial_4_Circuit_Scaling.ipynb
This notebook contains a series of cells focused on scaling the QSP generated circuit to larger problems using the Vlasov-Hermite Hamiltonian as a test case. 
