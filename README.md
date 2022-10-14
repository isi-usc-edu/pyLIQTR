<h1> pyLIQTR Version 0.2.0</h1>
pyLIQTR (<em>LI</em>ncoln Laboratory <em>Q</em>uantum algorithm <em>T</em>est and <em>R</em>esearch), is a python-based implementation of Quantum Algorithms. The codebase is actively developed, and currently supports Hamiltonian Simulation via the Quantum Signal Processing (QSP) algorithm. 
<br><br>
The current implementation of QSP supports real Hamiltonians, but development is ongoing to support Hamiltonians with imaginary components.
<br>
<br>
Several jupyter notebooks are included in the repo to demonstrate how the code works.

<h2> Installation and Environment Setup </h2>
It is recommended that conda is used to manage the environment. A yaml file is included to facilitate this.

- Create and activate the environment:

        conda env create -n <Environment Name> --file pyQSP_env.yml

        conda activate <Environment Name>
        
- Install MPSolve polynomial solver, which is used in the angle generation algorithm. Instructions for installation can be found here: https://numpi.dm.unipi.it/software/mpsolve
- Install the pyQuantumAlgos package

        pip install .

<h2> Overview of Examples Included as Jupyter Notebooks </h2>

<h3> Example Physical Problem - Quantum Computation for the Vlasov Equation </h3>
Many physical problems can reformulated from a quantum algorithms perspective. The notebooks, <b>~/Examples/Tutorial_1a_Vlasov_Evolution.ipynb</b> describes an approach to treating the Vlasov Euqation in such a manner. Additionally, the notebook <b>~/Examples/Tutorial_1b_QSP_Demonstration-Operator_Level.ipynb</b> implements QSP at an operator level, as opposed to a gate level, to solve example problems, including an Ising model problem and the Vlasov-Hermite problem.


<h3> QSP for Hamiltonian Simulation </h3>
The notebook <b>~/Examples/Tutorial_2_QSP_Simulation.ipynb</b> contains a step-by-step demo for generating a quantum circuit using QSP for the Vlasov Equation problem formulated in the Vlasov (1a) notebook. After the intial QSP circuit is generated, subsequent circuits are produced at future timesteps to time-evolve the system. The results of the quantum simulation are obtained, using cirq's simulator as a backend. 

<br>
A classical simulation of the system is also performed, following the methods outlined in the Vlasov (1a) noteobok. 

<br>
The two solutions, the QSP gate-based quantum circuit simulation, and the classical simulation, are compared to demonstrate agreement.

<h3>Quantum Circuit Generation</h3>
The notebook <b>~/Examples/Tutorial_3_Generate_Circuit.ipynb</b> contains a step-by-step demo for generating a quantum circuit for a given Hamiltonian. Note: The user must provide a Hamiltonian as a list of Pauli Strings. These can be obtained from previously referenced notebooks in the repository. 

<br>
The notebook outputs an OpenQASM 2.0 file containing the quantum circuit.

<h3> Circuit Scaling</h3>
The notebook <b>~/Examples/Tutorial_4_Circuit_Scaling.ipynb</b> containst a series of cells focused on scaling the QSP generated circuit to larger problems using the Vlasov-Hermite Hamiltonian as a test case. Note, some of the cells may take a significant amount of time to run, but the array dictating the problem sizes spanned can be trimmed as appropriate. Additionally, data can be saved to a csv file for later or further analysis

<h2> Citation </h2>
To be populated once code is posted. Cannot create DOI on zenodo without link to repo.

<h2> Disclaimer </h2>

DISTRIBUTION STATEMENT A. Approved for public release: distribution unlimited.

© 2022 MASSACHUSETTS INSTITUTE OF TECHNOLOGY
    
    Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014)
    SPDX-License-Identifier: BSD-2-Clause
    
This material is based upon work supported by the Under Secretary of Defense for Research and Engineering under Air Force Contract No. FA8702-15-D-0001. Any opinions, findings, conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the Under Secretary of Defense for Research and Engineering.

A portion of this research was sponsored by the United States Air Force Research Laboratory and the United States Air Force Artificial Intelligence Accelerator and was accomplished under Cooperative Agreement Number FA8750-19-2-1000. The views and conclusions contained in this document are those of the authors and should not be interpreted as representing the official policies, either expressed or implied, of the United States Air Force or the U.S. Government. The U.S. Government is authorized to reproduce and distribute reprints for Government purposes notwithstanding any copyright notation herein.

The software/firmware is provided to you on an As-Is basis



