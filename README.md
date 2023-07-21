# pyLIQTR
_Kevin Obenland, Justin Elenewski, Arthur Kurlej, Joe Belarge, John Blue &  Robert Rood_

---
## Description
`pyLIQTR` (<em>LI</em>ncoln Laboratory <em>Q</em>uantum algorithm <em>T</em>est and <em>R</em>esearch), is a python library for building quantum circuits derived from quantum algorithms. Once the circuits have been generated, one can simulate them (if they are small enough), or utilize them for resource estimations. The codebase is actively developed, and currently supports the following algorithms:
- Hamiltonian Simulation using Quantum Signal Processing (QSP) for real-valued Hamiltonians

- Ground State Estimation (GSE) using the Hadamard Test architecture with Trotterization
    - Requires openfermion (which is by default not installed), see note in Optional Installs section below. 

---

## Installation and Environment Setup
It is recommended that conda be used to manage the environment. A setup.py file is included to facilitate this.
Change directory to the location of setup.py, then perform the following commands.

- Create and activate the environment:

        conda create -n <Environment Name> python=3.8

        conda activate <Environment Name>
        
- Install the package

        pip install .

- Or, as a developer:

        pip install --editable .
        
### Optional Installs
- Install MPSolve polynomial solver, which is used in the angle generation algorithm. Instructions for installation can be found here: https://numpi.dm.unipi.it/software/mpsolve
- If not installed, then scipy will be used to calculate valid angles. Angle generation using scipy will generally take longer than angle generation using mpsolve.
Effort is underway to deprecate the existing angle generation in favor of a more portable and efficient method.
- Install openfermion & pyscf. 
    - Openfermion is used for the implementation of Suzuki-Trotter and as the input to our implementation of GSE. This is required to use the GSE implementation.
    - pyscf is used in conjunction with openfermion in order to generate new problem instances as input into our GSE implementation.

---

## Overview of Examples Included as Jupyter Notebooks

There are a variety of examples provided to demonstrate the capabilities of `pyLIQTR`. In particular, there are demonstrations of the following algorithms (and example problems) under `<pyLIQTR root>/Examples`:
* QSP
    * Vlasov-Hermite Model
    * Heisenberg Model
    * Transverse-field Ising Model
* GSE

Where each demonstration contains five tutorial notebooks, which each serve the following purpose:
* _Tutorial_1a (Problem Description & Classical Solving)_
    * This tutorial provides a brief introduction to the physical problem of interest and how `pyLIQTR` can be used to generate a representation of the model. This tutorial then showcases how `pyLIQTR` and/or `numpy`/`scipy` methods can then be used to classically solve the problem or simulate the system. We note that since we are interested in classically hard problems, these notebooks focus on small problems or special cases that can be executed on most personal computers.
* _Tutorial_1b (Solving with a Quantum Algorithm)_
    * This tutorial, in conjunction with 1a, shows how this problem can be mapped from a _"classical"_ method of solving the problem to a method conducive to running on a quantum computer, and shows how similar results can be achieved.
* _Tutorial_2 (Simulation of quantum circuit and classical comparison)_
    * This tutorial contains a step-by-step `pyLIQTR` demo for generating a representation of the problem of interest, generating quantum circuits that simulate or solve the problem, and then simulating the quantum circuit and comparing the results against a classical technique to show that the implementation of the quantum algorithm is correct. Again, since we are interested in classically hard problems and since, for the most part, classical computers cannot efficiently simulate a quantum computer, we focus on small problems or special cases that can be ran on most personal computers.
* _Tutorial_3 (Circuit manipulation and output)_
    * This tutorial shows how to take the `cirq` circuits that `pyLIQTR` generates in <i>Tutorial_2</i> and decompose them to simple gates, transform them to Clifford+T gates, and to export to an _OpenQASM_ format in order to allow for further analysis or manipulation in other tools.
* _Tutorial_4 (Circuit scaling and simple analysis)_
    * This tutorial is intended to provide a _"jumping off point"_ for using `pyLIQTR` to investigate problem/algorithm scaling by generating circuits of various different sizes and how they can be analyzed in python. 

**To summarize**, Tutorials 1a, 1b, and 2 are useful for those interested in better understanding the physical models or problems, how they are classically solved and/or simulated, and how a particular quantum algorithm can be used to solve and/or simulate the problem. Tutorials 3 and 4 are useful for those interested in generating circuits for downstream analyses, such as resource analysis, for a specific problem.

---

## Citation
<a href="https://zenodo.org/badge/latestdoi/545621986"><img src="https://zenodo.org/badge/545621986.svg" alt="DOI"></a>
---

## Disclaimer

DISTRIBUTION STATEMENT A. Approved for public release: distribution unlimited.

© 2023 MASSACHUSETTS INSTITUTE OF TECHNOLOGY
    
    Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014)
    SPDX-License-Identifier: BSD-2-Clause
    
This material is based upon work supported by the Under Secretary of Defense for Research and Engineering under Air Force Contract No. FA8702-15-D-0001. Any opinions, findings, conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the Under Secretary of Defense for Research and Engineering.

A portion of this research was sponsored by the United States Air Force Research Laboratory and the United States Air Force Artificial Intelligence Accelerator and was accomplished under Cooperative Agreement Number FA8750-19-2-1000. The views and conclusions contained in this document are those of the authors and should not be interpreted as representing the official policies, either expressed or implied, of the United States Air Force or the U.S. Government. The U.S. Government is authorized to reproduce and distribute reprints for Government purposes notwithstanding any copyright notation herein.

The software/firmware is provided to you on an As-Is basis



