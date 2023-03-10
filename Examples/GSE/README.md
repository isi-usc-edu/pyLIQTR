# GSE
This directory contains five jupyter notebooks that demonstrate how to calculate the ground state energy of a given molecule using both classical methods and the quantum GSE algorithm based on the Hadarmard Test architecture and Trotterization. Additionally, the notebooks outline how to transform the generated circuits for resource analyses, as well as how to export the generated quantum circuits to the _OpenQASM 2.0_ format.

## Environment Setup
To run the GSE portion of the `pyLIQTR` requires two third-party packages that are not installed with the initial `pyLIQTR` installation. Specifically, those packages are `openfermion` (https://quantumai.google/openfermion) and `openfermionpyscf` (https://github.com/quantumlib/OpenFermion-PySCF). To install them, do the following:
```bash
    # Activate your local conda environment
    conda activate <Environment Name>

    # Install openfermion
    python3 -m pip install --user openfermion

    # Install openfermionpyscf
    python -m pip install --user openfermionpyscf
```

<b>Note:</b> If the above commands do not work for the `openfermion` and `openfermionpyscf` installations, please refer to the package websites for more information.

<b>Note</b>: It is our understanding that `openfermionpyscf` does not support Windows installations. If you are unable to install `openfermionpyscf`, sample molecular hamiltonian data files have been provided so the tutorial notebooks can be utilized. The `pyLIQTR.GSE` functionality does not depend on `openfermionpyscf`, it is only utilized to generate molecular hamiltonians.

## Notebook Summaries
### Tutorial_1a_H2_DirectDiagonalization.ipynb
This notebook introduces the process of finding the lowest energy eigenvalue for a given Hamiltonian describing a physical system, in this case $H_2$, by direct diagonalization.

### Tutorial_1b_H2_Trotterization.ipynb
This notebook demonstrates how to determine the lowest energy eigenvalue for a given Hamiltonian describing a physical system, again using $H_2$, by Trotterization.

### Tutorial_2_GSE_H2_Simulation.ipynb
This notebook contains a step-by-step demonstration for how to generate a quantum circuit that will calculate the ground state energy of $H_2$. The circuit is displayed, and then the circuit is simulated a number of times using `cirq`'s built in simulation functionality. A brief analysis is performed to demonstrate that the simulation is performing as expected. 

### Tutorial_3_GSE_Generate_Circuit.ipynb
This notebook generates a quantum circuit for calculating the ground state of $H_2$, then uses `pyLIQTR` functionality to demonstrate how that circuit can be decomposed into Clifford+T gates and output to _OpenQASM 2.0_.

### Tutorial_4_GSE_Circuit_Scaling.ipynb
This notebook demonstrates how `pyLIQTR` can be used to perform scaling studies. In it, $H_2$, $LiH$ , and various length hydrogen chains are examined.

### Tutorial_pyscf.ipynb
This notebook describes how to generate molecular hamiltonians using `openfermionpyscf` and save the molecular hamiltonians to HDF5 files. Note, the HDF5 files for the molecules of interest have been pre-generated and are contained in the repo. They can be replicated using this notebook, or the code in this notebook can be used as a template for generating molecular hamiltonians for other molecules of interest. The molecules in this notebook are:
- $H_2$
- $H_4$
- $H_8$
- $LiH$
