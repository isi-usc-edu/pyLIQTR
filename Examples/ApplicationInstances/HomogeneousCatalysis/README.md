# Homogeneous Catalysis
The Jupyter notebook in this directory uses a small homogeneous catalyst model to demonstrate resource analysis for qubitized phase estimation using the `DoubleFactorized` block encoding based on Figure 16 of ["Even more efficient quantum computations of chemistry through tensor hypercontraction"](https://arxiv.org/abs/2011.03494) and the `ChemicalHamiltonian` ProblemInstance.

## Notebook Summaries
### double_factorized_encoding_and_phase_estimation.ipynb
This notebook walks through the steps for taking a catalyst as described by an fcidump file, generating a ProblemInstance, block encoding it using the `DoubleFactorized` algorithm, and estimating resources for the resulting qubitized phase estimation circuit. Printing the circuit to qasm is demonstrated at the end.

## Additional Files
### fcidump.32_2ru_III_3pl
Example of a seven-coordinate ruthenium complex from [*Inorg. Chem.* 2014, 53, 13, 6904–6913](https://doi.org/10.1021/ic500709h).