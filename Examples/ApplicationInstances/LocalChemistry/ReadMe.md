# Local Chemistry
The Jupyter notebook in this directory walks through resource analysis for qubitized phase estimation for small molecules in a local-basis representation using the generic `PauliLCU` block encoding and the `ChemicalHamiltonian` ProblemInstance.

## Notebook Summaries
### ChemicalHamiltonian_Example
This notebook shows how to create a `ChemicalHamiltonian` instance, either from our provided list of molecules (details found in `chemistry_instances.csv`) or from molecular parameters including geometry, charge, multiplicity and basis. It then goes through generating a `BlockEncoding` to create a qubitized phase estimation circuit, along with performing resource estimation for the circuit. 

## Additional Files
### chemistry_instances.csv
List of molecules and associated parameters including geometry, charge, multiplicity and basis.