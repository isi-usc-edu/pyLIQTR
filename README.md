# pyLIQTR

_Kevin Obenland, Justin Elenewski, Kaitlyn Morrell, Rylee Stuart Neumann, Arthur Kurlej, Robert Rood, John Blue, Joe Belarge, Benjamin Rempfer & Parker Kuklinski_

---

## Description

`pyLIQTR` (<em>LI</em>ncoln Laboratory <em>Q</em>uantum algorithm <em>T</em>est and <em>R</em>esearch), is a python library for building quantum circuits derived from quantum algorithms and generating Clifford+T resource estimates.

This package is built extensively atop `cirq` & the recent release of `qualtran`.

pyLIQTR is primarily structured as:

- pyLIQTR.ProblemInstances
  - A set of code that provides an easy interface (through `pyLIQTR.ProblemInstances.getInstance`) to generate Hamiltonians that capture various physical models of interest.
- pyLIQTR.BlockEncodings
  - A set of codes that provides an easy interface (through `pyLIQTR.BlockEncodings.getEncoding`) to generate BlockEncodings that encode the action of a ProblemInstance through various different encodings.
- pyLIQTR.clam
  - A set of code that provides various classical utilities supporting:
    - Integration of classical ODE’s (following application problems)
    - Conversion of spin / fermionic Hamiltonians to classical numerics
    - Simulation of quantum dynamics
    - Quadratic / Integrable Hamiltonians (fast; polynomial scaling)
    - General Hamiltonians (exponentially scaling)
    - Source / Notebooks: This is scattered diffusely throughout the examples (Vlasov, Nonequilibrium, Heisenberg / Hubbard).
- pyLIQTR.qubitization
  - A set of code that provides ways of building QSP/QSVT circuits in the context of the pyLIQTR package.
- pyLIQTR.phase_factors
  - A set of code that calculates the angles required for a given QSP/QSVT sequence
- pyLIQTR.circuits
  - A set of pyLIQTR cirq circuits/gates used in our implementation of block encodings and algorithms.
- pyLIQTR.gate_decomp
  - A set of code that performs gate synthesis (converting rotations to Clifford+T gates) for a specified precision.
- pyLIQTR.pest_interface
  - A pythonic interface to the PEST Julia package.
- pyLIQTR.utils
  - A set of various utility functions.
  - `pyLIQTR.utils.resource_analysis.estimate_resources` (adds robust estimation in addition to `qualtran.t_complexity`) can be used to determine the Clifford+T cost of any circuit generated with pyLIQTR
  - `pyLIQTR.utils.circuit_decomposition.circuit_decompose_multi` (adds robust decomposition in addition to `cirq`'s decomposition functions) can be used to decompose circuits to 1+2QB gates

In addition, a seperate Julia package for generating electronic structure hamiltonians (root/PEST) is included with pyLIQTR.

pyLIQTR is a work-in-progress, as time goes on, we hope to provide better and more extensive documentation and examples in addition to new features. Please reach out to us if anything is unclear, so we can prioritize the documentation of the aspects that are most frequently used or most unclear.

---

## Features

The features of this package are described in detail in [FEATURES.md](Features.md). This includes not only descriptions, but also source files and example notebooks.

---

## Installation and Environment Setup

- You can install the latest public tagged release from [PYPI](https://pypi.org/project/pyLIQTR/) using pip:

        pip install pyLIQTR

- You can install the latest state of the main branch:

        pip install git+https://github.com/isi-usc-edu/pyLIQTR.git

If you are going to be activelty developing the pyLIQTR source code, it is recommended that conda be used to manage the environment. A setup.py file is included to facilitate this.
Change directory to the location of setup.py, then perform the following commands.

- Create and activate the environment:

        on Windows use:
        conda create -n <Environment Name> "python>=3.8,<=3.11.5"
        on Mac use:
        conda create -n <Environment Name> python'>=3.8,<=3.11.5'

        conda activate <Environment Name>

- Install the package

        pip install .

- If you are working as a developer (for all platforms and shells):

        on Windows use:
        pip install -e .[dev]
        on Mac use:
        pip install -e ."[dev]"

<a id="optional-installs"></a>
### Optional Installs

- Install openfermionpyscf, which is used for the DoubleFactorized block encoding.
    pip install openfermionpyscf
- If not installed, then scipy will be used to calculate valid angles. Angle generation using scipy will generally take longer than angle generation using mpsolve.
Effort is underway to deprecate the existing angle generation in favor of a more portable and efficient method.
- Install pyscf.
  - pyscf is used in conjunction with openfermion in order to generate new problem instances as input into our GSE and Double Factorization implementation.
    pip install pyscf

---

## Overview of Examples Included as Jupyter Notebooks

Notebooks showcasing features are organized as follows in the /Examples directory.

- Algorithm_and_Infrastructure
  - A set of notebooks that go over the ProblemInstance and BlockEncoding infrastructure, in addition to a QSVT/QSP overview. Note that these are still being drafted.
- AngleGeneration
  - Contains three notebooks describing the reworked angle generation features
- ApplicationInstances
  - Contains four directories showing different types of physical models and how one may use the pyLIQTR package to generate circuits and provide resource estimates.
- PEST
  - An introduction to the generating electronic structure hamiltonians using the PEST package.

---

## Citation

Please use this DOI number reference, published on [Zenodo](https://zenodo.org), when citing the software:

![DOI](image.png)

---

## Disclaimer

DISTRIBUTION STATEMENT A. Approved for public release: distribution unlimited.

© 2023 MASSACHUSETTS INSTITUTE OF TECHNOLOGY

    Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014)
    SPDX-License-Identifier: BSD-2-Clause

This material is based upon work supported by the Under Secretary of Defense for Research and Engineering under Air Force Contract No. FA8702-15-D-0001. Any opinions, findings, conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the Under Secretary of Defense for Research and Engineering.

A portion of this research was sponsored by the United States Air Force Research Laboratory and the United States Air Force Artificial Intelligence Accelerator and was accomplished under Cooperative Agreement Number FA8750-19-2-1000. The views and conclusions contained in this document are those of the authors and should not be interpreted as representing the official policies, either expressed or implied, of the United States Air Force or the U.S. Government. The U.S. Government is authorized to reproduce and distribute reprints for Government purposes notwithstanding any copyright notation herein.

The software/firmware is provided to you on an As-Is basis
