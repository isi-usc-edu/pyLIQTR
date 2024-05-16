# Fermi-Hubbard Model
The Jupyter notebooks in this directory walk through several resource analysis tasks for the Fermi-Hubbard model using the `FermiHubbard` ProblemInstance. Target algorithms include qubitized quantum dynamics (via QSP/QET/QSVT) and qubitized ground-state energy estimation (GSEE). The block encodings used include the generic `PauliLCU` and the efficient Fermi-Hubbard specific model, `FermiHubbardSquare`, from Section V of ["Encoding Electronic Spectra in Quantum Circuits with Linear T Complexity"](https://arxiv.org/abs/1805.03662).

## Notebook Summaries
### fermi_hubbard-introduction_and_encoding.ipynb
This notebook introduces the Fermi-Hubbard model and takes a close look at several of its block encodings. Topics include circuit generation, resource analysis, and the generation of scaling estimates for larger systems. 

### fermi_hubbard-dynamics-qubitized.ipynb
This notebook introduces and demonstrates qubitized quantum dynamics (via QSP/QET/QSVT) for the Fermi-Hubbard model.  Topics include the generation of phase angles for quantum simulation, the construction of simulation circuits, resource analysis, and scaling estimates for larger systems. 


### fermi_hubbard-phase_estimation.ipynb
This notebook introduces and demonstrates qubitized ground-state energy estimation for the Fermi-Hubbard model.  Topics include the construction of simulation circuits, resource analysis, and scaling estimates for larger systems. 

