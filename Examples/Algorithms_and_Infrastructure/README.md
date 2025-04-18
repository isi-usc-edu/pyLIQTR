# Algorithms_and_Infrastructure
The Jupyter notebooks in this directory provide a more in-depth demonstration for certain components of pyLIQTR.


## Notebook Summaries
### block_encodings.ipynb
Provides a more nuanced discussion of pyLIQTR's BlockEncoding class.

### problem_instances.ipynb
Demonstrates how derived instances of the ProblemInstance class can be used to frame different calculations.  This includes the generation of bespoke models, such as arbitrary spin Hamiltonians.

### qsvt_and_qsp.ipynb
Provides additional context for our QSP/QET/QSVT tools, including their use with arbitrary Hamiltonians.

### scheduling_example.ipynb
Demonstrates how we build a Quantum Circuit and then use our internal Scheduler to create time-accurate, resource reliant circuit execution and advanced parallelism analysis.

### hardware_resource_estimation.ipynb
Demonstrates how we take a Hamiltonian Circuit and perform both logical and hardware profile specific resource estimation. We use Microsoft's Azure|trade| to perform the Hardware Estimate.
