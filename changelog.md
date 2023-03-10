# High-level changelog (not all inclusive of all changes)
## v0.3.0
* Implemented a version of GSE using Suzuki-Trotter, and provided examples
* Added ability to more easily generate random angles with the number of angles equal to required precision (see `compute_hamiltonain_angles` notes in the exmaple tutorials)
* Replaced SelectVBase implementation with a non-recursive version based off of the QROM circuit from `arxiv.org/pdf/1805.03662` (addressing a max recursion issue that sometimes occured for large problems)
* Changes to problem-generation/simulation related method organization (pyLIQTR.sim_methods). See examples for updates
* Added deprecation warning for "legacy" angle generation (to be fully replaced in future releases)
* Added QSP examples for more problem inputs
* Fixed issue related to a mpmath precision
* Speed up the decomposition of cirq circuits into Clifford+T gateset.
* Allow the input precision for gate decomposition to be any float, not just a power of 10.
* Allow users to input a desired decomposition accuracy for an entire circuit, instead of for each gate.
* Limit the precision for the gate decomposition algorithm so it cannot be smaller than the approximation error due to floating point rounding of angles contained in the cirq gates (fixes issues when decomposing rotations close to multiples of pi/4).
