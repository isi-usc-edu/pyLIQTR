# High-level changelog (not all inclusive of all changes)
## v1.1.1
* Double Factorization
* Assorted Bug Fixes

## v1.0.0
* Beginning mass deprecation of legacy code (primarily QSP and PhaseEstimation related, sim_methods, angle generation). Some old code may break!
* Beginning to leverage cirq-ft for faster resource estimation
* Introduction of various problem instances
* Introduction of various block encodings
* Rework of QSP circuit generation via pyLIQTR.qubitization
* Introduction of julia package for generating electronic structure hamiltonians
* Introduction of classical method package


## v0.3.1
* Update to QASM printing to be generator based & faster (see Examples/Utilities/Import_Export_and_Printing notebook)
* Updates to Clifford+T decomposition (see Examples/Utilities/Import_Export_and_Printing notebook)
    * Condensed printing of rx/ry/rz gates via exporting of rx_d/ry_d/rz_d gates
    * Randomized generation
* Setup now only requires cirq.core

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
