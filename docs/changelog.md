# High-level changelog (not all inclusive of all changes)
## v1.4.0
* Adds circuit decompositions for the first quantized block encoding
* Adds implementations of CFD matrices
* Adds scheduler handling of bloq.Allocate, bloq.Free, and bloq.Toffoli

## v1.3.6
* Contains the fix for [IndexError when intializing DoubleFactorized block encoding](https://github.com/isi-usc-edu/pyLIQTR/issues/42)
* Double factorized sf_error_threshold passed to alpha calculation in [#43](https://github.com/isi-usc-edu/pyLIQTR/pull/43)

## v1.3.5
* Contains the fix for [One-norm does not account for double factorization truncation threshold](https://github.com/isi-usc-edu/pyLIQTR/issues/31) in [#41](https://github.com/isi-usc-edu/pyLIQTR/pull/41)

## v1.3.4
* Contains the fix for [#26](https://github.com/isi-usc-edu/pyLIQTR/issues/26) as well as the citation update for [#32](https://github.com/isi-usc-edu/pyLIQTR/issues/32)

## v1.3.3
* Added [#23](https://github.com/isi-usc-edu/pyLIQTR/issues/23) the Fermi Hubbard Nearest Neighbor Instance

## v1.3.1
* Added missing picture from [#7](https://github.com/isi-usc-edu/pyLIQTR/issues/7) and missing csv file from [#24](https://github.com/isi-usc-edu/pyLIQTR/issues/24)

## v1.3.0
* Scheduler - allows for time-accurate, resource reliant circuit execution and advanced parallelism analysis
* Resource estimation for First Quantized block encoding
* Updated to [Qualtran 0.4.0](https://github.com/quantumlib/Qualtran/tree/v0.4.0)
* Call graph support
* Generator-based circuit decomposition
* [Online documentation](https://isi-usc-edu.github.io/pyLIQTR/)
* Various bugfixes to include
    * [Cannot instantiate Heisenberg model with zero local terms](https://github.com/isi-usc-edu/pyLIQTR/issues/12)
    * [DoubleFactorized block encoding fails when sf_error_threshold is zero](https://github.com/isi-usc-edu/pyLIQTR/issues/21)

## v1.2.1
* Contains the fix for [Qubit Hamiltonians not supported in phase estimation](https://github.com/isi-usc-edu/pyLIQTR/issues/18)

## v1.2.0
* Double Factorization improvements/tweaks (primarily to decrease resource requirements)
    * RotationsQROM changes to support measurement based uncompute
    * Binary to unary conversion optimization
    * Modified to enable controlled encoding generation
* Changes to multicontrolled gates in LinearT encoding to remove rotation and T gates unnecessarily inflating the resource counts
* Carleman Linearization
* Various tweaks to openqasm/estimate_resources and other helpers to support above
* Various bugfixes to include
    * [Cannot compare prepare operation to Ry operation](https://github.com/isi-usc-edu/pyLIQTR/issues/9)
    * [Invalid declaration of inverse CNOTs in clifford_plus_t_direct_transform](https://github.com/isi-usc-edu/pyLIQTR/issues/11)
    * [Ability to generate DF block encoding without performing JW](https://github.com/isi-usc-edu/pyLIQTR/issues/13)
* Beginning of coverage testing push
* to_openqasm was deprecated in favor of our internal openqasm() because of incomplete qasm() support within cirq
    * [to_openqasm fails for QubitizedPhaseEstimation](https://github.com/isi-usc-edu/pyLIQTR/issues/8)

## v1.1.1
* Double Factorization
* Migrated from cirq-ft to Qualtran v0.2 (cirq-ft deprecated/subsumed by Google)
* Assorted Bug Fixes
    * [QSVT dynamics circuit produces error when asked to provide QASM string](https://github.com/isi-usc-edu/pyLIQTR/issues/5)

## v1.0.0
* Beginning mass deprecation of legacy code (primarily QSP and PhaseEstimation related, sim_methods, angle generation). Some old code may break!
* Beginning to leverage cirq-ft for faster resource estimation
* Introduction of various problem instances
* Introduction of various block encodings
* Rework of QSP circuit generation via pyLIQTR.qubitization
* Introduction of julia package for generating electronic structure hamiltonians
* Introduction of classical method package
* Bug Fix for [Circuits with no rotation gates](https://github.com/isi-usc-edu/pyLIQTR/issues/4)


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
