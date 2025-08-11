# Best Practices
For the most part, this is a compilation of LIQTR Likes or comments about some of the non-circuit/operator features of pyLIQTR, in terms of making it a bit more user-friendly and more speedy to use. It is recommended to take a peek through this section periodically to confirm whether anything is new. If there is something that has not been made clear in notebooks or in communication, please let us know so we can try to address it with information in this section. Thanks :)

#### LIQTR Like 👍 : Resource Analysis
Resource analysis of `cirq` circuits can be done by using the `pyLIQTR.utils.resource_analysis.estimate_resources` function, which will provide T, qubit, and Clifford counts for most circuits!

Note that our estimate for T-count is done via experimentally derived statistical distribution for the Clifford count of rotations at a given precision. This can be passed explicitly using the `rotation_gate_precision` (default 1e-10) keyword argument, or can be implicitly calculated by passing the `circuit_precision`. If no precision argument is provided, the default value will be used.

#### LIQTR Like 👍 : Circuit Decomposition
`cirq` / `qualtran` provide the ability to decompose circuits/operators into their constituent parts (ie, a QSVT sequence is a cosine and then sine sequence). Different circuits will have different levels of decomposition, since they are composed of different operators. 

In pyLIQTR, we have introduced a set of utilities that wrap the `cirq` / `qualtran` methods and address shortcomings or tweaks. The set of utilities are:
- `pyLIQTR.utils.circuit_decomposition.circuit_decompose_multi`
    - Given a cirq.Circuit, will decompose the circuit the specified number of times (N).
    - Will assume a `cirq.DecompositionContext` that minimizes the ancilla usage
    - Note that you can use the similar operator `pyLIQTR.utils.circuit_decomposition.circuit_decompose_once` (which is just this function but with N=1)
- `pyLIQTR.utils.circuit_decomposition.generator_decompose`
    - This is the underlying engine in the above decompose_multi, and unveils a number of arguments that enable more control over the decomposition (fallbacks, what to keep, what to do on failure, etc).
    - This is a tweaked version of `cirq.decompose`, which modifies some under-the-hood functions and methods in `cirq`
    - In general, we do not necessarily recommend using this feature

In general, decompositions require operators to have decompositions defined. All `pyLIQTR.circuits.operators` will have decompositions defined, but not all `cirq` / `qualtran` operators necessarily have circuit-based decompositions defined. If this happens, you will get an error. You may also receive errors akin to `Duplicate QIDs`, which is an error that comes up in some `qualtran` bloqs due to an ancilla allocation issue. A temporary fix can be done by changing the context to `context=cirq.DecompositionContext(cirq.SimpleQubitManager())`, which generally resolves this issue. If you discover this issue please let us know so we can reach out to the `qualtran` developers or see if we can resolve the issues ourselves.

 All examples shown in `pyLIQTR` should support decomposition into "basic gates" (1-2Q gates + Toffolis), and an easy test is feeding your circuit of choice into `pyLIQTR.utils.printing.openqasm` and verifying that you can loop through the circuit to completion.


#### LIQTR Like 👍 : Export of circuits to OpenQASM
Export of circuits can be done using a provided `pyLIQTR` utility (`pyLIQTR.utils.printing.openqasm`) that wraps `cirq`'s native openQASM export. This is introduced since some new cirq-ft operators do not yet have a decomposition in terms of gates defined, or `cirq` has begun to use features (mid-circuit measurement, classical control, etc) not supported by OpenQASM2.

This utility fully decomposes circuits to either include rotation gates (`rotation_allowed=True`) or not (`rotation_allowed=False`). 

When use_random_decomp=True (this is the default setting) then the rotations are approximated from an experimentally derived statistical distribution for the Clifford count of rotations at a given precision. *When use_random_decomp=False then the execution will be orders of magnitude slower*.

In the latter case, we make use a of a Clifford+T representation outlined in https://arxiv.org/pdf/0806.3834.pdf (and further discussed in https://arxiv.org/pdf/1312.6584.pdf and http://www.cs.umd.edu/~amchilds/qa/qa.pdf, as potentially easier to digest forms).

When `rotation_allowed=False`, we introduce the "rx_d/ry_d/rz_d gate", followed by a line of the form:
```
// (leading_T, gate_sequence, sequence_length, clifford_part)
```
* `leading_T` : A bool indicating whether or not there is a `T` gate at the start of the sequence
* `gate_sequence`: An int whose binary representation encodes a sequence of `H`, `S`, and `T` gates
* `sequence_length`: The number of `T` gates in the sequence (note if `leading_T` is true, the total number of gates is `sequence_length+1`)
* `clifford_part`: A List[str] of clifford gates that come at the end of the H,S,T sequence.

Note that if your circuit has rotations, you may also want to specify a level of precision by passing `gate_precision` or `circuit_precision` to `pyLIQTR.utils.printing.openqasm` (see Clifford+T decomposition notes)


#### LIQTR Like 👍 : Circuit Scheduling
 
Circuits can be 'scheduled' using the `pyLIQTR.scheduler.scheduler.schedule_circuit` function. This will provide circuit runtime, realistic qubit counts, T-depth, and more!
 
`schedule_circuit` takes a number of optional arguments depending on desired outputs. They are:
* `architecture_config`: an `architecture` class object specifying gate times and state factory parameters. See `scheduling_example.ipynb` for details.

* `full_profile`: default `False`. If set to `True`, function output will include T-widths and parallelism distribution data. (NOTE: T-widths and parallelism data will only be accurate if `decomp_level` is set to `'Full'`.)

* `decomp_level`: int, default 0. Levels of decomposition implemented before scheduling. If set to `0`, each operator will be decomposed to one/two-qubit gates only on its first occurance and cached for each future occurance. If set to `'Full'`, entire circuit will be decomposed to one/two-qubit gates before scheduling. 

* `context` : cirq.DecompositionContext, default is none which will result in the use of a context manager that tries to minimize the number of ancilla generated. At times, certain Qualtran Bloqs may have an error in their decomposition which results in ancilla not being properly cleared. If this happens, we recommend passing `context=cirq.DecompositionContext(cirq.SimpleQubitManager())` to resolve this issue, at the cost of an overzealous qubit allocation.

* `rotation_gate_precision`: default 5e-6. Rotations are decomposed to a given precision. This can be passed explicitly using the `rotation_gate_precision` keyword argument. If no precision argument is provided, the default value will be used. Note that higher-precision will generally increase the resource cost.

For further details on scheduler capabilities, see `Algorithms_and_Infrastructure/scheduling_example.ipynb`.

### Clifford+T Decomposition
#### Overview
`pyLIQTR.gate_decomp.cirq_transforms.clifford_plus_t_direct_transform` is a utility to convert a circuit that has been fully decomposed (into 'simple'/'canonical' gates as opposed to complex operators or bloqs) into circuits consisting only of Clifford+T operations by performing exact decomposition of single qubit unitary matrices with entries from the ring D[ω] over the Clifford+T gateset. Makes use of the Matsumoto-Amano normal form for single qubit Clifford+T operators described in the following references:
[1] - arXiv:0806.3834
[2] - arXiv:1312.6584
[3] - https://www.cs.umd.edu/~amchilds/qa/qa.pdf Section 3

#### Arguments
The arguments for this function are:
```
    circuit: cirq.Circuit
        The circuit to be decomposed. Currently, the circuit can only use the following
        gates:
        - Multi-qubit gates. CCX and CCZ gates are decomposed into Clifford+T exactly.
        The rest are ignored by the decomposition function.
        - Single qubit gates from the list `pyLIQTR.gate_decomp.cirq_transforms.clifford_plus_T_gates`
        - Rx, Ry, and Rz gates, which are decomposed into clifford+T approximations.
    gate_precision: Union[int, float, None] (default = 10)
        Used to determine the maximum approximation error for each gate decomposition.
        If given an int, the decomposition returned by the function will be within a
        distance of 10^{-precision} of the given rotation in the operator norm, whereas
        if given a float the max error in the operator norm will simply be
        gate_precision.
    circuit_precision: Union[int, float, None] (default = None)
        If not None, the approximation error for each gate will be bounded by either
        `circuit_precision` divided by the number of rotation gates (if given a float),
        or 10^{-circuit_precision} (if given an int). Note that this requires doing an
        initial pass over the circuit to count the number of rotation gates, unless
        `num_rotation_gates` is specified by the caller.
    num_rotation_gates: Union[int, None]: (default = None)
        The number of rotation gates in the circuit. If known, pass this in as it will cut down
        on the amount of execution time, particularly if the number of gates is very large.
    use_rotation_decomp_gates: bool (default = False)
        If true, replace `cirq.r<x/y/z>` gates with
        `pyLIQTR.gate_decomp.r<x/y/z>_decomp` gates, which can be expanded later into a
        Clifford+T circuit. If false, return a fully expanded Clifford+T circuit.
    use_random_decomp: bool (default = False)
        If true, rotations will be "decomposed" into randomly selected Clifford+T
        sequences, with the length of the sequence depending on the given precision.
        For a precision of ε, the T count is sampled from a normal distribution with
        a mean of 3.02 * log2(1/ε) + 0.77 and a standard deviation of 2.06 (these
        numbers were obtained from decomposing 1000 random angles at varying
        precisions).
    warn_if_not_decomposed: bool (default = False)
        If true, raises a warning when the circuit includes a non-clifford+T gate that
        is not an X, Y, or Z rotation. Such gates are not decomposed.
```

#### Notes about speed and usage
The default arguments will result in a SLOW but TRUE decomposition of the input circuit. If one is interested in just understanding resource utilization, then it is recommended to set the `use_random_decomp` flag to be True, which will generate gates that reflect a realistic resource-count but will not implement the desired rotation. It can also be recommended to set the `use_rotation_decomp_gates` flag to be True in order to generate a smaller circuit, though one will need to interrogate it.

Also note the various precision arguments. If the gate rotation is smaller than the specified the precision, they will not be decomposed and are instead replaced as an Identity gate.

#### openqasm Note
Note, that in order to output a clifford+T circuit into openqasm form, it is not recommended to call this function ahead of time. See the arguments for `pyLIQTR.utils.printing.openqasm` and the relevant `LIQTR Like` for printing. 
