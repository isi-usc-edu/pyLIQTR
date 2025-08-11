"""
Copyright (c) 2024 Massachusetts Institute of Technology 
SPDX-License-Identifier: BSD-2-Clause
"""
import sys
import cirq
import qualtran
from cirq.ops.gate_operation import GateOperation

import cirq.protocols.decompose_protocol as cpdp
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    overload,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

def _controlled_decompose_dfs(current_pass:int, max_decomposition_passes: int,\
                              item: Any, args: cpdp._DecomposeArgs) -> Iterator['cirq.Operation']:
    """
    A modified version of cirq's cirq.protocol.decompose_protocol._decompose_dfs that limits the 
    number of times it will recurse.
    """
    from cirq.circuits import CircuitOperation, FrozenCircuit
    from cirq.protocols.decompose_protocol import ops

    if isinstance(item, ops.Operation):
        item_untagged = item.untagged
        if args.preserve_structure and isinstance(item_untagged, CircuitOperation):
            new_fc = FrozenCircuit(_decompose_dfs(item_untagged.circuit, args))
            yield item_untagged.replace(circuit=new_fc).with_tags(*item.tags)
            return
        if args.keep is not None and args.keep(item):
            yield item
            return

    decomposed = cpdp._try_op_decomposer(item, args.intercepting_decomposer, context=args.context)

    if decomposed is NotImplemented or decomposed is None:
        decomposed = cpdp.decompose_once(item, default=None, flatten=False, context=args.context)

    if decomposed is NotImplemented or decomposed is None:
        decomposed = cpdp._try_op_decomposer(item, args.fallback_decomposer, context=args.context)

    if decomposed is NotImplemented or decomposed is None:
        if not isinstance(item, ops.Operation) and isinstance(item, Iterable):
            decomposed = item

    if decomposed is NotImplemented or decomposed is None:
        if args.keep is not None and args.on_stuck_raise is not None:
            if isinstance(args.on_stuck_raise, Exception):
                raise args.on_stuck_raise
            elif callable(args.on_stuck_raise):
                error = args.on_stuck_raise(item)
                if error is not None:
                    raise error
        yield item
    else:
        if (max_decomposition_passes is None) or (current_pass < max_decomposition_passes):
            current_pass+=1
            do_decomp = True
        else:
            do_decomp = False
        for val in ops.flatten_to_ops(decomposed):
            if do_decomp:
                for op in _controlled_decompose_dfs(current_pass, max_decomposition_passes, val, args):
                    if isinstance(op,list):
                        for oop in op:
                            if isinstance(oop.gate,cirq.MatrixGate):
                                yield cirq.decompose(oop, context=args.context)
                            else:
                                yield oop
                    else:
                        if isinstance(op.gate,cirq.MatrixGate):
                            yield cirq.decompose(op, context=args.context)
                        else:
                            yield op
            else:
                if isinstance(val,list):
                    for vval in val:
                        if isinstance(vval.gate,cirq.MatrixGate):
                            yield cirq.decompose(vval, context=args.context)
                        else:
                            yield vval
                else:
                    if isinstance(val.gate,cirq.MatrixGate):
                        yield cirq.decompose(val, context=args.context)
                    else:
                        yield val
            
def generator_decompose(
    val: Any,
    *,
    intercepting_decomposer: Optional[cpdp.OpDecomposer] = None,
    fallback_decomposer: Optional[cpdp.OpDecomposer] = None,
    keep: Optional[Callable[['cirq.Operation'], bool]] = None,
    on_stuck_raise: Union[
        None, Exception, Callable[['cirq.Operation'], Optional[Exception]]
    ] = cpdp._value_error_describing_bad_operation,
    preserve_structure: bool = False,
    context: Optional[cpdp.DecompositionContext] = None,
    max_decomposition_passes : Union[None,int] = None
):
    """
    NOTE: This is a modified version of cirq's decompose function, which has been modified to return a generator and to stop decomposing after a number of passes
    
    Recursively decomposes a value into `cirq.Operation`s meeting a criteria.

    Args:
    val: The value to decompose into operations.
    intercepting_decomposer: An optional method that is called before the
        default decomposer (the value's `_decompose_` method). If
        `intercepting_decomposer` is specified and returns a result that
        isn't `NotImplemented` or `None`, that result is used. Otherwise the
        decomposition falls back to the default decomposer.

        Note that `val` will be passed into `intercepting_decomposer`, even
        if `val` isn't a `cirq.Operation`.
    fallback_decomposer: An optional decomposition that used after the
        `intercepting_decomposer` and the default decomposer (the value's
        `_decompose_` method) both fail.
    keep: A predicate that determines if the initial operation or
        intermediate decomposed operations should be kept or else need to be
        decomposed further. If `keep` isn't specified, it defaults to "value
        can't be decomposed anymore".
    on_stuck_raise: If there is an operation that can't be decomposed and
        also can't be kept, `on_stuck_raise` is used to determine what error
        to raise. `on_stuck_raise` can either directly be an `Exception`, or
        a method that takes the problematic operation and returns an
        `Exception`. If `on_stuck_raise` is set to `None` or a method that
        returns `None`, non-decomposable operations are simply silently
        kept. `on_stuck_raise` defaults to a `ValueError` describing the
        unwanted non-decomposable operation.
    preserve_structure: Prevents subcircuits (i.e. `CircuitOperation`s)
        from being decomposed, but decomposes their contents. If this is
        True, `intercepting_decomposer` cannot be specified.
    context: Decomposition context specifying common configurable options for
        controlling the behavior of decompose.
    max_decomposition_passes: If provided None (which is the default), then
        the generator_decompose acts equivalently to cirq.decompose. If provided
        a int>=0, then this decomposition will only decompose that number of 
        passes

    Returns:
    A list of operations that the given value was decomposed into. If
    `on_stuck_raise` isn't set to None, all operations in the list will
    satisfy the predicate specified by `keep`.

    Raises:
    TypeError:
        `val` isn't a `cirq.Operation` and can't be decomposed even once.
        (So it's not possible to return a list of operations.)

    ValueError:
        Default type of error raised if there's an non-decomposable
        operation that doesn't satisfy the given `keep` predicate.

    TError:
        Custom type of error raised if there's an non-decomposable operation
        that doesn't satisfy the given `keep` predicate.
    """

    if (
            on_stuck_raise is not cpdp._value_error_describing_bad_operation
            and on_stuck_raise is not None
            and keep is None
        ):
        raise ValueError(
                "Must specify 'keep' if specifying 'on_stuck_raise', because it's "
                "not possible to get stuck if you don't have a criteria on what's "
                "acceptable to keep."
            )

    reclim = sys.getrecursionlimit() 
    if context is None:
        context = cpdp.DecompositionContext(cpdp.ops.SimpleQubitManager(prefix='_decompose_protocol'))
    args = cpdp._DecomposeArgs(
            context=context,
            intercepting_decomposer=intercepting_decomposer,
            fallback_decomposer=fallback_decomposer,
            keep=keep,
            on_stuck_raise=on_stuck_raise,
            preserve_structure=preserve_structure,
        )
    sys.setrecursionlimit(min([reclim*100,100000]))
    for op in _controlled_decompose_dfs(0,max_decomposition_passes, val, args):
        if isinstance(op.gate,qualtran.cirq_interop._bloq_to_cirq.BloqAsCirqGate):
            if "qualtran.bloqs.basic_gates" in op._gate.__module__:
                yield op._gate.cirq_gate.on(*op.qubits)
            elif "qualtran.cirq_interop._bloq_to_cirq" in op._gate.__module__:
                gates = ['bloq.CNOT','bloq.XGate','bloq.YGate','bloq.ZGate','bloq.HGate','bloq.H','bloq.SGate','bloq.TGate','bloq.T','bloq.T†']
                cirq_gates = [cirq.CNOT,cirq.X,cirq.Y,cirq.Z,cirq.H,cirq.H,cirq.S,cirq.T,cirq.T,cirq.inverse(cirq.T)]
                gate2cirq = {g:cg for g,cg in zip(gates,cirq_gates)}
                if (str(op._gate) not in gate2cirq):
                    #Annoying, there are some 'non basic' gates mixed in with these -_-
                    if 'bloq.MultiAnd' in str(op._gate) or "bloq.Add" in str(op._gate) or 'bloq.TwoBitCSwap':
                        yield op
                    else:
                        raise NotImplementedError(str(op._gate))
                else:
                    yield gate2cirq[str(op._gate)].on(*op.qubits)
            else:
                raise NotImplementedError('Uncaught/bad qualtran -> cirq conversion')
        elif isinstance(op.gate,qualtran.cirq_interop._cirq_to_bloq.CirqGateAsBloq):
            yield op.gate.gate.on(*op.qubits)
        else:
            yield op
    sys.setrecursionlimit(reclim)


def test_for_bad_gate_op(gate_op:cirq.GateOperation) -> str:
    gate_op_str = str(gate_op)
    if hasattr(gate_op, '_gate'):
        gate_op_str = str(gate_op._gate)

    gates2check = ["reset","cirq.Measure","cirq.MeasurementGate"]
    gates2check += ["X","Y","Z","S","H","T"]
    gates2check += ["X**-1.0","Y**-1.0","Z**-1.0","H**-1.0"]
    gates2check += ["X**-1","Y**-1","Z**-1","H**-1"]
    if (gate_op_str in gates2check) or \
        (gate_op_str.startswith(('reset','Rx','Ry','Rz',\
            'CX','CZ','CCZ','CCX',\
            'TOFFOLI', 'CCXi', 'ccxi', 'cirq.Measure', 'CNOT'))):
            if ("**-1.0" in gate_op_str or "**-1" in gate_op_str) and \
                (("TOFFOLI" in gate_op_str) \
                    or ("CNOT" in gate_op_str) \
                    or ("CZ" in gate_op_str)\
                    or ("CX" in gate_op_str)
                ):
                return cirq.inverse(gate_op)
            else:
                return gate_op
            
def intercept_decompose(gate_op):
    top = test_for_bad_gate_op(gate_op)
    if  top is None:
        return None
    else:
        return [top]
    
def keep(gate_op):
    return test_for_bad_gate_op(gate_op) is not None


def circuit_decompose_multi(circuit,N, context=None):
    if context is None:
        import pyLIQTR.utils.global_ancilla_manager as gam
        context = cirq.DecompositionContext(gam.gam)
    circuit = cirq.align_left(cirq.Circuit(generator_decompose(circuit,\
                                               keep=keep, \
                                                on_stuck_raise = None, \
                                                context=context,\
                                                max_decomposition_passes=N)))
    return circuit


def decompose_once(circuit, debug=False, context=None):
    if context is None:
        import pyLIQTR.utils.global_ancilla_manager as gam
        context = cirq.DecompositionContext(gam.gam)
    circuit = cirq.align_left(cirq.Circuit(generator_decompose(circuit,\
                                               keep=keep, \
                                                on_stuck_raise = None, \
                                                context=context,\
                                                max_decomposition_passes=1)))
    return circuit

def decompose_op(op,debug=False, context=None):
    if context is None:
        import pyLIQTR.utils.global_ancilla_manager as gam
        context = cirq.DecompositionContext(gam.gam)
    gates = [op for op in generator_decompose(cirq.Circuit(op),\
                                               keep=keep, \
                                                on_stuck_raise = None, \
                                                context=context,\
                                                max_decomposition_passes=1)]
    return gates