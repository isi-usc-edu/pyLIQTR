"""
DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.

This material is based upon work supported by the Under Secretary of Defense for
Research and Engineering under Air Force Contract No. FA8702-15-D-0001. Any opinions,
findings, conclusions or recommendations expressed in this material are those of the
author(s) and do not necessarily reflect the views of the Under Secretary of Defense
for Research and Engineering.

© 2022 Massachusetts Institute of Technology.

The software/firmware is provided to you on an As-Is basis

Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS Part
252.227-7013 or 7014 (Feb 2014). Notwithstanding any copyright notice, U.S. Government
rights in this work are defined by DFARS 252.227-7013 or DFARS 252.227-7014 as detailed
above. Use of this work other than as specifically authorized by the U.S. Government
may violate any copyrights that exist in this work.
"""

from typing_extensions import deprecated
from warnings import warn
import cirq
from collections import defaultdict
from pyLIQTR.gate_decomp.rotation_gates import decomp_mixin
from pyLIQTR.utils.circuit_decomposition import decompose_op, circuit_decompose_multi
from typing import FrozenSet, Iterable, List, Tuple, Union
from cirq.type_workarounds import NotImplementedType
from pyLIQTR.gate_decomp.cirq_transforms import clifford_plus_t_direct_transform


from pyLIQTR.circuits.operators.AddMod import AddMod as pyLAM

def test_for_bad_gate_op(gate_op:cirq.GateOperation) -> str:
    if (str(gate_op) in ["X","Y","Z","S","H","T", \
                            "reset","cirq.Measure","cirq.MeasurementGate"]) or \
        (str(gate_op).startswith(('reset','Rx','Ry','Rz',\
            'CX','CZ','CCZ','CCX',\
            'TOFFOLI', 'CCXi', 'ccxi', 'cirq.Measure', 'CNOT'))):
            if "**-1.0" in str(gate_op) and \
                (("TOFFOLI" in str(gate_op)) \
                    or ("CNOT" in str(gate_op)) \
                    or ("CZ" in str(gate_op))\
                    or ("CX" in str(gate_op))
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
    
    

def _build_qasm_qubit_map(circuit_in) -> Tuple[cirq.QasmArgs, dict, str]:
    str_out = f'// Qubits: ['
    qubits_ = list(circuit_in.all_qubits())
    qubits_.sort()
    qubit_map = defaultdict(None)
    for idx, qubit_ in enumerate(qubits_):
        
        qubit_map[(qubit_)] = f'q[{idx}]'
        
        if idx==0:
            str_out += f'{str(qubit_)}'
        else:
            str_out += f', {str(qubit_)}'
    str_out += f']\n'
    str_out += f'qreg q[{len(qubits_)}];\n\n'

    qubit_map = (dict(qubit_map))
    qasm_args = cirq.QasmArgs(qubit_id_map=qubit_map)
    return qasm_args,qubit_map,str_out

def openqasm(circuit: cirq.AbstractCircuit,
             basis: str = 'rotations',
             gate_precision: Union[int, float, None] = 10,
             circuit_precision: Union[int, float, None] = None,
             skip_header : bool = False,
             context : cirq.DecompositionContext = None,
             rotation_allowed=False,
             on_stuck_raise: Union[Exception, None] = None,
             use_random_decomp: bool = True) -> str:
    #This function always will decompose to 1-3Q gates
    if context is None:
        gqm = cirq.GreedyQubitManager(prefix="_ancilla", maximize_reuse=True)
        context = cirq.DecompositionContext(gqm)
    
    circuit = cirq.Circuit(cirq.decompose(circuit,\
                                            keep=keep,\
                                            on_stuck_raise=on_stuck_raise,\
                                            context = context))
    
    if not rotation_allowed:
        if circuit_precision is not None:
            circuit = clifford_plus_t_direct_transform(circuit,
                                                    use_rotation_decomp_gates=True,
                                                    use_random_decomp=use_random_decomp,
                                                    circuit_precision=circuit_precision)
        else:
            circuit = clifford_plus_t_direct_transform(circuit,
                                                    use_rotation_decomp_gates=True,
                                                    use_random_decomp=use_random_decomp,
                                                    gate_precision=gate_precision)
    
    str_out = ""
    if not skip_header:
        str_out =  '// Generated from Cirq, Openfermion, and MIT LL\n\n'
        str_out += 'OPENQASM 2.0;\n'
        str_out += 'include \"qelib1.inc\";\n\n'
        
        
        qasm_args, qubit_map, tmp = _build_qasm_qubit_map(circuit)
        str_out += tmp
    else:
        qasm_args, qubit_map, tmp = _build_qasm_qubit_map(circuit)


    yield str_out
    for moment in circuit:
        for op in moment:
            try:
                test_op = test_for_bad_gate_op(op)
                if test_op is None:
                    #does not handle XPowGates properly if rotation_allowed=False...
                    yield cirq.qasm(op,args=qasm_args).strip()
                else:
                    yield cirq.qasm(test_op, args=qasm_args).strip()
            except Exception as e:
                if str(op).startswith("cirq.MeasurementGate"):
                    #This measurement result is being stored as a bit
                    if len(op.qubits) > 1:
                        print("Cant handle more than one measurement")
                        raise(e)
                    
                    yield "measure {} -> b[{}]".format(qubit_map[op.qubits[0]],\
                                                     qubit_map[op.qubits[0]]).strip()
                elif isinstance(op,cirq.ClassicallyControlledOperation):
                    #This is an op that is classically controlled
                    if len(op.classical_controls)>1:
                        print("Cant handle more than one classical control")
                        raise(e)
                    
                    newstr = ""
                    for cc in op.classical_controls:
                        tmp = str(cc)
                        tmp = tmp.replace(context.qubit_manager._prefix+"_","")
                        qnum = int(tmp)
                        prefix = context.qubit_manager._prefix

                        thistmpqb = cirq.NamedQubit(name="{}_{}".format(prefix,str(qnum)))
                        newstr+="if (b[{}]==1) ".format(\
                            qubit_map[thistmpqb])
                    
                    newstr+=cirq.qasm(op._sub_operation,args=qasm_args)
                    yield newstr.strip()
                elif (str(op).startswith("cirq_ft.AddMod")):
                    #this is an op that needs to be casted to 
                    qbs = op.qubits
                    top = pyLAM(bitsize=op.gate.bitsize, add_val=op.gate.add_val, \
                               mod=op.gate.mod,cv=op.gate.cv).\
                            on(*qbs)
                    yield from openqasm(cirq.Circuit(top),skip_header=True,context=context,rotation_allowed=rotation_allowed)
                else:
                    raise e
    

@deprecated("Use openqasm() instead.")
def to_openqasm(circuit_in,use_rotation_decomp_gates=False) -> str:
    warn('\n\n\n*********************************************************************************************\nThis function is deprecated -- Recommend switching to pyLIQTR.utils.printing.openqasm\n*********************************************************************************************\n\n\n', DeprecationWarning, 2)
    str_out =  '// Generated from Cirq, Openfermion, and MIT LL\n\n'
    str_out += 'OPENQASM 2.0;\n'
    str_out += 'include \"qelib1.inc\";\n\n'
    
    
    qasm_args, qubit_map, tmp = _build_qasm_qubit_map(circuit_in)
    str_out += tmp
    try:
        yield str_out
        for moment in circuit_in:
            for gate_op in moment:
                if not use_rotation_decomp_gates:
                    try:
                        if isinstance(gate_op._gate,decomp_mixin):
                            for gop in decompose_op(gate_op)[1]:
                                out = cirq.qasm(gop, args=qasm_args)
                                for line in out.split("\n"):
                                    if line.strip() == "":
                                        continue
                                    yield "{}\n".format(line.strip())
                                continue
                    except GeneratorExit:
                        return

                try:
                    out = cirq.qasm(gate_op, args=qasm_args)  
                    for line in out.split('\n'):
                        if line.strip() == "":
                            continue
                        yield ("{}\n".format(line.strip()))
                except GeneratorExit:
                    return
                except:
                    try: 
                        out = cirq.qasm(test_for_bad_gate_op(gate_op), args=qasm_args)  
                        for line in out.split('\n'):
                            if line.strip() == "":
                                continue
                            yield ("{}\n".format(line.strip()))
                    except GeneratorExit:
                        return
                    except:
                    
                        str_out          = f'\n// Gate: {(gate_op)}\n'
                        decomposed_gates = cirq.decompose(gate_op, preserve_structure=True)
                    
                        for d_gate_op in decomposed_gates:
                            try:
                                str_out += cirq.qasm(d_gate_op, args=qasm_args)
                            except GeneratorExit:
                                return
                            except:
                                str_out += cirq.qasm(test_for_bad_gate_op(d_gate_op), args=qasm_args)
                        for line in str_out.split('\n'):
                            if line.strip() == "":
                                continue
                            yield ("{}\n".format(line.strip()))
    except GeneratorExit:
        return