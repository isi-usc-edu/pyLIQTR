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

import cirq
from collections import defaultdict
from pyLIQTR.gate_decomp.rotation_gates import decomp_mixin
from pyLIQTR.utils.circuit_decomposition import decompose_op, circuit_decompose_multi
from typing import FrozenSet, Iterable, List, Tuple, Union
from cirq.type_workarounds import NotImplementedType
from pyLIQTR.gate_decomp.cirq_transforms import clifford_plus_t_direct_transform

from warnings import warn


from pyLIQTR.circuits.operators.AddMod import AddMod as pyLAM
import qualtran

import pyLIQTR.utils.global_ancilla_manager as gam

def test_for_bad_gate_op(gate_op:cirq.GateOperation) -> str:
    gate_op_str = str(gate_op)
    if hasattr(gate_op, '_gate'):
        gate_op_str = str(gate_op._gate)

    gates2check = ["reset","cirq.Measure","cirq.MeasurementGate"]
    gates2check += ["X","Y","Z","S","H","T"]
    gates2check += ["X**-1.0","Y**-1.0","Z**-1.0","S**-1.0","H**-1.0","T**-1.0"]
    gates2check += ["X**-1","Y**-1","Z**-1","S**-1","H**-1","T**-1"]
    if (gate_op_str in gates2check) or \
        (gate_op_str.startswith(('reset','Rx','Ry','Rz',\
            'CX','CZ','CCZ','CCX',\
            'TOFFOLI', 'CCXi', 'ccxi', 'cirq.Measure', 'CNOT'))):
            if ("**-1.0" in gate_op_str or "**-1" in gate_op_str) and \
                (("TOFFOLI" in gate_op_str) \
                    or ("CNOT" in gate_op_str) \
                    or ("CZ" in gate_op_str)\
                    or ("CX" in gate_op_str)
                    or (gate_op_str.startswith("S**-1"))
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
        context = cirq.DecompositionContext(gam.gam)
    
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
    qubit_map_str2key = {str(k):k for k in qubit_map.keys()}


    yield str_out
    for moment in circuit:
        for op in moment:
            try:
                try:
                    if "qualtran.bloqs.basic_gates" in op._gate.__module__:
                        op = op._gate.cirq_gate.on(*op.qubits)
                        #raise ValueError('do something here')
                except:
                    pass
                if isinstance(op._gate,cirq.GlobalPhaseGate):
                    continue
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
                elif isinstance(op,cirq.ClassicallyControlledOperation) or isinstance(op,cirq.ops.controlled_operation.ControlledOperation):
                    #This is an op that is classically controlled
                    if len(op.classical_controls)>1:
                        print("Cant handle more than one classical control")
                        raise(e)
                    
                    newstr = ""
                    for cc in op.classical_controls:
                        tmp = str(cc)
                        thistmpqb = qubit_map_str2key[str(tmp)]
                        newstr+="if (b[{}]==1) ".format(\
                            qubit_map[thistmpqb])
                    newstr+=cirq.qasm(op._sub_operation,args=qasm_args)
                    yield newstr.strip()
                elif (str(op).startswith("AddConstantMod")):
                    #this is an op that needs to be casted to 
                    qbs = op.qubits
                    top = pyLAM(bitsize=op.gate.bitsize, add_val=op.gate.add_val, \
                               mod=op.gate.mod,cvs=op.gate.cvs).\
                            on(*qbs)
                    yield from openqasm(cirq.Circuit(top),skip_header=True,context=context,rotation_allowed=rotation_allowed)
                else:
                    print('Issue with either decomposing or generating qasm with {} ()'.format(str(op),type(op)))
                    raise e
    


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