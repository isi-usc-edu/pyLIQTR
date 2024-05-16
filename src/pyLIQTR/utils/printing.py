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
    
class QASMInfo:
    def __init__(self, circuit_in, header=""):

        str_out = f'// Qubits: ['
        qubits_ = list(circuit_in.all_qubits())
        qubits_.sort()
        qubit_map = defaultdict(None)
        self.maxq = 0
        for idx, qubit_ in enumerate(qubits_):
            
            qubit_map[(qubit_)] = f'q[{idx}]'
            
            if idx==0:
                str_out += f'{str(qubit_)}'
            else:
                str_out += f', {str(qubit_)}'
        self.maxq = len(qubits_)
        str_out += f']\n'
        str_out += f'qreg q[{len(qubits_)}];\n\n'

        qubit_map = (dict(qubit_map))
        qasm_args = cirq.QasmArgs(qubit_id_map=qubit_map)
        self.qasm_args = qasm_args
        self.qubit_map = qubit_map
        self.str_out = header+str_out
        self.measurement_result = {}

    def update_info(self, circuit):
        qubits = list(circuit.all_qubits())
        qubits.sort()
        didAdd = False
        for idx, qubit_ in enumerate(qubits):
            if qubit_ not in self.qubit_map:
                didAdd = True
                self.qubit_map[qubit_] = f'q[{idx+self.maxq}]'
        
        if didAdd:
            self.qasm_args = cirq.QasmArgs(qubit_id_map=self.qubit_map)
        #no need to modify str_out

    def add_measurement_result(self,result,bits,doClear=False,single_op=False):
        bit_name = "{}[{}]".format(result,bits)
        if single_op:
            result = bit_name
            doClear = True
        if (result not in self.measurement_result) or doClear:
            self.measurement_result[result] = [bit_name]
        else:
            self.measurement_result[result].append(bit_name)
        return bit_name

    def get_measurement_result(self,result,doClear=False):
        if result not in self.measurement_result:
            raise ValueError("Did not find {} in {}".format(result,self.measurement_result))
        
        return self.measurement_result[result]

    def sanitize_meas_name(self,measname):
        keys_as_strings = [str(k) for k in self.qubit_map.keys()]
        if (measname not in self.measurement_result \
                and (measname in self.qubit_map.keys() or measname in keys_as_strings)) \
            or measname == "":
            if isinstance(measname,str):
                qn = {str(a):b for a,b in self.qubit_map.items()}
                return "b[{}]".format(qn[measname])
            else:
                return "b[{}]".format(self.qubit_map[measname])
        return measname
        
    
            
myQASMInfo = None
myOp = None
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
    global myQASMInfo
    if context is None:
        context = cirq.DecompositionContext(gam.gam)
    
    circuit = cirq.Circuit(cirq.decompose(circuit,\
                                            keep=keep,\
                                            on_stuck_raise=on_stuck_raise,\
                                            context = context))
    #print(circuit)
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
        
        
        #qasm_args, qubit_map, tmp = _build_qasm_qubit_map(circuit)
        myQASMInfo = QASMInfo(circuit,str_out)
    else:
        #qasm_args, qubit_map, tmp = _build_qasm_qubit_map(circuit)
        myQASMInfo.update_info(circuit)
    qubit_map_str2key = {str(k):k for k in myQASMInfo.qubit_map.keys()}


    yield myQASMInfo.str_out
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
                    yield cirq.qasm(op,args=myQASMInfo.qasm_args).strip()
                else:
                    yield cirq.qasm(test_op, args=myQASMInfo.qasm_args).strip()
            except Exception as e:
                if str(op).startswith("cirq.MeasurementGate"):
                    
                    for q in op.qubits:
                        meas_name = op._measurement_key_name_()
                        if meas_name == "":
                            meas_name = 'b'
                        elif meas_name == str(q):
                            meas_name = 'b'
                        meas_result = myQASMInfo.add_measurement_result(meas_name,str(myQASMInfo.qubit_map[q]),\
                                                                        single_op=(len(op.qubits)==1))

                        yield 'measure {} -> {}'.format(myQASMInfo.qubit_map[q],meas_result).strip()
                    #yield "measure {} -> b[{}]".format(myQASMInfo.qubit_map[op.qubits[0]],\
                                                     #myQASMInfo.qubit_map[op.qubits[0]]).strip()
                elif isinstance(op,cirq.ClassicallyControlledOperation) or isinstance(op,cirq.ops.controlled_operation.ControlledOperation):
                    #This is an op that is classically controlled
                    if len(op.classical_controls)>1:
                        print("Cant handle more than one classical control")
                        raise(e)
                    
                    newstr = ""
                    for cc in op.classical_controls:
                        if type(cc) != cirq.value.condition.KeyCondition:
                            #This works with the type that we implement
                            #We require a "get_condition_format_string" be provided...
                            keys = cc.keys
                            if len(keys)!=1:
                                raise NotImplementedError("Need to update to support additional keys")
                            sanitized_keys = []
                            for key in keys:
                                key = myQASMInfo.sanitize_meas_name(key)
                                meas_result = myQASMInfo.get_measurement_result(key)
                                meas_result = ",".join(meas_result)
                            newstr+="if ({}) ".format(cc.get_condition_format_string().format(meas_result))
                        else:
                            #This works with default cirq.value.condition.KeyCondition
                            tmp = myQASMInfo.sanitize_meas_name(str(cc))
                            #thistmpqb = qubit_map_str2key[str(tmp)]
                            meas_result = myQASMInfo.get_measurement_result(tmp)
                            
                            checkVal = ','.join(len(meas_result)*['1'])
                            checkResult = ','.join(meas_result)
                            newstr+="if ({}=={}) ".format(checkResult,checkVal)
                    newstr+=cirq.qasm(op._sub_operation,args=myQASMInfo.qasm_args)
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