"""
Copyright (c) 2024 Massachusetts Institute of Technology 
SPDX-License-Identifier: BSD-2-Clause
"""


import cirq
from collections import defaultdict
from types import GeneratorType
from pyLIQTR.gate_decomp.rotation_gates import decomp_mixin
from pyLIQTR.utils.circuit_decomposition import generator_decompose, keep, test_for_bad_gate_op
from typing import FrozenSet, Iterable, List, Tuple, Union
from cirq.type_workarounds import NotImplementedType
from pyLIQTR.gate_decomp.cirq_transforms import clifford_plus_t_direct_transform, determine_gate_precision, _perop_clifford_plus_t_direct_transform

from warnings import warn
import copy

from warnings import warn


from pyLIQTR.circuits.operators.AddMod import AddMod as pyLAM
import qualtran
import numpy as np

import pyLIQTR.utils.global_ancilla_manager as gam



def _build_qasm_qubit_map(circuit_in) -> Tuple[cirq.QasmArgs, dict, str]:
    """
    Internal class
    """
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
    """
    Internal class
    """
    def __init__(self, circuit_in, header=""):
        self.measurement_count = 0
        if isinstance(circuit_in, cirq.Circuit):
            qubits_ = list(circuit_in.all_qubits())
        elif isinstance(circuit_in,GeneratorType):
            #if its a generator, we need to go through each thing and 
            #get the qubits involved :\
            qubits_  = set()
            for op in circuit_in:
                qubits_.update(set(op.qubits))
                if _check_for_measurement(op):
                    self.measurement_count += 1
            qubits_ = list(qubits_)
        qubits_.sort()
    
        str_out = f'// Qubits: ['
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
        str_out += f'qreg q[{len(qubits_)}];\n'
        str_out += f'creg c[{self.measurement_count}];\n\n'

        qubit_map = (dict(qubit_map))
        qasm_args = cirq.QasmArgs(qubit_id_map=qubit_map)
        self.qasm_args = qasm_args
        self.qubit_map = qubit_map
        self.str_out = header+str_out
        self.measurement_result = {}
        self.measurement_counter = 0

    def update_info(self, circuit):
        if isinstance(circuit, cirq.Circuit):
            qubits = list(circuit.all_qubits())
        elif isinstance(circuit,GeneratorType):
            #if its a generator, we need to go through each thing and 
            #get the qubits involved :\
            qubits  = set()
            for op in circuit:
                qubits.update(set(op.qubits))
                if _check_for_measurement(op):
                    self.measurement_count += 1
            qubits = list(qubits)
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
        mbit = "c[{}]".format(self.measurement_counter)
        self.measurement_counter += 1
        
        if single_op:
            result = bit_name
            doClear = True
        if (result not in self.measurement_result) or doClear:
            self.measurement_result[result] = [mbit]
        else:
            self.measurement_result[result].append(mbit)
        return mbit

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
    
def _check_for_measurement(op : cirq.Operation):
    return str(op).startswith("cirq.MeasurementGate")
    
            
myQASMInfo = None
myOp = None
def openqasm(circuit: cirq.AbstractCircuit,
             gate_precision: Union[int, float, None] = 10,
             circuit_precision: Union[int, float, None] = None,
             skip_header : bool = False,
             context : cirq.DecompositionContext = None,
             rotation_allowed: bool =False,
             on_stuck_raise: Union[Exception, None] = None,
             use_random_decomp: bool = True,
             max_decomposition_passes:Union[int,None]=None) -> str:
    """
    This function is an extension and wrapper around Cirq's existing decomposition and openqasm printing workflow, and the primary intent is to address short-comings and usability that accompanies using the default cirq openqasm workflow.

    This function is a generator, and yields a string which corresponds to QASM output.

    NOTE: For the most part, this function requires that any operation has a '''_qasm_''' defined, otherwise you will receive errors on output (with some small execptions).

    Args:
        circuit: A cirq circuit that you wish to output as QASM
        gate_precision: Union[int, float, None]
            Used to determine the maximum approximation error for each gate decomposition.
            If given an int, the decomposition returned by the function will be within a
            distance of 10^{-precision} of the given rotation in the operator norm, whereas
            if given a float the max error in the operator norm will simply be gate_precision.
        circuit_precision: Union[int, float, None]
            If not None, the approximation error for each gate will be bounded by either
            `circuit_precision` divided by the number of rotation gates (if given a float),
            or 10^{-circuit_precision} (if given an int).
        skip_header: bool
            Whether or not to output the QASM header
        rotation_allowed: bool
            Whether or not to allow rotations (Rx/Ry/Rz) to remain in the circuit.
            If set to false, we make use a of a Clifford+T representation outlined in https://arxiv.org/pdf/0806.3834.pdf 
            (and further discussed in https://arxiv.org/pdf/1312.6584.pdf and http://www.cs.umd.edu/~amchilds/qa/qa.pdf, 
            as potentially easier to digest forms).

            When rotation_allowed=False, we introduce the "rx_d/ry_d/rz_d gate", followed by a line of the form:

            // (leading_T, gate_sequence, sequence_length, clifford_part)
            leading_T : A bool indicating whether or not there is a T gate at the start of the sequence
            gate_sequence: An int whose binary representation encodes a sequence of H, S, and T gates
            sequence_length: The number of T gates in the sequence (note if leading_T is true, the total number of 
                gates is sequence_length+1)
            clifford_part: A List[str] of clifford gates that come at the end of the H,S,T sequence.
            Note that when rotation_allowed=False, you may also want to specify a level of precision by 
                passing gate_precision or circuit_precision to pyLIQTR.utils.printing.openqasm
        on_stuck_raise:  Union[Exception, None]
            See cirq.decompose documentation for this
        use_random_decomp: bool
            Used if rotation_allowed==False. By default this is set true in the interest of time, and assumes
            a heuristic for how rotations are decomposed into Clifford+T. NOTE: Unless you need the circuit to be
            to implement the correct unitary, you should leave this False for the interest of speed (resource counts,
            parallelism, etc) will all be of the right order

            The heuristic for determining the length of the sequence depends on the given precision.
            For a precision of ε, the T count is sampled from a normal distribution with
            a mean of 3.02 * log2(1/ε) + 0.77 and a standard deviation of 2.06 (these
            numbers were obtained from decomposing 1000 random angles at varying
            precisions).
        max_decomposition_passes: Union[int,None]
            This sets how many decomposition passes to do before outputting to QASM (None implies until we stop) 

        

    """
    #This function always will decompose to 1-3Q gates
    global myQASMInfo
    #Provide custom context if one is not provided
    if context is None:
        context = cirq.DecompositionContext(gam.gam)
    #Determine gate precision if necessary. NOTE: Num_rotations cannot be found without looking through
    #entire circuit
    #This decomposes+does clifford transform
    def convertMe(circuit):
        #A deep copy MUST be done here in order to keep qubit naming conventions consistent.
        tcontext = copy.deepcopy(context)
        for op in generator_decompose(circuit, keep=keep, on_stuck_raise = on_stuck_raise, context=tcontext,max_decomposition_passes=max_decomposition_passes):
            if not rotation_allowed:
                for cptdt_op in _perop_clifford_plus_t_direct_transform(op, use_rotation_decomp_gates=True,\
                                                    use_random_decomp=use_random_decomp,\
                                                    gate_precision=gate_precision):
                    yield cptdt_op
            else:
                yield op
    gate_precision = determine_gate_precision(convertMe(circuit), gate_precision, circuit_precision, None)
    

    str_out = ""
    if (not skip_header):
        str_out =  '// Generated from Cirq, Openfermion, and MIT LL\n\n'
        str_out += 'OPENQASM 2.0;\n'
        str_out += 'include \"qelib1.inc\";\n\n'
        
        
        myQASMInfo = QASMInfo(convertMe(circuit),str_out)
    else:
        myQASMInfo.update_info(convertMe(circuit))
    qubit_map_str2key = {str(k):k for k in myQASMInfo.qubit_map.keys()}
    
    if not skip_header:
        yield myQASMInfo.str_out
    for op in convertMe(circuit):
        try:
            try:
                if "qualtran.bloqs.basic_gates" in op._gate.__module__:
                    op = op._gate.cirq_gate.on(*op.qubits)
                elif str(op._gate).startswith(('Ry_d(','Rx_d(','Rz_d(')):
                    op = op._gate.get_gate().on(*op.qubits)
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
            if _check_for_measurement(op):
                for q in op.qubits:
                    meas_name = op._measurement_key_name_()
                    if meas_name == "":
                        meas_name = 'b'
                    elif meas_name == str(q):
                        meas_name = 'b'
                    meas_result = myQASMInfo.add_measurement_result(meas_name,str(myQASMInfo.qubit_map[q]),\
                                                                    single_op=(len(op.qubits)==1))

                    yield 'measure {} -> {}'.format(myQASMInfo.qubit_map[q],meas_result).strip()
            elif isinstance(op,cirq.ClassicallyControlledOperation) or isinstance(op,cirq.ops.controlled_operation.ControlledOperation):
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
                        meas_result = myQASMInfo.get_measurement_result(tmp)

                        checkVal = ','.join(len(meas_result)*['1'])
                        checkResult = ','.join(meas_result)
                        newstr+="if ({}=={}) ".format(checkResult,checkVal)
                newstr+=cirq.qasm(op._sub_operation,args=myQASMInfo.qasm_args)
                yield newstr.strip()
            elif (str(op).startswith("AddConstantMod") or str(op).startswith('ModAddK')):
                #this is an op that needs to be casted to 
                qbs = op.qubits
                top = pyLAM(bitsize=op.gate.bitsize, add_val=op.gate.add_val, \
                           mod=op.gate.mod,cvs=op.gate.cvs).\
                        on(*qbs)
                yield from openqasm(cirq.Circuit(top),skip_header=True,context=context,rotation_allowed=rotation_allowed)
            elif "qualtran.cirq_interop._bloq_to_cirq" in op._gate.__module__:
                skip_gates = [
                    'bloq.Allocate','bloq.Free'
                ]
                if str(op.gate) in skip_gates:
                    continue

                bloq2cirq = {
                    'bloq.Toffoli': cirq.CCX,
                    'bloq.TwoBitCSwap': cirq.CSWAP,
                }
                if str(op.gate) not in bloq2cirq:
                    raise NotImplementedError("Uncpatured bloq2cirq conversion {}".format(str(op.gate)))
                else:
                    yield cirq.qasm(bloq2cirq[str(op._gate)].on(*op.qubits),args=myQASMInfo.qasm_args).strip()
            
            else:
                for k,v in qubit_map_str2key.items():
                    print((k,v))
                print(myQASMInfo.qasm_args.qubit_id_map)
                print('Issue with either decomposing or generating qasm with {} ({})'.format(str(op),type(op.gate)))
                raise e
