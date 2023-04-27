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

def test_for_bad_gate_op(gate_op):
    if (str(gate_op).startswith(('reset','Rx','Ry','Rz',\
            'X','Y','Z','S', 'H',\
            'CX','CZ','CCZ','CCX',\
            'TOFFOLI', 'CCXi', 'ccxi', 'cirq.Measure', 'CNOT'))):
            if "**-1.0" in str(gate_op) and \
                (("TOFFOLI" in str(gate_op)) \
                    or ("CNOT" in str(gate_op))):
                return cirq.inverse(gate_op)
            else:
                return gate_op


def _build_qasm_qubit_map(circuit_in):
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

def to_openqasm(circuit_in):
    str_out =  '// Generated from Cirq, Openfermion, and MIT LL\n\n'
    str_out += 'OPENQASM 2.0;\n'
    str_out += 'include \"qelib1.inc\";\n\n'
    
    
    qasm_args, qubit_map, tmp = _build_qasm_qubit_map(circuit_in)
    str_out += tmp
    try:
        yield str_out
        for moment in circuit_in:
            for gate_op in moment:
                try:
                    out = cirq.qasm(gate_op, args=qasm_args)  
                    for line in out.split('\n'):
                        yield ("{}\n".format(line))
                except GeneratorExit:
                    return
                except:
                    try: 
                        out = cirq.qasm(test_for_bad_gate_op(gate_op), args=qasm_args)  
                        for line in out.split('\n'):
                            yield ("{}\n".format(line))
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
                            yield ("{}\n".format(line))
    except GeneratorExit:
        return