"""
DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.

This material is based upon work supported by the Under Secretary of Defense for
Research and Engineering under Air Force Contract No. FA8702-15-D-0001. Any opinions,
findings, conclusions or recommendations expressed in this material are those of the
author(s) and do not necessarily reflect the views of the Under Secretary of Defense
for Research and Engineering.

© 2023 Massachusetts Institute of Technology.

The software/firmware is provided to you on an As-Is basis

Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS Part
252.227-7013 or 7014 (Feb 2014). Notwithstanding any copyright notice, U.S. Government
rights in this work are defined by DFARS 252.227-7013 or DFARS 252.227-7014 as detailed
above. Use of this work other than as specifically authorized by the U.S. Government
may violate any copyrights that exist in this work.
"""
import cirq
from cirq.ops.gate_operation import GateOperation





def circuit_decompose_multi(circuit,N):

    if ( N==1 ):
        return(cirq.align_left(decompose_once(circuit)))
    else:
        return(cirq.align_left(decompose_once(circuit_decompose_multi(circuit,N-1))))



def decompose_once(circuit, debug=False):
    new_qubits = circuit.all_qubits()
    decomposed_gates = []
    for moment in circuit:
        for op in moment:
            if debug:
                print(f'>> circuit op = {op}')
            if (str(op) in ["X","Y","Z","S","H","T", \
                            "reset","cirq.Measure","cirq.MeasurementGate"]) or \
                (str(op).startswith(("X(","Y(","Z(","S(","H(","T(",\
                                    "X**","Y**","Z**","S**","H**","T**",
                                    'CX','CZ','CCZ','CCX',\
                                    "cirq.MeasurementGate",\
                                    "reset",\
                                    "CNOT(","CNOT**-1",
                                    "Rx","Ry","Rz",\
                                    'TOFFOLI', 'CCXi', 'ccxi'))):
                if "**-1.0" in str(op) and \
                    (("TOFFOLI" in str(op)) \
                     or ("CNOT" in str(op))):
                    decomposed_gates.append(cirq.inverse(op))
                else:
                    decomposed_gates.append(op)
                continue
            try:
                tmp_gates    = cirq.decompose_once(op)
                decomp_gates = []

                # Note: If this decomposes into a MatrixGate, 
                #  lets decompose it once more so its not 
                #  a MatrixGate
                for gate in tmp_gates:
                    if str(gate).startswith('[['):
                        decomp_gates.extend(cirq.decompose_once(gate))
                    else:
                        decomp_gates.append(gate)
                if debug:
                    print(f"\t>> ops = {decomp_gates}")
                decomposed_gates.append(decomp_gates)
            except Exception as e:
                print(op)
                raise(e)

    return cirq.Circuit(decomposed_gates, strategy=cirq.InsertStrategy.NEW)

def decompose_op(op,debug=False):
    decomposed_gates = []
    if debug:
        print(f'>> circuit op = {op}')
    if (str(op) in ["X","Y","Z","S","H","T", \
                    "reset","cirq.Measure","cirq.MeasurementGate"]) or \
        (str(op).startswith(("X(","Y(","Z(","S(","H(","T(",\
                            "X**-1","Y**-1","Z**-1","S**-1","H**-1","T**-1",
                            'CX','CZ','CCZ','CCX',\
                            "cirq.MeasurementGate",\
                            "reset",\
                            "CNOT(","CNOT**-1",
                            "Rx","Ry","Rz",\
                            'TOFFOLI', 'CCXi', 'ccxi'))):
        if "**-1.0" in str(op) and \
            (("TOFFOLI" in str(op)) \
                or ("CNOT" in str(op))):
            decomposed_gates.append(cirq.inverse(op))
        else:
            decomposed_gates.append(op)
    try:
        tmp_gates    = cirq.decompose_once(op)
        decomp_gates = []
        # Note: If this decomposes into a MatrixGate, 
        #  lets decompose it once more so its not 
        #  a MatrixGate
        for gate in tmp_gates:
            if str(gate).startswith('[['):
                decomp_gates.extend(cirq.decompose_once(gate))
            else:
                decomp_gates.append(gate)
        if debug:
            print(f"\t>> ops = {decomp_gates}")
        decomposed_gates.append(decomp_gates)
    except Exception as e:
        print(op)
        raise(e)
    
    return decomposed_gates
