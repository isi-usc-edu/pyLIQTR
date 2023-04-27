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

from typing import List, Tuple
import cirq
import copy
import numpy as np

def count_qubits(circuit):
    
    # Get qubit info
    ttl_qubits = (len(circuit.all_qubits()))

    anc_qubits = 0
    ctl_qubits = 0
    for qb in circuit.all_qubits():
        try:
            if 'anc' in qb.name:
                anc_qubits += 1
        except:
            continue

        try:
            if 'ctl' in qb.name: 
                ctl_qubits += 1
        except:
            continue

    return (ttl_qubits, ctl_qubits, anc_qubits)

def time_step_comparison_mpe(clscl, gtbs, verbose=False):

    mpe = []
    N   = len(clscl[0])

    for i in range(len(gtbs)):
        mean_clscl = np.mean(np.abs(clscl[i,:]))
        mpe.append(np.sum(np.abs(clscl[i] - gtbs[i])/(N*mean_clscl)))

    return mpe

def get_state_vector(state,lsb_first=False):
    if lsb_first:
        return state
    
    sz = int(np.log2(len(state)))
    new_state = np.zeros(len(state),dtype=np.complex128)
    for idx,amp in enumerate(state):
        idx = int(bin(idx)[2:].zfill(sz)[::-1],base=2)
        new_state[idx] = amp
    return new_state

def print_to_openqasm(f,circuit,print_header=True,qubits=None):
    '''
    Frustratingly required because of circ very 'kindly' converting ccz/ccx/cz/X^1/T for us...
    '''

    op_list = list(circuit.all_operations())

    gateCounter = -1
    myReg = None
    skipLines = False
    skip_gate_counter_count = 0
    if qubits is None:
        qasm_circuit = circuit.to_qasm().split("\n")
    else:
        qasm_circuit = circuit.to_qasm(qubit_order=qubits).split("\n")

    for idx,line in enumerate(qasm_circuit):
        if (len(line) == 0) and skipLines:
            skipLines = False
            continue

        if len(line)==0:
            continue

        if not print_header and (("// Generated" in line) or ("OPENQASM" in line) or ("include" in line) or ('qreg' in line)):
            continue

        #book keep
        if ("// Qubits:" in line):
            if print_header:
                f.write("{}\n".format(line))
            myReg = line.replace("// Qubits: ",'')[1:-1].replace(' ','').split(',')
            myReg = {qname : 'q[{}]'.format(qidx) for qidx,qname in enumerate(myReg)}
            qubit_map = str(myReg).replace('{','').replace('}','')
            if print_header:
                f.write("// Cirq -> OpenQASM Map : {}\n".format(qubit_map))
            continue

        #book keep
        if ("q[" in line) and ("qreg" not in line):
            if not skipLines:
                if skip_gate_counter_count > 0:
                    skip_gate_counter_count -= 1
                else:
                    gateCounter+=1

        if "// Gate:" in line:
            #get gate
            gateCounter += 1
            # These gates have already been handled
            # Additionally, for each of these cirq ops multiple gates get written out.
            # In order to make sure that we keep a valid index for the cirq.circuit, we
            # need to skip incrementing the gate counter for each gate written out.
            if "Rx_d" in line:
                skip_gate_counter_count = 2
                continue
            elif "Ry_d" in line:
                skip_gate_counter_count = 5
                continue
            elif "Rz_d" in line:
                skip_gate_counter_count = 1
                continue
            skipLines = True
            op = op_list[gateCounter]
            ogline = str(op)

            line = str(op).replace("**-1.0",'').replace('**-1','')\
                .replace('TOFFOLI','ccx').replace("CCX","ccx")\
                .replace('CX','cx').replace('CNOT','cx')\
                .replace("CCZ","ccz").replace("CZ","cz")#.replace(')','')

            for r in myReg:
                if r in line:
                    line = line.replace(r,myReg[r])

            line = line.replace(' ','').replace('(',' ').replace(')','')
            #write it.
            f.write("{};\n".format(line))
            skipLines = True

        if not skipLines:
            line = line.replace("rx(pi*-1)",'x')
            line = line.replace("rz(pi*0.25)","t")
            f.write("{}\n".format(line))

def prettyprint_qsp_to_qasm(f, circuit):
    '''
    NOTE: This requires that the circuit is NOT decomposed or have any alignment applied to it.
    Frustratingly required because of circ very 'kindly' converting ccz/ccx/cz/X^1/T for us...
    '''
    
    # Do a loop to count Phased Iterates.
    # This is not efficient, but (I think) it is effective, so w/e
    selv_counter    = 0
    reflect_counter = 0
    for moment in circuit:
        for op in moment:
            if "SelectV" in str(op):
                selv_counter += 1
            elif "Reflect" in str(op):
                reflect_counter += 1

    num_phased_iterates = selv_counter - 1 # I think this is correct, confirm with collabs

    # With that ^^ out of the way, proceed
    in_phased_iterate      = False
    in_select_v            = False
    in_reflect             = False
    phased_iterate_counter = 0
    for cidx,moment in enumerate(circuit):
        for op in moment:
            if "SelectV" in str(op):
                # SelectV triggers the start of a 'phased iterate'
                if phased_iterate_counter < num_phased_iterates:
                    f.write("// START PHASED_ITERATE_{}\n".format(phased_iterate_counter))    
                    in_phased_iterate = True
                f.write("// START SELECT_V\n")
                in_select_v       = True
            elif "Reflect" in str(op):
                f.write("// START REFLECT\n")
                in_reflect        = True

        #print to qasm
        #decompose the operation.
        subcircuit = qsp_decompose_once(qsp_decompose_once(cirq.Circuit(moment)))
        if cidx == 0:
            print_to_openqasm(f,subcircuit,print_header=True, qubits=circuit.all_qubits())
        else:
            print_to_openqasm(f,subcircuit,print_header=False, qubits=circuit.all_qubits())

        doHeader = False
        
        if in_select_v:
            f.write("// END SELECT_V\n")
            in_select_v = False

        if in_reflect:
            f.write("// END REFLECT\n")
            in_reflect = False
            if phased_iterate_counter < num_phased_iterates:
                f.write("// END PHASED_ITERATE_{}\n".format(phased_iterate_counter))
                phased_iterate_counter += 1
                in_phased_iterate = False


def prettyprint_qsp_to_qasm_og(f,circuit):
    '''
    ORIGINAL PROTOTYPE FUNCTION - DOESNT WORK FULLY

    NOTE: This requires that the circuit is NOT decomposed or have any alignment applied to it.
    Frustratingly required because of circ very 'kindly' converting ccz/ccx/cz/X^1/T for us...
    '''
    
    phased_iterate_start   = False
    need2print_pi          = False
    select_v_start         = False
    need2print_sv          = False
    reflect_start          = False
    need2print_r           = False
    phased_iterate_counter = 0
    doHeader = True
    for cidx,moment in enumerate(circuit):
        for op in moment:
            if "SelectV" in str(op):
                phased_iterate_start = not phased_iterate_start
                select_v_start = not select_v_start
                need2print_pi = True
                need2print_sv = True
            elif "Reflect" in str(op):
                reflect_start = not reflect_start
                need2print_r = True

        if need2print_pi and phased_iterate_start:
            f.write("// START PHASED_ITERATE_{}\n".format(phased_iterate_counter))
        if need2print_sv and select_v_start:
            f.write("// START SELECT_V\n")
        if need2print_r and reflect_start:
            f.write("// START REFLECT\n")
        #print to qasm
        #decompose the operation.
        subcircuit = qsp_decompose_once(qsp_decompose_once(cirq.Circuit(moment)))
        print_to_openqasm(f,subcircuit,print_header=doHeader, qubits=circuit.all_qubits())
        doHeader = False
        #
        if need2print_r and not reflect_start:
            f.write("// END REFLECT\n")
        if need2print_sv and not select_v_start:
            f.write("// END SELECT_V\n")
        if need2print_pi and not phased_iterate_start:
            f.write("// END PHASED_ITERATE_{}\n".format(phased_iterate_counter))
            phased_iterate_counter += 1
        
            

def circuit_compare(circuit1,circuit2,ignoreLen=False):
    c1 = cirq.align_left(cirq.Circuit([op for op in circuit1.all_operations()]))
    c2 = cirq.align_left(cirq.Circuit([op for op in circuit2.all_operations()]))
    if not ignoreLen:
        if len(c1) != len(c2):
            print("Lens not the same {} != {}".format(len(c1),len(c2)))
            return False
    isSame = True
    for idx,(mom1,mom2) in enumerate(zip(c1,c2)):
        if not cirq.equal_up_to_global_phase(mom1,mom2):
            #double check...
            if not cirq.allclose_up_to_global_phase(cirq.unitary(mom1),cirq.unitary(mom2)):
                print(f"@idx = {idx}\n{mom1}\n????\n{mom2}\n")
                isSame = False
    return isSame

def snorm(a:float, b:float) -> float:
    if b<0:
        return -np.linalg.norm([np.abs(a), np.abs(b)])
    else:
        return np.linalg.norm([np.abs(a), np.abs(b)])

def splitInPairs(inList : List[float]) -> List[Tuple[float, float]]:
    '''
    Given [a,b,c,d,e,f,....]
    Return [(a,b),(c,d),(e,f),....]
    '''
    aV = [inList[idx] for idx in range(0,len(inList),2)]
    bV = [inList[idx+1] for idx in range(0,len(inList),2)]
    return [(a,b) for a,b in zip(aV,bV)]
    
def splitIn2(list_in:List):
    '''
    Given [a,b,c,d,e,f]
    Return ([a,b,c],[d,e,f])
    '''
    mid_point = int(np.floor(len(list_in)/2))
    first_list = list_in[0:mid_point]
    second_list = list_in[mid_point:]
    return (first_list, second_list)

def first_two_items(in_list : List) -> Tuple[List, List]:
    # Doing this out of abundance of caution:
    use_list = copy.deepcopy(in_list)
    
    if len(in_list) < 2: 
        return [[], use_list]
    else:
        a = use_list[0]
        b = use_list[1]
        use_list.remove(a)
        use_list.remove(b)
        return ([a,b], use_list)

# Can use a better name:
def getLineQubitIndexMap(qubit_line, name) -> List[tuple]:
    out_map = []
    for ii in range(len(qubit_line)):
        tmp = (ii, name)
        out_map.append(tmp)
    return out_map

# Same - could use a better name...
def getQubitFromMap(tuple_in, ctl_q, tgt_q, anc_q):

    if tuple_in[1] == 'ctl':
        return ctl_q[tuple_in[0]]
    elif tuple_in[1] == 'tgt':
        return tgt_q[tuple_in[0]]
    elif tuple_in[1] == 'anc':
        return anc_q[tuple_in[0]]
    else:
        raise RuntimeError('Problem with Reflect algorithm...')

def decompose_CCZ(circuit, debug=False):
    decomposed_gates = []
    for moment in circuit:
        for op in moment:
            if debug:
                print(f'>> circuit op = {op}')
            if (str(op).startswith('CCZ')):
                decomp_ccz = cirq.decompose_once(op)
                decomposed_gates.append(decomp_ccz)
                if debug:
                    print(f'>> CCZ Decomposed into {decomp_ccz}')
                # for dc in decomp_ccz:
                #     print(f'>>> DC = {dc}')
                #     if "**-1" in str(dc):
                #         print(f'>> new circuit op (did an inverse)= {cirq.inverse(dc)}')
                #         decomposed_gates.append(cirq.inverse(dc))
                #     else:
                #         decomposed_gates.append(dc) 
                #         print(f'>> new circuit op = {dc}')              
            else:
                decomposed_gates.append(op)
    
    return cirq.Circuit(decomposed_gates, strategy=cirq.InsertStrategy.NEW)


def qsp_decompose_once(circuit, debug=False):
    new_qubits = circuit.all_qubits()
    decomposed_gates = []
    for moment in circuit:
        for op in moment:
            if debug:
                print(f'>> circuit op = {op}')
            if (str(op).startswith(('reset','Rx','Ry','Rz',\
                'X','Y','Z','S', 'H',\
                'CX','CZ','CCZ','CCX',\
                'TOFFOLI', 'CCXi', 'ccxi', 'cirq.Measure'))):
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

class QSPFilesIO:
    """
    A class for reading and writing QSP relevant files.

    Attributes:
    -----------
    None

    Methods:
    --------
    readHaml(fileName) -> List[Tuple[str, float]]
        Reads in a .haml file and outputs a list.

    readAngles(fileName) -> List[float]
        Reads in a .angles file and outputs a list.

    """

    def __init__(self):
        pass

    @staticmethod
    def readHaml(fileName: str) -> List[Tuple[str, float]]:
        """
        Reads in a .haml file and outputs a list.

        Parameters:
            fileName : str
                The name of the file, with path if necessary
        
        Returns:
            outList: List[Tuple[str, float]]
                The list of 2-tuples containing Hamiltonian info
        """
        outList = list()
        with open(fileName, 'r') as fileIn:
            for line in fileIn:
                val = (line.split()[0], float(line.split()[1]))
                outList.append(val)
        return outList

    @staticmethod
    def readAngles(fileName: str) -> List[float]:
        """
        Readds in a .angles file and outputs a list.

        Parameters:
            fileName : str
                The name of the file, with path if necessary.

        Returns:
            outList : List[float]
                The list of phi angles in [radians?].
        """
        outList = list()
        with open(fileName, 'r') as fileIn:
            for line in fileIn:
                val = float(line)
                outList.append(val)
        return outList
