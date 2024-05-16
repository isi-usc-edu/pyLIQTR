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


def circuit_decompose_once(circuit, debug=False):
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
