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

from abc import abstractmethod
import cirq
from typing import (List, Tuple, Dict, Optional)

from pyLIQTR.utils.circuit_decomposition import decompose_once
from pyLIQTR.gate_decomp.cirq_transforms import clifford_plus_t_direct_transform, get_approximate_t_depth

#BASE CLASS, ANY METHODS
class pyLOperator(cirq.Gate):
    def __init__(self, *args, **kwargs):
        """
        pyLOperator

        Every operator is required to implement the abstract methods below, as well as the
        "_total_decomp" member.

        The intention with the total_decomp is that it specifies how many levels above Clifford+T
        gates this operator sits at:
            * Clifford + T gates sit at level 0
            * Gates/operators that are composed of clifford+t gates sit at level 1
            * and so on

        Basically, our total_decomp is always one level greater than our decomposition.

        """
        super().__init__(*args, **kwargs)
        self._total_decomp=0
        

    @property
    def total_decomp(self):
        return self._total_decomp

    @total_decomp.setter
    def total_decomp(self,val):
        level_difference = (val - self.total_decomp)
        if level_difference > 0:
            self._total_decomp = level_difference
            
    #FUNCTIONALITY
    @abstractmethod
    def _decompose_(self,qubits):
        """This should implement a decomposition of this operator into additional operators or base cirq.Gates"""
    
    @abstractmethod
    def _num_qubits_(self):
        """This should return an int specifying the total number of qubits this operator acts on"""

    @abstractmethod
    def _circuit_diagram_info_(self, args) -> List[str]:
        """Represents how this operator will be printed by cirq, should be a list of strings for each qubit it acts upon"""

    @abstractmethod
    def __str__(self) -> str:
        """Represents the representation of this operator when the operator is used as an argument in str()"""
        raise NotImplementedError
    
    @abstractmethod
    def _qasm_(self, args: 'cirq.QasmArgs', qubits: Tuple['cirq.Qid', ...]) -> Optional[str]:
        """Represents the representation of this operator  in openqasm2.0 format"""
        #Note here is an example of how one can construct the qasm representation in a cirq specific way.
        #However, somtimes its nice to order the qubits in a certain order
        #qasm str = str(self)+",".join([args.format(str(x)) for x in qubits])
        raise NotImplementedError

    @abstractmethod
    def __eq__(self, other):
        """
        The equality operator is required to be implemented in order to satisfy the resource analysis/caching functionality
        
        This method is just used to test for equivalence between two qubits
        """
        raise NotImplementedError

    @abstractmethod
    def _get_as_circuit(self):
        #Get the operator as a circuit.
        raise NotImplementedError

    
    def count_exact(self,gate_precision=1e-6) -> Dict[str,int]:
        """
        This provides a default implementation of the count exact method

        NOTE: This should be overriden if the operator level is >2
        """
        gate_dict = {}
        #Decompose once only works on circuits
        mySelf = self._get_as_circuit()
        for _ in range(self.total_decomp-1):
            mySelf = decompose_once(mySelf)
        tmp = decompose_once(mySelf)
        #Iterate through all the operations that compose this gate.
        for x in tmp.all_operations():
            gateStr = "PRECLIFFT_"+str(x.gate)
            if gateStr not in gate_dict:
                gate_dict[gateStr] = 1
            else:
                gate_dict[gateStr] += 1
        tmp = cirq.align_left(tmp)
        gate_dict["PRECLIFFT_ApproxDepth"] = len(tmp)
        gate_dict["PRECLIFFT_ApproxTDepth"] = get_approximate_t_depth(tmp,depthToff=1)

        tmp = cirq.align_left(clifford_plus_t_direct_transform(tmp,gate_precision=gate_precision))


        gate_dict["ApproxDepth"] = len(tmp)
        gate_dict["ApproxTDepth"] = get_approximate_t_depth(tmp,depthToff=1)
        for x in tmp.all_operations():
            gateStr = str(x.gate)
            if gateStr not in gate_dict:
                gate_dict[gateStr] = 1
            else:
                gate_dict[gateStr] += 1 

        return gate_dict
    

class fakeOperator(pyLOperator):
    def __init__(self,decomp_level):
        super(fakeOperator,self).__init__()

        self.total_decomp = decomp_level
    
    def _decompose_(self,qubits):
        child_decomp_level = self.total_decomp-1
        print("Decomposing fake operator to level {}".format(child_decomp_level))
        if child_decomp_level == 0:
            yield cirq.X.on(qubits[0])
        else:
            yield fakeOperator(decomp_level=child_decomp_level).on(qubits[0])
    
    def __str__(self):
        return f"fakeoperator_lvl_{self.total_decomp}"

    def _num_qubits_(self):
        return 1
    
    def _circuit_diagram_info_(self, args) -> List[str]:
        return [f"fakeoperator_lvl_{self.total_decomp}"] * self.num_qubits()

    def __eq__(self,other):
        return (self.total_decomp == other.total_decomp) and (self.__class__ == other.__class__)

    def _qasm_(self, args: 'cirq.QasmArgs', qubits: Tuple['cirq.Qid', ...]) -> Optional[str]:
        args.validate_version('2.0')
        allQStr = ",".join([args.format(str(x)) for x in qubits])
        return f"fakeoperator_lvl_{self.total_decomp}({allQStr})"

    def count_exact(self,gate_precision=1e-8):
        return {'Fake':self.total_decomp}