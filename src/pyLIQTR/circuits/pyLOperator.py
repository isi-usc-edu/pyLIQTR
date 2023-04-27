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

from abc import abstractmethod
import cirq
from typing import (List, Tuple, Dict, Optional)

#BASE CLASS, ANY METHODS
class pyLOperator(cirq.Gate):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    #FUNCTIONALITY
    @abstractmethod
    def _decompose_(self,qubits):
        """This should implement a decomposition of this operator into additional operators or base cirq.Gates"""
    
    @abstractmethod
    def _num_qubits_(self,qubits):
        """This should return an int specifying the total number of qubits this operator acts on"""

    @abstractmethod
    def _circuit_diagram_info_(self, args) -> List[str]:
        """Represents how this operator will be printed by cirq, should be a list of strings for each qubit it acts upon"""

    @abstractmethod
    def __str__(self) -> str:
        raise NotImplementedError
    
    @abstractmethod
    def _qasm_(self, args: 'cirq.QasmArgs', qubits: Tuple['cirq.Qid', ...]) -> Optional[str]:
        #args.
        raise NotImplementedError

    def to_qasm(self,decomp_level):
        raise NotImplementedError("IMPLEMENT ME")

    #@abstractmethod
    #def get_resouces(self) -> Dict:
    #    """This method must return a dictionary of the resouces of the child class"""

    #METRICS 
    def get_qubits(self) -> Tuple:
        return (self.num_qubits,0)
    
    def gate_count(self) -> Dict[str,int]:
        return {}
    
    


    
    
