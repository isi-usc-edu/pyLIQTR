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

### Note: This code is under development

from math import floor, log2
from typing import List, Tuple, Optional

from pyLIQTR.circuits.pyLOperator import pyLOperator,Dict
from pyLIQTR.utils.qsp_helpers import splitInPairs, splitIn2, snorm
from pyLIQTR.utils.circuit_decomposition import decompose_once
from pyLIQTR.gate_decomp.cirq_transforms import clifford_plus_t_direct_transform, get_approximate_t_depth
# Improve theses imports
import cirq
import numpy as np

class Prepare(pyLOperator):
    """
    Implements the Prepare operator from <cite>.
    """
    
    def __init__(self, qubit_reg:List, alphas, \
            pp_exp=cirq.Y, pp_ctl=cirq.X, pp_com=cirq.I):
        # check and make sure that the user is sending a power of 2 of the len(alphas)
        if floor(log2(len(alphas))) != len(qubit_reg):
            raise ValueError("The length of the alphas MUST be a power of 2, exiting...")

        self.__reg_q  = qubit_reg
        self.__alphas = alphas
        self.__pp_exp = pp_exp
        self.__pp_ctl = pp_ctl
        self.__pp_com = pp_com
        self.__sel_q = None
        
        super(Prepare, self).__init__()
        self.total_decomp = 2
        
    def __str__(self) ->str:
        qStr = ",".join([str(x) for x in self.__reg_q])
        return "Prepare {}".format(qStr)

    def _qasm_(self, args:'cirq.QasmArgs', qubits:Tuple['cirq.Qid', ...]) -> Optional[str]:
        args.validate_version('2.0')
        allQ = [*self.__reg_q]
        allQStr = ",".join([args.format(str(x)) for x in allQ])
        return f"Prepare({allQStr})\n"
        
    def _num_qubits_(self,):
        return len(self.__reg_q)
        
    def _circuit_diagram_info_(self, args):
        return ["Prepare"] * self.num_qubits()
        
    def _get_as_circuit(self):
        return cirq.Circuit(self.on(*self.__reg_q))

    @property
    def alphas(self):
        return self.__alphas

    @property 
    def reg_qubits(self):
        return self.__reg_q
    
    @property
    def pp_exp(self):
        return self.__pp_exp

    @property
    def pp_ctl(self):
        return self.__pp_ctl

    @property
    def pp_com(self):
        return self.__pp_com

    def _decompose_(self, qubits):
        return self.disentangle_recursive(alt=False, flip=True, pp_tgt=self.pp_exp, pp_com=self.pp_com, pp_ctl=self.pp_ctl)

    def __eq__(self,other):
        if not hasattr(other,'total_decomp'):
            return False
        return (self.total_decomp == other.total_decomp) and (self.__class__ == other.__class__)

    def alpha_prep(self, flip: bool, alpha_pair:Tuple[float, float]):
        if flip:
            if alpha_pair[0] == 0:
                return (0, alpha_pair[1])
            else:
                new_a = np.pi/2 - np.arccos(alpha_pair[0] / snorm(alpha_pair[0], alpha_pair[1]))
                new_b = snorm(alpha_pair[0], alpha_pair[1])
                return (new_a, new_b)        
        else:
            if alpha_pair[0] == 0:
                return (np.pi/2, alpha_pair[1])
            else:
                new_a = np.arccos(alpha_pair[0] / snorm(alpha_pair[0], alpha_pair[1]))
                new_b = snorm(alpha_pair[0], alpha_pair[1])
                return (new_a, new_b)

    def prepare_angles(self, flip : bool, alphas:List[float]) -> Tuple[List[float], List[float]]:

        # Get these right from this function, and initialize other lists
        alphas_in_pairs         = splitInPairs(alphas)
        prepped_alphas          = [self.alpha_prep(flip=flip, alpha_pair=alpha_pair) for alpha_pair in alphas_in_pairs]
    
        # unzip using list comprehension:
        angles    = [i for i,j in prepped_alphas]
        new_coefs = [j for i,j in prepped_alphas]
        
        return angles, new_coefs

    def expOp(self, phi, tgt_q, debug=False):
        if np.isclose(phi,0):
            return []
        if self.pp_exp==cirq.X:
            return [cirq.Rx(rads=phi*2).on(tgt_q)]
        elif self.pp_exp==cirq.Y:
            if debug:
                print(f'> Ry with phi={phi*2} on tgt={tgt_q}')
            return [cirq.Ry(rads=phi*2).on(tgt_q)]
        else: # assuming one of the three...
            return [cirq.Rz(rads=phi*2).on(tgt_q)]

    def ctrlOp(self, ctl, tgt, debug=False):
        if self.pp_ctl == cirq.X:
            if debug:
                print(f'> Cx ctl={ctl} and tgt={tgt}')
            if self.__sel_q == None:
                return [cirq.CX.on(ctl, tgt)]
            else:
                return [cirq.CCX.on(self.__sel_q, ctl, tgt)]
        elif self.pp_ctl == cirq.Z:
            if self.__sel_q == None:
                return [cirq.CZ.on(ctl, tgt)]
            else:
                return [cirq.CCZ.on(self.__sel_q, ctl, tgt)]
        else:
            print('help, dont see CY in cirq doc.')

    # TODO: unit test this against the qp equivalent.
    # Note: I think this basically works, just needs to be debugged
    def split_angles(self, alt:bool, alphas):
        l1,l2 = splitIn2(alphas)
        if len(l1) != len(l2):
            raise ValueError('Need same size lists in split_angles!!!')

        multiplier = 1
        if (self.pp_com==cirq.X or self.pp_com==cirq.Y):
            multiplier = -1
        
        p = lambda w,x : (w+x)/2
        m = lambda w,x : multiplier*(w-x)/2
        if alt:
            return ([m(a,b) for a,b in zip(l1,l2)],\
                    [p(a,b) for a,b in zip(l1,l2)])
        else:
            return ([p(a,b) for a,b in zip(l1,l2)],\
                    [m(a,b) for a,b in zip(l1,l2)])
        
    def disentangle_recursive(self, alt:bool, flip:bool, pp_tgt, pp_ctl, pp_com):
        # Not used:
        #  flip   : bool
        #  pp_tgt : pauli
        #  pp_ctl : pauli
        #  pp_com : pauli

        gates = []
        gates.extend(self.disentangle_recursion(alt=alt, alphas=self.alphas, qubit_reg=self.reg_qubits))
        # Assuming no gate_sh_inv right now...
        gates.reverse()
        return gates
   
    # TODO: Check angles and signs and stuff
    # TODO: Get to work kindly with existing infra
    def disentangle_recursion(self, alt:bool, alphas, qubit_reg):
        gates = []
        if not qubit_reg:
            # Need to figure out what to do here.
            return gates
        else:
            qh = qubit_reg[0] # first qubit
            qi = qubit_reg[0:-1]
            ql = qubit_reg[-1]
            # print('---------')
            # print(f'> Alphas = {alphas}')
            angles, new_coefs = self.prepare_angles(flip=True, alphas=alphas) # Note: Where does flip come from?

            if True: #gopts.contains(rXYZ) 
                # -> TODO pp_exp = pp_tgt
                pp_clf = self.pp_ctl
            else:
                pp_exp = cirq.Z
                pp_ctl = cirq.X

            if pp_clf != self.pp_exp:
                pp_clf_ = pp_clf # note, i dont understand what's happening here in the haskell code

            gates.extend(self.multiplexor(alt=alt, alphas=angles, ctl_qubits=qi, tgt_qubit=ql))
            gates.extend(self.disentangle_recursion(alt=alt, alphas=new_coefs, qubit_reg=qi))
            return gates

    # TODO: Check angles and signs and stuff
    # TODO: Get to work kindly with existing infra
    def multiplexor(self, alt:bool, alphas, ctl_qubits, tgt_qubit):
        gates = []

        if not ctl_qubits:
            gates.extend(self.multiplex_aux(alt=False, alphas=alphas, ctl_qubits=ctl_qubits, tgt_qubit=tgt_qubit))
            return gates
        else:
            if alt:
                gates.extend(self.ctrlOp(ctl=ctl_qubits[0], tgt=tgt_qubit))
                gates.extend(self.multiplex_aux(alt=True, alphas=alphas, ctl_qubits=ctl_qubits, tgt_qubit=tgt_qubit))
                return gates
            else:
                gates.extend(self.multiplex_aux(alt=False, alphas=alphas, ctl_qubits=ctl_qubits, tgt_qubit=tgt_qubit))
                gates.extend(self.ctrlOp(ctl=ctl_qubits[0], tgt=tgt_qubit))
                return gates
    
    # TODO: Check angles and signs and stuff
    # TODO: Get to work kindly with existing infra
    def multiplex_aux(self, alt:bool, alphas, ctl_qubits, tgt_qubit, debug=False):
        gates = []
        # ---------
        # Haskell recursive structure:
        # ---------
        if not ctl_qubits:
            # is a return needed?
            gates.extend(self.expOp(phi=alphas[0], tgt_q=tgt_qubit))
            
        else:
            q0 = ctl_qubits[0]
            qt = ctl_qubits[1:]
            as0, as1 = self.split_angles(alt=alt, alphas=alphas)

            if debug:
                print('---------------------')
                print(f'> q0 = {q0}')
                print(f'> qt = {qt}')
                print(f'> alphas = {alphas}')
                print(f'> as0    = {as0}')
                print(f'> as1    = {as1}')

            gates.extend(self.multiplex_aux(alt=False, alphas=as0, ctl_qubits=qt, tgt_qubit=tgt_qubit))
            gates.extend(self.ctrlOp(ctl=q0, tgt=tgt_qubit))
            gates.extend(self.multiplex_aux(alt=True, alphas=as1, ctl_qubits=qt, tgt_qubit=tgt_qubit))
            # Is a return needed?

        return gates
