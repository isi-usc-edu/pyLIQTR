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
import numpy as np
from pyLIQTR.QSP.qsp_helpers import splitInPairs, splitIn2, snorm
import cirq
# from Hamiltonian import Hamiltonian

class QSP_Prepare(cirq.Gate):
    
    def __init__(self, qubit_reg,
                 alphas,
                 pp_exp, pp_ctl, pp_com, select_q=None): # Which paulis belong here?
        self.__reg_q  = qubit_reg
        self.__alphas = alphas
        self.__pp_exp = pp_exp
        self.__pp_ctl = pp_ctl
        self.__pp_com = pp_com
        self.__sel_q = select_q

        super(QSP_Prepare, self)

    def _circuit_diagram_info_(self, args):
        return ["Prepare"] * self.num_qubits()

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

    def _num_qubits_(self):
        if self.__sel_q == None:
            return len(self.__reg_q)
        else:
            return len(self.__reg_q)+1

    def _decompose_(self, qubits):
        # print('Not yet working...')
        return self.disentangle_recursive(alt=False, flip=True, pp_tgt=self.pp_exp, pp_com=self.pp_com, pp_ctl=self.pp_ctl)

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

# generate a multi-qubit ccx gate using the specified sense out of CCX gates
def multi_ccx(in_qubits, sense, out_qubit, anc_qubits):
    num_anc = len(in_qubits) - 2
    assert(num_anc <= len(anc_qubits))
    assert(len(in_qubits) == len(sense))
    
    circuit = cirq.Circuit()

    circuit_f = cirq.Circuit()
    src_qubits = [q for q in in_qubits]
    a_qubits = [q for q in anc_qubits]
    t_sense = [v for v in sense]
        
    while len(src_qubits) > 2:
        circuit_f += [cirq.X.on(src_qubits[ii]) for ii in range(2) if not t_sense[ii]]
        circuit_f.append(cirq.CCX.on(src_qubits[0], src_qubits[1], a_qubits[0]))
        circuit_f += [cirq.X.on(src_qubits[ii]) for ii in range(2) if not t_sense[ii]]
        
        src_qubits = [a_qubits[0]] + src_qubits[2:]
        t_sense = [1] + t_sense[2:]
        a_qubits = a_qubits[1:]
        
    circuit_m = cirq.Circuit()
    circuit_m += [cirq.X.on(src_qubits[ii]) for ii in range(2) if not t_sense[ii]]
    circuit_m.append(cirq.CCX.on(src_qubits[0], src_qubits[1], out_qubit))
    circuit_m += [cirq.X.on(src_qubits[ii]) for ii in range(2) if not t_sense[ii]]

    circuit_r = cirq.inverse(circuit_f)

    circuit = circuit_f + circuit_m + circuit_r

    return circuit

    
def gen_pre_prepare(nb, qubit_dict):

    def ripple_adder_outofplace(nb, qubit_dict, subtract=False):
        abits = qubit_dict['a']
        bbits = qubit_dict['b']
        dbits = qubit_dict['d']
        cbit = qubit_dict['c']
        zbit = qubit_dict['z']
        
        adder_twosc = cirq.Circuit()
        adder_core = cirq.Circuit()
    
        if subtract:
            # two's complement of the second input
            adder_twosc += [cirq.X.on(bbits[ii]) for ii in range(nb)]
            adder_twosc.append(cirq.X.on(cbit))
            if not zbit == None:
                adder_twosc.append(cirq.X.on(zbit))

        carry_in = [cbit] + [abits[qi] for qi in range(nb-1)]
        carry_out = [abits[qi] for qi in range(nb)]
        areg = [abits[qi] for qi in range(nb)]
        breg = [bbits[qi] for qi in range(nb)]
        if not dbits == None:
            dreg = [dbits[qi] for qi in range(nb)]
        else:
            dreg = None

        for qi in range(nb):
            if not dreg == None:
                adder_core.append(cirq.CX.on(carry_in[qi], dreg[qi]))
            adder_core.append(cirq.CX.on(areg[qi], breg[qi]))
            if not dreg == None:
                adder_core.append(cirq.CX.on(breg[qi], dreg[qi]))
            adder_core.append(cirq.CX.on(areg[qi], carry_in[qi]))
            adder_core.append(cirq.CCX.on(carry_in[qi], breg[qi], carry_out[qi]))

        if not zbit == None:
            adder_core.append(cirq.CX.on(carry_out[-1], zbit))

        adder = adder_twosc + adder_core

        return adder
    
    def select_terms(nb, qubit_dict):
        pbits = qubit_dict['p']
        qbits = qubit_dict['q']
        rbits = qubit_dict['r']
        ubit = qubit_dict['u']
        vbit = qubit_dict['v']
        sbits = qubit_dict['s']
        ancbits = qubit_dict['anc']
    
        circuit_ab = cirq.Circuit()
        circuit_ab.append(cirq.CX.on(pbits[-1],qbits[-1]))

        # circuit to compute q-p with result placed in msb of r
        adder_dict = dict([['a',qbits[:-1]], ['b',pbits[:-1]], ['c',ancbits[0]], ['d',None], ['z',rbits[-1]]])
        circuit_qmp_frwd = ripple_adder_outofplace(nb-1, adder_dict, True)

        # circuit to compute first part of selects
        circuit_sel_qmp = cirq.Circuit()
        (u,v,re,b) = (ubit,vbit,rbits[-1],qbits[-1])
        (s0,s1,s2) = (sbits[0],sbits[1],sbits[2])
        circuit_sel_qmp += multi_ccx([u,v,re,b], [1,0,1,0], s0, ancbits[1:3])
        circuit_sel_qmp += multi_ccx([u,v,re,b], [0,0,1,0], s1, ancbits[1:3])
        circuit_sel_qmp += multi_ccx([u,v,re,b], [0,1,1,0], s2, ancbits[1:3])

        # reverse the computation of q-p
        circuit_qmp_rvrs = cirq.inverse(circuit_qmp_frwd)
        #circuit_qmp_rvrs = copy.deepcopy(circuit_qmp_frwd)
        #circuit_qmp_rvrs.reverse_circuit()

        circuit_qmp = circuit_qmp_frwd + circuit_sel_qmp + circuit_qmp_rvrs

        # circuit to compute p-q into r
        adder_dict = dict([['a',pbits[:-1]], ['b',qbits[:-1]], ['c',ancbits[0]], ['d',rbits[:-1]], ['z',rbits[-1]]])
        circuit_pmq_frwd = ripple_adder_outofplace(nb-1, adder_dict, True)
        # reverse the garbage, but keep the sum
        adder_dict = dict([['a',pbits[:-1]], ['b',qbits[:-1]], ['c',ancbits[0]], ['d',None], ['z',None]])
        circuit_pmq_rvrs = cirq.inverse(ripple_adder_outofplace(nb-1, adder_dict, True))

        # circuit to compute second part of selects
        circuit_sel_pmq = cirq.Circuit()
        circuit_sel_pmq += multi_ccx([u,v,re,b], [1,0,1,0], s0, ancbits[1:3])
        circuit_sel_pmq += multi_ccx([u,v,re,b], [0,0,1,0], s1, ancbits[1:3])
        circuit_sel_pmq += multi_ccx([u,v,re,b], [0,1,1,0], s2, ancbits[1:3])
    
        circuit_pmq = circuit_pmq_frwd + circuit_sel_pmq + circuit_pmq_rvrs

        circuit_sel = cirq.Circuit()
        circuit_sel += multi_ccx([u,v,b], [1,0,0], s0, ancbits[1:3])
        circuit_sel += multi_ccx([u,v,b], [0,1,1], s2, ancbits[1:3])

        circuit = circuit_ab + circuit_qmp + circuit_pmq + circuit_sel + circuit_ab

        return circuit
    

    circuit_sel = select_terms(nb, qubit_dict)
    
    yield circuit_sel

class FermionPrePrepare(cirq.Gate):
    def __init__(self, nb, control_qubits, ancilla_qubits):

        self.__nb = nb  # width of p+alpha in bits
        self.__num_qubits = len(control_qubits) + len(ancilla_qubits)
        
        self.__qubit_dict = dict()
        self.__qubit_dict['p'] = control_qubits[:nb]         # p register + alpha
        self.__qubit_dict['q'] = control_qubits[nb:2*nb]     # q register + beta
        self.__qubit_dict['r'] = ancilla_qubits[:nb]         # r = p-q (n bits)
        self.__qubit_dict['u'] = control_qubits[2*nb]        # U bit
        self.__qubit_dict['v'] = control_qubits[2*nb+1]      # V bit
        self.__qubit_dict['s'] = ancilla_qubits[nb:nb+3]     # 3 select qubits used to select terms
        self.__qubit_dict['anc'] = ancilla_qubits[nb+3:nb+6] # 3 ancilla used for toffolis

        super(FermionPrePrepare, self)
        
    def _num_qubits_(self):
        return self.__num_qubits

    def _decompose_(self, qubits):
        yield gen_pre_prepare(self.__nb, self.__qubit_dict)
    
    def _circuit_diagram_info_(self, args):
        return ["FermionPrePrepare"] * self.num_qubits()
        

    
class FermionPrepare(cirq.Gate):

    def __init__(self, hamiltonian, control_qubits, ancilla_qubits):

        self.__ctl_q = control_qubits
        self.__anc_q = ancilla_qubits
        self.__hamiltonian = hamiltonian
    
        # the Hamiltonian contains three components:
        # U(p), T(p-q), V(p-q)
        # Each component contains a number of terms: [(p,alpha,q,beta), amp]
        # The number of orbitals + spin is specified as N
        N = self.__hamiltonian.problem_size
        Nu = N  # number of U terms in the Hamiltonian (p,sigma,p,sigma)
        Nt = 2*((N//2)**2 - N//2)   # number of Utterms in the Hamiltonian (p,sigma,q,sigma) (p != q)
        #Nt = 2*Nt    # for now the prep does all combination of alpha,beta
        Nv = (N**2) - N  # number of V terms in the Hamiltonian (p,alpha) != (q,beta)
        print("N = {}, Nu = {}, Nt = {}, Nv = {}".format(N, Nu, Nt, Nv))
        
        # separate the coefficients for each term (there are either N or 2N unique terms)
        self.__coefU = [self.__hamiltonian.terms[ii][1] for ii in range(0,N)]
        self.__coefT = [self.__hamiltonian.terms[ii][1] for ii in range(N,2*N)]
        self.__coefV = [self.__hamiltonian.terms[ii][1] for ii in range(2*N,4*N)]

        # Create the three select iterators used
        fermY = [([("Y",ni)], 1.0) for ni in range(N)]   # Fermionic YpZp-1...Z0
        self.__selFermY = fermY
        fermX = [([("X",ni)], 1.0) for ni in range(N)]   # Fermionic XpZp-1...Z0
        self.__selFermX = fermX
        fermZ = [([("Z",ni)], 1.0) for ni in range(N)]   # Zp
        self.__selAplyZ = fermZ

        self.__N = N
        self.__nb = int(np.ceil(np.log2(N)))
        
        super(FermionPrepare, self)
        
    def _num_qubits_(self):
        return len(self.__ctl_q)+len(self.__anc_q)

    def _decompose_(self, qubits):
        from pyLIQTR.QSP.qsp_select_v import UnaryIterate
        
        circuit_uv = cirq.Circuit()
        (u,v) = (self.__ctl_q[2*self.__nb], self.__ctl_q[2*self.__nb+1])
        circuit_uv.append(cirq.H.on(u))
        circuit_uv.append(cirq.H.on(v))
        circuit_uv.append(cirq.CCX.on(u, v, self.__anc_q[0]))
        circuit_uv.append(cirq.measure(self.__anc_q[0]))

        circuit_pq = cirq.Circuit()
        circuit_pq += [cirq.H.on(pb) for pb in self.__ctl_q[:self.__nb]]
        circuit_pq += [cirq.H.on(qb) for qb in self.__ctl_q[self.__nb:2*self.__nb]]
    
        # circuit to post-select on non-zero select bits
        circuit_collapse = cirq.Circuit()
        [s0, s1, s2] = self.__anc_q[self.__nb:self.__nb+3]     # 3 select qubits used to select terms
        circuit_collapse += multi_ccx([s0, s1, s2], [0,0,0], self.__anc_q[0], self.__anc_q[1:3])
        circuit_collapse.append(cirq.measure(self.__anc_q[0]))
    
        yield circuit_uv
        yield circuit_pq
        yield FermionPrePrepare(self.__nb, self.__ctl_q, self.__anc_q).on(*(self.__ctl_q + self.__anc_q))
        yield circuit_collapse

        preg = self.__ctl_q[:self.__nb-1]
        yield QSP_Prepare(qubit_reg = preg, alphas = self.__coefU, pp_exp=cirq.Y, pp_ctl = cirq.X, pp_com = cirq.I, select_q=s0).on(*(preg+[s0]))
        rreg = self.__anc_q[:self.__nb]
        yield QSP_Prepare(qubit_reg = rreg, alphas = self.__coefT, pp_exp=cirq.Y, pp_ctl = cirq.X, pp_com = cirq.I, select_q=s1).on(*(rreg + [s1]))
        yield QSP_Prepare(qubit_reg = rreg, alphas = self.__coefV, pp_exp=cirq.Y, pp_ctl = cirq.X, pp_com = cirq.I, select_q=s2).on(*(rreg + [s2]))

        """
        This is intended to set the phase angle, but I think that it requires more ancilla than I currently have allocated
        I can either add more ancilla or calculate p-q in place
        # set the angles for the p-q case
        # this logic needs to be checked
        ham_even = []
        ham_odd = []
        for ni in range(self.__N):
            if (self.__coefT[ni] > 0.0) and (self.__coefV[ni] > 0.0):
                ham_even += [[("I",0),1.0]]
                ham_odd += [[("I",0),1.0]]
            elif (self.__coefT[ni] < 0.0) and (self.__coefV[ni] > 0.0):
                ham_even += [[("X",0),1.0]]
                ham_odd += [[("I",0),1.0]]
            elif (self.__coefT[ni] > 0.0) and (self.__coefV[ni] < 0.0):
                ham_even += [[("I",0),1.0]]
                ham_odd += [[("X",0),1.0]]
            else:
                ham_even += [[("X",0),1.0]]
                ham_odd += [[("X",0),1.0]]
        ham = ham_even + ham_odd
        
        circuit = cirq.Circuit()
        circuit.append(cirq.CX.on(s1, s2))
        yield circuit

        yield UnaryIterate(ham, s2, [self.__ctl_q[-1]], self.__anc_q[:self.__nb], self.__anc_q[:2*self.__nb],'select').\
            on(*([s2] + [self.__ctl_q[-1]] + self.__anc_q[:self.__nb] + self.__anc_q[:2*self.__nb]))
        yield circuit
        """
        yield cirq.inverse(FermionPrePrepare(self.__nb, self.__ctl_q, self.__anc_q).on(*(self.__ctl_q + self.__anc_q)))
    
    def _circuit_diagram_info_(self, args):
        return ["FermionPrepare"] * self.num_qubits()
    

        
# Select the type of function to use based on the Hamiltonian type
#
def PrepareOracle(hamiltonian, control_qubits, ancilla_qubits):
    if hamiltonian.is_fermionic:
        yield FermionPrepare(hamiltonian, control_qubits, ancilla_qubits).on(*(control_qubits + ancilla_qubits))
    else:
        yield QSP_Prepare(qubit_reg = control_qubits, alphas = hamiltonian.alphas, pp_exp=cirq.Y, pp_ctl = cirq.X, pp_com = cirq.I).\
              on(*(control_qubits))
              
