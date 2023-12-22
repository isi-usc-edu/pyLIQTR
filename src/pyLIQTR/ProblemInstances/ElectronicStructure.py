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
import numpy       as np
import pyLIQTR.pest_interface.pest_python as pp

from pyLIQTR.ProblemInstances.ProblemInstance import ProblemInstance

class ElectronicStructure(ProblemInstance):
    """
    This ProblemInstance represents periodic electronic structure Hamiltonians generated using PEST. The Hamiltoninas generally take on the form 
    $$ 
    H = \sum_{p,q,\sigma} T(p-q) a^\dagger_{p,\sigma}a_{q,\sigma} + \sum_{p,\sigma} U(p) n_{p,\sigma} + \sum_{(p,\alpha)\neq(q,\beta)} V(p-q)n_{p,\alpha}n_{q,\beta}
    $$
    where $a^\dagger_{p,\sigma}$ and $a_{p,\sigma}$ are fermionic creation and annihilation operators on spatial orbital $p$ with spin $\sigma$ in {up,down}, and $n_{p,\sigma} = a^\dagger_{p,\sigma}a_{p,\sigma}$ is the number operator.

    Given the PEST files, this ProblemInstance can yield the information needed to encode the Hamiltonian using the LinearT block encoding. It also provides descriptions of the electronic structure system size with M_vals, the number of grid points along each spatial dimension, and N, the total number of orbitals.

    Keyword arguments:
    filenameH -- filename for PEST generated ham.hdf5 file containing the Hamiltonian coefficients
    filenmaeG -- filename for PEST generated grid.hdf5 file containing the basis grid
    """
    def __init__(self,filenameH:str, filenameG:str,**kwargs):

        self._filenameH = filenameH
        self._filenameG =filenameG
        self._H = pp.HamiltonianCoefficients()  
        try:
            self._H.open_grid(self._filenameG)
            self._H.open_hamiltonian(self._filenameH)
        except OSError:
            print(f"Unable to open {self._filenameH} and {self._filenameG}. Check filename and path. To generate new hdf5 files see 'pest-intro.ipynb' ")

        self._M_vals = self._H.n_points() # Number of grid points (orbitals) along each spatial dimension.
        self._N = np.prod(self._M_vals) # doesn't include spin
        self._model = "ElectronicStructure"

        super(ProblemInstance, self).__init__(**kwargs)

    def __str__(self):
        return f"{self._model}\r\n\tHamiltonian filename:{self._filenameH}\tGrid filename:{self._filenameG}\n\r\tN:{self._N}\tM_vals:{self._M_vals}"

    def n_qubits(self):
        # equal to number of basis functions including spin
        return int(2*self._N)

    def get_alpha(self,encoding:str='LinearT'):
        # sum of the absolute value of the hamiltonian coefficients
        # TODO: can this be more efficient
        if encoding == 'LinearT':
            T_pq_mag_sum = 0
            V_pq_mag_sum = 0
            U_p_mag_sum = 0
            for p in range(self._N):
                U_p_mag_sum += abs(self._H.U(p))
                for q in range(self._N):
                    T_pq_mag_sum += abs(self._H.T(p,q))
                    if p != q: # exclude p=q for V
                        V_pq_mag_sum += abs(self._H.V(p,q))    
            return T_pq_mag_sum + U_p_mag_sum + V_pq_mag_sum

    def yield_PauliLCU_Info(self, do_pad=0,return_as='arrays'):

        if return_as=='arrays':

            if do_pad:
                for __ in range(int(np.floor(do_pad/2))):
                    yield ((),'I',0)

            for p in range(self._N):

                for q in range(p+1,self._N):

                    xx_str = 'X' + ('Z'*abs(p-q)) + 'X'
                    yy_str = 'Y' + ('Z'*abs(p-q)) + 'Y'
                    idx_set = tuple(np.arange(p,p+abs(p-q)+2,1))
                    
                    # NOTE: return only non zero coefficients?
                    yield (idx_set, xx_str, self._H.c_xzx(p,q))
                    yield (idx_set, yy_str, self._H.c_xzx(p,q))
                    yield ((p,q), 'ZZ', self._H.c_zz(p,q))

                yield ((p,),'Z',self._H.c_z(p))

            if do_pad:
                for __ in range(int(np.ceil(do_pad/2))):
                    yield ((),'I',0)
        else:
            # TODO
            raise NotImplementedError()

    def yield_LinearT_Info(self,termSelect:str):
        
        if termSelect == 'T':
            # this corresponds to tilde(T)**2 in https://arxiv.org/pdf/1805.03662.pdf
            for p in range(self._N):
                term = 2*self._H.c_xzx(p,0) # factor of 2 since coefficients occur for each xzx/yzy term only once depending on p<q or p>q
                sign = (1-np.sign(term))/2 if np.sign(term) else 0
                yield (int(sign),abs(term))

        if termSelect == 'U':
            for p in range(self._N):
                term = self._H.c_z(p)
                sign = (1-np.sign(term))/2 if np.sign(term) else 0
                yield (int(sign),abs(term))

        if termSelect == 'V':
            for p in range(self._N):
                term = self._H.c_zz(p,0)
                sign = (1-np.sign(term))/2 if np.sign(term) else 0
                yield (int(sign),abs(term))