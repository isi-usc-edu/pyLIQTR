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

################################################################################
###                                                                          ### 
###          PEST_python  -  pyLIQTR's Python Interface to PEST              ###         
###                                        (A Herpetological Annoyance!)     ### 
################################################################################



import  numpy     as     np
import  h5py      as     h5py


bohr_to_angstrom = 0.529177







class HamiltonianCoefficients:

    def __init__(self):

        self._n_points = [0,0,0]

        return(None)

    def open_hamiltonian(self,filename):

        self.fh = h5py.File(filename, 'r')

        self.coeffs_fermi = True
        self.coeffs_pauli = self.fh['coeffs_pauli']

        self._T       =  self.fh['T']
        self._U       =  self.fh['U']
        self._V       =  self.fh['V']

        if (self.coeffs_pauli):
            self._c_i     =  self.fh['c_i']
            self._c_z     =  self.fh['c_z']
            self._c_zz    =  self.fh['c_zz']
            self._c_xzx   =  self.fh['c_xzx']

        self.n_qb    =  len(self._U)
        self.n_dist  =  len(self._T)
        


    def distance_dict(self,coord):

        return(self.dtable_dict[tuple(coord)])
    
    def gridpt_to_id(self,coord):
        return(self.crd_to_id[tuple(coord)])

    def open_grid(self,filename):

        self.fg = h5py.File(filename, 'r')
        self.dtable     =  self.fg['dtable']
        self.vecs       =  self.fg['vecs']
        self._n_points  =  self.fg['n_points']
        self.n_qb       =  np.prod(self._n_points)

        self.n_dist   =  len(self.dtable)


        self.points   = [(i,j,k) for i in range(self._n_points[0]) 
                            for j in range(self._n_points[1]) 
                                for k in range(self._n_points[2])]


        self.crd_to_id = dict((self.points[n],n) for n in range(self.n_qb))

        self.build_dtable_dict()


    def id_to_gridpt(self,idx):
        return(self.points[idx])

    def close_hamiltonian(self):
        self.fh.close()

    def close_grid(self):
        self.fg.close()

    def close(self):
        self.close_hamiltonian()
        self.close_grid()


    def build_dtable_dict(self):

        self.dtable_dict = {}

        for n in range(self.n_dist):
            self.dtable_dict[tuple(self.dtable[n])] = n

        for n in range(self.n_dist):
            self.dtable_dict[tuple(-self.dtable[n])] = n



    def pair_coeff_lookup(self,p,q,C):

        dist = (np.array(self.points[p]) 
                    - np.array(self.points[q]))

        tdist = tuple(dist)

        C_pq = C[self.dtable_dict[tdist]]


        return(C_pq)
    


    def T(self,p,q):
        return(self.pair_coeff_lookup(p,q,self._T))

    def V(self,p,q):
        return(self.pair_coeff_lookup(p,q,self._V))
    
    def U(self,p):
        return(self._U[p])
    
    def c_i(self):
        val = self._c_i[()]
        return(val)
    
    def c_z(self,p):
        return(self._c_z[p])

    def c_zz(self,p,q):
        return(self.pair_coeff_lookup(p,q,self._c_zz))
    
    def c_xzx(self,p,q):
        return(self.pair_coeff_lookup(p,q,self._c_xzx))
    

    def pair_coeff_array(self,T):
        T_pq = np.zeros(self.n_qb,self.n_qb)
        for p in range(self.n_qb):
            for q in range(self.n_qb):
                T_pq[p,q] = self.pair_coeff_lookup(p,q,T)
        return(T_pq)
    
    def n_points(self):
        return(np.array(list(self._n_points)))
          
    def T_arr(self):
        return(self.pair_coeff_array(self._T))
    
    def V_arr(self):
         return(self.pair_coeff_array(self._V))
    
    def U_arr(self):
         return(np.array(self._U[p] for p in range(self.n_qb)))
    
    def zz_arr(self):
        return(self.pair_coeff_array(self._c_zz))
    
    def xzx_arr(self):
         return(self.pair_coeff_array(self._xzx))
    
    def z_arr(self):
         return(np.array(self._c_z[p] for p in range(self.n_qb)))
    





##
##
class dpw_jw_alternating_spin():

    def __init__(self,ham):

        self._ham           =   ham
        self.dims_physical  =  self._ham.n_points() 

        self.dims           =  [self.dims_physical[0], self.dims_physical[1], self.dims_physical[2]]
    
    def xzx(self,p,q):

        terms = []
        
        if (p != q):

            terms.append([(2*p,2*q),"XZX",self._ham.c_xzx(p,q)])
            terms.append([(2*p,2*q),"YZY",self._ham.c_xzx(p,q)])

            terms.append([(2*p+1,2*q+1),"XZX",self._ham.c_xzx(p,q)])
            terms.append([(2*p+1,2*q+1),"YZY",self._ham.c_xzx(p,q)])

        return(terms)
    
    
    def zz(self,p,q):

        terms = []

        if (p == q):

            terms.append([(2*p,2*p+1),"ZZ",self._ham.c_zz(p,p)])      # (p,u) ,(p,d)

        else:
            
            terms.append([(2*p,2*q),"ZZ",self._ham.c_zz(p,q)])          # (p,d) ,(q,d)
            terms.append([(2*p,2*q+1),"ZZ",self._ham.c_zz(p,q)])        # (p,d) ,(q,u)

            terms.append([(2*p+1,2*q),"ZZ",self._ham.c_zz(p,q)])        # (p,u) ,(q,d)
            terms.append([(2*p+1,2*q+1),"ZZ",self._ham.c_zz(p,q)])      # (p,u) ,(q,u)


        return(terms)

    def z(self,p):
            
        terms = []

        terms.append([(2*p,),"Z",self._ham.c_z(p)])          
        terms.append([(2*p+1,),"Z",self._ham.c_z(p)])       

        return(terms)   








###
###  pw_to_dpw_cutoff(E_cut, node_scale=1.0,unit="bohr")
### ======================================================================
###   
###   Convert plane-wave cutoff <E_cut> to DPW lattice spacing.  Unit of 
###   spacing is given by <unit>.  The keyword <node_scale> rescales the 
###   real-space DPW grid to give additional points (i.e., node_scale=0.5
###   halves the lattice spacing).  Energy is given in Rydberg units.
###


def pw_to_dpw_cutoff(E_cut, node_scale=1.0,unit="bohr"):

    E_cut_ry  =  E_cut/2

    l_cut_bohr      =  np.sqrt(2*np.pi**2/E_cut_ry)*node_scale
    
    if (unit.lower() == "angstrom"):
        return(l_cut_bohr*bohr_to_angstrom)
    else:
        return(l_cut_bohr)
    

###
###  dpw_to_pw_cutoff(a0, node_scale=1.0,unit="bohr")
### ======================================================================
###   
###   Convert a real-space lattice spacing <a0> to a "pure" plane-wave energy 
###   cutoff.  See pw_to_dpw_cutoff for arguments.  Energy is returned in 
###   Rydberg units.

def dpw_to_pw_cutoff(a0, node_scale=1.0,unit="bohr"):


    if (unit.lower() == "bohr"):
        E_cut_ha      =      (2*np.pi**2)/((a0/node_scale)**2)
    if (unit.lower() == "angstrom"):
        E_cut_ha      =      (2*np.pi**2)/((a0/bohr_to_angstrom/node_scale)**2)

    E_cut_ry = 2*E_cut_ha

    return(E_cut_ry)