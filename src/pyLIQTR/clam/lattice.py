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


################################################################################
###                                                                          ###
###      Lattice - Routines to generate and manipulate periodic systems      ###
###                                                                          ###
################################################################################



import  numpy                    as      np

from    copy                     import  deepcopy
from    pyLIQTR.clam.cell_complex        import  Cell, CellComplex


############################################################################
###
###   CLASS     UnitCell                                    (CellComplex)
###
############################################################################
###
###   DESCRIPTION
###
###      Class to define the UnitCell in a periodic system.  Derived from 
###      CellComplex.
###
###   CONSTRUCTOR ARGUMENTS
###
###      sites      -    List containing embers of the unit cell.  
###                      Sites, edges, plaquettes, etc may all be included.
###
###   EXAMPLE
###
###         Generate a unit cell containing the site s0 and edges e0,e1
###         which will connect  different cells in the periodic systems
###
###         s0  =  UnitCell([s0,e0,e1])
###
###
###   RETURNS
###
###      Site       -    Instance of the UnitCell Class
###
###   REQUIRES
###
###      numpy          as  np
###      utilities      import listify, tupleize, tupleize_sites
###
###   PUBLIC MEMBERS (see also parent class Cell)
###
###      skeletons   -   (K+1)-dimensional array, where each element is a list
###                      of cells in the k-skeleton.  The 0-skeleton 
###                      (e.g, skeletons[0]) is a list of points, the 1-skeleton
###                      (e.g., skeletons[1]) is a list of edges, the 
###                      2-skeleton is a list of three-site operators (defined using
###                      plaquetttes for everything higher-order than an edge).  
###      dim        -    Dimension of basis (number of basis elements)
###      K          -    Topological dimension of the cell; equal to (dim - 1)
###
###   PUBLIC ROUTINES  (see parent class CellComplex)
###
###   MAGIC ROUTINES  
###
###     __str__()
###
###           Pretty-printing of comprehensive unit cell members and attributes
###           (inherited from cell complex).
###

class UnitCell(CellComplex):

    def __init__(self,sites):

        self.Kmax = 0

        for site in sites:
            if (site.K > self.Kmax):
                self.Kmax = site.K

        self.dim = self.Kmax + 1

        CellComplex.__init__(self,self.Kmax)

        for site in sites:
            self.add(site)





############################################################################
###
###   CLASS     Lattice                                    
###
############################################################################
###
###   DESCRIPTION
###
###      Parent class for periodic systems.
###      
###
###   CONSTRUCTOR ARGUMENTS
###
###      vectors      -   List of lattice vectors.  Defines dimensionality
###                       of the lattice (REQUIRED).
###      size=        -   Size of the lattice as a list or tuple of integers. 
###                       Defined as number repetitions of the unit cell in each
###                       dimension.
###      unit=        -   Specifies the unit cell using a UnitCell object.  Optional
###                       keyword; a single-point unit cell is used if this argument
###                       is not provided.
###      periodic=    -   Tuple or list of booean values indiciating if the system is
###                       periodic in each dimension.  Default is a completely aperiodic
###                       system (all false).
###      overhang=    -   Keyword indicating how to handle dangling edges/plaquettes/...
###                       in an aperiodic system:
###
###                             'ignore'    -   retain hanging chains even if no lattice
###                                             site is present.
###                             'drop'      -   drop hanging chains
###                             'fill'      -   add in sites that would be linked by these
###                                             edges by referencing back to the unit cell.
###
###   EXAMPLE
###
###         Generate a lattice based on the unit cell UC with lattice vectors [[0,2],[2,0]]
###         for a 4 x 4 supercell in a fully-periodic system
###
###             lattice_vectors = np.array([[0,2],[2,0]])
###
###             LT  =  Lattice(lattice_vectors,unit=UC,periodic=(True,True),size=(4,4))
###             LT.build()
###
###         Rebuild the same lattice for an 8x8 system:
###
###             LT.build((8,8))
###
###   RETURNS
###
###      Site       -    Instance of the UnitCell Class
###
###   REQUIRES
###
###      numpy          as  np
###      utilities      import listify, tupleize, tupleize_sites
###
###   PUBLIC MEMBERS (see also parent class Cell)
###
###      skeletons   -   (K+1)-dimensional array, where each element is a list
###                      of cells in the k-skeleton.  The 0-skeleton 
###                      (e.g, skeletons[0]) is a list of points, the 1-skeleton
###                      (e.g., skeletons[1]) is a list of edges, the 
###                      2-skeleton is a list of three-site operators (defined using
###                      plaquetttes for everything higher-order than an edge).  
###      dim        -    Dimension of basis (number of basis elements)
###      K          -    Topological dimension of the cell; equal to (dim - 1)
###
###   PUBLIC ROUTINES  (see parent class CellComplex)
###
###   MAGIC ROUTINES  
###
###     __str__()
###
###           Pretty-printing of comprehensive unit cell members and attributes
###           (inherited from cell complex).
###

class Lattice():

    def __init__( self, 
                  vectors,
                  size=None,
                  unit=None,
                  periodic=None,
                  overhang='ignore'   ):

        self.dim               =  len(vectors[0])
        self.basis_vectors     =  np.array(vectors)
        self.size              =  size

        self.n_sites           =  None

        self.overhang          =  overhang 
 
        ## Set array denoting periodicity
        ##
        if (periodic == None):
            self.periodic  =  np.full(self.dim, False)
        else:
            self.periodic  =  deepcopy(periodic)


        ## Fill unit cell with a single site if none was specified
        ##
        if (unit is None):
 #           self.unit  =  UnitCell([Site([0]*self.dim)])
            self.unit  =  UnitCell([Cell([[0]*self.dim])])

        else:
            self.unit  =  unit

        self.skeletons         =  []        # Array to hold k-skeletons 
        self.props             =  []        # Array to hold properties 
        self.labels            =  []        # Array to hold labels

        self.__populate_skeletons()


    def __populate_skeletons(self):
        for n in range(self.unit.dim):
            self.skeletons.append([])
            self.props.append([])
            self.labels.append([])



    def purge_skeletons(self):
        self.skeletons  =  []
        self.props      =  []
        self.labels     =  []
        self.__populate_skeletons()





    ##
    ##  build()
    ## ==================================================
    ##  Driver to generate lattice with repetitions given
    ##  by tuple reps.
    ##

    def build(self,size=None,overhang=None):

        if (overhang is not None):
            self.overhang = overhang

        self.purge_skeletons()

        if (size is not None):
            self.size = np.array(size,dtype=int)
        elif (self.size is None):
            raise Exception("Must specify number of lattice repetitions [ size = (n_0, n_1, ... n_d) ].")
        
        self.max_disp = np.zeros(self.dim)
        for n in range(self.dim):
            self.max_disp = self.max_disp + (size[n])*self.basis_vectors[n]

        for k in range(self.unit.dim):
            if (len(self.unit.skeletons[k]) != 0):
                self.__generate_lattice(k_level=k)


    def num_sites(self):
        if (self.n_sites is None):
            self.n_sites = len(self.skeletons[0])

        return(self.n_sites)            



    ##
    ##  displace_unit_cell()
    ## ==================================================
    ##  Displace chains of index <k_level> in unit cell
    ##  by multiple <Nxyz> of lattice basis vectors.
    ##

    def __displace_unit_cell(self,Nxyz,k_level=0):

        new_chain  =  []
        new_prop   =  []

        for cell in self.unit.skeletons[k_level]:

            new_basis  =  []
            drop_flag  =  False

            for point in cell.basis:

                new_point = deepcopy(point)

                for d in range(len(Nxyz)):
                    new_point  = new_point + Nxyz[d]*self.basis_vectors[d]
                
                for n, pbc_flag in enumerate(self.periodic):

                    if ( pbc_flag and (new_point[n] > self.max_disp[n]-1) ):
                        new_point[n] = new_point[n] - self.max_disp[n]

                    elif ( (not pbc_flag) and (new_point[n] > self.max_disp[n]-1) and 
                          (self.overhang.lower() == 'drop') ):

                        drop_flag = True 

                    elif ( (not pbc_flag) and (new_point[n] > self.max_disp[n]-1) and 
                          (self.overhang.lower() == 'fill')):

                        raise NotImplementedError


                        # if (self.sites is not None):
                        #     # find wrapback into unit and add an associated point to points and
                        #     # to properties
                        #     continue

                    elif (self.overhang.lower == 'ignore'):
                        continue
      
                new_basis.append(new_point)

            if (not drop_flag): 
                self.skeletons[k_level].append(new_basis)
                self.props[k_level].append(cell.props)
                self.labels[k_level].append(cell.label)





    ##
    ##  __generate_lattice()
    ## ==================================================
    ##  Recursively generate lattce of sites, edges,
    ##  or plaquettes by displacement of the unit cell.
    ##

    def __generate_lattice(self,n=0,x0=None,k_level=0):

        is_base = False

        if ( np.prod(x0 == None) ):

            x0       =  np.zeros(self.dim)
            is_base  =  True

        for k in range(self.size[n]):

            x    = x0
            x[n] = k

            if (n < self.dim-1):
                self.__generate_lattice(n=n+1,x0=x,k_level=k_level)
            else:
                self.__displace_unit_cell(x,k_level=k_level)

        if (is_base == True):
            return
        




