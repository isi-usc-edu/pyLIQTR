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
###       CellComplex - Routines to generate cell / similical complexes      ###
###                                                                          ###
################################################################################
###
###      NOTE:  Routes in this module are specified with a heavy abuse 
###             of notation and terminology.  
###



import  numpy           as      np
from    pyLIQTR.clam.utilities  import  listify, tupleize, tupleize_sites



############################################################################
###
###   CLASS     Cell
###
############################################################################
###
###   DESCRIPTION
###
###      Defines a K-th order cell in a cell complex, where the topological 
###      dimension K is defined by the speicified basis elements (e.g., the
###      designation must be internally consistent).
###
###   CONSTRUCTOR ARGUMENTS
###
###      basis      -    Basis elements that specify the cell. Dimensionality
###                      is defined by the basis; to borrow from simplicial 
###                      complexes, a 0-cell is a single point, a 1-cell is 
###                      a line, a 2-cell is face / plaquette, etc.  
###
###      props      -    List of properties that are tagged to the cell. These
###                      are formally arbitrary though derived classes can
###                      associate Pauli / Fermionic operators with the cell
###                      (e.g., operators acting on a lattice site or along 
###                      a bond.)   (KEYWORD, OPTIONAL)
###
###     boundary    -    List of cells that constitute the boundary of the current
###                      cell.  For a 2-chain these would be bounding 1-chains, etc.
###                      (KEYWORD, OPTIONAL)
###
###     label       -    Label associated with cell.  Essentially a one-off property that
###                      is distinguished from other properties for referrential simplicity.
###                      (KEYWORD, OPTIONAL)
###
###
###   RETURNS
###
###      Cell       -    Instance of the Cell Class
###
###   REQUIRES
###
###      numpy          as  np
###      utilities      import listify, tupleize, tupleize_sites
###
###   PUBLIC MEMBERS
###
###      basis      -    Basis elements that define the cell
###      dim        -    Dimension of basis (number of basis elements)
###      edim       -    Dimension of the embedding space; e.g. the 
###                      dimension of a single basis element.  
###      K          -    Topological dimension of the cell; equal to 
###                      (dim - 1)
###      label      -    Label associated with the cell
###      clabel     -    Additional label for internal use by user-defined
###                      routines
###      props      -    Cell properties
###      boundary   -    Cell boundary
###
###   PUBLIC ROUTINES
###
###      tupleize()    
###        
###           Convert basis elements of the cell from python lists / 
###           numpy arrays to tuples.
###
###     set_label(label)
###
###           Set the label of the cell to label
###
###   MAGIC ROUTINES  
###
###     __str__()
###
###           Pretty-printing of comprehensive cell attributes.
###

class Cell():


    ##  __init__()
    ## ---------------------------------------
    ##  Cell constructor. 
    ##

    def __init__(self,basis,props=None, boundary=None,label=None):

        self.basis        =  []                   # Cell basis vectors
        self.basis_cells  =  []                   # Basis cells


        # Determine if basis is passed as tuples or as 
        # other cells.

        if ( isinstance(basis[0], Cell) ):
            self.basis = []
            for cell in basis:
                self.basis.append(cell.basis)
                self.basis_cells.append(cell)
        elif ( isinstance(basis[0],tuple) or 
               isinstance(basis[0],list)  or 
               isinstance(basis[0],np.ndarray) ):
                self.basis        =  np.array(basis)
        else:
            raise Exception("Cell:  Invalid basis specification.")


        self.dim       =  len(self.basis)           # Dimension basis 
                                                    # (e.g., number of basis elements)
        self.edim      =  len(self.basis[0])        # Dimension of embedding space 
                                                    # (e.g., basis array dimension)
        self.K         =  len(self.basis)-1         # Topological dimension of cell
        self.label     =  label                     # Cell label
        self.clabel    =  None
        self.props     =  props                     # Cell properties
        self.boundary  =  boundary                  # Boundary cells            



    ##  tupleize()
    ## ---------------------------------------
    ##  Convert basis elements of the cell
    ##  from python lists / numpy arrays to 
    ##  tuples. 
    ##

    def tupleize(self):
        self.basis = tupleize_sites(self.basis)



    ##  set_label(label)
    ## ---------------------------------------
    ##  Set the label for a given cell to 
    ##  <label>.  Useful for routines that 
    ##  relabel cells within a complex by 
    ##  connectivity or some other property.
    ##

    def set_label(self,label):
        self.label = label



    ##  __str__()
    ## ---------------------------------------
    ##  Canonical printing routines for cell
    ##  class instances.
    ##

    def __str__(self):

        string = str(self.K)+"-Cell"
        
        if (self.label is not None):
            string += " ("+str(self.label)+")"
        
        string += ": [ "
        string +=  "K="       +  str(self.K)
        string +=  ", dim="    +  str(self.dim)
        string +=  ", edim="  +  str(self.edim)
        string +=  ", Basis="  +  str(self.basis)

        if (self.props is not None):
            string += ", Prop="+str(self.props)

        string += " ]"

        return(string)
    




############################################################################
###
###   CLASS     CellComplex
###
############################################################################
###
###   DESCRIPTION
###
###      Associates a set of cells into a cell complex.  Comprised of cells
###      up to a maximal topological dimension K.  This class is globally
###      tracked so that each complex has a unique index (for parsing cells 
###      shared between overlapping complexes)
###
###   CONSTRUCTOR ARGUMENTS
###
###     NOTE:  Either a <dim> or a <K> keyword argument must be specified 
###            at instantiation or an exception will be thrown.  Setting one 
###            attribute will automatically set the other.
###
###      K          -    Maximum topological dimension of complex (KEYWORD, QUASI-REQUIRED).
###      dim        -    Maximal "natural" dimension of the cell complex. Equals (K+!)
###                      (KEYWORD, QUASI-REQUIRED).
###
###     label       -    Label associated with the cell complex.
###
###   RETURNS
###
###      CellComplex  -    Instance of the CellComplex class
###
###   REQUIRES
###
###      numpy          as  np
###      utilities      import listify, tupleize, tupleize_sites
###
###   PUBLIC MEMBERS
###
###      skeletons   -   (K+1)-dimensional array, where each element is a list
###                      of cells in the k-skeleton.
###      dim        -    Dimension of basis (number of basis elements)
###      K          -    Topological dimension of the cell; equal to (dim - 1)
###
###   PUBLIC ROUTINES
###
###      add(cell_set) 
###        
###           Add cells in the list <cell_set> to the k-skeleton of the 
###           complex.  The topological dimension k is automatically determined
###           for each cell in the list.
###
###   MAGIC ROUTINES  
###
###     __str__()
###
###           Pretty-printing of comprehensive cell attributes.
###

class CellComplex():

    __complex_count_global = 0

    # Global to number each cell complex as it is added; not currently used but
    # slated for future use.

    def __init__(self,K=None,dim=None,label=None):

        # self.complex_id = deepcopy(self.__complex_count_global)
        # self.__complex_count_global += 1
        #
        # Add unique label for position in each complex

        self.skeletons  =  []
        self.label      =  label

        if   ( (K is not None) and (dim is None) ):
            self.K     =  K
            self.dim   =  K+1
        elif ( (K is None) and (dim is not None) ):
            self.dim   =  dim
            self.K     =  dim-1
        elif ( (K is not None) and (dim is not None) ): 
            raise Exception("Cannot specify both dim and K for CellComplex.")
        else:
            raise Exception("Must specify either K or dim for CellComplex.")

        for n in range(self.dim):
            self.skeletons.append([])
    

    
    ##  add(S_set)
    ## 
    ##      Add cells in the list S_set to a CellComplex instance.
    ##   
     
    def add(self,S_set):                         # add a cell to the complex

        S_set = listify(S_set,d=1)

        for S in S_set:

            if (S.K > self.K):
                raise Exception("Cell dimension ("+str(S.K)+") is greater that maximal skeleton dimension ("+str(self.K)+")")

            if (S.clabel is None):

                cell_label    =  "c."+str(S.K)+"."+str(len(self.skeletons[S.K])+1)
                S.cell_label  =  cell_label

            # check if subcells are already part of complex; if not, we add them.

            self.skeletons[S.K].append(S)
            
        return
    


    ##  __str__()
    ## 
    ##      Canonical printing routines for CellComplex instances.
    ##

    def __str__(self):

        string = ""

        for k in range(self.dim):

            skeleton  = self.skeletons[k]

            string += "\n"+str(k)+"-Skeleton:"+"\n\n"

            for cell in skeleton:
                string += "\t"+cell.__str__()+"\n"

        return(string)





############################################################################
###
###   CLASS     OperatorCell
###
############################################################################
###
###   DESCRIPTION
###
###      Cell explicitly associated with quantum operators. Derived from 
###      Cell class.
###
###   CONSTRUCTOR ARGUMENTS
###
###      basis      -    Basis elements used to specify the cell.  Dimensionality
###                      is defined by it's basis; to borrow from the simplicial 
###                      context, a 0-cell is a single point, a 1-cell is a line, 
###                      a 2-cell is face / plaquette, etc.   (REQUIRED)
###
###      ops        -    List of operators associated with cell.
###      cons       -    List of constants associated with the operators.
###
###     boundary    -    List of cells that constitute the boundary of the current
###                      cell.  For a 2-chain these would be bounding 1-chains, etc.
###                      (KEYWORD, OPTIONAL)
###
###     label       -    Label associated with cell.  Essentially a one-off property that
###                      is distinguished from other properties for referrential simplicity.
###                      (KEYWORD, OPTIONAL)
###
###
###   RETURNS
###
###      Cell       -    Instance of the Cell Class
###
###   REQUIRES
###
###      numpy          as  np
###      utilities      import listify, tupleize, tupleize_sites
###
###   PUBLIC MEMBERS (see also parent class Cell)
###
###      basis      -    Basis elements that define the cell
###      dim        -    Dimension of basis (number of basis elements)
###      K          -    Topological dimension of the cell; equal to 
###                      (dim - 1)
###      label      -    Label associated with the cel
###      ops        -    List of operators associated with cell complex
###      cons       -    List of constants associated with the operators
###      props      -    Cell properties
###      boundary   -    Cell boundary
###
###   PUBLIC ROUTINES  (see parent class Cell)
###
###   MAGIC ROUTINES  
###
###     __str__()
###
###           Pretty-printing of comprehensive cell attributes.
###

class OperatorCell(Cell):
    
    def __init__(self,basis,ops=None,cons=None,label=None,boundary=None):

        basis = listify(basis, d=2)

        #EDIT: Changed d=2 to d=1 in the ops listify
        ops   = listify(ops,   d=1)
        cons  = listify(cons,  d=2)

        if ( (ops is not None) and (cons is not None) ):
            props = []
            for op,c in zip(ops,cons):
                props.append([op,c])
        else:
            props = None
                
        Cell.__init__(self, np.array(basis), props=props,label=label,boundary=boundary)



    def ops(self):
        return(list([prop[0] for prop in self.props]))



    def cons(self):
        return(list([prop[1] for prop in self.props]))



    def __str__(self):

        string  = "\n"
        string += "\tBasis:\t\t"


        for n in range(len(self.basis)):
            string += "s"+str(n)+" = "+str(self.basis[n])+"   "

        # Fix hanging +
        if (self.props[0][0] is not None):
            string += "\n\tOperators:\t"
            for prop in self.props:
                string += "("+str(prop[1])+")*"+str(prop[0])
                if (self.props.index(prop) != len(self.props)-1):
                    string += "  +  "
            string += "\n"

        return(string)



