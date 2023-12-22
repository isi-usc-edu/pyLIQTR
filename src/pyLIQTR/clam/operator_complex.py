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
###    Operator Complex - Cell complex components tagged with operators      ###
###                                                                          ###
################################################################################


import  numpy                    as      np

from    pyLIQTR.clam.cell_complex        import  OperatorCell



############################################################################
###
###   CLASS     Site                                         (OperatorCell)
###
############################################################################
###
###   DESCRIPTION
###
###      Site representes a single site in a lattice or network (a 0-cell)
###      and tags this cell with operators.
###
###   CONSTRUCTOR ARGUMENTS
###
###      basis      -    Single basis element that specifies a point in the embedding
###                      space (e.g., [0,0,0] three dimensional Euclidean space.  If the
###                      problem is defined in terms of a model Hamiltonian these can just
###                      be points in a grid (e.g., only connectivity is important instead
###                      of metric properties).
###
###      ops        -    List of operators associated with the site.
###      cons       -    List of constants associated with the operators.
###     label       -    Label associated with site.  
###
###   EXAMPLE
###
###         Generate a single site with a Pauli X operator = -1.0*X
###
###         s0  =  Site([0,0],['X'],[-1.0])
###
###
###   RETURNS
###
###      Site       -    Instance of the Site Class
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
###      ops        -    List of perators associated with cell complex
###      cons       -    List of constants associated with the operators
###      props      -    Cell properties
###
###   PUBLIC ROUTINES  (see parent class Cell)
###
###   MAGIC ROUTINES  
###
###     __str__()
###
###           Pretty-printing of comprehensive cell attributes.
###

class Site(OperatorCell):

    def __init__(self,base,ops=None,cons=None,label=None):
        OperatorCell.__init__(self, np.array([base]),ops=ops,cons=cons,label=label)

    def __str__(self):

        if (self.label is not None):
            string = "\nSite "+str(self.label)+":\n"
        else:
            string = "\nSite:"

        string += OperatorCell.__str__(self)

        return(string)





############################################################################
###
###   CLASS     Edge                                            (Cell)
###
############################################################################
###
###   DESCRIPTION
###
###      Edge representes a single bond/edge/link in a graph/lattice/network 
###      (a 1-cell) and tags this edge with operators.
###
###   CONSTRUCTOR ARGUMENTS
###
###      basis      -    Pair of basis elements (or sites) that are linked by the Edge.  
###                      (see class Site and the example below)
###      ops        -    List of operators associated with the edge.  Tensor products
###                      are specified by pairs of single-site operators in a list. 
###      cons       -    List of constants associated with the operators.
###     label       -    Label associated with site.  
###
###   EXAMPLE
###
###         Generate a link between sities [0,0] and [0,1] in a two-dimensional
###         lattice using the operator -1.0*(X \otimes X + Y \otimes Y)
###
###             e0  =  Edge([0,0],[0,1],[['X','X'],['Y','Y']],[-1.0,-1.0])
###
###         Do the same if the sites have already been specified as s0 and s1
###
###             e0  =  Edge([s0,s1,[['X','X'],['Y','Y']],[-1.0,-1.0])
###
###   RETURNS
###
###      Edge       -    Instance of the Edge class
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
###      ops        -    List of perators associated with cell complex
###      cons       -    List of constants associated with the operators
###      props      -    Cell properties
###
###   PUBLIC ROUTINES  (see parent class Cell)
###
###   MAGIC ROUTINES  
###
###     __str__()
###
###           Pretty-printing of comprehensive cell attributes.
###

class Edge(OperatorCell):
    
    def __init__(self,s1,s2,ops=None,cons=None,label=None):
        OperatorCell.__init__(self, np.array([s1,s2]),ops=ops,cons=cons,label=label)

    def __str__(self):

        if (self.label is not None):
            string = "\nEdge "+str(self.label)+":\n"
        else:
            string = "\nEdge:"

        string += OperatorCell.__str__(self)

        return(string)





############################################################################
###
###   CLASS     Plaquette                                       (Cell)
###
############################################################################
###
###   DESCRIPTION
###
###      Catch-all cass for operators defined on a plaquette or higher-
###      dimensional object in a lattice/graph/network (e.g., on something
###      higher order than a 1-chain).
###
###   CONSTRUCTOR ARGUMENTS
###
###      basis      -    List of basis elements (or sites) that define  the plaquette.  
###                      (see class Site and the example below).  This can be a two 
###                      dimensional region, an extended string, or a higher volume.
###      ops        -    List of operators associated with the plaquette.  Tensor products
###                      are specified by pairs of single-site operators in a list. 
###      cons       -    List of constants associated with the operators.
###     label       -    Label associated with site.  
###
###   EXAMPLE
###
###         Generate a plaquette operator for sites [0,0], [0,1], [1,0], [1,1] 
###         in a two-dimensional 0.5*(Z \otimes Y \otimes Z \otimes I).  Order in the
###         tensor product follows the order of basis elements in the list <basis>.
###
###             p0  =  Plaquette([s0,s1,s2,s3],[['Z','X','X','Z']],[0.5])
###
###         where we assume that s0 - s3 specify the sites:
###
###             s0 ---- s1                      X ---- Y
###              |      |           ---\        |      |
###              |      |           ---/        I ---- Z
###             s3 ---- s2
###
###         Generate a string operator for sites s0,s1,s2,s3,s4,s5,s6
###         in a two-dimensional lattice 0.5*(X \otimes Z \otimes Z \otimes ... \otimes Z \otimes X).  
###         Order in the tensor product follows the order of basis elements in the list <basis>.
###
###             p0  =  Plaquette([s0,s1,s2,s3,s4,s5,s6],[['Z']+['X']*5+['Z'],[0.5])
###
###   RETURNS
###
###      Plaquette       -    Instance of the Plaquette class
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
###      ops        -    List of perators associated with cell complex
###      cons       -    List of constants associated with the operators
###      props      -    Cell properties
###
###   PUBLIC ROUTINES  (see parent class Cell)
###
###   MAGIC ROUTINES  
###
###     __str__()
###
###           Pretty-printing of comprehensive cell attributes.
###

class Plaquette(OperatorCell):
    def __init__(self,basis,ops=None,cons=None,label=None):
        OperatorCell.__init__(self, np.array(basis), ops=ops,cons=cons,label=label)

    def __str__(self):

        if (self.label is not None):
            string = "\nPlaquette "+str(self.label)+":\n"
        else:
            string = "\nPlaquette:"

        string += OperatorCell.__str__(self)

        return(string)
    



#############################################################################
###
###   CLASS     OString                                         (Cell)
###
#############################################################################
###
###   DESCRIPTION
###
###      Class for operator strings defined in a lattice.
###
###   CONSTRUCTOR ARGUMENTS
###
###      basis      -    List of basis elements (or sites) that define  the plaquette.  
###                      (see class Site and the example below).  This can be a two 
###                      dimensional region, an extended string, or a higher volume.
###      ops        -    List of operators associated with the plaquette.  Tensor products
###                      are specified by pairs of single-site operators in a list. 
###      cons       -    List of constants associated with the operators.
###     label       -    Label associated with site.  
###
###   EXAMPLE
###
###         Generate a string operator for sites s0,s1,s2,s3,s4,s5,s6
###         in a two-dimensional lattice 0.5*(X \otimes Z \otimes Z \otimes ... \otimes Z \otimes X).  
###         Order in the tensor product follows the order of basis elements in the list <basis>.
###
###             p0  =  Plaquette([s0,s1,s2,s3,s4,s5,s6],[['Z']+['X']*5+['Z'],[0.5])
###
###   RETURNS
###
###      Plaquette       -    Instance of the Plaquette class
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
###
###   PUBLIC ROUTINES  (see parent class Cell)
###
###   MAGIC ROUTINES  
###
###     __str__()
###
###           Pretty-printing of comprehensive cell attributes.
###
   
class OString(OperatorCell):
    def __init__(self,basis,ops=None,cons=None,label=None):
        OperatorCell.__init__(self, np.array(basis), ops=ops,cons=cons,label=label)

    def __str__(self):

        if (self.label is not None):
            string = "\nPlaquette "+str(self.label)+":\n"
        else:
            string = "\nPlaquette:"

        string += OperatorCell.__str__(self)

        return(string)