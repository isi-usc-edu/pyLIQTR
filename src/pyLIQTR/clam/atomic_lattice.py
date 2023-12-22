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

############################################################################
###
###   CLASS     Atom                                            (Cell)
###
############################################################################
###
###   DESCRIPTION
###
###      Class to represent a single-atom site in a lattice
###
###   CONSTRUCTOR ARGUMENTS
###
###      basis      -    Single basis element that specifies a point in the embedding
###                      space (e.g., [0,0,0] three dimensional Euclidean space. This
###                      class is intended for use in cases where the metric *is* important,
###                      such as when specifying a primitive cell.
###     props       -    Properties to be associated with the atomic site; e.g., for 
###                      writing to a pdb or extended xyz file (isotope, etc.)
###     label       -    Label associated with site.  This is used for the element name.
###     element     -    Keyword alias for the label.
###
###   EXAMPLE
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

class Atom(Cell):

    def __init__(self,basis,label=None,props=None,element=None):

        basis = listify(basis, d=2)

        if (label is None):
            label = 'X'
        if (element is not None):
            label = element
                
        Cell.__init__(self, np.array([basis]),props=props, label=label)



class AtomicLattice(Lattice):

    def __init__( self, 
                  vectors,
                  size=None,
                  unit=None,
                  periodic=None,
                  do_wrap=None,
                  cell_vectors=None,
                  centered=True        ):

        Lattice.__init__(self,vectors,size=size,unit=unit,periodic=(False,False,False))

        self.build(size)

        if (cell_vectors is None):
            self.cell_vecs = []
            for n in range(len(vectors)):
                self.cell_vecs.append(size[n]*vectors[n])
        else:
            self.cell_vecs = np.array(cell_vectors)
    
        if (do_wrap is None):
            self.do_wrap  = []
            for n in range(self.dim):
                self.do_wrap.append(False)
        else:
            self.do_wrap = do_wrap

#        Set labels to something generic like x if none are specified but unit is specified?
#       or should this be done in geometry
        # if (unit is None):
        #     self.labels = 


        self.build_geometry(do_wrap)
    
        return
    

    def wrap(self,coords):

        new_coords = []

        for x in coords:
            for n in range(len(x[0])): 
                if ( self.do_wrap[n] == True):
                    if (x[0][n] >= self.cell_vecs[n][n]):
                        x[0][n] -= self.cell_vecs[n][n]

                    if (x[0][n] < 0.0):
                        x[0][n] += self.cell_vecs[n][n]

            new_coords.append(x[0])

        return(new_coords)


    def build_geometry(self,do_wrap=None):

        labels = self.labels[0]

        coords = []
        for coord in self.skeletons[0]:
            coords.append(coord)

        new_coords = self.wrap(coords)

        self.geometry = Geometry(coords=new_coords,labels=labels)

        return