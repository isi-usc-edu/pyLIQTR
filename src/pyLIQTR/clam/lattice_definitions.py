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
###                Lattice Definitions - Predefined Unit Cells               ###
###                                                                          ###
################################################################################



import  numpy                  as      np
from    pyLIQTR.clam.operator_lattice  import  OperatorUnitCell



######                            #############################################
######  2D LATTICES               #############################################
######                            #############################################



##  SquareLattice()            
## -------------------------------------------------------------
##
##  Family of square (2d) lattices.  Arguments:
##
##      td       =   Term dictionary
##      double   =   Flag to double lattice and apply terms from
##                   onsite= dictionary to these pairs.
##      onsite   =   Dictionary of pair terms for doubled sites
##      type     =   
##                      'default'         :  Same as 'regular'
##                      'regular'         :  Single site type per unit cell
##
##                                                   A --- A
##                                                   |     | 
##                                                   A --- A
##
##                      'striped-2'       :  Alternating striped bands
##
##                                                   A --- B
##                                                   |     | 
##                                                   A --- B
##
##                      'checkerboard-2'  :  Checkerboard lattice, two sites
##
##                                                   A --- B
##                                                   |     | 
##                                                   B --- A
##
##                      'checkerboard-4'  :  Checkerboard lattice, four sites
##
##                                                   A --- B
##                                                   |     | 
##                                                   D --- C
##

class SquareLattice(OperatorUnitCell):

    def __init__(self,td,double=False,onsite=None,type='default',dimension=2):

        self.__sites        =  {}
        self.__dummy_sites  =  {}
        self.__edges        =  []

        self.__vectors      =  None

        self.type_string    =  "SquareLattice"

        if (type == 'default'):
            self.subtype_string =  'regular'
        else:
            self.subtype_string  =  type

        if (self.subtype_string == 'regular'):
            self.__square_lattice_regular()
        elif (self.subtype_string == 'striped-2'):
            self.__square_lattice_striped_2()
        elif (self.subtype_string == 'checkerboard-2'):
            self.__square_lattice_checkerboard_2()
        elif (self.subtype_string == 'checkerboard-4'):
            self.__square_lattice_checkerboard_4()
        else:
            raise NotImplementedError
        
        OperatorUnitCell.__init__( self,
                               td,
                               sites=self.__sites,
                               edges=self.__edges,
                               dummy_sites=self.__dummy_sites,
                               vectors=self.__vectors,
                               double=double,
                               onsite=onsite )

        
    def __square_lattice_regular(self):

        self.__sites[(0,0)] = 'A'

        self.__dummy_sites[(0,1)] = 'A'
        self.__dummy_sites[(1,0)] = 'A'

        self.__edges.append( ( (0,0), (0,1) ) )
        self.__edges.append( ( (0,0), (1,0) ) ) 

        self.__vectors =  [[1,0],[0,1]]
   

    def __square_lattice_striped_2(self):

        self.__sites[(0,0)] = 'A'
        self.__sites[(1,1)] = 'B'
        self.__sites[(0,1)] = 'A'
        self.__sites[(1,0)] = 'B'

        self.__dummy_sites[(0,2)] = 'A'
        self.__dummy_sites[(2,0)] = 'B'
        self.__dummy_sites[(2,1)] = 'A'
        self.__dummy_sites[(1,2)] = 'B'

        self.__square_edges_4()

        self.__vectors  =  [[2,0],[0,2]]        


    def __square_lattice_checkerboard_2(self):

        self.__sites[(0,0)] = 'A'
        self.__sites[(1,1)] = 'A'
        self.__sites[(0,1)] = 'B'
        self.__sites[(1,0)] = 'B'

        self.__dummy_sites[(0,2)] = 'A'
        self.__dummy_sites[(2,0)] = 'A'
        self.__dummy_sites[(2,1)] = 'B'
        self.__dummy_sites[(1,2)] = 'B'

        self.__square_edges_4()

        self.__vectors  =  [[2,0],[0,2]]        



    def __square_lattice_checkerboard_4(self):

        self.__sites[(0,0)] = 'A'
        self.__sites[(1,1)] = 'C'
        self.__sites[(0,1)] = 'B'
        self.__sites[(1,0)] = 'D'

        self.__dummy_sites[(0,2)] = 'A'
        self.__dummy_sites[(2,0)] = 'B'
        self.__dummy_sites[(2,1)] = 'C'
        self.__dummy_sites[(1,2)] = 'D'

        self.__square_edges_4()

        self.__vectors  =  [[2,0],[0,2]]        


    def __square_edges_4(self):

        self.__edges.append( ( (0,0), (0,1) ) )
        self.__edges.append( ( (0,0), (1,0) ) ) 
        self.__edges.append( ( (0,1), (1,1) ) ) 
        self.__edges.append( ( (1,0), (1,1) ) ) 

        self.__edges.append( ( (0,1), (0,2) ) ) 
        self.__edges.append( ( (1,0), (2,0) ) ) 
        self.__edges.append( ( (1,1), (2,1) ) ) 
        self.__edges.append( ( (1,1), (1,2) ) ) 





##  TriangularLattice()            
## -------------------------------------------------------------
##
##  Family of triangular (2d) lattices.  Arguments:
##
##      td       =   Term dictionary
##      double   =   Flag to double lattice and apply terms from
##                   onsite= dictionary to these pairs.
##      onsite   =   Dictionary of pair terms for doubled sites
##      type     =   
##                      'default'         :  Same as 'triangle-1'
##                      'triangle-1'      :  Single site type per triangular cell
##
##                                                      A
##                                                    /   \ 
##                                                   A --- A
##
##                      'triangle-3'      :  Three site types per triangular cell
##
##                                                      A
##                                                    /   \ 
##                                                   B --- C
##

class TriangularLattice(OperatorUnitCell):

    def __init__(self,td,double=False,onsite=None,type='default',dimension=2):

        self.__sites        =  {}
        self.__dummy_sites  =  {}
        self.__edges        =  []

        self.__vectors      =  None

        self.type_string    =  "Triangular Lattice"

        if (type == 'default'):
            self.subtype_string =  'triangle-1'
        else:
            self.subtype_string  =  type

        ## Should actually use a dict of these things
        ##
        if (self.subtype_string == 'triangle-1'):
            self.__triangular_lattice_1()
        elif (self.subtype_string == 'triangle-3'):
            self.__triangular_lattice_3()
        else:
            raise NotImplementedError
        
        OperatorUnitCell.__init__( self,
                               td,
                               sites=self.__sites,
                               edges=self.__edges,
                               dummy_sites=self.__dummy_sites,
                               vectors=self.__vectors,
                               double=double,
                               onsite=onsite )


    def __triangular_lattice_1(self):


        self.__sites[(0,0)] = 'A'
        self.__sites[(0,1)] = 'A'
        self.__sites[(0,2)] = 'A'

        self.__sites[(1,0)] = 'A'
        self.__sites[(1,1)] = 'A'
        self.__sites[(1,2)] = 'A'

        self.__dummy_sites[(0,3)] = 'A'
        self.__dummy_sites[(1,3)] = 'A'
        self.__dummy_sites[(2,0)] = 'A'


        self.__dummy_sites[(2,1)] = 'A'
        self.__dummy_sites[(2,2)] = 'A'
        self.__dummy_sites[(2,3)] = 'A'

        self.__triangle_edges_3()

        self.__vectors = [[2,0],[0,3]]


    def __triangular_lattice_3(self):

        self.__sites[(0,0)] = 'A'
        self.__sites[(0,1)] = 'B'
        self.__sites[(0,2)] = 'C'

        self.__sites[(1,0)] = 'C'
        self.__sites[(1,1)] = 'A'
        self.__sites[(1,2)] = 'B'

        self.__dummy_sites[(0,3)] = 'A'
        self.__dummy_sites[(1,3)] = 'C'


        self.__dummy_sites[(2,1)] = 'B'
        self.__dummy_sites[(2,2)] = 'C'
        self.__dummy_sites[(2,3)] = 'A'
        self.__dummy_sites[(2,0)] = 'B'


        self.__triangle_edges_3()

        self.__vectors = [[2,0],[0,3]]


    def __triangle_edges_3(self):

        self.__edges.append( ( (0,0), (0,1) ) )
        self.__edges.append( ( (0,0), (1,0) ) )
        self.__edges.append( ( (0,1), (0,2) ) )

        self.__edges.append( ( (0,1), (1,0) ) )
        self.__edges.append( ( (0,1), (1,1) ) )
        
        self.__edges.append( ( (0,2), (0,3) ) )
        self.__edges.append( ( (0,2), (1,1) ) )
        self.__edges.append( ( (0,2), (1,2) ) )
        
        self.__edges.append( ( (1,0), (1,1) ) )
        self.__edges.append( ( (1,0), (2,0) ) )
        self.__edges.append( ( (1,0), (2,1) ) )
        
        self.__edges.append( ( (1,1), (1,2) ) )
        self.__edges.append( ( (1,1), (2,1) ) )
        self.__edges.append( ( (1,1), (2,2) ) )
        
        self.__edges.append( ( (1,2), (1,3) ) )
        self.__edges.append( ( (1,2), (0,3) ) )
        self.__edges.append( ( (1,2), (2,3) ) )
        self.__edges.append( ( (1,2), (2,2) ) )





##  KagomeLattice()            
## -------------------------------------------------------------
##
##  Family of Kagome lattices.  Arguments:
##
##      td       =   Term dictionary
##      double   =   Flag to double lattice and apply terms from
##                   onsite= dictionary to these pairs.
##      onsite   =   Dictionary of pair terms for doubled sites
##      type     =   
##                      'default'         :  Same as 'kagome-1'
##                      'kagome-1'        :  Single site type 
##                      'kagome-3'        :  Three site types 
##

class KagomeLattice(OperatorUnitCell):


    ###
    ### THIS SPECIFICATION MUST BE UPDATED TO ELIMINATE THE DANGLING TRIANGLE
    ###

    def __init__(self,td,double=False,onsite=None,type='default',dimension=2):

        self.__sites        =  {}
        self.__dummy_sites  =  {}
        self.__edges        =  []

        self.__vectors      =  None

        self.type_string    =  "Triangular Lattice"

        if (type == 'default'):
            self.subtype_string =  'kagome-1'
        else:
            self.subtype_string  =  type

        if (self.subtype_string == 'kagome-1'):
            self.__kagome_lattice_1()
        elif (type == 'kagome-3'):
            self.__kagome_lattice_3()
        else:
            raise NotImplementedError
        
        OperatorUnitCell.__init__( self,
                               td,
                               sites=self.__sites,
                               edges=self.__edges,
                               dummy_sites=self.__dummy_sites,
                               vectors=self.__vectors,
                               double=double,
                               onsite=onsite )


    def __kagome_lattice_1(self):

        self.__sites[(0,0)] = 'A'
        self.__sites[(0,2)] = 'A'
        self.__sites[(1,1)] = 'A'


        self.__dummy_sites[(2,0)] = 'A'
        self.__dummy_sites[(2,2)] = 'A'
        self.__dummy_sites[(0,4)] = 'A'

        self.__kagome_edges_3()

        self.__vectors = [[2,2],[0,4]]



    def __kagome_lattice_3(self):

        self.__sites[(0,0)] = 'A'
        self.__sites[(0,2)] = 'B'
        self.__sites[(1,1)] = 'C'


        self.__dummy_sites[(2,0)] = 'A'
        self.__dummy_sites[(2,2)] = 'B'
        self.__dummy_sites[(0,4)] = 'C'

        self.__kagome_edges_3()

        self.__vectors = [[2,2],[0,4]]


    def __kagome_edges_3(self):

        self.__edges.append( ( (0,0), (0,2) ) )
        self.__edges.append( ( (0,0), (1,1) ) )
        self.__edges.append( ( (1,1), (0,2) ) )

        self.__edges.append( ( (1,1), (2,2) ) )
        self.__edges.append( ( (0,2), (0,4) ) )        
        self.__edges.append( ( (1,1), (2,0) ) )





######                            #############################################
######  3D LATTICES               #############################################
######                            #############################################



##  CubicLattice()            
## -------------------------------------------------------------
##
##  Family of cubic lattices.  Arguments:
##
##      td       =   Term dictionary
##      double   =   Flag to double lattice and apply terms from
##                   onsite= dictionary to these pairs.
##      onsite   =   Dictionary of pair terms for doubled sites
##      type     =   
##                      'default'   :   Equivalent to 'regular'
##                      'regular'   :   Single site per unit cell
##

class CubicLattice(OperatorUnitCell):


    def __init__(self,td,double=False,onsite=None,type='default',dimension=3):

        self.__sites        =  {}
        self.__dummy_sites  =  {}
        self.__edges        =  []

        self.__vectors      =  None

        self.type_string    =  "CubicLattice"

        if (type == 'default'):
            self.subtype_string =  'regular'
        else:
            self.subtype_string  =  type

        if (self.subtype_string == 'regular'):
            self.__cubic_lattice_regular()
        else:
            raise NotImplementedError

        OperatorUnitCell.__init__( self,
                                td,
                                sites=self.__sites,
                                edges=self.__edges,
                                dummy_sites=self.__dummy_sites,
                                vectors=self.__vectors,
                                double=double,
                                onsite=onsite )



    def __cubic_lattice_regular(self):

        self.__sites[(0 ,0, 0)] = 'A'

        self.__dummy_sites[(1, 0, 0)] = 'A'
        self.__dummy_sites[(0, 1, 0)] = 'A'
        self.__dummy_sites[(0, 0, 1)] = 'A'


        self.__edges.append( ( (0,0,0), (1,0,0) ) ) 
        self.__edges.append( ( (0,0,0), (0,1,0) ) ) 
        self.__edges.append( ( (0,0,0), (0,0,1) ) )

        self.__vectors =  [[1,0,0],[0,1,0],[0,0,1]]
