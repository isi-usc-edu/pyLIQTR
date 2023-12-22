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
###     GEOMETRY  -  Specification and manipulation of atomic geometries     ###
###                                                                          ###
################################################################################


import numpy               as  np
import pyLIQTR.clam.physical_data  as  pd
import os



############################################################################
###
###   CLASS     Geometry                                           (none)
###
############################################################################
###
###   DESCRIPTION
###
###      Class for handling and manipulating atomic geometries.  Assumes
###      orthorhombic coordinate system.
###
###   CONSTRUCTOR ARGUMENTS
###
###      coords      -   Coordinates specified as an array of tuples
###      labels      -   Atom labels if coordinates are provided as unlableled
###                      xyz tuples
###      filename    -   Path to an xyz file containing the coordinates
###      units       -   Units for coordinates set (default: 'angstrom')
###      process     -   That to converts to atomic units (Bohr) and centers
###                      coordinates geometrically about the origin.
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
###      Geometry    -    Instance of the Geometry class
###
###   REQUIRES
###
###      clam.physical_data  as  pd
###      numpy               as  np
###
###   PUBLIC MEMBERS 
###
###      units      -    Current unit specification for the coordinates
###      dim        -    Dimension of basis (number of basis elements)
###      coords     -    List of 3-tuples specifying atomic coordinates
###      labels     -    List of labels for each of the coordinates 
###                      from <coords>
###      center     -    Center of coordinate set
###      minmax     -    List containing arrays with minimal and maximal
###                      bounding coordinates for set:
###         
###                         [ [x_min,y_min,z_min], [x_max,y_max,z_max] ]  
###
###      lengths    -    Lengths of bounding cell in canonical orthorhombic
###                      frame
###
###   PUBLIC ROUTINES  
###
###      load_coordinates          -   Loads coordinate set into geometry
###      align_to_principal_axes   -   Aligns principal axes to those of 
###                                    the orthorhombic frame (unimplemented)
###      get_minmax                -   Calculate minmax bounding values of
###                                    coordinates in orthorhombic frame
###      get_center                -   Calculate geometric center of coordinates
###      recenter                  -   Recenter coordinates at origin
###      set_units                 -   Set unit specifciation for coordinates
###                                    ( *** no conversion performed *** )
###      convert                   -   Convert coordinates to new units
###      pairs                     -   Return list of coordinates paired with 
###                                    labels:
###
###                                        ['He', 1.000, 1.500, 3.000]
###
###      print_xyz                 -   Print coordinates to file or <stdout> in
###                                    .xyz format.
###      write_xyz                 -   Shortcut for writting coordinates as .xyz
###                                    file.
###      process                   -   Convert coordinates to Bohr and recenter
###                                      
###
###
###   MAGIC ROUTINES  
###
###     __str__()
###
###           Pretty-printing of comprehensive cell attributes.
###

class Geometry():

    ##  __init__()
    ## -------------------------------------------------------------
    ##
    ##  Constructor for geometry class.  (see above)
    ##

    def __init__(self, coords=None, labels=None, filename=None, units='angstrom', process=False):

        self.dim      =  3

        self.units    =  'angstrom'

        self.coords   =  coords
        self.labels   =  labels

        self.center   =  None
        self.minmax   =  None
        self.lengths  =  None

        if (filename is not None):
            self.load_coordinates(filename)

        # if (self.coords is None):
        #     raise Exception("Geometry: Coordinates or a coordinate file must be specified.")

        if (self.coords is not None):
            self.__set_bounds()

        if (process == True):
            self.process()
            self.__set_bounds()

        return


    ##  load_coordinates()
    ## -------------------------------------------------------------
    ##
    ##  Driver for loading coordinates into the Geometry class from 
    ##  a file.  The file <filename> is assumed to have format
    ##  <ctype> with units <units>.
    ##

    def load_coordinates(self,filename,ctype='xyz',units='angstrom'):

        if ctype is None:
            base, suffix = os.path.splitext(filename)       
            ctype = suffix[1:]    

        if (ctype.lower() == 'xyz'):
            self.__load_coordinates_xyz(filename)
        elif(ctype.lower()== 'pdb'):
            self.__load_coordinates_pdb(filename)
        else:
            print("Unsupported coordinate format "+ctype+"\n")
            return
        
        self.units = units

        return
    

    ##  __load_coordinates_xyz()
    ## -------------------------------------------------------------
    ##
    ##  Loads coordinates from .xyz format file.
    ##

    def __load_coordinates_xyz(self,filename):

        self.coords = []
        self.labels = []

        fp    =  open(filename, 'r')
        lines =  [line.rstrip() for line in fp]
        fp.close()

        # could just skim over the rest of the line to get property fields,
        # but leaving this functionality undefined for now
        #
        for line in lines[2:]:
            ents = line.split()
            self.labels.append(str(ents[0]))
            self.coords.append(np.array([float(ents[1]), float(ents[2]), float(ents[3])]))

        self.N        = len(self.coords)
        self.__set_bounds()

        return
    


    ##  __load_coordinates_xyz()
    ## -------------------------------------------------------------
    ##
    ##  Loads coordinates from .pdb format file.
    ##

    def __load_coordinates_pdb(self,filename):
        raise NotImplementedError()



    ##  __load_coordinates_xyz()
    ## -------------------------------------------------------------
    ##
    ##  Loads coordinates from .xyz format file.
    ##


    ## Calculate principal axes and rotate coordinates to align these axes
    ## with an orthorhombic coordinate system.
    ##
    def align_to_principal_axes(self):
        raise NotImplementedError()



    ##  __calculate_principal_axes()            (not implemented)
    ## -------------------------------------------------------------
    ##
    ##  Calculates principal axes for the specificed geometry.
    ##

    def __calculate_principal_axes(self):
        raise NotImplementedError()



    ##  __calculate_principal_axes()            (not implemented)
    ## -------------------------------------------------------------
    ##
    ##  Rotates principal axes of geometry to align with canonical
    ##  orthorhombic frame.
    ##

    def __rotate_to_principal_axes(self):
        raise NotImplementedError()




    ##  get_minmax()            
    ## -------------------------------------------------------------
    ##
    ##  Get bounding minimum and maximum values of specified  
    ##  coordinates.  If <silent=True> this function returns
    ##  the minmax arrays and a vector containing lengths of the 
    ##  minimal cell that bounds the coordinates.
    ##

    def get_minmax(self,silent=False):

        lims = []

        for n in range(self.dim):
            lims.append(np.zeros(2,dtype=float))

        for coord in self.coords:
            for n in range(self.dim):
                if (coord[n] < lims[n][0]):
                    lims[n][0] = coord[n]
                elif (coord[n] > lims[n][1]):
                    lims[n][1] = coord[n]

        lengths = np.zeros(self.dim,dtype=float)

        for n in range(3):
            lengths[n] = lims[n][1] - lims[n][0]
    
        if (not silent):
            return(lims,lengths)
    




    ##  __set_bounds()            
    ## -------------------------------------------------------------
    ##
    ##  Set bounding values and lengths in geometry class as 
    ##  obtained from get_minmax() 
    ##

    def __set_bounds(self):

        lims, lengths = self.get_minmax()

        self.minmax   = lims
        self.lengths  = lengths

        return



    ##  get_center()            
    ## -------------------------------------------------------------
    ##
    ##  Driver for calculating center of specified geometry
    ## 
    ##      type  =  'minmax'     :   Center of bounding cell 
    ##               'geometric'  :   Geometric center of coordinates
    ##               'barycenter' :   Center of mass of coordinates
    ##

    def get_center(self,type='minmax'):
        
        ttype = type.lower()

        if   (ttype == 'minmax'):
            center = self.__get_center_minmax()
        elif (ttype == 'geometric'):
            center = self.__get_center_geometric()
        elif (ttype == 'barycenter'):
            center = self.__get_center_barycenter()
        else:
            print ("Unknown centering: ", type)
            return

        return(center)



    ##  __get_center_geometric()            
    ## -------------------------------------------------------------
    ##
    ##  Caculates geometric center of coordinates
    ##

    def __get_center_geometric(self):

        coord_center = np.zeros(self.dim)

        for coord in self.coords:
            coord_center += coord

        return(coord_center/self.N)
    


    ##  __get_center_geometric()            
    ## -------------------------------------------------------------
    ##
    ##  Caculates minmax center of coordinates
    ##

    def __get_center_minmax(self):

        center = np.zeros(self.dim)

        for n in range(self.dim):
            center[n] = (self.minmax[n][0] + self.minmax[n][1]) / 2.0

        return(center)
    

    ##  __get_center_barycenter()            
    ## -------------------------------------------------------------
    ##
    ##  Caculates barycenter of coordinates
    ##

    def __get_center_barycenter(self):
         raise NotImplementedError()







    ##  recenter()            
    ## -------------------------------------------------------------
    ##
    ##  Recenters coordinates at origin (or point <point>).  
    ##  (*** point *** unimplemented)


    def recenter(self, mode='symmetric', point=None):

        lims, lengths = self.get_minmax()

        shift = np.zeros(3)

        for n in range(3):
            shift[n] = lengths[n]/2 - lims[n][1]

        new_coords = []

        for coord in self.coords:
#           new_coord = np.array(coord[1]) + shift
            new_coord = np.array(coord) + shift
            new_coords.append(new_coord)

        self.coords = new_coords

        self.__set_bounds()

        self.center = self.get_center()
        return



    ##  get_units()            
    ## -------------------------------------------------------------
    ##
    ##  Returns current unit specification
    ##

    def get_units(self):
        return(self.units)



    ##  set_units()            
    ## -------------------------------------------------------------
    ##
    ##  Sets unit specification for coordinates without doing a 
    ##  conversion.
    ##

    def set_units(self,units):
        self.units=units
        return
    


    ##  convert()            
    ## -------------------------------------------------------------
    ##
    ##  Converts existing units to  <new_units> ('angstrom' or 'bohr')
    ##

    def convert(self,new_units):

        if (new_units.lower() == self.units):
            return
        
        if ( (self.units == 'angstrom') and (new_units.lower() == 'bohr')):
            cfactor = pd.angstrom_to_bohr
        elif ( (self.units == 'bohr') and (new_units.lower() == 'angstrom') ):
            cfactor = pd.bohr_to_angstrom

        new_coords = []

        for coord in self.coords:
            new_coords.append(coord*cfactor)

        self.units   =  new_units
        self.coords  =  new_coords
        self.__set_bounds()

        return
    


    ##  pairs()            
    ## -------------------------------------------------------------
    ##
    ##  Returns list of coordinates paired with their atom labels
    ##

    def pairs(self,as_list=False):
        
        if (as_list):
            pairs = []
            for n in range(len(self.labels)):
                pairs.append([self.labels[n], self.coords[n].tolist()])
        else:
            pairs = [ pair for pair in zip(self.labels,self.coords)]

        self.coord_pairs = pairs

        return(pairs)
    


    ##  print_xyz()            
    ## -------------------------------------------------------------
    ##
    ##  Print coordinates to a file or <stdout> in .xyz format. 
    ##  Coordinates are numbered if <print_number=True>
    ##

    def print_xyz(self,filename=None, print_number=False):

        self.pairs(as_list=True)

        n_pairs = len(self.coord_pairs)

        if (filename is not None):
            with open(filename,'w+') as fp:
                fp.write(str(n_pairs)+"\n\n")
                for pair in self.coord_pairs:
                    xyz_string=str(pair[0])+"   "+str(pair[1][0])+"   "+str(pair[1][1])+"   "+str(pair[1][2])+"\n"
                    fp.write(xyz_string)

        else:
            print(n_pairs,"\n")
            atidx = 0
            for pair in self.coord_pairs:
                if (print_number):
                    print(atidx,":   ", pair[0],"  ",pair[1][0],"  ",pair[1][1],"  ",pair[1][2])
                    atidx += 1
                else:
                    print(pair[0],"  ",pair[1][0],"  ",pair[1][1],"  ",pair[1][2])
        return



    ##  write_xyz()            
    ## -------------------------------------------------------------
    ##
    ##  Write coordinates to file path <filename> in .xyz format. 
    ##

    def write_xyz(self,filename):
        self.print_xyz(filename=filename)
        return



    ##  process()            
    ## -------------------------------------------------------------
    ##
    ##  Convert coodinates to atomic units (Bohr) and recenter. 
    ##

    def process(self):
        self.convert('bohr')
        self.recenter()
        return


    ## modify xyz printing to return coordinates as numbered set
    ## and return that here.
    ##
    def __str__(self):
        return
        # coord_text = print_xyz(print_file=True)

        # return(coord_text)







##  AtomicLattice(Lattice)
## -------------------------------------------------------------
##
##  Class for building periodic lattices from coordinate 
##  specifications.  
##


# class AtomicLattice(Lattice):

#     def __init__( self, 
#                   vectors,
#                   size=None,
#                   unit=None,
#                   periodic=None,
#                   do_wrap=None,
#                   cell_vectors=None,
#                   centered=True        ):

#         Lattice.__init__(self,vectors,size=size,unit=unit,periodic=(False,False,False))

#         self.build(size)

#         if (cell_vectors is None):
#             self.cell_vecs = []
#             for n in range(len(vectors)):
#                 self.cell_vecs.append(size[n]*vectors[n])
#         else:
#             self.cell_vecs = np.array(cell_vectors)
    
#         if (do_wrap is None):
#             self.do_wrap  = []
#             for n in range(self.dim):
#                 self.do_wrap.append(False)
#         else:
#             self.do_wrap = do_wrap

# #        Set labels to something generic like x if none are specified but unit is specified?
# #       or should this be done in geometry
#         # if (unit is None):
#         #     self.labels = 


#         self.build_geometry(do_wrap)
    
#         return
    

#     def wrap(self,coords):

#         new_coords = []

#         for x in coords:
#             for n in range(len(x[0])): 
#                 if ( self.do_wrap[n] == True):
#                     if (x[0][n] >= self.cell_vecs[n][n]):
#                         x[0][n] -= self.cell_vecs[n][n]

#                     if (x[0][n] < 0.0):
#                         x[0][n] += self.cell_vecs[n][n]

#             new_coords.append(x[0])

#         return(new_coords)


#     def build_geometry(self,do_wrap=None):

#         labels = self.labels[0]

#         coords = []
#         for coord in self.skeletons[0]:
#             coords.append(coord)

#         new_coords = self.wrap(coords)

#         self.geometry = Geometry(coords=new_coords,labels=labels)

#         return







##  geometry_angstrom_to_bohr()            
## -------------------------------------------------------------
##
##  Convert coodinates <geom> in angstrom to bohr (or bohr to 
##  angstrom via <invert=True>.
##

def geometry_angstrom_to_bohr(geom,invert=False):

    new_geom = []

    if (invert):
        sf = pd.angstrom_to_bohr
    else:
        sf = pd.bohr_to_angstrom

    for coord in geom:
        new_geom.append([coord[0], (coord[1][0]*sf,coord[1][1]*sf,coord[1][2]*sf)])

    return(new_geom)









