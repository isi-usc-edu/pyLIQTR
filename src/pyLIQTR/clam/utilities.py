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

import numpy  as  np



## Convert tuples to lists up to a specified depth
## (keyword d= )
##

def listify(item,d=1):

    if (d == 0):
        return(item)
    
    if (d == 1):
        if (hasattr(item, "__len__") and (type(item) != str)):
            return(item)
        else:
            return([item])
        
    if (d == 2):
        if (hasattr(item, "__len__") and (type(item) != str) ):

            if (hasattr(item[0], "__len__") and (type(item[0]) != str) ):
                return(item)
            elif (hasattr(item[0], "__len__") and (type(item[0]) == str) ):
                return([[ x for x in item]])
            else:
                return([x for x in item])
            
        elif (hasattr(item, "__len__") and (type(item) == str)):
            return( [[item]] )
        else:
            return([[item]])
        




## Recursively convert lists to tuples
##

def tupleize(item,as_int=False):

    if (isinstance(item,list)):
        return tuple(tupleize(x) for x in item)
    else:
        return( int(item) if as_int else item )





## Recursively convert sites objects containing lists to
## tuples
##

def tupleize_sites(item):

    if (isinstance(item,list) or isinstance(item,np.ndarray)):
        return(tuple(tupleize_sites(x) for x in item))
    else:
        return(int(item))





## Return entries from arrays <target> and <source> for which
## elements of <target> are nonzero.
##

def get_nonzero_match(target,source):

    set_target = []
    set_source = []

    for n in range(len(target)):
        if (target[n] != 0.0):
            set_target.append(target[n])
            set_source.append(source[n])

    return(set_target,set_source)





## Return limits of coordinate tuples in list <geom> as 
## well as the length of a minimum orthorhombic cell
## bounding these coordinates.
##

def minmax(geom):

    lims = []

    for n in range(3):
        lims.append(np.zeros(2,dtype=float))

    for coord in geom:
        for n in range(3):
            if (coord[1][n] < lims[n][0]):
                lims[n][0] = coord[1][n]
            elif (coord[1][n] > lims[n][1]):
                lims[n][1] = coord[1][n]

    lengths = np.zeros(3,dtype=float)

    for n in range(3):
        lengths[n] = lims[n][1] - lims[n][0]

    return(lims,lengths)