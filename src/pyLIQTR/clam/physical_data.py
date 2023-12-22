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
###              PHYSICAL_DATA  -  Physical data and functions               ###
###                                                                          ###
################################################################################



import numpy as np



######                            #############################################
######  CONVERSION FACTORS        #############################################
######                            #############################################



bohr_radius       =   0.529117 

ev_to_hartree     =   0.0367493
hartree_to_ev     =   1.0/ev_to_hartree
ev_to_rydberg     =   ev_to_hartree/2
rydberg_to_ev     =   1.0/ev_to_rydberg

bohr_to_angstrom  =   0.529117 
angstrom_to_bohr  =   1.0/bohr_to_angstrom



######                            #############################################
######  ATOMIC DATA               #############################################
######                            #############################################



dt_atomic_numbers = {
 'H':   1,   'He':  2,   'Li':  3,   'Be':  4,   'B':   5,   'C':   6,   'N':   7,    
 'O':   8,   'F':   9,   'Ne': 10,   'Na': 11,   'Mg': 12,   'Al': 13,   'Si': 14,  
 'P':  15,   'S':  16,   'Cl': 17,   'Ar': 18,   'K':  19,   'Ca': 20,   'Sc': 21,
 'Ti': 22,   'V':  23,   'Cr': 24,   'Mn': 25,   'Fe': 26,   'Co': 27,   'Ni': 28,
 'Cu': 29,   'Zn': 30,   'Ga': 31,   'Ge': 32,   'As': 33,   'Se': 34,   'Br': 35,
 'Kr': 36,   'Rb': 37,   'Sr': 38,   'Y':  39,   'Zr': 40,   'Nb': 41,   'Mo': 42,
 'Tc': 43,   'Ru': 44,   'Rh': 45,   'Pd': 46,   'Ag': 47,   'Cd': 48,   'In': 49,
 'Sn': 50,   'Sb': 51,   'Te': 52,   'I':  53,   'Xe': 54,   'Cs': 55,   'Ba': 56,
 'La': 57,   'Ce': 58,   'Pr': 59,   'Nd': 60,   'Pm': 61,   'Sm': 62,   'Eu': 63,
 'Gd': 64,   'Tb': 65,   'Dy': 66,   'Ho': 67,   'Er': 68,   'Tm': 69,   'Yb': 70,
 'Lu': 71,   'Hf': 72,   'Ta': 73,   'W':  74,   'Re': 75,   'Os': 76,   'Ir': 77,
 'Pt': 78,   'Au': 79,   'Hg': 80,   'Tl': 81,   'Pb': 82,   'Bi': 83,   'Po': 84,
 'At': 85,   'Rn': 86,   'Fr': 87,   'Ra': 88,   'Ac': 89,   'Th': 90,   'Pa': 91, 
 'U':  92,   'Np': 93,   'Pu': 94,   'Am': 95,   'Cm': 96,   'Bk': 97,   'Cf': 98, 
 'Es': 99,   'Fm': 100,  'Md': 101,  'No': 102,  'Lr': 103    }



######                            #############################################
######  PHYSICAL FUNCTIONS        #############################################
######                            #############################################



##
##  Fermi distribution function
##

fermi  =  lambda E,mu,T : 1.0/(np.exp((E-mu)/T)+1.0)