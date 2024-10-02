"""
Copyright (c) 2024 Massachusetts Institute of Technology 
SPDX-License-Identifier: BSD-2-Clause
"""
################################################################################
###                                                                          ###
###    CONVERSIONS   -    Conversion between QSP phase angle conventions.    ###
###                                                                          ###
################################################################################


import  numpy   as   np


def phases_Wx_to_R(phis):
    N = len(phis)
    ph = np.zeros(N)
    ph[0] = phis[0] + np.pi/4
    for n in range(1,N-1):
        ph[n] = phis[n] + np.pi/2
    ph[-1] = phis[-1] + np.pi/4
    return(ph)
