"""
Copyright (c) 2024 Massachusetts Institute of Technology 
SPDX-License-Identifier: BSD-2-Clause
"""



import  numpy  as  np
from    pyLIQTR.phase_factors.named_factors  import  f_cosine, f_sine



def simulation_phases(times,precompute=True,sequence="qsvt",convention="r", **kwargs):

    is_array = False

    if ( isinstance(times,tuple) or
         isinstance(times,list)  or
         isinstance(times,np.ndarray) ):
            is_array = True

    if (sequence.lower() == "qsvt"):
        if is_array:
            cos_terms = list([f_cosine(tau=time, precompute=precompute ,convention=convention, **kwargs) for time in times])
            sin_terms = list([f_sine(tau=time, precompute=precompute, **kwargs) for time in times])
            return(list(zip(cos_terms,sin_terms)))
        else:
            cos_terms = f_cosine(tau=times, precompute=precompute ,convention=convention, **kwargs)
            sin_terms = f_sine(tau=times, precompute=precompute ,convention=convention, **kwargs)
            return((cos_terms,sin_terms))
        
    elif (convention.lower == "qsp-fourier_response"):
        if is_array:
            exp_terms = list([f_exp_fourier_response(tau=time, precompute=precompute ,convention="fr", **kwargs) for time in times])
            return(list(exp_terms))
        else:
            exp_terms = f_exp_fourier_response(tau=time, precompute=precompute ,convention="fr", **kwargs) 
            return(exp_terms)