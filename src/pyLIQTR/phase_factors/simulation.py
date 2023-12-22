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