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

from    pyLIQTR.phase_factors.phase_factors  import  PhaseFactors
from    pyLIQTR.phase_factors.named_factors  import  f_cosine, f_sine, f_exp_fourier_response
from    pyLIQTR.qubitization.qsvt            import  QSVT_real_polynomial_sum, QSP_fourier_response



def simulation_phases( times,
                       sequence="qsvt",
                       convention="r", 
                       precompute=True,
                       **kwargs ):

    is_array = False

    if ( isinstance(times,tuple) or
         isinstance(times,list)  or
         isinstance(times,np.ndarray) ):
            is_array = True

    if (sequence.lower() == "qsvt"):

        if is_array:
            cos_terms = list( [f_cosine( tau=time, 
                                         convention=convention, 
                                         precompute=precompute,
                                         **kwargs) for time in times] )
            sin_terms = list( [f_sine(   tau=time, 
                                         convention=convention,
                                         precompute=precompute,
                                         **kwargs) for time in times] )
            return(list(zip(cos_terms,sin_terms)))
        
        else:

            cos_terms = f_cosine( tau=times, 
                                  convention=convention,
                                  precompute=precompute, **kwargs )
            sin_terms = f_sine(   tau=times, 
                                  convention=convention,
                                  precompute=precompute, **kwargs)
            return((cos_terms,sin_terms))
        
    elif (sequence.lower() in ["fr", "qsp-fourier_response"]):

        if is_array:

            exp_terms = list( [f_exp_fourier_response( tau=time, 
                                                       precompute=precompute, 
                                                       **kwargs) for time in times] )
            return(list(exp_terms))
        
        else:

            exp_terms = f_exp_fourier_response( tau=time, 
                                                precompute=precompute, 
                                                **kwargs) 
            return(exp_terms)









def qsvt_dynamics( encoding,
                   instance=None,
                   times=[],
                   eps=1e-3,
                   sequence="qsvt",
                   phase_algorithm="optimization",
                   phase_sets=None,
                   time_rescale=True,
                   **kwargs ):

    is_array           =  True
    is_array_instance  =  False

    if (isinstance(times,float) or isinstance(times,int)):
        is_array  =  False
        times     =  np.array([times])


    if (isinstance(instance,list) or isinstance(instance,tuple)):
        is_array_instance  =  True
        times = np.array(times)

    if instance is not None:
        alpha = encoding(instance).alpha        
    else:
        alpha = encoding.alpha




    if (phase_sets is None):

        if time_rescale:
            time_set = times*alpha
        else:
            time_set = times

        phase_sets  =  simulation_phases( time_set,
                                          sequence=sequence,
                                          phase_algorithm=phase_algorithm,
                                          eps=eps ) 
    else:

        if (isinstance(phase_sets[0],PhaseFactors)):
            is_array  =  False
            phase_sets = (phase_sets,)


    if (sequence.lower() == "qsvt"):
        sequence_gate = QSVT_real_polynomial_sum
    elif (sequence.lower() in ["fourier_response", "fr"]):
        sequence_gate = QSP_fourier_response
    else:
        raise Exception("Unsupported QSP/QET/QSVT sequence.")


    gates     =  []

    if is_array_instance:
        gates = [ sequence_gate( encoding, 
                                 instance=inst,
                                 phis=phis,
                                 **kwargs ) for phis,inst in zip(phase_sets,instance) ]
    else:
        gates = [ sequence_gate( encoding,
                                 instance=instance,
                                 phis=phis,**kwargs ) for phis in phase_sets ]

    if is_array:
        return(gates)
    else:
        return(gates[0])
