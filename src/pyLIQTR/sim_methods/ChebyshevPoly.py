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
import numpy                as np
import scipy.special        as sfn



class ChebyshevPoly:



    def __init__(self,coeffs=None,parity=None):

        self.N                 =  None
        self.deg               =  None
        self.coeffs            =  None
        self.parity            =  parity
        self.type              =  None

        if (coeffs is not None):
            self.coeffs           = coeffs
            self.N                = len(coeffs)
            self.deg              = self.N - 1
            self.parity           = parity





    def set_type(self,string):

        self.type = string
        return





    def set_parity(self,parity):

        self.parity = parity
        return





    def zero_poly(self,N,parity=None):

        self.N      = N
        self.deg    = N-1
        self.coeffs = np.zeros(N)

        if (parity is not None):
            self.parity = parity
        else:
            self.parity = N % 2

        return






    def set_coeffs(self,cfs_set,terms="all",parity=None):

        if (terms == "all"):
            self.coeffs = cfs_set
            self.parity = parity
            self.N      = len(cfs_set)
            self.deg    = self.N - 1
        elif (terms == "even"):
            N_cf = 2*len(cfs_set)
            self.coeffs[0:N_cf:2] = cfs_set
        elif (terms == "odd"):
            N_cf = 2*len(cfs_set)+1
            self.coeffs[1:N_cf:2] = cfs_set
        else:
            print("Invalid term specification. Must be even, odd, or all.")

        return

    # 
    # def coeffs(self):
    #     return(self.coeffs)

    def evens(self):
        return(self.coeffs[0:self.N:2])

    def odds(self):
        return(self.coeffs[1:self.N:2])


    def set_evens(self,cfs_set):

        N_cf = 2*len(cfs_set)

        # need to ultimately check and make sure coefficients will fit
        # into the target array, make sure array exists, etc.

        self.coeffs[0:N_cf:2] = cfs_set
        return





    def set_odds(self,cfs_set):

        N_cf = 2*len(cfs_set)+1

        # need to ultimately check and make sure coefficients will fit
        # into the target array, make sure array exists, etc.
        self.coeffs[1:N_cf:2] = cfs_set
        return






    def coeffs(self,coeffs=None,parity=None):

        # do we really want parity to be defined?  shouldn't we just assume
        # it is none if the value of parity is None or or otherwise?

        cflag = (coeffs is not None)
        pflag = (parity is not None)

        if (cflag and pflag):
            self.N       =  len(coeffs)
            self.deg     =  N-1
            self.coeffs  =  coeffs
            self.parity  =  parity

        elif (not (cflag and pflag)):
            print("\nCoefficients and parity are required for a ChebyshevPoly.\n")
            exit()

        else:
            return (self.coeffs)






    def eval(self,x,terms=None):


        ### really should split this out into itself and its mpmath variant,
        ### so it is available for user manipualation - and just put a
        ### wrapper here for eval (can also have it evaluate for cosh terms)

        x     = self._handle_scalar(x)      # Transforms scalar input to a 0-d numpy array

        fx    = np.zeros(len(x))      # Store evaluation of of f(x)
        acx   = np.arccos(x)          # Arccos appearing in Chebyshev terms

        ## Evaluates only even or odd terms in the expansion, assuming
        ## that the list of coefficients contains both even and odd parity
        ## terms.
        ##
        if ((terms is not None) and (self.parity not in [0,1])):

            if   (terms == 0):
                for k in range(0,self.N,2):
                    fx += self.coeffs[k] * np.cos(k*acx)
            elif (terms == 1):
                for k in range(1,self.N,2):
                    fx += self.coeffs[k] * np.cos(k*acx)
            else:
                for k in range(0,self.N):
                    fx += self.coeffs[k] * np.cos(k*acx)

        ## For a Chebyshev polynomial with a defnite parity we evaluate terms
        ## assuming that coefficients all correspond to terms with that parity.
        ##
        else:

            if (self.parity == 0):
                for k in range(0,self.N):
                    fx += self.coeffs[k] * np.cos((2*k)*acx)
            elif (self.parity == 1):
                for k in range(0,self.N):
                    fx += self.coeffs[k] * np.cos((2*k+1)*acx)
            else:
                for k in range(0,self.N):
                    fx += self.coeffs[k] * np.cos(k*acx)

        return(fx)




    def _handle_scalar(self,x):

        x = np.asarray(x)

        if x.ndim == 0:
            x = x[np.newaxis]

        return(x)


    def write_poly(self,filename):

        return




    def read_poly(self,filename):

        return
