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
from abc import ABC, abstractmethod

import  numpy                as  np
import  numpy.linalg         as  nla
import  scipy.sparse         as  sps


from    functools            import  cached_property


from  pyLIQTR.ProblemInstances.ProblemInstance  import   ProblemInstance
from  pyLIQTR.BlockEncodings                    import   VALID_ENCODINGS






class MatrixInstance(ProblemInstance):

    def __init__( self, data, sparse=False, do_normalize=False, norm="Frobenius",  **kwargs ):

        super(ProblemInstance, self).__init__(**kwargs)

        self._sparse               =  sparse                # store as sparse matrix data (if not provided as such)
        self._do_normalize         =  do_normalize          # normalize matrix data upon instantiation 
                                                            # (may provide other stuff for finer-grained control)

        self._norm_type            =  norm                  # type of matrix norm to compute


        if not self._sparse:                            # set prefix for printing the instance via __str__
            self._model_prefix     =  "Dense Matrix"    
        else:                                           
            self._model_prefix     =  "Sparse Matrix"


        # string used to describe the model
        if not self._sparse:
            self._model  =   "Matrix "
        else:
            self._model  =   "Sparse Matrix "




        if (self._do_normalize):
            self._set_data(data / self.get_norm(data))
        else:
            self._set_data(data)




    # store matrix data in the class instance
    def _set_data(self,data):
        if self._sparse:
            self._data             =    sps.coo_array(data)
        else:
            self._data             =    data




    # string that describes the instance
    def __str__(self):
        return f"{self._model}:\tdims = {self._data.shape}"


    # return the number of qubits needed to represent the 
    # problem
    def n_qubits(self):
        return ()





    # -return dense matrix (use @cached_property to store value 
    #  after computation (if it makes sense to do so)
    # -property allows us to access this as self.matrix as oppsoed to self.matrix()
    @property   
    def matrix(self):
        if self._sparse:
            return(self._data.toarray())
        else:
            return(self._data)



    # return sparse matrix
    @cached_property   
    def sparse_matrix(self):
        return(sps.coo_array(self._data))
    


    # return sparse data as list of tuples ((column index, row index), value)
    @property
    def sparse_tuples(self):
        data     =  self.sparse_matrix
        coords   =  list(zip(data.row,data.col))
        tuples   =  list(zip(coords,data.data))
        return(tuples)



    # return sparse data as tuple of arrays ([col idx], [row idx], [values])
    @property
    def sparse_arrays(self):
        data     =  self.sparse_matrix
        return((data.row,data.col,data.data))


    # calculate norm of matrix 
    def get_norm(self,matrix):
        if (self._norm_type.lower() == "frobenius"):
            if not self._sparse:
                matrix_norm = nla.norm(matrix,ord='fro')
            else:
                matrix_norm = sps.linalg.norm(matrix,ord='fro')
        # else:
        #  #   other cases
        
        return(matrix_norm)



    def normalize(self,norm_type='frobenius'):
        self._norm_type  =  norm_type                                   # need to to some exception catching here
        self._set_data(self._data / self.get_norm(self._data))
        


    ## add other functions, like option to normalize data after instantation etc
    


    # return the norm of the stored matrix
    @property
    def norm(self):
        return(self.get_norm(self._data))


    # add something later for mapping this to a Pauli string LCU representation
    # def yield_PauliLCU_Info(self,return_as='arrays',do_pad=0,pad_value=1.0):
    #     return

    # use this to yield details for the fable encoding (the matrix, other factors, etc.)
    def yield_fable_info(self):
        return

    # define this to yield details for the fable encoding (if ingested differently than fable)
    # def yield_sparse_fable_info(self):
    #     return


        # for term in terms:
        #     yield term
