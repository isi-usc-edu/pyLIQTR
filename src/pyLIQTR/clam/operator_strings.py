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
###       OPERATOR_STRINGS  -  String-based operator representations         ###
###                                                                          ###
################################################################################

import numpy                               as  np
import pyLIQTR.clam.multilinear_algebra    as  mla
import copy

import pyLIQTR.clam.operators        as     qo    


##
## Make op_strings derived from operator_strings
##




## CLASS op_strings(circ)
## ---------------------------------------x
##
##  Stores LCU of operator as Pauli strings
##

class op_strings():

    def __init__(self,N=0,N_qb=0, indices=None,types=None,coeffs=None):

        self.N        =    N
        self.N_qb     =    N_qb
        self.alpha    =   -1

        if ((coeffs == None) or (types == None) or (indices == None)):
            self.indices  =   []
            self.types    =   []
            self.coeffs   =   []
        else:
            self.indices  =   copy.deepcopy(indices)
            self.types    =   copy.deepcopy(types)
            self.coeffs   =   copy.deepcopy(coeffs)

    ## append(self,sites,term,coeff)
    ## ---------------------------------------x
    ## Add string on sites of index <sites>
    ## corresponding to the Pauli operators
    ## <term>  with coefficients <coeff>.
    ## 
    ##   Ex: append((1,3),'XX',0.25)
    ##

    def append(self,sites,term,coeff):
        self.indices.append(sites)
        self.types.append(term)
        self.coeffs.append(coeff)
        self.N +=1 
        return


    ## append_tuple(self,data)
    ## ---------------------------------------x
    ## Add string based on <data>, which 
    ## contains a tuple followin the convetions
    ## of append() 
    ## 
    ##   Ex: append(((1,3),'XX',0.25))
    ##

    def append_tuple(self,data):
        self.indices.append(data[0])
        self.types.append(data[1])
        self.coeffs.append(data[2])
        self.N +=1 
        return


    ## matrix()
    ## ---------------------------------------x
    ## Return matrix representation of the 
    ## operator formed by this LCU.
    ##

    def matrix(self):

        M = ham_from_pauli_tuples(self.indices,self.types,self.coeffs,self.N)
        return(M)
    

    ## sparse_matrix()
    ## ---------------------------------------x
    ## Return sparse matrix representation of  
    ## the operator formed by this LCU.
    ## 
    
    def sparse_matrix():
        return()
    

    ## tensor()
    ## ---------------------------------------x
    ## Return low-rank tensor representation of  
    ## the operator formed by this LCU.
    ## 

    def tensor():
        return()

    ## print()
    ## ---------------------------------------x
    ## Print table of Pauli strings within this
    ## object.
    ##

    def print(self):

        ps_print_set(self.indices,self.types,self.coeffs)
        return()



    def set_num_qb(self,N):
        self.N_qb = N
        return


    def terms(self,do_pad=0,pad_value=1.0):

        term_set = []

        if (do_pad > 0):
            term_set = [([0],'I',pad_value) for _ in range(do_pad)]

        term_set += zip(self.indices,self.types,self.coeffs)
        return(term_set)


    ## strings()
    ## ---------------------------------------x
    ## Return tuples containing identity-padded
    ## Pauli strings and their coefficients
    ## for ingestion by circuit generation code
    ##
    
    def strings(self,coeffs=True,do_pad=0,pad_value=1.0):

        if (coeffs == True):
            pstr = ps_text_full_set_withzero( self.indices, 
                                              self.types,
                                              self.N_qb,
                                              coeffs=self.coeffs,
                                              do_pad=do_pad,
                                              pad_value=pad_value )
        else:
            pstr = ps_text_full_set_withzero(self.indices,self.types,self.N_qb,do_pad=do_pad)

        return(pstr)


    ## expand_projectors()
    ## ---------------------------------------x
    ## Expand Z eigenstate projectors from
    ## Pauli strings into constituent Pauli
    ## operators.
    ##
    
    def expand_projectors(self):

        new_indices = []
        new_types  = []
        new_coeffs = []

        for pair in zip(self.indices,self.types,self.coeffs):
            for cpn in  convert_projector(pair):

                new_indices.append(cpn[0])
                new_types.append(cpn[1])
                new_coeffs.append(cpn[2])

        self.indices = new_indices
        self.types   = new_types
        self.coeffs  = new_coeffs

        return()


    ## remove_zeros()
    ## ---------------------------------------x
    ## Remove strings with coefficients of 0.0
    ##

    def remove_zeros(self):

        new_indices = []
        new_types   = []
        new_coeffs  = []

        for pair in zip(self.indices,self.types,self.coeffs):
            if (pair[2] == 0.0):
                pass
            else:
                new_indices.append(pair[0])
                new_types.append(pair[1])
                new_coeffs.append(pair[2]) 

        self.indices = new_indices
        self.types   = new_types
        self.coeffs  = new_coeffs

        return()


    ## cleanup()
    ## ---------------------------------------x
    ## Expand projectors and remove zero terms
    ##

    def cleanup(self):
        
        self.remove_zeros()
        self.expand_projectors()

        return()
    

    ## rescale_coeff_norm()
    ## ---------------------------------------x
    ## Rescale LCU coefficients by 1-norm of
    ## coefficient vector and store the
    ## rescaling factor <alpha>:
    ##
    ##   \alpha = \sum_k |\alpha_k|
    ##
    ## where the LCU is:
    ##
    ##   \sum_{k} \alpha_k U_k
    ##

    def rescale_coeff_norm(self):

        alpha = np.sum(np.abs(self.coeffs))

        new_indices = []
        new_types   = []
        new_coeffs  = []

        for n in range(len(self.coeffs)):

            new_indices.append(self.indices[n])
            new_types.append(self.types[n])
            new_coeffs.append(self.coeffs[n] / alpha)

        self.indices = new_indices
        self.types   = new_types
        self.coeffs  = new_coeffs 

        if (self.alpha == -1):
            self.alpha = alpha

        return()
        
    

    ## get_alpha()
    ## ---------------------------------------x
    ## Return the LCU rescaling parameter 
    ## <alpha>.
    ##

    def get_alpha(self):
 #       alpha = np.sum(np.abs(self.coeffs))
        return(self.alpha)


    ## get_coeff_norm()
    ## ---------------------------------------x
    ## Calculate the 1-norm of the coefficient
    ## vector but do not rescale coefficients.
    ##

    def get_coeff_norm(self):
        scf = np.sum(np.abs(self.coeffs))
        return(scf)


    ## rescale_coeff_scalar(val)
    ## ---------------------------------------x
    ## Rescale LCU coefficients by value
    ## <val> and store coefficient as <alpha>
    ##

    def rescale_coeff_scalar(self,val):

        new_indices = []
        new_types   = []
        new_coeffs  = []

        for n in range(len(self.coeffs)):

            new_indices.append(self.indices[n])
            new_types.append(self.types[n])
            new_coeffs.append(self.coeffs[n] / val)

        self.indices = new_indices
        self.types   = new_types
        self.coeffs  = new_coeffs 

        if (self.alpha == -1):
            self.alpha = val

        return()


    ## rescale_minmax(e_min,e_max)
    ## ---------------------------------------x
    ## Perform minmax rescaling based on 
    ## bounds <e_min>, <e_max>
    ##

    def rescale_minmax(self,e_min,e_max):

        e_diff = e_max - e_min

        new_indices = []
        new_types   = []
        new_coeffs  = []

        for n in range(len(self.coeffs)):

            new_indices.append(self.indices[n])
            new_types.append(self.types[n])
            new_coeffs.append(self.coeffs[n] / e_diff)
        
        new_indices.append(tuple(range(self.N)))
        new_types.append('I'*self.N)
        new_coeffs.append(-e_min/(e_diff))

        self.indices = new_indices
        self.types   = new_types
        self.coeffs  = new_coeffs

        return()
    

    ## consolidate_strings()
    ## ---------------------------------------x
    ## Sum Pauli strings with identical 
    ## elements
    ##
    
    def consolidate_strings(self):

        return()


    def delete_term(self,tid):

        if (tid < self.N):

            del self.indices[tid]
            del self.types[tid]
            del self.coeffs[tid]
            self.N -=1 

        return
    


    ## Is delete_site the best term here?
    ##
    def delete_site(self,site):

        exist_flag = False 

        for k in range(len(self.indices)):
            if site in self.indices[k]:
                del self.indices[k]
                del self.types[k]
                del self.coeffs[k]
                exist_flag = True
                self.N -=1 

        return
    

    def modify_type(self,k,new_type):
        self.types[k] = new_type
        return
    
    def modify_coeff(self,k,new_coeff):
        self.coeffs[k] = new_coeff
        return  
    



################################################################################
###
###   FUNCTION     convert_projector(pstr)
###
################################################################################
###
###   DESCRIPTION
###
###      Converts the Z-eigenstate projectors ('+','-') in the string <pstr>
###      into their constituent Pauli operations.  Performs rescaling of
###      coefficients to match.
###
###
###
###   ARGUMENTS
###
###      pstr   =   Target Pauli string
###
###   RETURNS
###
###      new     =  Modified Pauli string
###
###   REQUIRES
###
###      numpy as np
###

def convert_projector(pstr):

    proj_set = ('+','-')

    new      =  []
    indices  =  pstr[0];      opstr  =  pstr[1];       coeff  =  pstr[2]

    if ( ('+' not in opstr) and ('-' not in opstr)):
        new.append(pstr)
    else:
        for op in proj_set:
            if (op in opstr):

                idx = opstr.index(op)

                ns_1 = opstr[0:idx]+'I'+opstr[(idx+1):]
                ns_2 = opstr[0:idx]+'Z'+opstr[(idx+1):]
                
                sgn = (-1)**proj_set.index(op)

                nsp_1 = ((indices,ns_1,(1/2)*coeff))
                nsp_2 = ((indices,ns_2,sgn*(1/2)*coeff))

                [new.append(string) for string in convert_projector(nsp_1)]
                [new.append(string) for string in convert_projector(nsp_2)]

                break

    return(new)






################################################################################
###
###   FUNCTION     ps_text_pad(string)
###
################################################################################
###
###   DESCRIPTION
###
###      Pads the shorthand text string a mulit-qubit Pauli operator with
###      identity elements to give an N-qubit Pauli string. That is,
###      an operator described by <string> acting on spins/qubits
###      in the tuple <sites> is padded by identity operators to give a string
###      acting on <N> qubits. If <sites> = (0,2,3) and <string> = 'XYZ' with
###      N = 6, this would return XIYZII. Note that the indices in sites
###      follow pythonic array indexing, while <N> is the absolute number of
###      qubits (NOT the largest array index).
###
###
###
###   ARGUMENTS
###
###      sites  =   Tuple containing site/qubit indices.
###      string =   Text string describing the operator
###      N      =
###
###   RETURNS
###
###      op      =  Matrix representation of the operator
###
###   REQUIRES
###
###      numpy as np
###

def ps_text_pad(sites,string,N):

    padded = ''

    ## Check if we have been given a scalar or a tuple.
    ##
    is_tuple = hasattr(sites, "__len__")

    if (is_tuple):
        for k in range(N):
            if (k in sites):
                sidx    = sites.index(k)
                padded += string[sidx]
            else:
                padded += 'I'
    else:
        padded = 'I'*(sites) + string + 'I'*(N-sites-1)

    return(padded)





################################################################################
###
###   FUNCTION     ps_text_to_ops(string)
###
################################################################################
###
###   DESCRIPTION
###
###      Returns an array containing the Pauli operators represented by the
###      Pauli string <string>, starting from the least significant element.
###      This method is case insensitive.
###
###   ARGUMENTS
###
###      string  =  Pauli string consisting of X,Y,Z and I (projectors and the
###                 like are not yet implemented).
###
###   RETURNS
###
###      op_set  =  Array of Pauli operators in the canonical matrix
###                 representation for su(2)
###
###
###   REQUIRES
###
###      numpy as np
###

def ps_text_to_ops(string):

    N = len(string)

    op_set = []

    for j in range(N):
        op = label_to_pauli(string[j])
        op_set.append(op)

    return(op_set)





################################################################################
###
###   FUNCTION     ps_text_to_matrix(string)
###
################################################################################
###
###   DESCRIPTION
###
###      Generates a matrix representation of the operator corresponding to
###      the text-based Pauli string <string> (example: 'IXXIYII').
###
###
###   ARGUMENTS
###
###      string  =  Pauli string consisting of X,Y,Z and I (projectors and the
###                 like are not yet implemented).
###
###   RETURNS
###
###      op      =  Matrix representation of the operator
###
###   REQUIRES
###
###      numpy as np
###

def ps_text_to_matrix(string):

    ops = ps_text_to_ops(string)
    M   = mla.mk(ops)

    return(M)





################################################################################
###
###   FUNCTION     ps_label_to_pauli(label)
###
################################################################################
###
###   DESCRIPTION
###
###      Returns an a matrix representation of the Pauli operator corresponding
###      to the label <label>.  This method is case insensitive.
###
###   ARGUMENTS
###
###      string  =  Text label for the Pauli; may be 'X', 'Y', 'Z' or 'I'
###
###   RETURNS
###
###      op      =  Matrix representation of the operator
###      -1      =  If label is not recognized.
###
###   REQUIRES
###
###      numpy as np
###

def label_to_pauli(label):

    if  (label.lower() == 'x'):
        return(qo.X)
    elif(label.lower() == 'y'):
        return(qo.Y)
    elif(label.lower() == 'z'):
        return(qo.Z)
    elif(label.lower() == 'i'):
        return(qo.I)
    elif(label.lower() == '+'):
        return(qo.przp)
    elif(label.lower() == '-'):
        return(qo.przm)
    else:
        ## Error Condition
        print("\nNo matching Pauli operator: ", label,"\n")
        return(-1)
    




################################################################################
###
###   FUNCTION     ham_from_pauli_tuples(tuples,types,coeffs,N)
###
################################################################################
###
###   DESCRIPTION
###
###      Expand Pauli strings into matrix representation of Hamiltonian.
###
###   ARGUMENTS
###
###      string  = 
###
###   RETURNS
###
###      op      =  
###
###   REQUIRES
###
###      numpy as np
###

def ham_from_pauli_tuples(tuples,types,coeffs,N):

    H = 0.0j

    for j in range(len(tuples)):

        p_string = ps_text_pad(tuples[j],types[j],N)
        H       += coeffs[j]*ps_text_to_matrix(p_string)

    return(H)





################################################################################
###
###   PAULI STRING PRINTING ROUTINES
###
################################################################################


def ps_text_full_set(Sites,Strings,N,coeffs=None,expand_projector=False):

    Padded = []

    if (coeffs is not None):
        for pair in zip(Sites,Strings,coeffs):
            if (pair[2] == 0.0):
                pass
            else:
                Padded.append((ps_text_pad(pair[0],pair[1],N),pair[2]))
    else:
        for pair in zip(Sites,Strings,coeffs):
            if (pair[2] == 0.0):
                pass
            else:
                Padded.append(ps_text_pad(pair[0],pair[1],N))

    return(Padded)


    



def ps_text_full_set_withzero(Sites,Strings,N,coeffs=None,expand_projector=False,do_pad=0,pad_value=1.0):

    Padded = []

    if (coeffs is not None):
        for n in range(do_pad):
            Padded.append((ps_text_pad([0],'I',N),pad_value))
        for pair in zip(Sites,Strings,coeffs):
            Padded.append((ps_text_pad(pair[0],pair[1],N),pair[2]))
    else:
        for n in range(do_pad):
            Padded.append(ps_text_pad([0],'I',N))
        for pair in zip(Sites,Strings):
            Padded.append(ps_text_pad(pair[0],pair[1],N))

    return(Padded)


def ham_from_pauli_tuples(tuples,types,coeffs,N):

    H = 0.0j

    for j in range(len(tuples)):

        p_string = ps_text_pad(tuples[j],types[j],N)
        H       += coeffs[j]*ps_text_to_matrix(p_string)

    return(H)




def ps_print_set(tuples,types,coeffs):

    print("\n\tSites     \t Type \t\t Coefficient\n",end="")
    print("-"*80,"\n")
    for j in range(len(tuples)):
        print(str(j),":\t",tuples[j],"\t\t", types[j], "\t\t", coeffs[j])
    print("")

    return()



def ps_print_set_merged(tuples):

    print("\nStrings     \t Coefficients\n",end="")
    print("-"*50,"\n")
    for j in range(len(tuples)):
        print(tuples[0],"\t\t", tuples[1])
    print("")