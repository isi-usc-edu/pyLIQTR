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
###     LINEARIZATION_TOOLS - Linearization for nonlinear ODE systems        ###
###                                                                          ###
################################################################################



import  numpy                as  np
import  scipy.integrate      as  spi
import  scipy.sparse         as  sps
import  multilinear_algebra  as  mla


class Linearizer:

    ##  Linearizer(coeffs,sparse=False)
    ## ------------------------------------------------------------------------
    ##
    ##  Set up linearization for the coefficient system <coeffs> using 
    ##  coordinate-based sparse matrix routines if the boolean keyword <sparse> 
    ##  is True.
    ##

    def __init__(self,coeffs,sparse=False):

        self._sparse      =  sparse
        self._N           =  coeffs[0].shape[0]

        if self._sparse:
            self._coeffs  =  []
            for F in coeffs:
                self._coeffs.append(sps.coo_array(F))
        else:
            self._coeffs  =  coeffs

        self._orders      =  []

        for F in self._coeffs:
            if (self._N != F.shape[0]):
                raise Exception('Coefficients must have consistent row count!')

            order = self._coeff_order(F)
            self._orders.append(order)

        
        self.__complex     =  False

        for F in self._coeffs:
            if (F.dtype == np.dtype('complex')):
                self.__complex  =  True



    ##  _coeff_order(F)
    ## -----------------------------------------------------------------------
    ##
    ##  Return the order of the term corresponding to 
    ##  coefficient set <F>
    ##

    def _coeff_order(self,F):
        order = int(np.log2(F.shape[1])/np.log2(F.shape[0]))
        return(order)

#######                              ########################################## 
#######    LINEARIZATION ROUTINES    ########################################## 
#######                              ########################################## 

    ##  carleman(N_lin)
    ## ------------------------------------------------------------------------
    ##
    ##  Perform an Carleman linearization to order <N_lin> for the system
    ##  specified in class constructor.  Sparse methods are used if the 
    ##  sparse=True keyword argument was passed to the constructor.
    ##

    def carleman(self,N_lin):
        
        ord_max  =  max(self._orders)
        ord_min  =  min(self._orders)
       
        # dimension of the system
        #
        udim   = int((self._coeffs[self._orders.index(ord_min)].shape[1])**(1/ord_min))
            
        # maximal column dimension
        #
        cdim   = int((udim**(ord_max+N_lin)-1)/(udim-1) - 1)

        self._ord_max  =  ord_max
        self._ord_min  =  ord_min
        self._cdim     =  cdim
        self._udim     =  udim

        self._N_lin    =  N_lin

        if self._sparse:
            self.__linop  =  self.__carleman_linearize_sparse(N_lin,ord_max,udim,cdim)
        else:
            self.__linop  =  self.__carleman_linearize_dense(N_lin,ord_max,udim,cdim)



    ##  __carleman_linearize_dense(N_lin,ord_max,udim,cdim)
    ## ------------------------------------------------------------------------
    ##
    ##  Perform an Carleman linearization using dense matrix methods.
    ##

    def __carleman_linearize_dense(self,N_lin,ord_max,udim,cdim):

        # template for column buffer
        #
        CB     = np.empty(shape=[0,cdim])
        
        for n in range(1,N_lin+ord_max):
            
            if self.__complex:        
                RB = np.empty(shape=[udim**n,0],dtype='complex')
            else:
                RB = np.empty(shape=[udim**n,0])
            
            for k in range(1,ord_max + N_lin):   
                kidx = k-n+1
                if (kidx in self._orders):
                    F_k = self._coeffs[self._orders.index(kidx)]
                    Ank = mla.mk_ksum(F_k,n,dim=udim)
                else:
                    F_k = np.zeros((udim**n,udim**k))
                    Ank = F_k

                RB = np.hstack((RB,Ank))
            CB = np.vstack((CB,RB))
        return(CB)
    


    ##  __carleman_linearize_dense(N_lin,ord_max,udim,cdim)
    ## ------------------------------------------------------------------------
    ##
    ##  Perform an Carleman linearization using sparse matrix methods.
    ##

    def __carleman_linearize_sparse(self,N_lin,ord_max,udim,cdim):

        # template for column buffer
        #
        if self.__complex:
            CB     = sps.coo_array(np.empty(shape=[0,cdim],dtype='complex'))
        else:
            CB     = sps.coo_array(np.empty(shape=[0,cdim]))

        
        for n in range(1,N_lin+ord_max):
            
            if self.__complex:
                RB = sps.coo_array(np.empty(shape=[udim**n,0],dtype='complex'))
            else:
                RB = sps.coo_array(np.empty(shape=[udim**n,0]))
            
            for k in range(1,ord_max + N_lin):   
                kidx = k-n+1
                if (kidx in self._orders):
                    F_k = self._coeffs[self._orders.index(kidx)]
                    Ank = mla.mk_ksum_s(F_k,n,dim=udim)
                else:
                    F_k = sps.coo_array((udim**n,udim**k))
                    Ank = F_k

                RB = sps.hstack((RB,Ank))
            CB = sps.vstack((CB,RB))
        return(CB)



    ##  ivp(u_0)
    ## ------------------------------------------------------------------------
    ##
    ##  Calcualte initial value vector for linearized system. Initial conditions
    ##  for the nonlinear system are passed as <u_0>.
    ##

    def ivp(self,u_0):
        
        if (self._udim != len(u_0)):
            raise Exception("ERROR: Initial value array is smaller than ODE system.")

        if (self._sparse):
            self.__uc_0  =   self.__ivp_sparse(u_0)
        else:
            self.__uc_0  =   self.__ivp_dense(u_0)



    ##  __ivp_dense(u_0)
    ## ------------------------------------------------------------------------
    ##
    ##  Calcualte initial value vector for linearized system. Initial conditions
    ##  for the nonlinear system are passed as <u_0>. Handle dense case.
    ##

    def __ivp_dense(self,u_0):

        if self.__complex:
            uc_0   = np.zeros(self._cdim,dtype='complex')
        else:
            uc_0   = np.zeros(self._cdim)

        
        lb = 0
        
        for n in range(1,self._ord_max+self._N_lin):
            
            ub           =  lb + self._udim**n
            uc_0[lb:ub]  =  mla.mk_n(u_0,n)
            lb           =  ub
            
        return(uc_0)


    ##  __ivp_sparse(u_0)
    ## ------------------------------------------------------------------------
    ##
    ##  Calcualte initial value vector for linearized system. Initial conditions
    ##  for the nonlinear system are passed as <u_0>. Handle dense case.
    ##    

    def __ivp_sparse(self,u_0):

        if self.__complex:
            uc_0   = sps.coo_array(np.empty(shape=[1,0],dtype='complex'))
        else:
            uc_0   = sps.coo_array(np.empty(shape=[1,0]))

                
        for n in range(1,self._ord_max+self._N_lin): 
            uc_0         =  sps.hstack((uc_0,mla.mk_n_s(u_0,n)))
            
        return(uc_0)



#######                              ########################################## 
#######    INTERFACE ROUTINES        ########################################## 
#######                              ########################################## 


    ## coeff(F)
    ## ------------------------------------------------------------------------
    ##
    ## Return orders of the supplied coefficient sets
    ##

    def orders(self):
        return(self._orders)
    


    ##  op()
    ## ------------------------------------------------------------------------
    ##
    ##  Return dense linearized evolution operator (op must be calculated first).
    ##

    def op(self):
        
        if self._sparse:
            return(self.__linop.toarray())
        else:
            return(self.__linop)
        


    ##  op_s()
    ## ------------------------------------------------------------------------
    ##
    ##  Return sparse linearized evolution operator if sparse methods were 
    ##  specified (op must be calculated first).
    ##

    def op_s(self,dense=False):
        
        if self._sparse:
            return(self.__linop)



    ##  op_s_data()
    ## ------------------------------------------------------------------------
    ##
    ##  Return constituent data from a sparse, linearized evolution operator as
    ##  a list with elements ((row,col), value).
    ##

    def op_s_data(self):

        if self._sparse:

            data     =  self.__linop.data
            col_idx  =  self.__linop.col
            row_idx  =  self.__linop.row

            coords   = list(zip(row_idx,col_idx))

            data     = list(zip(coords,data))

            return(data)



    ##  iv()
    ## ------------------------------------------------------------------------
    ##
    ##  Return dense initial value array for the linearized system. 
    ## 
    ##  The linearized system must be calculated and initial values for the
    ##  nonlinear system specified via ivp() before this method can be invoked.
    ##

    def iv(self):
        if self._sparse:
            return(self.__uc_0.toarray())
        else:
            return(self.__uc_0)
        

    ##  iv_s()
    ## ------------------------------------------------------------------------
    ##
    ##  Return sparse initial value array for the linearized system. 
    ## 
    ##  The sparse, linearized system must be calculated and initial values 
    ##  for the nonlinear system specified via ivp() before this method can 
    ##  be invoked.
    ##

    def iv_s(self):
        if self._sparse:
            return(self.__uc_0)


    ##  iv_s_data()
    ## ------------------------------------------------------------------------
    ##
    ##  Return data from sparse initial value array of the linearized system. 
    ##  Format is a list with elements ((row,col), value). 
    ##
    ##
    ##  The sparse, linearized system must be calculated and initial values 
    ##  for the nonlinear system specified via ivp() before this method can 
    ##  be invoked. 
    ##

    def iv_s_data(self):

        if self._sparse:

            data     =  self.__uc_0.data
            col_idx  =  self.__uc_0.col
            row_idx  =  self.__uc_0.row

            coords   = list(zip(row_idx,col_idx))

            data     = list(zip(coords,data))

            return(data)






# ## carlin(coeffs,orders,Nmax)
# ## -------------------------------------------------------------
# ##
# ## Perform Carleman lineariation for a
# ## nonlinear system with coefficient tensors
# ## as a list in <coeffs> of orders 
# ## specified in the list <orders>. Linearziation
# ## is taken out to an order of Nmax
# ##
# ## coeffs  =  list containing coefficient tensors
# ## orders  =  list with orders of coefficient
# ##            tensors
# ## Nmax    =  maximum order of 
# ##            linearization
# ##

# def carlin(coeffs,orders,Nmax):
    
#     maxord = max(orders)
#     minord = min(orders)
    
#     # dimension of the system
#     #
#     udim   = int((coeffs[orders.index(minord)].shape[1])**(1/minord))
        
#     # maximal column dimension
#     #
#     cdim   = int((udim**(maxord+Nmax)-1)/(udim-1) - 1)

#     # template for column buffer
#     #
#     CB     = np.empty(shape=[0,cdim])
    
#     for n in range(1,Nmax+maxord):
        
#         RB = np.empty(shape=[udim**n,0])
        
#         for k in range(1,maxord + Nmax):   
#             kidx = k-n+1
#             if (kidx in orders):
#                 F_k = coeffs[orders.index(kidx)]
#                 Ank = mla.mk_ksum(F_k,n,dim=udim)
#             else:
#                 F_k = np.zeros((udim**n,udim**k))
#                 Ank = F_k

#             RB = np.hstack((RB,Ank))
#         CB = np.vstack((CB,RB))
#     return(CB)
    



# ## carlin_ivp(u_0,maxord,Nmax)
# ## -------------------------------------------------------------
# ##
# ## Set up the initial value problem for 
# ## a Carleman linearization problem.
# ##
# ## u_0     =  initial condition vector
# ## maxord  =  maximum order of system
# ## Nmax    =  maximum order of 
# ##            linearization
# ##

# def carlin_ivp(u_0,maxord,Nmax):
    
#     udim   = len(u_0)
#     cdim   = int((udim**(maxord+Nmax)-1)/(udim-1) - 1)

#     uc_0   = np.zeros(cdim)
    
#     lb = 0
    
#     for n in range(1,maxord+Nmax):
        
#         ub           =  lb + udim**n
#         uc_0[lb:ub]  =  mla.mk_n(u_0,n)
#         lb           =  ub
        
#     return(uc_0)






## euler_int_fw(A,u_0,dt,T,b=0)
## -------------------------------------------------------------
##
## Forward Euler integrator.
##
## A    =  coefficient matrix for linear 
##         system
## u_0  =  initial solution vector
## dt   =  timestep
## T    =  maximal time
## b=0  =  inhomogeneous term 
##

# def euler_int_fw(A,u_0,dt,T,b=0):
    
#     Neq    =  len(u_0)
#     Nstep  =  int(np.ceil(T/dt))
     
#     u_sol  =  np.zeros((Neq,Nstep))
    
#     u = u_0
    
#     u_sol[:,0] = u_0[:]
    
#     for j in range(1,Nstep):
#         u_sol[:,j] = u_sol[:,j-1] + (A @ u_sol[:,j-1] + b)*dt
        
#     return(u_sol)


def euler_int_fw(A,u_0,dt,T,b=0):
    
    Neq    =  u_0.shape[-1]
    Nstep  =  int(np.ceil(T/dt))
     
    u_sol  =  np.zeros((Neq,Nstep))
    
    u = u_0
    
    u_sol[:,0] = u_0[:]
    
    for j in range(1,Nstep):
        u_sol[:,j] = u_sol[:,j-1] + (A @ u_sol[:,j-1] + b)*dt
        
    return(u_sol)
            


## prop_nonlinear(ops)
## -------------------------------------------------------------
##
## Propagator for a nonlinear system, for
## use with ODEPACK routines (e.g., odeint)
##
## u  =  current solution vector
## t  =  times
## coeffs  =  coefficient set
## orders  =  list of orders for terms
##

def prop_nonlinear(u,t,coeffs):

    get_order = lambda F : int(np.log2(F.shape[1])/np.log2(F.shape[0]))

    orders      =  []

    for F in coeffs:
        order = get_order(F)
        orders.append(order)
    
    is_complex     =  False

    for F in coeffs:
        if (F.dtype == np.dtype('complex')):
            is_complex  =  True

    if is_complex:
        du = np.zeros(len(u),dtype='complex')
    else:
        du = np.zeros(len(u))


    for od in orders:
        F  = coeffs[orders.index(od)]
        up = mla.mk_n(u,od)
        du += F @ up
        
    return du
    

