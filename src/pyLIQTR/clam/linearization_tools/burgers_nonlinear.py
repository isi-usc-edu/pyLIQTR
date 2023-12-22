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
###           BURGERS_NONLINEAR - Viscous Burger's Equation in 1D            ###
###                                                                          ###
################################################################################


import  numpy                as  np
import  scipy.sparse         as  sps
import  multilinear_algebra  as  mla



def vbe_initial_conditions(N,dx,vmax,nk):

    xb = N*dx/2

    x    =  np.linspace(-xb,xb,N)
    u_0  = -vmax*np.sin(2*np.pi*nk*x/(2*xb))

    return(u_0)



def vbe_system(L,N,nu,gamma=0.0,
                      fixed_boundary=True,
                      driven=False,
                      T_max=1.0,
                      dt=0.1,
                      f_drive=2.0,
                      vmax=1.0):

    F_1  =  vbe_coeffs_linear_1d(L,N,nu,gamma=gamma,fixed_boundary=fixed_boundary)
    F_2  =  vbe_coeffs_nonlinear_1d(L,N,fixed_boundary=fixed_boundary)

    if driven:
        F_0        =  vbe_drive(L,N,T_max,dt,vmax,f_drive=f_drive)
        coeff_set  =  [F_0,F_1,F_2]
    else:
        coeff_set = [F_1,F_2]

    return(coeff_set)    



def vbe_coeffs_linear_1d(L,N,nu,gamma=0.0,fixed_boundary=True):
    
   dx = L/(N-1)

   D  = np.ones(N)   * (-2*nu/dx**2)
   Dp = np.ones(N-1) * (nu/dx**2)

   F = np.diag(D) + np.diag(Dp, 1) + np.diag(Dp, -1) - gamma*np.eye(N)

   # Dirichlet conditions
   if fixed_boundary:
      F[0,:]   =  0
      F[-1,:]  =  0

   return(F)



def vbe_coeffs_nonlinear_1d(L,N,fixed_boundary=True):

   dx = L/(N-1)

   F = np.zeros((N,N**2))

   F[1,0] = 1.0/(4.0*dx)

   for k in range(1,N-1):
      F[k-1,N*k] = -1.0/(4.0*dx)
      F[k+1,N*k] = 1.0/(4.0*dx)

   F[N-2,(N**2)-1] = -1.0/(4.0*dx)

   # fixed boundary (Dirichlet) conditions
   if fixed_boundary:
      F[0,:]   =  0
      F[-1,:]  =  0

   return(F)




def vbe_drive(L,N,T_max,dt,vmax,f_drive=2.0):

    dx   =  L/(N-1)
    xb   =  N*dx/2
    Nt   =  int(np.ceil(T_max / dt))

    x_vec    =  np.linspace(-xb,xb,N)
    t_vec    =  np.linspace(0.0,T_max,Nt+1)

    Nt = int(np.ceil(T_max / dt))

    f = lambda x, t : vmax * np.exp( -(x-L/4)**2 / (2*(L/32)**2) ) * np.cos(f_drive*np.pi*t)

    F0 = np.zeros((N,Nt+1))

    ix = 0

    for x in x_vec:
        it = 0
        for t in t_vec:
            F0[ix,it] = f(x,t)
            it += 1
        ix += 1

    return(F0)
            