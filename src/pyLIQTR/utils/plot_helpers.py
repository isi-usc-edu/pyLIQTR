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
import numpy.linalg         as nla
import scipy.linalg         as sla
import scipy.special        as sfn
import matplotlib.pyplot    as plt
from IPython.display        import display, clear_output

from   pyLIQTR.sim_methods.expander               import Expander

from IPython.display import set_matplotlib_formats
set_matplotlib_formats('svg')

plt.rcParams['text.usetex'] = False


def ig_plot_scalar_qsp(xvec,p_x,qsp_x):

    plt.plot(xvec,p_x,linewidth=3,color='k')
    plt.plot(xvec,qsp_x,"--",linewidth=2,color='bisque')
    plt.legend(["$\pi_d$","QSP"],fontsize=11)
    plt.tick_params(axis='both',which='both',direction='in')
    plt.xlabel('$x$',fontsize=11)
    plt.ylabel('$f(x)$',fontsize=11)
    plt.title('QSP Simulation',fontsize=11)
    plt.grid()

def ig_plot_convergence(conv):

    plt.semilogy(conv,'.-')
    plt.tick_params(axis='both',which='both',direction='in')
    plt.xlabel('$x$',fontsize=11)
    plt.ylabel('$L^\infty$  Error',fontsize=11)
    plt.title('Remez Exchange Convergence',fontsize=11)
    plt.grid()



def ig_plot_fit(xpts,p_x,f_x,r_x):

    # p_x = poly fit;  f_x = exact function, r_x = residuals

    fig,axs = plt.subplots(2)


    axs[0].plot(xpts,f_x,linewidth=2,color="k")
    axs[0].plot(xpts,p_x,"--",color="r")
    axs[0].tick_params(axis='both',which='both',direction='in')
    axs[0].set_xlabel('$x$',fontsize=11)
    axs[0].set_ylabel('$f(x)$',fontsize=11)
    axs[0].set_title("Function Approximation",fontsize=11)
    axs[0].grid()

    # axs[1].plot(xpts,r_x,linewidth=2,color="b")
    # axs[1].tick_params(axis='both',which='both',direction='in')
    # axs[1].set_xlabel('$x$',fontsize=11)
    # axs[1].set_ylabel('$\pi_d(x) - f(x)$',fontsize=11)
    # axs[1].set_title("Fitting Residuals",fontsize=11)
    # axs[1].grid()

    axs[1].semilogy(xpts,np.abs(r_x),linewidth=2,color="b")
#    axs[1].plot(xpts,r_x,linewidth=2,color="b")
    axs[1].tick_params(axis='both',which='both',direction='in')
    axs[1].set_xlabel('$x$',fontsize=11)
    axs[1].set_ylabel('$| \pi_d(x) - f(x) |$',fontsize=11)
    axs[1].set_title("Fitting Residuals",fontsize=11)
    axs[1].grid()

    fig.tight_layout()








def plot_func_error(xpts,f_vals,p_vals,eps0,name):

    fig,axs = plt.subplots(2)

    axs[0].plot(xpts,f_vals,linewidth=3,color="k")
    axs[0].plot(xpts,p_vals,"--",color="w")
    axs[0].tick_params(axis='both',which='both',direction='in')
    axs[0].set_xlabel('$x$',fontsize=11)
    axs[0].set_ylabel('$f(x)$',fontsize=11)
    axs[0].set_title(name,fontsize=11)
    axs[0].grid()

    a = np.min(xpts)
    b = np.max(xpts)

    err = np.abs(f_vals - p_vals)

    axs[1].semilogy(xpts,err,linewidth=3,color="k")
    axs[1].semilogy([a,b],[eps0,eps0],"--",color="r")
    axs[1].tick_params(axis='both',which='both',direction='in')
    axs[1].set_xlabel('$x$',fontsize=11)
    axs[1].set_ylabel('$ | f(x) - p(x) |$',fontsize=11)
    axs[1].grid()
#    axs[1].set_ylim(np.min(err)/10,eps0*10)


    fig.tight_layout()





#
#
#
# fig,axs = plt.subplots(2,2)
#
# fval_cos = f_cos(xpts,tau)
# fval_sin = f_sin(xpts,tau)
#
# err_cos = np.abs(fval_cos - p_cos)
# err_sin = np.abs(fval_sin - p_sin)
#
# axs[0,0].plot(xpts,fval_cos,linewidth=3,color="k")
# axs[0,0].plot(xpts,p_cos,"--",color="w")
# #axs[0,0].legend(["Exact","Polynomial"],fontsize=10)
# axs[0,0].tick_params(axis='both',which='both',direction='in')
# axs[0,0].set_xlabel('$x$',fontsize=11)
# axs[0,0].set_ylabel('$f(x)$',fontsize=11)
# axs[0,0].set_title('$f(x) = \cos(t \cdot x)$',fontsize=11)
# axs[0,0].grid()
#
# axs[0,1].plot(xpts,fval_sin,linewidth=3,color="k")
# axs[0,1].plot(xpts,p_sin,"--",color="w")
# #axs[0,1].legend(["Exact","Polynomial"],fontsize=10)
# axs[0,1].tick_params(axis='both',which='both',direction='in')
# axs[0,1].set_xlabel('$x$',fontsize=11)
# axs[0,1].set_ylabel('$f(x)$',fontsize=11)
# axs[0,1].set_title('$f(x) = \sin(t \cdot x)$',fontsize=11)
# axs[0,1].grid()
#
# axs[1,0].semilogy(xpts,err_cos,linewidth=3,color="k")
# axs[1,0].semilogy([-1.0,1.0],[eps0,eps0],"--",color="r")
# #axs[1,0].legend(["Exact","Polynomial"],fontsize=10)
# axs[1,0].tick_params(axis='both',which='both',direction='in')
# axs[1,0].set_xlabel('$x$',fontsize=11)
# axs[1,0].set_ylabel('$ | f(x) - p(x) |$',fontsize=11)
# #axs[1,0].set_title('$f(x) = \sin(t \cdot x)$',fontsize=11)
# axs[1,0].grid()
# axs[1,0].set_ylim(1e-16,1e-5)
#
#
# axs[1,1].semilogy(xpts,err_sin,linewidth=3,color="k")
# axs[1,1].semilogy([-1.0,1.0],[eps0,eps0],"--",color="r")
# #axs[1,0].legend(["Exact","Polynomial"],fontsize=10)
# axs[1,1].tick_params(axis='both',which='both',direction='in')
# axs[1,1].set_xlabel('$x$',fontsize=11)
# axs[1,1].set_ylabel('$| f(x) - p(x) |$',fontsize=11)
# #axs[1,1].set_title('$f(x) = \sin(t \cdot x)$',fontsize=11)
# axs[1,1].grid()
# axs[1,1].set_ylim(1e-16,1e-5)
#
#
# fig.tight_layout()
