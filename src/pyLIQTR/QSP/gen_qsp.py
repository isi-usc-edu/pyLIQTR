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


import re          as _re
import os          as _os
import sys         as _sys
import numpy       as _np
import argparse    as _argp
import subprocess  as _subp
import warnings    as _warn
import time        as _time
from   collections import OrderedDict as _odict

from numpy        import pi
from scipy        import special as _spc
from scipy        import sparse  as _sps
from scipy        import optimize as _spo

import pyLIQTR.QSP.Hamiltonian as _pyHam
import pyLIQTR.QSP.qsp_helpers as _pyQSPh
import pyLIQTR.QSP.QSP         as _pyQSP
import pyLIQTR.QSP.qspangles   as _qa
import cirq

import warnings



class QSP_Simulator():

    def __init__(self, timestep_vec, angles, init_state:cirq.Circuit, qsp_H):
        self.timestep_vec = timestep_vec
        self.angles       = angles
        self.init_state   = init_state
        self.qsp_H        = qsp_H
        self.sim_results  = []
        self.simulator    = cirq.Simulator(dtype=_np.complex128)

    def do_sim(self):
        for idx, (t_, phis_) in enumerate(zip(self.timestep_vec, self.angles)):
            if idx==0:
                self.sim_results.append(self.simulator.simulate(self.init_state).final_state_vector)
            else:
                tmp_sim_circuit = self.init_state +\
                                  _pyQSP.QSP(phis_, self.qsp_H, self.qsp_H.problem_size).circuit()
                self.sim_results.append(self.simulator.simulate(tmp_sim_circuit).final_state_vector)

        return self.sim_results

def getlogepsilon(tau,steps,nsegs=1):
    if _np.isclose(tau,0):
        val = -_np.inf
    else:
        q  = steps//2 + 1
        val  = _np.log(32)
        val += _np.log(tau/2) * q
        val -= _spc.gammaln(q+1) # == log(q!)
        val -= _np.log(nsegs) * (q-1)
    return val

def get_steps_from_logeps(true_eps,tau,nsegs=1):
    def f(x):
        return abs(true_eps - getlogepsilon(tau,steps=x,nsegs=1))
    root = _spo.minimize_scalar(f)

    approx_steps = int(root.x)
   
    #find closest step divisible by 4
    steps1 = approx_steps - (approx_steps % 4)
    steps2 = (approx_steps + 4) - (approx_steps % 4)
    if (approx_steps - steps1) > (steps2-approx_steps):
        steps = steps2 
    else:
        steps = steps1
    
    return steps,f(steps)

def diffphis(phis):
    phis = _np.append(phis[:1],_np.append(_np.diff(phis),-phis[-1:])+pi)
    phis = ((phis+pi) % (2*pi)) - pi
    return phis


def get_phis(hamiltonian, simtime,req_prec, balance = False, ex=False, steps_only=False):
    msgblk = """\n
    get_phis will be deprected by 'compute_hamiltonian_angles' in a future release

    The equivalent call is:
    \tcompute_hamiltonian_angles(<Hamiltonian>,<simtime>,<req_prec>,<mode>)
    \t\tSupported modes: random (random angles),legacy("old" generation)
    \t\tUpcoming modes: expander (a more powerful, faster, flexible angle calculation tool)
    \n
    """
    warnings.warn(msgblk)
    
    return legacy_get_phis(hamiltonian, simtime, req_prec, balance, ex, steps_only)

def compute_hamiltonian_angles(hamiltonian, simtime, req_prec, mode : str = "random" , max_iter = None):
    """
    """
    if mode not in ["legacy","random"]:
        raise ValueError("Mode must be one of legacy/random")
    
    if mode == "legacy":
        return legacy_get_phis(hamiltonian, simtime, req_prec)
    elif mode == "random":
        tau = simtime*hamiltonian.alpha
        steps,closeval = get_steps_from_logeps(_np.log(req_prec),tau,1)
        while getlogepsilon(tau,steps)>_np.log(req_prec):
            steps+=4
        phis = _np.pi*_np.random.random(steps+1)
        error = None
        return phis,error
    elif mode == "expander":
        return None


def legacy_get_phis(hamiltonian,simtime,req_prec, balance=False, ex=False, steps_only=False):
    #return anything required for QSP to be properly setup.
    tau = simtime*hamiltonian.alpha
    #have tau and epsilon, backtrack in order to get steps
    steps,closeval = get_steps_from_logeps(_np.log(req_prec),tau,1)
    # print(':------------------------------------------')
    # print(f': Steps = {steps}')
    while getlogepsilon(tau,steps)>_np.log(req_prec):
        steps+=4
    logepsilon = getlogepsilon(tau,steps)
    actepsilon = _np.exp(logepsilon)
    
    if steps_only:
        return steps

    phis,err = _calc_phis(tau,steps,prec=None, balance = balance, ex=ex)
    # print(f': Steps again = {steps}')
    # print(f': phis        = {phis}')
    # print(f': err         = {err}')
    print(f"Req eps = {req_prec}\tAct eps = {err}")
    return phis,err

def _calc_phis(tau,steps,prec,balance,ex):
    err=0
    if tau == 0:
        phis = _np.zeros(steps+1)
    elif steps >= 4:
        # calculate the angles
        phis,err = _qa.calc_angles(tau,steps,prec=prec)
    else:
        phis = _np.array([0,pi/2,0])
        

    # diff = _np.append(phis[:1],_np.append(_np.diff(phis),-phis[-1:])+pi)
    # diff = ((diff+pi) % (2*pi)) - pi
    if len(phis) % 2 == 0:
        phis = diffphis(phis)
    # else:
    #     phi0 = phis[0]
    #     phis = diffphis(phis[1:])
    #     phis[0] += phi0

    if balance:
        phis[ ::2] = (phis[ ::2] + phis[  ::-2]) / 2
        phis[1::2] = (phis[1::2] - phis[-2::-2]) / 2

    if ex:
        phis = [0] + phis[::-1].tolist() + [0]

    return phis,err

def my_simulate(init, hamiltonian, simtime, rseed=0):
    
    reps = 1
    tau = simtime*hamiltonian.alpha
    rand = _np.random.RandomState(rseed)
    terms = hamiltonian.terms
    paulis = [t[0] for t in terms]
    alphas = _np.array([t[1] for t in terms])
    size = len(paulis[0])
    #init = ''.join(rand.choice(tuple('01+-yn'),size=size))
    state = initstate(init,size)
    if size >= 15: logger('constructing hamiltonian matrix')
    hmat = get_hmat(paulis,alphas,dp=True)

    if size >= 15: logger('sparsifying hamiltonian matrix')
    hmat = 1j*tau*_sps.csc_matrix(hmat)

    if size >= 15: logger('exponentiating hamiltonian matrix ({} reps)'.format(reps))
    if reps > 1:
        states = _sps.linalg.expm_multiply(hmat, state,1,reps,reps,endpoint=True)
    else:
        states = _sps.linalg.expm_multiply(hmat, state)
        states = states.reshape(1,*states.shape)

    #del hmat # its big
    return states, hmat

def initstate(init,size):
    vec = _np.zeros(2**size,dtype=_np.complex)
    val = 1.0 # 2.0**(-len(tuple(c for c in init if c in '+-yn')))
    idx = 0
    for i,b in enumerate(init[::-1].lower()):
        if b == '1':
            idx += 2**i
    st = {idx:val}
    cs = {'+':1,'-':-1,'y':1j,'n':-1j}
    for i,b in enumerate(init[::-1].lower()):
        if b in '+-yn':
            for k,v in list(st.items()):
                st[k+2**i] = cs[b] * v
    for i,v in st.items():
        vec[i] = v
    vec /= _np.sqrt(sum(_np.square(_np.abs(vec))))
    return vec

