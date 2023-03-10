"""
DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.

This material is based upon work supported by the Under Secretary of Defense for
Research and Engineering under Air Force Contract No. FA8702-15-D-0001. Any opinions,
findings, conclusions or recommendations expressed in this material are those of the
author(s) and do not necessarily reflect the views of the Under Secretary of Defense
for Research and Engineering.

© 2022 Massachusetts Institute of Technology.

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
import time        as _time
import pickle      as _pkl
import numbers     as _num
import numpy       as _np
import argparse    as _argp
import subprocess  as _subprc
import cachetools  as _ct
import mpmath      as _mp
from   collections import OrderedDict as _odict
from   scipy       import special     as _spc
from   scipy       import optimize    as _opt
from   scipy       import signal      as _sgnl


_dir_path = _os.path.dirname(_os.path.realpath(__file__))
_cache    = _os.path.join(_dir_path,'.qspangles.cache')
_cache_size = 100000

parser = _argp.ArgumentParser()
parser.add_argument('-o','--destination',nargs='?',default=None,const=True,help='output file (default: stdout)')
parser.add_argument('-t','--tau'  ,nargs='+' ,default=None,help='tau')
parser.add_argument('-m','--steps',type=int  ,default=None,help='number of steps in iterate, must be a multiple of 4')
parser.add_argument('-e','--epsilon',type=float,default=None,help='fit tau to epsilon')
parser.add_argument('-p','--prec' ,type=int  ,default=None,help='bits to use for angle generation, smaller=faster but too small will break')
parser.add_argument('-c','--check',action='store_true')
parser.add_argument('-f','--force',action='store_true')
parser.add_argument('--mt',type=int,default=None,help='number of threads used by mpsolve')
parser.add_argument('--cache',nargs='?',default=_cache,const=None,help='phase cache')

check = False
if __name__ == '__main__':
    args,rem = parser.parse_known_args()
    check = args.check

# these things take a while to load
if not check:
    import sympy       as _sym
    import portalocker as _pl

class PrecErr(Exception): pass

def logger(*args,**kwargs):
    end = kwargs.pop('end','\n')
    app = kwargs.pop('append',None)
    if app is None:
        # _sys.stderr.write('[qspangles] '+' '.join(map(str,args))+end,**kwargs)
        _sys.stderr.write('[{}] {}{}'.format(_time.strftime('%H:%M:%S'),' '.join(map(str,args)),end),**kwargs)
    else:
        _sys.stderr.write('{}'.format(app))
    _sys.stderr.flush()

def parse_tau(s,steps=None,prec=3):
    if isinstance(s,str) and s.startswith('-'):
        assert steps is not None
        return round(steps/_np.exp(1)-float(s[1:]),prec)
    elif isinstance(s,str) and s.startswith('+'):
        assert steps is not None
        return round(steps/_np.exp(1)+float(s[1:]),prec)
    elif isinstance(s,_num.Number) and s < 0:
        assert steps is not None
        return round(steps/_np.exp(1)+float(s),prec)
    elif s:
        return round(float(s),prec)

bcache = _ct.LRUCache(800000)
@_ct.cached(bcache)
def _cbinomial(n,k):
    return _mp.binomial(n,k)

dcache = {}
def cbinomial(n,k):
    k = min(k,n-k)
    if k > 0:
        # return _cbinomial(n,k)
        val = dcache.get((n,k))
        if val is None:
            val = _mp.binomial(n,k)
            dcache[(n,k)] = val
        return val
    elif k == 0:
        return _mp.mpf(1.0)
    else:
        return _mp.mpf(0.0)
    # return _cbinomial(n,min(k,n-k))

def empirical_err(tau,l,N=500):
    q = l // 2 + 1
    #  _mp.mpf(4.0)*(tau/2)**(N//2+1) / _mp.factorial(N//2+1)
    rl = sum( abs(_mp.besselj(k,tau)) for k in range(q  ,q+2*N  ,2) )
    im = sum( abs(_mp.besselj(k,tau)) for k in range(q+1,q+2*N+1,2) )
    return 2 * _mp.sqrt( rl**2 + im**2 )


def empirical_tau(steps,epsilon,N=500):
    q = steps//2 + 1
    def fn(tau):
        rl = abs(_spc.jn(_np.arange(q  ,q+N,2),tau)).sum()
        im = abs(_spc.jn(_np.arange(q+1,q+N,2),tau)).sum()
        return (epsilon - 2*_np.sqrt( rl**2 + im**2 ))**2
    res = _opt.minimize_scalar(fn, method='bounded', bounds=(max(1,steps/_np.exp(1)-30),steps+30))
    if not res.success:
        print(res)
    assert res.success
    return round(res.x,2)

def j0int(tau):
    return tau*( (_np.pi/2)*(_spc.struve(0,tau)*_spc.j1(tau) - _spc.struve(1,tau)*_spc.j0(tau)) + _spc.j0(tau) )

def emperr(tau,steps,mp=False):
    q = steps//2 + 1
    if mp:
        rl = 1 - _mp.besselj(0,tau)    - 2*sum(_mp.besselj(k,tau) for k in range(2,q,2))
        im =     _mp.besselj(0,tau,-1) - 2*sum(_mp.besselj(k,tau) for k in range(1,q,2))
        return _mp.sqrt(rl**2 + im**2)
    else:
        rl = 1 - _spc.j0(tau) - 2*_spc.jn(_np.arange(2,q,2),tau).sum()
        im =       j0int(tau) - 2*_spc.jn(_np.arange(1,q,2),tau).sum()
        er = abs(rl + 1j*im)
        if er < 1e-10:
            return emperr(tau,steps,mp=True)
        else:
            return er

def empirical_tau(steps,epsilon,mp=False,**kwargs):
    q = steps//2 + 1
    if epsilon > 1.0:
        tau1 = empirical_tau(steps,1.0,mp=mp,**kwargs)
        return tau1 + epsilon - 1.0
    elif mp:
        def fn(tau):
            rl = 1 - _mp.besselj(0,tau)    - 2*sum(_mp.besselj(k,tau) for k in range(2,q,2))
            im =     _mp.besselj(0,tau,-1) - 2*sum(_mp.besselj(k,tau) for k in range(1,q,2))
            return epsilon - _mp.sqrt(rl**2 + im**2)
    else:
        q1s = _np.arange(1,q,2)
        q2s = _np.arange(2,q,2)
        def fn(tau):
            rl = 1 - _spc.j0(tau) - 2*_spc.jn(q2s,tau).sum()
            im =       j0int(tau) - 2*_spc.jn(q1s,tau).sum()
            return epsilon - abs(rl + 1j*im)
    res = _opt.root_scalar(fn,bracket=(0,steps),**kwargs)
    assert res.converged, 'root_scalar failed:\n{}'.format(res)
    return round(res.root,2)


def norm_fourier(tau,l,win=None,*args):
    assert l % 4 == 0
    trunc = l // 2
    q     = trunc + 1
    tau   = _mp.mpf(tau)
    # error = _mp.mpf(4.0)*(tau/2)**q / _mp.factorial(q)
    error = empirical_err(tau,l)
    logger('minimum error (empirical): {}'.format(float(error)))
    nudge = _mp.mpf(1.0)/(1.0 + error)

    if win is None:
        am = [       nudge * _mp.besselj(0,tau) ] \
           + [ 2.0 * nudge * _mp.besselj(2*k+2,tau) for k in range(trunc//2) ]
        cm = [ 2.0 * nudge * _mp.besselj(2*k+1,tau) for k in range(trunc//2) ]
    else:
        assert 0
        wd = win(l+1,*args,sym=1)[(l+1)//2:]
        print(wd)
        am = [       wd[0    ] * _mp.besselj(0,tau) ] \
           + [ 2.0 * wd[2*k+1] * _mp.besselj(2*k+2,tau) for k in range(trunc//2) ]
        cm = [ 2.0 * wd[2*k  ] * _mp.besselj(2*k+1,tau) for k in range(trunc//2) ]
        nudge /= sum(am)
        am = [ a*nudge for a in am ]
        cm = [ c*nudge for c in cm ]
        
    # am2ap():
    ap = _mp.matrix(trunc+1,1)
    for u in range(trunc+1):
        ap[u] = _mp.mpf(-1)**u * \
            sum( am[k] * sum( cbinomial(4*k,2*j) * cbinomial(j,2*k-u) for j in range(2*k-u,2*k+1) ) \
                 for k in range((u+1)//2,trunc//2+1) )
    del am

    # for u in range(trunc+1):
    #     sgn = -1 if u&1 else 1
    #     for k in range((u+1)//2,trunc//2+1):
    #         avals[u,k] = sgn * sum( cbinomial(4*k,2*j) * cbinomial(j,2*k-u) for j in range(2*k-u,2*k+1) )

    # for u in range(trunc+1):
    #     sgn = -1 if u&1 else 1
    #     for k in range((u+1)//2,trunc//2):
    #         cvals[u,k] = sgn * sum( cbinomial(4*k+2,2*j+1) * cbinomial(2*k-j,2*k-u) for j in range(u+1) )
        
    # cm2cp():
    cp = _mp.matrix(trunc  ,1)
    for u in range(trunc):
        cp[u] = _mp.mpf(-1)**u * \
            sum( cm[k] * sum( cbinomial(4*k+2,2*j+1) * cbinomial(2*k-j,2*k-u) for j in range(u+1) ) \
                 for k in range((u+1)//2,trunc//2) )
    del cm
    return ap, cp
    
def amcm(tau,l):
    assert l % 4 == 0
    trunc = l // 2
    q     = trunc + 1
    tau   = _mp.mpf(tau)
    # error = _mp.mpf(4.0)*(tau/2)**q / _mp.factorial(q)
    error = empirical_err(tau,l)
    logger('minimum error (empirical): {}'.format(float(error)))
    nudge = _mp.mpf(1.0)/(1.0 + error)

    am = [       nudge * _mp.besselj(0,tau) ] \
       + [ 2.0 * nudge * _mp.besselj(2*k+2,tau) for k in range(trunc//2) ]
    cm = [ 2.0 * nudge * _mp.besselj(2*k+1,tau) for k in range(trunc//2) ]
    return am,cm

def am2ap(am,cm):
    trunc = 2*len(cm)
    l     = 2*trunc
    q     = trunc + 1
    assert l % 4 == 0
    assert len(am) == len(cm) + 1

    ap = _mp.matrix(trunc+1,1)
    for u in range(trunc+1):
        ap[u] = _mp.mpf(-1)**u * \
            sum( am[k] * sum( cbinomial(4*k,2*j) * cbinomial(j,2*k-u) for j in range(2*k-u,2*k+1) ) \
                 for k in range((u+1)//2,trunc//2+1) )
    del am

    # for u in range(trunc+1):
    #     sgn = -1 if u&1 else 1
    #     for k in range((u+1)//2,trunc//2+1):
    #         avals[u,k] = sgn * sum( cbinomial(4*k,2*j) * cbinomial(j,2*k-u) for j in range(2*k-u,2*k+1) )

    # for u in range(trunc+1):
    #     sgn = -1 if u&1 else 1
    #     for k in range((u+1)//2,trunc//2):
    #         cvals[u,k] = sgn * sum( cbinomial(4*k+2,2*j+1) * cbinomial(2*k-j,2*k-u) for j in range(u+1) )
        
    # cm2cp():
    cp = _mp.matrix(trunc  ,1)
    for u in range(trunc):
        cp[u] = _mp.mpf(-1)**u * \
            sum( cm[k] * sum( cbinomial(4*k+2,2*j+1) * cbinomial(2*k-j,2*k-u) for j in range(u+1) ) \
                 for k in range((u+1)//2,trunc//2) )
    del cm
    return ap, cp
    
    am = _mp.matrix(trunc//2+1,1)
    cm = _mp.matrix(trunc//2  ,1)
    am[0] = _mp.besselj(0,tau)
    for k in range(0,trunc,2):
        am[k//2+1] = 2*_mp.besselj(k+2,tau)
        cm[k//2  ] = 2*_mp.besselj(k+1,tau)
    am /= 1 + error
    cm /= 1 + error

       
    ap1 = _mp.matrix([
            sum( am[k//4] * \
                sum( (-1)**((k-u)//2)  \
                    * cbinomial(k,j) \
                    * cbinomial(j//2,(k-u)//2) \
                for j in range(k-u,k+1,2) ) \
            for k in range(u,l+1,2) \
            if  k % 4 == 0 ) \
        for u in range(0,l+1,2) ])

    # cm2cp():
    cp1 = _mp.matrix([ \
            sum( cm[k//4] * (-1)**((u-1)//2) \
                      * cbinomial(k,j) \
                      * cbinomial((k-j-1)//2,(k-u-1)//2) \
            for j in range(1,u+1,2) \
            for k in range(u+1,l+1,2) \
            if k % 4 == 2 ) \
        for u in range(1,l+1,2) ])
    
    return ap,ap1,cp,cp1

# Step 4: Constructing unspecified pairs without the unitarity condition
# fcache = {}
# def ftilde(t,au,cu):
#     l = 2*(len(au)-1)
#     assert l == 2*len(cu)
# 
#     p1_2 = _mp.matrix([cbinomial(2*l,k) for k in range(2*l+1)])
#     p2   = _mp.matrix(
#     sum( (-1)**k * cbinomial(2*l
#     
#     p1 = (1+t)**l
#     # atilde:
#     p2 = sum( aa * (1-t)**(2*i) * (1+t)**(l-2*i) for i,aa in enumerate(au) )
#     # ctilde * ((1-t**2)/(1+t**2)):
#     p3 = 2 * (1-t) * sum( cc * (4*t)**i * (1+t)**(l-2*i-2) for i,cc in enumerate(cu) )
# 
#     p2s = [ (1-t)**(2*i) * (1+t)**(l-2*i) for i in range(len(au)) ]
#     p3s = [ (4*t)**i * (1+t)**(l-2*i-2)   for i in range(len(cu)) ]
#     
#     fcache['p1'] = p1.copy()
#     fcache['p2'] = p2.copy()
#     fcache['p3'] = p3.copy()
#     
#     p2 = 2 * (1-t) * 
#     return p1**2 - p2**2 - t * p3**2

def ftilde(t,au,cu):
    l = 2*(len(au)-1)
    assert l == 2*len(cu)

    p1 = (1+t)**l
    # atilde:
    p2 = sum( aa * (1-t)**(2*i) * (1+t)**(l-2*i) for i,aa in enumerate(au) )
    # ctilde * ((1-t**2)/(1+t**2)):
    p3 = 2 * (1-t) * sum( cc * (4*t)**i * (1+t)**(l-2*i-2) for i,cc in enumerate(cu) )

    return p1**2 - p2**2 - t * p3**2

    p1 = (1+t**2)**l
    # atilde:
    # p2 = sum(cbinomial(2*i,j)*cbinomial(l-2*i,i-j)  (1-t**2)**(2*i) * (1+t**2)**(l-2*i) for i,aa in enumerate(au) )
    p2 = sum( aa * (1-t**2)**(2*i) * (1+t**2)**(l-2*i) for i,aa in enumerate(au) )
    # ctilde * ((1-t**2)/(1+t**2)):
    p3 = (1-t**2) * sum( cc * (2*t)**(2*i+1) * (1+t**2)**(l-2*i-2) for i,cc in enumerate(cu) )

    return p1**2 - p2**2 - p3**2


def iCondVague   (x,tol): return abs(x.real)   < tol and abs(x.imag-1) < tol
def rCondVague   (x,tol): return abs(x.imag)   < tol and x.real > tol + 1
def iotaCondVague(x,tol): return abs(x.real)   < tol and x.imag > tol + 1
def uCondVague   (x,tol): return abs(abs(x)-1) < tol and x.real > tol and x.imag > tol # on unit circle
def cCondVague   (x,tol): return abs(x)-1      > tol and x.real > tol and x.imag > tol

def div2(x): return x//2 if x%2 == 0 else x/2

def frVague(t,roots,tol):
    prod = t**div2(roots.count(0)) * (t**2-1)**div2(roots.count(1))
    
    # n0    = sum( 1 for x in roots if abs(x) < tol )
    # n1    = sum( 1 for x in roots if abs(abs(x)-1) < tol )

    # roots[k] == conj(roots[k+1]), so only got through first
    for x in roots:
        if rCondVague(x,tol):
            e = x**2 + x**(-2)
            # prod *= t**4 - e*t**2 + 1
            prod *= _sym.sqrt(t**4 - t**2*(x**2+x**(-2))+1)
        # elif abs(x) < tol:
            
    return prod

def fiVague(t,roots,tol):
    # nsiv = len(SiVague(roots))
    nsiv = sum(iCondVague(x,tol) for x in roots)
    return (t**2+1)**div2(nsiv)

def twosquares(r,s,u,v):
    g = r*u + s*v
    h = r*v - s*u
    return g,h

def fiota(t,rootlist):
    root = rootlist[0]
    r = t**2-1
    s = t*(root.imag + 1/root.imag)
    if len(rootlist) == 1:
        return r,s
    else:
        u,v = fiota(t,rootlist[1:])
        return twosquares(r,s,u,v)

def fu(t,rootlist):
    root = rootlist[0]
    r = t**2-1
    s = 2*t*_mp.sin( _mp.arg(root) )
    if len(rootlist) == 1:
        return r,s
    else:
        u,v = fu(t,rootlist[1:])
        return twosquares(r,s,u,v)

def fc(t,rootlist):
    root = rootlist[0]
    rarg = _mp.arg(root)
    XX = 1/abs(root)**4 + abs(root)**4 + 2*_mp.cos(4*rarg) + 2
    YY = -(1/abs(root)**2 + abs(root)**2) * 2*_mp.cos(2*rarg)
    aa = 2 + _mp.sqrt(XX + 2*YY + 2)
    bb = _mp.sqrt(YY + 2*aa)/2
    r  = t**4 - t**2*aa + 1
    s  = 2*(t**3-t)*bb
    if len(rootlist) == 1:
        return r,s
    else:
        u,v = fc(t,rootlist[1:])
        return twosquares(r,s,u,v)

def calghVague(t,roots,tol):
    SuVague    = [ x for x in roots if uCondVague    (x,tol) ]
    SiotaVague = [ x for x in roots if iotaCondVague (x,tol) ]
    ScVague    = [ x for x in roots if cCondVague    (x,tol) ]
    nu     = len(SuVague)
    niota  = len(SiotaVague)
    nc     = len(ScVague)
    logger('calgh: nu={}, nc={}, niota={}'.format(nu,nc,niota))

    if not nu>=1 and niota>=1 and nc>=1:
        raise PrecErr('Calgh error')
    
    if nu + niota == 0:
        logger('calgh path 1')
        g,h = fc(t,ScVague)
    elif nu>=1 and niota>=1 and nc>=1:
        logger('calgh path 2')
        gu,hu = fu(t,SuVague)
        giota,hiota = fiota(t,SiotaVague)
        gc,hc = fc(t,ScVague)
        g,h = twosquares(gu,hu,giota,hiota)
        g,h = twosquares(g,h,gc,hc)
    elif nu>=1 and niota>=1 and nc==0:
        logger('calgh path 3')
        gu,hu = fu(t,SuVague)
        giota,hiota = fiota(t,SiotaVague)
        g,h = twosquares(gu,hu,giota,hiota)
    elif nu>=1 and niota==0 and nc>=1:
        logger('calgh path 4')
        gu,hu = fu(t,SuVague)
        gc,hc = fc(t,ScVague)
        g,h = twosquares(gu,hu,gc,hc)
    elif nu>=1 and niota==0 and nc==0:
        logger('calgh path 5')
        g,h = fu(t,SuVague)
    elif nu==0 and niota>=1 and nc>=1:
        logger('calgh path 6')
        giota,hiota = fiota(t,SiotaVague)
        gc,hc = fc(t,ScVague)
        g,h = twosquares(giota,hiota,gc,hc)
    elif nu==0 and niota>=1 and nc==0:
        logger('calgh path 7')
        g,h = fiota(t,SiotaVague)
    else:
        raise ValueError('..')
    return g,h

def calBDtildeVague(t,g,h,roots,tol):
    n0    = roots.count(0)
    n1    = roots.count(1)
    # n0    = sum( 1 for x in roots if abs(x) < tol )
    # n1    = sum( 1 for x in roots if abs(abs(x)-1) < tol )
    nu    = sum( uCondVague    (x,tol) for x in roots )
    niota = sum( iotaCondVague (x,tol) for x in roots )
    pfr = frVague(t,roots,tol)
    pfi = fiVague(t,roots,tol)

    logger('calbd: n0={}, n1={}, nu={}, niota={}'.format(n0,n1,nu,niota))
    if not (nu+niota)%2==0 and (n0//2)%2==0 and (n1//2)%2==0:
        raise PrecErr('Calbd error')
    
    if   (nu+niota)%2==1 and (n0//2)%2==0 and (n1//2)%2==1:
        logger('calbd path 1')
        bt = pfr*pfi*g
        dt = pfr*pfi*h
    elif (nu+niota)%2==1 and (n0//2)%2==1 and (n1//2)%2==0:
        logger('calbd path 2')
        bt = pfr*pfi*h
        dt = pfr*pfi*g
    elif (nu+niota)%2==0 and (n0//2)%2==0 and (n1//2)%2==0:
        logger('calbd path 3')
        bt = pfr*pfi*g
        dt = pfr*pfi*h
    elif (nu+niota)%2==0 and (n0//2)%2==1 and (n1//2)%2==1:
        logger('calbd path 4')
        bt = pfr*pfi*h
        dt = pfr*pfi*g
    else:
        for x in roots:
            if iotaCondVague (x,tol): logger(' ??>> {}'.format(x))
        raise ValueError('BDtilde counts: {} {} {} {} (probably not using enough precision)'.format(n0,n1,nu,niota))
    return bt, dt

# retired
def calPhi(l,j,vec1,vec2):
    if j % 2 == 0:
        Phi = sum( (vec1[n//2] + 1j*vec2[n//2]) * cbinomial((l-n)//2,j//2) \
                    for n in range(0,l-j+1,2) )
    else:
        Phi = sum( (1j*vec1[(n-1)//2] - vec2[(n-1)//2]) * cbinomial((l-n)//2,(j-n)//2) \
                    for n in range(1,j+1,2) )
    Phi *= (1j**j)
    return Phi

def calPhase(Phivec):
    return sum(Phivec[1::2]) / sum(Phivec[0::2])

def calupperPhi(Phivec,phase):
    l = len(Phivec)-1
    upper  = _mp.matrix(l,1)
    invph  = _mp.mpc(1) / phase
    accph0 = _mp.mpc(0)
    accph1 = _mp.mpc(0)
    
    for j in range(l):
        if j & 1 == 0:
            accph0 += Phivec[j]
            upper[j] = accph0 - accph1 * invph
        else:
            accph1 += Phivec[j]
            upper[j] = accph1 - accph0 * phase
    
    return upper
    
    upperPhivec = _mp.matrix(l,1)
    accup0 = _mp.mpc(0)
    accup1 = _mp.mpc(0)
    accup00 = sum( Phivec[2*j  ]*((l+1)//2-j) for j in range((l+1)//2) )
    accup01 = sum( Phivec[2*j+1]*((l-1)//2-j) for j in range((l-1)//2) )
    accup10 = sum( Phivec[2*j  ]*( l   //2-j) for j in range( l   //2) )
    accup11 = sum( Phivec[2*j+1]*( l   //2-j) for j in range( l   //2) )

    newphase = (accup11 - accup10*phase) / (accup00 - accup01*invph)
    
    logger('!!! max calphase diff = {:.8}'.format(complex(newphase-calPhase(upper))))
            
    for j in range(l):
        for k in range(j+1):
            if (j+k) % 2 == 0:
                upperPhivec[j] += Phivec[k]
            else:
                upperPhivec[j] -= Phivec[k] * phase**(-(-1)**j)
    logger('!!! max calupper diff = {:.8}'.format(float(max(map(abs,upper-upperPhivec)))))
    
    return upper

def calPhases(ab,cd):
    assert len(ab) == len(cd)+1
    l = 2*(len(ab)-1)
    q = l//2

    phivec = _mp.matrix(l+1,1)
    
    # do calPhi():
    for i in range(q+1):
        phivec[2*i  ] =  (-1)**i * sum( ab[n] * cbinomial(q-n,i) for n in range(q-i+1) )
    for i in range(q):
        phivec[2*i+1] = -(-1)**i * sum( cd[n] * cbinomial(q-n-1,i-n) for n in range(i+1) )

    # Phivec = _mp.matrix(l+1,1)
    # for i in range(0,l+1,2):
    #     Phivec[i] = calPhi(l,i,au,bu)
    # for i in range(1,l+1,2):
    #     Phivec[i] = calPhi(l,i,cu,du)
    # logger('!!! max calPhases diff = {:.8}'.format(float(max(map(abs,phivec-Phivec)))))

    phases = _mp.matrix(l+1,1)
    for i in range(l,0,-1):
        phases[i] = calPhase(phivec)
        phivec = calupperPhi(phivec,phases[i])
    phases[0] = phivec[0]
    return phases

    phases = _mp.matrix(l,1)
    phases[l-1] = calPhase(phivec)
    for i in range(l-1,0,-1):
        phivec = calupperPhi(phivec,phases[i])
        # phases[i-1] = newph
        # phivec = calupperPhi(phivec,phases[i])
        phases[i-1] = calPhase(phivec)
    return phases

def merge(re,im):
    merger = lambda r,i: r+1j*i if r.imag or i.imag else _mp.mpc(r,i)
    ri = _mp.matrix([merger(r,i) for r,i in zip(re,im)])
    return ri
    # ri = _mp.matrix([r + 1j*i for r,i in zip(re,im)])

def diffphis(phis):
    twopi = 2 * _mp.pi
    if len(phis) & 1:
        m = len(phis) - 1
        ph = _np.zeros(m+1,dtype=_np.float64)
        ph[0] = ((phis[0] + phis[1] + _mp.pi) % twopi) - _mp.pi
        ph[m] = ((        - phis[m]         ) % twopi) - _mp.pi
        for k in range(1,m):
            ph[k] = ((phis[k+1] - phis[k]) % twopi) - _mp.pi
    return ph

# Step 7: Recovering the angles
def calAngles(ab,cd):
    ph = calPhases(ab,cd)
    if not all( abs(abs(x)-1) < 1e-12 for x in ph ):
        logger('!!! phase strayed from unity (max = {})'.format(max(abs(abs(x)-1) for x in ph)) )
    else:
        logger('max phase stray from unity: {:.8}'.format(float(max(abs(abs(x)-1) for x in ph))) )
        
    # assert all( abs(abs(x)-1) < TOL for x in ph )
    th = list(map(_mp.arg,ph))
    # th = _np.array(list(map(_mp.arg,ph)),dtype=float)
    # th[1::2] += _np.pi
    return th

   
def mproots(coeffs,rational=True,mt=None,mpargs=()):

    cmd = 'mpsolve -Of -Ga -au -o{}'.format(_mp.mp.dps+50)
    for arg in mpargs:
        cmd += ' {}'.format(arg)
    if mt is not None:
        cmd += ' -j{}'.format(mt)

    logger('calling mpsolve...')
    logger('>>> {}'.format(cmd))

    header = 'Degree={};\n' \
             'Monomial;\n' \
             'Dense;\n' \
             'Real;\n' \
             .format(len(coeffs)-1)
    

    prc = _subprc.Popen(cmd.split(),stdin=_subprc.PIPE,stdout=_subprc.PIPE,stderr=_subprc.PIPE,universal_newlines=True)
    # assert prc.poll() is None, 'mpsolve returned unexpectedly ({})'.format(prc.returncode)
    
    prc.stdin.write(header)

    if rational:
        prc.stdin.write('Rational;\n')
        for c in coeffs:
            prc.stdin.write('{}\n'.format(_sym.Rational(_sym.Float(c))))
    else:
        prc.stdin.write('FloatingPoint;\n')
        for c in coeffs:
            prc.stdin.write('{}\n'.format(c))

    # logger('finding roots..',end='')

    res,err = prc.communicate()
    logger('....done.')
    
    for line in err.splitlines():
        logger('!!! MPSOLVE: {}'.format(line.strip()))
        
    assert prc.returncode == 0, 'mpsolve returned with code {}'.format(prc.returncode)
    
    rxv = _re.compile(r'\s*\(([-+0-9eE.]+)\s*,\s*([-+0-9eE.]+)\)')
    rxe = _re.compile(r'\s*([-+0-9.]+x[-+0-9.]+)')
    roots = []
    errs  = []
    for line in res.splitlines():
        m = rxv.match(line)
        if m:
            val = _mp.mpc(*m.groups())
            roots.append(val)
            continue
        m = rxe.match(line)
        if m:
            err = _mp.mpf(m.group(1).replace('x','e'))
            errs.append(err)
            continue
        # assert 'Approx' not in line
        
    assert len(errs) == len(roots)
    err = max(errs)
    return roots,err
    
def factor_t(ft):
    t = _sym.Symbol('t')
    f = ft.subs(t**2,t).as_poly()
    assert f.gens == (t,)
    return f

def find_roots(ft,tol,mt=None,mpargs=()):
    
    has_mpsolve = True
    try:
        _subprc.check_call(['mpsolve','-v'])
    except:
        logger('!!! MPSolve not found, using sympy::nroots')
        assert ft.total_degree() <= 260, 'MPSolve required to generate phases for query depths above m=128'
        has_mpsolve = False

    if has_mpsolve:
        try:
            roots,err = mproots(ft.all_coeffs()[::-1],rational=True,mt=mt,mpargs=mpargs)
        except:
            _sys.set_int_max_str_digits(0)
            roots,err = mproots(ft.all_coeffs()[::-1],rational=True,mt=mt,mpargs=mpargs)
            _sys.set_int_max_str_digits(4300)
    else:
        prec  = 50+_mp.mp.dps
        roots = ft.nroots(n=prec,maxsteps=50+prec)
        err   = 10**(-prec)

    zcut = 0
    rcut = 0
    icut = 0
    err = _mp.sqrt(err)
    e = max(err,tol) #,_mp.mpf('1e-{}'.format(_mp.mp.dps))))
    for i,root in enumerate(map(_mp.sqrt,roots)):
        if abs(root) < e:
            logger(' --> cutting {:.8}'.format(complex(root)))
            root = _mp.mpc(0.0,0.0)
            zcut += 1
        elif abs(root.real) < e:
            logger(' --> cutting {:.8}'.format(complex(root)))
            root = _mp.mpc(0.0,root.imag)
            icut += 1
        elif abs(root.imag) < e:
            logger(' --> cutting {:.8}'.format(complex(root)))
            root = _mp.mpc(root.real,0.0)
            rcut += 1
        roots[i] = root
    logger('rootfinding resolution: {:.3} (tol={:.3})'.format(float(err),float(e)))
    if rcut: logger(' --> {} values made purely real'.format(rcut))
    if icut: logger(' --> {} values made purely imaginary'.format(icut))
    if zcut: logger(' --> {} values taken as zero'.format(zcut))

    # exponents are always multiples of two, so substitute t**2 --> t'
    # f = factor_t(ft)
    # prec  = _mp.mp.dps
    # roots = f.nroots(n=prec,maxsteps=50+prec)
    
    # roots  = list(map(_mp.sqrt,roots))
    # roots += [ 1.0/x for x in roots ]
    roots += [   - x for x in roots ]
    
    # roots += [ -x for x in reversed(roots) ]
    # roots  = sorted(roots, key=lambda c:(c.real,abs(c.imag),c.imag))
    # roots  = sorted(roots, key=lambda c:(abs(c),c.real,abs(c.imag),c.imag))
    return roots,err

def solve_beta(bt,**kwargs):
    prec = _mp.mp.dps
    t    = _sym.Symbol('t')
    l    = (bt.total_degree() + 1) // 2

    btcs = bt.all_coeffs()
    diff = max( abs(hi-lo) for lo,hi in zip(btcs,btcs[::-1]) )
    logger('beta asymmetry: {:.3}'.format(float(diff)))

    fx = lambda k,x: sum( cbinomial(k,d)*cbinomial(l-k,x-d)*(-1)**d for d in range(max(0,k-l+x),min(x,k)+1) )
    m  = _mp.matrix([ [ fx(k,x) for k in range(0,l+1,2)  ] for x in range(0,l//2+1) ])
    v  = _mp.matrix([ [ ( btcs[2*x] + btcs[-1-2*x] ) / 2 ] for x in range(0,l//2+1) ])
    # bu = _mp.inverse(m) * v
    logger('solving beta...')
    bu,err = _mp.qr_solve(m,v)
    logger(' --> solution found, radius: {:.3}'.format(float(err)))
    return bu

    # tosolve = [ (btcs[2*x] + btcs[-1-2*x])/2 - sum( beta[k] * fx(k,x) for k in range(0,l+1,2) ) for x in range(0,l//2+1) ]

    # beta = _odict( (k, _sym.Symbol('b{}'.format(k))) for k in range(0,l+1,2) )
    # logger(',,,')
    # btarget = sum( beta[k] * (1-t**2)**k * (1+t**2)**(l-k) for k in range(0,l+1,2) ).expand()
    # logger(',,')
    # tosolve = [ (bcoeffs[i] - btarget.coeff(t,i)) for i in range(0,2*l+1,2) ]
    # logger(',,,')

    # 
    # t = _sym.Poly('t')
    # logger(',,,')
    # btarget = [ ((1-t**2)**k * (1+t**2)**(l-k)).all_coeffs() for k in range(0,l+1,2) ]
    # logger(',,,,')
    # # tosolve = [ bcoeffs[x] - sum( beta[k] * btarget[k//2].coeff_monomial('t**{}'.format(x)) for k in range(0,l+1,2) )
    # tosolve = [ bcoeffs[x] - sum( beta[k] * btarget[k//2][x] for k in range(0,l+1,2) ) \
    #             for x in range(0,2*l+1,2) ]
    for i in range(3):
        init_beta = _np.random.normal(size=len(beta),loc=10,scale=5).tolist()
        try:
            bu = _sym.nsolve( tosolve, list(beta.values()), init_beta, prec=prec, dict=False, maxsteps=10000*prec, **kwargs )
        except:
            print('retrying...')
            bu = None
            continue
        break
    assert bu is not None
    bu = _mp.matrix(bu.T.tolist()[0])
    return bu

def solve_delta(dt,**kwargs):
    prec  = _mp.mp.dps
    l     = (dt.total_degree() + 1) // 2
    delta = _odict( (k, _sym.Symbol('d{}'.format(k))) for k in range(1,l+1,2) )

    dtcs = dt.all_coeffs()

    diff = max( abs(hi+lo) for lo,hi in zip(dtcs,dtcs[-2::-1]) )
    logger('delta asymmetry: {:.3}'.format(float(diff)))

    two = _mp.mpf(2.0)
    fx = lambda k,x: two**k * ( cbinomial(l-k-1,(x-k)//2) - cbinomial(l-k-1,(x-k)//2-1) )
    m  = _mp.matrix([ [ fx(k,x) for k in range(1,l+1,2) ] for x in range(1,l+1,2) ])
    v  = _mp.matrix([ [ ( dtcs[-x-1] - dtcs[x-1] ) / 2  ] for x in range(1,l+1,2) ])
    #du = _mp.inverse(m) * v
    du,err = _mp.qr_solve(m,v)
    logger('delta solution radius: {:.3}'.format(float(err)))
    return du

    # t       = _sym.Symbol('t')
    # dtarget = (1-t**2) * sum( delta[k] * (2*t)**k * (1+t**2)**(l-k-1) for k in range(1,l+1,2) )
    # dtarget = dtarget.expand()
    # tosolve = [ (dt.coeff_monomial(t**i) - dtarget.coeff(t,i)).expand() for i in range(1,2*l+1,2) ]
    # tosolve = [ (dcoeffs[-x-1]-dcoeffs[x-1])/2 - sum( delta[k] * fx(k,x) for k in range(1,l+1,2) ) for x in range(1,l+1,2) ]

    for i in range(3):
        init_delta = _np.random.normal(size=len(delta),loc=0,scale=5).tolist()
        try:
            du = _sym.nsolve( tosolve, list(delta.values()), init_delta, prec=prec, dict=False, maxsteps=10000*prec, **kwargs )
        except:
            du = None
            continue
        break
    assert du is not None
    du = _mp.matrix(du.T.tolist()[0])
    return du

# defunct
def calrenorm(au,bu):
    # Step 5: satisfy the unitarity condition
    au_sum = sum(au)
    bu_sum = sum(bu)
    au2 = au*au_sum + bu*bu_sum
    bu2 = bu*au_sum - au*bu_sum
    ab  = _mp.matrix([_mp.mpc(a,b) for a,b in zip(au2,bu2)])
    return ab
    # return au2, bu2

def setprec(prec):
    if _mp.mp.dps != prec:
        _mp.mp.dps = prec
        dcache.clear()
        bcache.clear()

def check_cache(tau=None,steps=None,epsilon=None,prec=None,cache=_cache,**kwargs):
    if 'm' in kwargs:
        assert steps is None
        steps = kwargs.pop('m')
    if 'e' in kwargs:
        assert epsilon is None
        epsilon = kwargs.pop('e')
    if epsilon is not None:
        assert tau is None
        tau = empirical_tau(steps,epsilon)
        
    if prec is not None:
        prec = int(prec)
    if steps is not None:
        steps = int(steps)
        if steps <= 0: return False
        if tau is not None:
            tau = parse_tau(tau,steps)
            if tau <= 0: return False
            if tau >= steps: return False
    
    data = None
    try:
        with open(cache,'rb') as f:
            data = _pkl.load(f)
    except _pkl.UnpicklingError as e:
        logger('!!! check_cache unpickling error ({}), trying again...'.format(e))
        _time.sleep(0.1)
        with open(cache,'rb') as f:
            data = _pkl.load(f)
        
        
    
    if tau is None:
        keys = [ (t,m,p) for t,m,p in data if steps in (m,None) and prec in (p,None) ]
    elif steps is None:
        keys = [ (t,m,p) for t,m,p in data if t == parse_tau(tau,m) and prec in (p,None) ]
    else: 
        keys = [ (t,m,p) for t,m,p in data if tau in (t,None) and steps in (m,None) and prec in (p,None) ]
    
    keys = [ (t,float('{:.2}'.format(float(emperr(t,m)))),m,p) for t,m,p in keys ]
    return sorted(keys)
        
       
def read_angles(tau=None,steps=None,epsilon=None,prec=None,cache=_cache,**kwargs):
    if 'm' in kwargs:
        assert steps is None
        steps = kwargs.pop('m')
    if 'e' in kwargs:
        assert epsilon is None
        epsilon = kwargs.pop('e')
    if epsilon is not None:
        assert tau is None
        tau = empirical_tau(steps,epsilon)
        
    assert steps is not None and steps > 0
    assert tau   is not None
    if prec is not None:
        prec = int(prec)
        
    tau = parse_tau(tau,steps)
    if tau == 0:
        return _np.zeros(steps)
    else:
        assert tau > 0

    data = None
    with open(cache,'rb') as f:
        data = _pkl.load(f)
    
    keys = max( ((t,m,p) for t,m,p in data if tau==t and steps==m and prec in (p,None)), default=None )
    assert keys is not None
    return data[keys]
    
def calc_abcd(tau,steps,mt=None,mpargs=()):
    prec = _mp.mp.dps
    tol  = _mp.mpf('1e-{}'.format(prec//2))
    tau  = _mp.mpf(tau)

    logger(' --> tau = ',round(tau,6))

    # Steps 1-3: Finding a polynomial approximation
    au,cu = norm_fourier(tau,steps)
    logger('found Fourier approximation: len(au)={}, len(cu)={}'.format(len(au),len(cu)))

    # Step 4: Constructing unspecified pairs without the unitarity condition
    t  = _sym.poly('t')
    ft = ftilde(t,au,cu)
    assert isinstance(ft,_sym.Poly), 'ftilde expression did not result in a polynomial'
    assert ft.gens == t.gens, 'ftilde couldnot be reduced'

    logger('finding roots')
    roots,err = find_roots(ft,tol,mt=mt,mpargs=mpargs)
    logger('found {} roots'.format(len(roots)))
    
    g ,h  = calghVague(t,roots,tol)
    logger('found g,h ({},{})'.format(g.__class__.__name__,h.__class__.__name__))

    bt,dt = calBDtildeVague(t,g,h,roots,tol)
    logger('found bt,dt ({},{})'.format(bt.__class__.__name__,dt.__class__.__name__))

    K  = _mp.sqrt(ft.coeff_monomial(1))
    bt *= K
    dt *= K

    bu = solve_beta (bt)
    logger('found beta (len {})'.format(len(bu)))
    du = solve_delta(dt)
    logger('found delta (len {})'.format(len(du)))
    return au,bu,cu,du


def cal_prec(tau,steps,empfn=empirical_err):
    min_prec = 3 * int(10-_mp.log10(empfn(tau,steps)/steps))

    prec = max(100, min_prec, 10 * int(round(8*(steps/64)))) + 200
    
    if steps > 512:
        prec = 10 * int(round(_np.ceil(_np.log2(steps))*(steps/64)))
    if tau < 4 and prec > 100:
        prec = 10*int(round(prec*0.15))
    prec += steps // 2
    return prec


### main run routine:
def calc_angles(tau=None,steps=None,epsilon=None,prec=None,cache=_cache,force=False,mt=None,mpargs=(),verb=1,win=None):
    if win is not None:
        cache = None
    steps = int(steps)
    assert steps > 0 and (steps%4 == 0 or steps < 4), 'm must be a positive multiple of 4'
    
    if epsilon is None:
        tau = parse_tau(tau,steps)
    else:
        assert tau is None
        epsilon = float(epsilon)
        tau = empirical_tau(steps,epsilon)
    
    tmptol = empirical_err(tau,steps)
    logger(' ~ minimum error (empirical): {}'.format(float(tmptol)))

    if cache and not force:
        # if not _os.path.exists(_cache) and not _os.path.isfile(_cache):
        try:
            if not _os.path.exists(_cache):
                cachedata = {}
                with _pl.Lock(cache,'ab') as f:
                    f.truncate(0)
                    _pkl.dump(cachedata,f)
            else:
                with _pl.Lock(cache,'rb') as f:
                    cachedata = _pkl.load(f)
    
            if _os.path.getmtime(__file__) > _os.path.getmtime(cache):
                logger('WARNING: qspangles.py has been changed since cache was last accessed')

            assert isinstance(cachedata,dict)

            if prec is None:
                precs = [ kprec for ktau,ksteps,kprec in cachedata if (ktau,ksteps) == (tau,steps) ]
                if precs:
                    prec = max(precs)
                    if verb >= 1:
                        if len(precs) > 1:
                            logger('{} cached results with tau={}, m={}: prec={{{}}}'.format(len(precs),tau,steps,','.join(map(str,sorted(precs)))))
                            logger('using prec={}'.format(prec))
                        else:
                            logger('{} cached results with tau={}, m={}: prec={}'.format(len(precs),tau,steps,prec))
                        logger('retrieved cached angles from {}'.format(cache))
                    return cachedata[(tau,steps,prec)],tmptol
            else:
                prec = int(prec)
                cachekey = (tau,steps,prec)
                if cachekey in cachedata:
                    logger('retrieved cached angles from {}'.format(cache))
                    return cachedata[cachekey],tmptol
        except TypeError as e:
            logger(str(e))
            logger('unhashable arguments, bypassing cache')
            cache = None
            return 0,tmptol

    logger(' === QSPANGLES ===')
    if bcache: logger('starting bcache size:',len(bcache))
    if dcache: logger('starting dcache size:',len(dcache))
    
    if _mp.libmp.BACKEND != 'gmpy':
        logger(' !!! libmp using backend {}, routines would be *much* faster if gmpy was installed'.format(_mp.libmp.BACKEND))
    
    
    if prec is None:
        # if   steps > 480: prec = 800
        # elif steps > 320: prec = 600
        # elif steps > 160: prec = 400
        # elif steps >  80: prec = 200
        # else            : prec = 100
        min_prec = 3 * int(10-_mp.log10(empirical_err(tau,steps)/steps))

        prec = max(100, min_prec, 10 * int(round(8*(steps/64)))) + 200
        
        if steps > 512:
            prec = 10 * int(round(_np.ceil(_np.log2(steps))*(steps/64)))
        if tau < 4 and prec > 100:
            prec = 10*int(round(prec*0.15))
        prec += steps // 2
        logger('no cache entry for tau={}, m={}'.format(tau,steps))

    else:
        prec = int(prec)
        logger('no cache entry for tau={}, m={} with prec={}'.format(tau,steps,prec))
    cachekey = (tau,steps,prec)

    logger(' --> computing new angles with precision {}'.format(prec))
    
    # Step 0: Initialization
    setprec(prec)

    tol = _mp.mpf('1e-{}'.format(prec//2))
    tau = _mp.mpf(tau)

    logger(' --> tau = ',round(tau,6))

    # au,bu,cu,du = calc_abcd(tau,steps,mt=mt,mpargs=mpargs)

    # Steps 1-3: Finding a polynomial approximation
    au,cu = norm_fourier(tau,steps,win=win)
    logger('found Fourier approximation: len(au)={}, len(cu)={}'.format(len(au),len(cu)))

    # Step 4: Constructing unspecified pairs without the unitarity condition
    t  = _sym.poly('t')
    ft = ftilde(t,au,cu)
    assert isinstance(ft,_sym.Poly), 'ftilde expression did not result in a polynomial'
    assert ft.gens == t.gens, 'ftilde couldnot be reduced'

    logger('finding roots')
    roots,err = find_roots(ft,tol,mt=mt,mpargs=mpargs)
    logger('found {} roots'.format(len(roots)))
    
    g ,h  = calghVague(t,roots,tol)
    logger('found g,h ({},{})'.format(g.__class__.__name__,h.__class__.__name__))

    bt,dt = calBDtildeVague(t,g,h,roots,tol)
    logger('found bt,dt ({},{})'.format(bt.__class__.__name__,dt.__class__.__name__))

    K  = _mp.sqrt(ft.coeff_monomial(1))
    bt *= K
    dt *= K

    bu = solve_beta (bt)
    logger('found beta (len {})'.format(len(bu)))
    du = solve_delta(dt)
    logger('found delta (len {})'.format(len(du)))

    ab = merge(au,bu)
    cd = merge(cu,du)
    # return ab,cd
    # th = calAngles(au,bu,cu,du) # (*The list of angles output by Low-Chuang algorithm.*)
    th = calAngles(ab,cd) # (*The list of angles output by Low-Chuang algorithm.*)
    phis = diffphis(th)

    logger('found',len(phis),'angles')
    if bcache: logger('bcache size:',len(bcache))
    if dcache: logger('dcache size:',len(dcache))

    # Quipper understands phis/2 as phis
    # phis /= 2 ### (actually we'll deal with this in the Quipper code)

    if cache:
        try:
            with _pl.Lock(cache,'rb') as f:
                cachedata = _pkl.load(f)
            with _pl.Lock(cache,'ab') as f:
                cachedata[cachekey] = phis
                while len(cachedata) > _cache_size:
                    cachedata.pop(list(cachedata.keys())[0])
                f.truncate(0)
                _pkl.dump(cachedata,f)
                suffix = 'y' if len(cachedata) == 1 else 'ies'
                logger('cache {} updated ({} entr{})'.format(_os.path.basename(cache),len(cachedata),suffix))
        except Exception as e:
            logger('could not write to cache: {}'.format(e))
        
    return phis,tmptol


def gettau(steps,epsilon):
    q   = steps//2 + 1
    val = _spc.gammaln(q+1) + _np.log(float(epsilon)) - _np.log(32)
    val = 2 * _np.exp(val/q)
    return round(val,3)
