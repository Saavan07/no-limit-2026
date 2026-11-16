"""Legal catalog perturbations and differentiable joint-population fitting."""
import hashlib
import itertools
from functools import lru_cache
import numpy as np
from scipy.special import expit


STANDARD_SCOUTS=np.array([.2,.3,.2,.1,.05,.05,.035,.035,.03])
CATALOG_VARIANTS=['original','zero_scout_targets','broad_scout_targets','high_scout_targets',
    'uniform_active_targets','equal_active_targets','value_active_targets','corner_active_targets',
    'c4_threshold_upweight','c24_threshold_family','nonatom_transfer1','nonatom_transfer5']


def fingerprint(x,w):
    h=hashlib.sha256();h.update(np.ascontiguousarray(x,dtype=np.uint8).tobytes())
    h.update(np.ascontiguousarray(np.asarray(w,dtype=float)/np.sum(w)).tobytes())
    return h.hexdigest()


@lru_cache(maxsize=512)
def compositions(total,size,minimum=8):
    if size==1:
        return np.array([[total]],dtype=int)if total>=minimum else np.empty((0,1),dtype=int)
    result=[]
    for first in range(minimum,total-minimum*(size-1)+1):
        for rest in compositions(total-first,size-1,minimum):result.append([first,*rest])
    return np.asarray(result,dtype=int).reshape(-1,size)


def target_family(active,kind,n,seed):
    active=np.array(active,dtype=int);outside=np.array([j for j in range(5)if j not in active]);rng=np.random.default_rng(seed)
    rows=np.zeros((n,5),dtype=int)
    if kind=='zero_scout_targets':scouts=np.zeros((n,len(outside)),dtype=int)
    elif kind=='broad_scout_targets':scouts=rng.integers(0,25,size=(n,len(outside)))
    elif kind=='high_scout_targets':scouts=rng.integers(5,17,size=(n,len(outside)))
    else:scouts=rng.choice(9,size=(n,len(outside)),p=STANDARD_SCOUTS)
    rows[:,outside]=scouts;budgets=100-scouts.sum(axis=1)
    for budget in np.unique(budgets):
        ids=np.flatnonzero(budgets==budget);comps=compositions(int(budget),len(active))
        if not len(comps):raise ValueError('Infeasible target scouts')
        eq=budget/len(active);value=budget*(active+1)/(active+1).sum()
        equal=np.exp(-np.abs(comps-eq).sum(axis=1)/12)
        proportional=np.exp(-np.abs(comps-value).sum(axis=1)/12)
        if kind=='uniform_active_targets':prob=np.ones(len(comps))/len(comps)
        elif kind=='equal_active_targets':prob=np.exp(-np.abs(comps-eq).sum(axis=1)/6)
        elif kind=='value_active_targets':prob=np.exp(-np.abs(comps-value).sum(axis=1)/6)
        elif kind=='corner_active_targets':prob=np.exp((comps.max(axis=1)-comps.min(axis=1))/12)
        else:prob=.2/len(comps)+.4*equal/equal.sum()+.4*proportional/proportional.sum()
        prob/=prob.sum();rows[np.ix_(ids,active)]=comps[rng.choice(len(comps),len(ids),p=prob)]
    return rows.astype(np.uint8),np.ones(n)/n


def make_catalog(original,kind,n,seed):
    if kind not in CATALOG_VARIANTS:raise ValueError('Unknown catalog variant')
    catalog={name:(x.copy(),w.copy())for name,(x,w)in original.items()}
    if kind=='original':return catalog
    for j,(name,(x,w))in enumerate(original.items()):
        if kind in CATALOG_VARIANTS[1:8]and name.startswith('individual_target_'):
            active=[int(v)-1 for v in name.split('_')[-1]]
            catalog[name]=target_family(active,kind,n,seed+j)
        elif kind=='c4_threshold_upweight':
            ww=w*(1+5*((x[:,3]>=61)&(x[:,3]<=72)))
            catalog[name]=(x.copy(),ww/ww.sum())
        elif kind.startswith('nonatom_transfer')and not name.startswith('exact_'):
            radius=int(kind[-1]);rng=np.random.default_rng(seed+j)
            rows=x[rng.choice(len(x),n,p=w/w.sum())].astype(int)
            for step in range(radius):
                eligible=rows>0;random=rng.random(rows.shape);random[~eligible]=-1
                donor=random.argmax(axis=1);recipient=rng.integers(4,size=n);recipient+=recipient>=donor
                rows[np.arange(n),donor]-=1;rows[np.arange(n),recipient]+=1
            catalog[name]=(rows.astype(np.uint8),np.ones(n)/n)
    if kind=='c24_threshold_family':
        rows=np.array([[a,100-a-c-d-four,c,four,d]for a,c,d in itertools.product(range(4),repeat=3)for four in range(60,76)],dtype=np.uint8)
        catalog['individual_target_24']=(rows,np.ones(len(rows))/len(rows))
    for x,w in catalog.values():
        if np.any(x.sum(axis=1)!=100)or not np.isfinite(w).all()or np.any(w<0):raise ValueError('Illegal perturbed catalog')
    return catalog


def ordered_index(ids,k):
    lookup={tuple(map(int,combo)):j for j,combo in enumerate(ids)}
    return np.array([lookup[tuple(sorted(combo))]for combo in itertools.product(range(k),repeat=3)],dtype=int)


class CubicCDF:
    """A sampled cubic payoff polynomial, exact derivatives of that surrogate."""
    def __init__(self,coefficients,ids,families,thresholds,observed,importance,bandwidth):
        a=len(families);self.k=int(max(families))+1
        order=ordered_index(ids,self.k)
        tensor=np.ascontiguousarray(coefficients[:,order],dtype=float)
        self.flat=tensor.reshape(a*self.k,self.k*self.k)
        self.byfamily=np.eye(self.k)[families]/np.bincount(families)[families,None]
        self.thresholds=np.asarray(thresholds);self.observed=np.asarray(observed)
        self.importance=np.asarray(importance);self.bandwidth=bandwidth;self.last=None

    def scores_and_gradient(self,w):
        du=3*(self.flat@np.outer(w,w).ravel()).reshape(-1,self.k)
        return du@w/3,du

    def stats(self,w):
        if self.last is not None and np.array_equal(self.last[0],w):return self.last[1:]
        u,du=self.scores_and_gradient(w);mass=self.byfamily@w
        cdf_terms=expit((self.thresholds[:,None]-u[None,:])/self.bandwidth)
        predicted=cdf_terms@mass
        dcdf=cdf_terms@self.byfamily-(cdf_terms*(1-cdf_terms)*mass)@du/self.bandwidth
        residual=predicted-self.observed
        loss=float(self.importance@(residual**2)+1e-5*(w@w))
        grad=2*(self.importance*residual)@dcdf+2e-5*w
        self.last=(w.copy(),loss,grad,u,predicted)
        return self.last[1:]


def choose_diverse(records,maximum,min_l1=.08):
    result=[]
    for row in sorted(records,key=lambda r:(r['loss'],r['run'])):
        if all(np.abs(np.array(row['weights'])-other['weights']).sum()>min_l1 for other in result):result.append(row)
        if len(result)==maximum:break
    return result


def split_actions():
    return np.array([[0,b,0,100-b,0]for b in range(101)],dtype=np.uint8)


def paired_moments(pay12,counts):
    """Exact integer quarter-payout sums and all pair cross-products.

    Every multiply/add below is a nonnegative integer below 2**53, so float64
    matrix multiplication represents these integer moments exactly.
    """
    if np.any(pay12%3):raise ValueError('Non-quarter placement payout')
    q=(pay12//3).astype(float);weights=np.asarray(counts,dtype=float)
    n=int(np.sum(counts))
    if 1296*n>=2**53:raise ValueError('Moment accumulation exceeds exact float64 integers')
    sums=q@weights;cross=(q*weights[None,:])@q.T
    if np.any(sums!=np.rint(sums))or np.any(cross!=np.rint(cross)):raise ValueError('Inexact integer moments')
    return sums.astype(np.uint64),cross.astype(np.uint64)


def cubic_minimum(coefficients,maximum_share=1.):
    """Minimum of the cubic Bernstein polynomial on [0, maximum_share]."""
    c=np.asarray(coefficients,dtype=float)
    a=c[...,3]-3*c[...,2]+3*c[...,1]-c[...,0]
    b=3*(c[...,2]-2*c[...,1]+c[...,0]);d=3*(c[...,1]-c[...,0]);e=c[...,0]
    def value(t):return ((a*t+b)*t+d)*t+e
    result=np.minimum(e,value(maximum_share));disc=4*b*b-12*a*d
    rootdisc=np.sqrt(np.maximum(disc,0))
    with np.errstate(divide='ignore',invalid='ignore'):
        roots=[(-2*b+rootdisc)/(6*a),(-2*b-rootdisc)/(6*a),-d/(2*b)]
    for j,t in enumerate(roots):
        valid=(t>0)&(t<maximum_share)&np.isfinite(t)
        valid&=(np.abs(a)<1e-12)if j==2 else (np.abs(a)>=1e-12)&(disc>=0)
        result=np.minimum(result,np.where(valid,value(np.where(valid,t,0)),np.inf))
    return result


def empirical_bernstein_error(variance,n,comparisons,alpha=.05,value_range=9.):
    """Two-sided union of Maurer-Pontil (2009), Theorem 4.

    Source: https://www.cs.mcgill.ca/~colt2009/papers/012.pdf
    Use alpha/(2*M) for each one-sided event, hence log(4*M/alpha).
    """
    if n<2 or comparisons<1 or not 0<alpha<1:raise ValueError('Invalid confidence inputs')
    log=np.log(4*comparisons/alpha)
    return np.sqrt(2*np.maximum(variance,0)*log/n)+7*value_range*log/(3*(n-1))
