"""Cubic population weights, mixture bounds, and representation controls."""
from collections import Counter
from fractions import Fraction
from itertools import combinations_with_replacement,product
from math import factorial
import heapq
import numpy as np
from blotto_lab.core import Population,validate_allocation
from .core import shared_triplets
from .arena import class_name,metric_arrays,initial_shares,static_triplets

A=(0,0,0,0,100)
FAMILY_IDS=list(combinations_with_replacement(range(4),3))


def residual_draw(family,size,rng):
    if family not in [1,2,3]:raise ValueError('Residual family must be 1,2,3')
    if family==1:fifth=rng.choice([0,0,0,1,2,3],size=size)
    elif family==2:fifth=rng.integers(4,80,size=size)
    else:fifth=rng.integers(80,100,size=size)
    # Rotated priorities and several concentration levels keep all first-four bids legal.
    profiles=np.array([[1,1,1,1],[1,4,1,5],[1,1,4,5],[3,1,4,5],[1,3,4,1]],dtype=float)
    profile=profiles[rng.integers(len(profiles),size=size)];profile/=profile.sum(axis=1,keepdims=True)
    concentration=rng.choice([.5,2,8,24],size=size)
    probs=rng.gamma(profile*concentration[:,None]);probs/=probs.sum(axis=1,keepdims=True)
    rows=np.column_stack((rng.multinomial(100-fifth,probs),fifth))
    # Retain exact human/analyst anchors within the corresponding fifth-bid range.
    anchors={1:[(1,29,3,64,3),(2,2,45,51,0),(1,29,3,67,0),(1,9,40,50,0)],
             2:[(0,39,0,0,61),(9,3,0,27,61),(20,20,20,20,20),(3,8,4,44,41)],
             3:[(2,2,3,4,89),(1,1,1,1,96),(0,0,0,1,99)]}[family]
    use=rng.random(size)<.25;rows[use]=np.array(anchors)[rng.integers(len(anchors),size=int(use.sum()))]
    assert np.all(rows.sum(axis=1)==100)and np.all(rows[:,4]<100)
    return rows


def without_copies(pop):
    mass=sum(n for x,n in pop.entries if x==A)
    if mass==pop.total_mass:raise ValueError('No residual population')
    return Population.from_counts([(x,n)for x,n in pop.entries if x!=A]),Fraction(mass,pop.total_mass)


def pooled_population(populations):
    # Population snapshot totals are equal; support unequal totals exactly as well.
    from math import lcm
    total=lcm(*(p.total_mass for p in populations));counts=Counter()
    for p in populations:
        for x,n in p.entries:counts[x]+=n*(total//p.total_mass)
    return Population.from_counts(list(counts.items()))


def conditional_triplets(cell,size,seed,cloud):
    rng=np.random.default_rng(seed)
    if cell['kind']=='family':
        rows=np.empty((size,3,5),dtype=np.int64)
        for j,family in enumerate(cell['ids']):rows[:,j]=A if family==0 else residual_draw(family,size,rng)
        return rows
    k=cell['copies'];q=cell['reference'];rows=np.empty((size,3,5),dtype=np.int64);rows[:,:k]=A
    if k==3:return rows
    n=size*(3-k)
    if q['kind']=='population':
        pop=Population.from_dict(q['population']);draw=shared_triplets(pop,(n+2)//3,int(rng.integers(2**32)))['triplets'].reshape(-1,5)[:n]
    else:
        draw=[];remaining=n
        while remaining:
            batch_size=max(16,(remaining+2)//3);batch=[]
            for variant in [0,1]:batch.append(static_triplets(q['family'],variant,batch_size,int(rng.integers(2**32)),cloud).reshape(-1,5))
            # Choose a component independently for every row before copy rejection.
            labels=rng.integers(2,size=len(batch[0]));sample=np.stack(batch)[labels,np.arange(len(labels))]
            sample=sample[sample[:,4]!=100][:remaining];draw.append(sample);remaining-=len(sample)
        draw=np.concatenate(draw)
    assert np.all(draw[:,4]!=100);rows[:,k:]=draw.reshape(size,3-k,5)
    return rows


def family_tensor(means):
    a=np.asarray(means);result=np.empty(a.shape[:-1]+(4,4,4),dtype=a.dtype)
    for ids in product(range(4),repeat=3):result[(...,)+ids]=a[...,FAMILY_IDS.index(tuple(sorted(ids)))]
    return result


def polynomial_value(tensor,weights):return np.einsum('ijk,i,j,k->',tensor,weights,weights,weights)


def copy_coefficients(tensor,q):
    # a_k conditions on k exact copies, not the polynomial's monomial coefficients.
    q=np.r_[0,np.asarray(q,dtype=float)];a=np.array([1.,0,0,0])
    return np.stack([np.einsum('...ijk,i,j,k->...',tensor,q,q,q),
                     np.einsum('...ijk,i,j,k->...',tensor,a,q,q),
                     np.einsum('...ijk,i,j,k->...',tensor,a,a,q),tensor[...,0,0,0]],axis=-1)


def cubic_value(coeff,p):
    c=np.asarray(coeff);return c[...,0]*(1-p)**3+3*c[...,1]*p*(1-p)**2+3*c[...,2]*p*p*(1-p)+c[...,3]*p**3


def continuous_minimum(coeff,cap):
    if not 0<=cap<=1:raise ValueError('Copy cap outside [0,1]')
    c=np.asarray(coeff,dtype=float);b=3*(c[...,1]-c[...,0]);d=3*(c[...,0]-2*c[...,1]+c[...,2]);e=-c[...,0]+3*c[...,1]-3*c[...,2]+c[...,3]
    values=np.minimum(c[...,0],cubic_value(c,cap));where=np.where(c[...,0]<=cubic_value(c,cap),0.,cap)
    qa=3*e;qb=2*d;qc=b;disc=qb*qb-4*qa*qc
    with np.errstate(divide='ignore',invalid='ignore'):
        roots=[np.where(abs(qa)>1e-12,(-qb+np.sqrt(np.maximum(disc,0)))/(2*qa),-qc/qb),
               np.where(abs(qa)>1e-12,(-qb-np.sqrt(np.maximum(disc,0)))/(2*qa),np.nan)]
    for root in roots:
        valid=np.isfinite(root)&(root>=0)&(root<=cap)&((abs(qa)<=1e-12)|(disc>=0));safe=np.where(valid,root,0)
        y=cubic_value(c,safe);take=valid&(y<values);values=np.where(take,y,values);where=np.where(take,safe,where)
    return values,where


def domain_simplices(cap):
    cap=float(cap)
    if not 0<=cap<=1:raise ValueError('Copy cap outside [0,1]')
    eye=np.eye(4)
    if cap==0:return [eye[1:]]
    if cap==1:return [eye]
    base=eye[1:];top=cap*eye[0]+(1-cap)*base;v=np.concatenate((base,top))
    return [v[list(ids)]for ids in [(0,1,2,5),(0,1,4,5),(0,3,4,5)]]


def robust_partition(tensor,cap,tolerance=.015,max_splits=20000):
    """Empirical polynomial bounds; saved bisections support an exact rational audit."""
    tensor=np.asarray(tensor,dtype=float);roots=domain_simplices(cap);nodes=[];heap=[];best=float('inf');witness=None;next_id=0
    def add(v,parent=None):
        nonlocal best,witness,next_id
        coeff=np.einsum('ijk,ai,bj,ck->abc',tensor,v,v,v);lower=float(coeff.min())-1e-10
        points=np.concatenate((v,v.mean(axis=0,keepdims=True)));scores=[polynomial_value(tensor,w)for w in points];j=int(np.argmin(scores))
        if scores[j]<best:best=float(scores[j]);witness=points[j].copy()
        record={'id':next_id,'parent':parent,'vertices':v.tolist(),'lower':lower,'split':None};nodes.append(record);heapq.heappush(heap,(lower,next_id));next_id+=1
    for v in roots:add(v)
    splits=0
    while heap and best-heap[0][0]>tolerance and splits<max_splits:
        lower,index=heapq.heappop(heap);node=nodes[index];v=np.array(node['vertices']);dist=((v[:,None]-v[None,:])**2).sum(axis=2);i,j=np.unravel_index(int(dist.argmax()),dist.shape)
        midpoint=(v[i]+v[j])/2;left=v.copy();right=v.copy();left[i]=midpoint;right[j]=midpoint
        node['split']=[int(i),int(j)];add(left,index);add(right,index);splits+=1
    lower=min(x[0]for x in heap)
    return {'copy_cap':cap,'lower':lower,'upper':best+1e-10,'witness':witness.tolist(),'gap':best+1e-10-lower,
            'target_met':best+1e-10-lower<=tolerance+1e-9,'splits':splits,'nodes':nodes,
            'scope':'Entire continuous A/L/M/H family domain, empirical tensor only'}


def representation_path(actions,multiplicity,balanced,cfg,seed):
    actions=[validate_allocation(x)for x in actions];m=np.asarray(multiplicity,dtype=float)
    if len(set(actions))!=len(actions)or m.shape!=(len(actions),)or np.any(m<1):raise ValueError('Invalid unique support or multiplicities')
    prior=initial_shares(actions,'balanced',0)if balanced else m/m.sum();q=initial_shares(actions,'balanced',0);rng=np.random.default_rng(seed);trace=[];average=np.zeros(len(actions))
    for step in range(1,cfg['representation_steps']+1):
        ids=rng.choice(len(actions),size=(cfg['representation_triplets'],3),p=q);rows=np.asarray(actions)[ids];rank=metric_arrays(actions,rows)[0]
        fitness=rank.mean(axis=1)/4;proposal=prior*fitness
        if proposal.sum()<=0:proposal=q.copy()
        proposal/=proposal.sum();mu=cfg['representation_mutation'];q=(1-mu)*proposal+mu*prior;q/=q.sum();average+=q
        trace.append(q.copy())
    return {'final':q,'average':average/cfg['representation_steps'],'trace':np.array(trace),'prior':prior}
