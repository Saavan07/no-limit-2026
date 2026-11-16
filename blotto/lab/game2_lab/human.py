"""Human Game 2 hypotheses, exact population weights, and shared-opponent scoring."""
from collections import Counter
from fractions import Fraction
from itertools import combinations_with_replacement,product
from math import factorial
import numpy as np
from blotto_lab.core import DEFAULT_GAME,validate_allocation
from .core import PREFIX4,_rows

LITERALS={'human_89':(2,2,3,4,89),'human_96':(1,1,1,1,96),'human_100':(0,0,0,0,100)}
ANCHORS={
 'all_in':[(0,0,0,0,100)],
 'abandon':[(1,9,40,50,0),(3,15,37,45,0),(0,0,50,50,0),(5,20,30,44,1),(2,8,40,47,3)],
 'broad':[(20,20,20,20,20),(7,13,20,27,33),(5,15,25,35,20),(10,20,40,20,10),(10,15,20,25,30)],
 'near_all_in':[(2,2,3,4,89),(1,1,1,1,96)]}


def payoff_arrays(candidates,triplets,candidate_batch=16,triplet_batch=8192):
    """Exact focal payout/score arrays, reusing full-vector opponent maxima."""
    opponents=_rows(triplets,3,DEFAULT_GAME)
    actions=np.array([validate_allocation(x)for x in candidates],dtype=np.int64)
    if candidate_batch<=0 or triplet_batch<=0:raise ValueError('Positive chunk sizes required')
    payout=np.empty((len(actions),len(opponents)),dtype=np.uint8);castle=np.empty_like(payout)
    for start in range(0,len(opponents),triplet_batch):
        rows=opponents[start:start+triplet_batch];maximum=rows.max(axis=1);winners=rows==maximum[:,None,:];ties=winners.sum(axis=1)
        for first in range(0,len(actions),candidate_batch):
            xs=actions[first:first+candidate_batch];own=np.zeros((len(xs),len(rows)),dtype=np.int64)
            enemy=np.zeros((len(xs),len(rows),3),dtype=np.int64)
            for i,value in enumerate(range(1,6)):
                below=xs[:,i,None]<maximum[None,:,i];equal=xs[:,i,None]==maximum[None,:,i]
                own+=value*((~(below|equal))*12+equal*(12//(ties[:,i]+1))[None,:])
                factor=value*(below*(12//ties[:,i])[None,:]+equal*(12//(ties[:,i]+1))[None,:])
                enemy+=factor[:,:,None]*winners[None,:,:,i]
            above=(enemy>own[:,:,None]).sum(axis=2);tied=1+(enemy==own[:,:,None]).sum(axis=2)
            numerator=PREFIX4[above+tied]-PREFIX4[above]
            if np.any(numerator%tied):raise AssertionError('Nonintegral rank payout units')
            payout[first:first+len(xs),start:start+len(rows)]=(numerator//tied).astype(np.uint8)
            castle[first:first+len(xs),start:start+len(rows)]=own.astype(np.uint8)
    return payout,castle


def anchor_support():return [x for rows in ANCHORS.values()for x in rows]


def triples_for_support(support):
    ids=np.array(list(combinations_with_replacement(range(len(support)),3)),dtype=np.int64)
    return ids,np.asarray(support,dtype=np.int64)[ids]


def triple_weights(ids,counts):
    counts=list(map(int,counts));weights=[]
    for triple in ids:
        used=Counter(map(int,triple));w=factorial(3)
        for k in used.values():w//=factorial(k)
        for i in triple:w*=counts[int(i)]
        weights.append(w)
    result=np.array(weights,dtype=np.int64);assert int(result.sum())==sum(counts)**3
    return result


def counts_for_mix(mix,total=100):
    shares=[Fraction(v)for v in mix]
    if len(shares)!=4 or min(shares)<0 or sum(shares)!=1:raise ValueError('Four shares must sum to one')
    result=[]
    for share,rows in zip(shares,ANCHORS.values()):
        value=share*total/len(rows)
        if value.denominator!=1:raise ValueError('Mass scale cannot encode this mixture exactly')
        result.extend([int(value)]*len(rows))
    return result


def candidate_field(seed=2026111101):
    rows={};
    def add(x,role):
        x=validate_allocation(x);rows.setdefault(x,[])
        if role not in rows[x]:rows[x].append(role)
    for label,x in LITERALS.items():add(x,label)
    for group,anchors in ANCHORS.items():
        for x in anchors:add(x,'anchor_'+group)
    add((3,8,4,44,41),'game1_selected')
    for a,b,c,e in product([0,1,2,3,5,8,12],[0,1,2,3,5,8,12],range(10,81,5),[0,1,3]):
        d=100-a-b-c-e
        if d>=0:add((a,b,c,d,e),'scout_34_grid')
    rng=np.random.default_rng(seed)
    for j in range(384):
        if j%3==0:
            bars=np.sort(rng.choice(104,4,replace=False));x=tuple(map(int,np.diff(np.r_[-1,bars,104])-1))
        else:
            concentration=[.5,2,8,25][j%4];p=rng.dirichlet(np.ones(5)*concentration);x=tuple(map(int,rng.multinomial(100,p)))
        add(x,'broad_seeded')
    return rows


def fresh_mixes():
    result=[]
    for raw in ['0','0.10','0.25','0.50','0.75','1']:
        p=Fraction(raw);rest=1-p;result.append(('copy_'+raw,[p,rest*Fraction(2,5),rest*Fraction(2,5),rest*Fraction(1,5)]))
    for name,values in [('abandon_heavy',['0','.7','.2','.1']),('copy_abandon',['.1','.7','.1','.1']),
        ('broad_heavy',['.1','.1','.7','.1']),('equal_families',['.25']*4),('near_heavy',['.1','.2','.1','.6']),
        ('mostly_broad',['.05','.05','.85','.05'])]:result.append((name,list(map(Fraction,values))))
    return result


def fresh_triplets(mix,style,size,seed):
    if style not in ['noisy','rounded','contrarian']or type(size)is not int or size<=0:raise ValueError('Invalid fresh draw')
    probs=np.array([float(Fraction(v))for v in mix]);probs/=probs.sum();rng=np.random.default_rng(seed)
    types=rng.choice(4,size=(size,3),p=probs);rows=np.zeros((size*3,5),dtype=np.int64);flat=types.ravel()
    for family in range(4):
        positions=np.flatnonzero(flat==family);n=len(positions)
        if not n:continue
        if family==0:rows[positions,4]=100
        elif family==1:
            if style=='rounded':scouts=rng.choice([0,1,2,5,10],size=(n,2));fifth=rng.choice([0,0,0,1,3],size=n)
            else:scouts=rng.integers(0,13 if style=='noisy'else 21,size=(n,2));fifth=rng.choice([0,0,1,3],size=n)
            remaining=100-scouts.sum(axis=1)-fifth
            share=rng.beta(8,9,size=n)if style!='contrarian'else rng.beta(1.5,1.5,size=n)
            third=rng.binomial(remaining,share);rows[positions]=np.column_stack((scouts,third,remaining-third,fifth))
        elif family==2:
            profile=np.array([1,2,3,4,5],dtype=float);profile/=profile.sum()
            p=rng.dirichlet((30 if style=='noisy'else 12)*profile,size=n);bids=rng.multinomial(100,p)
            if style=='rounded':
                units=bids//5;left=20-units.sum(axis=1);order=np.argsort(-(bids%5+rng.random(bids.shape)*.1),axis=1)
                for j in range(5):
                    chosen=np.flatnonzero(left>j);units[chosen,order[chosen,j]]+=1
                bids=units*5
            elif style=='contrarian':
                fifth=rng.integers(20,66,size=n);p=rng.dirichlet(np.ones(4)*3,size=n);bids=np.column_stack((rng.multinomial(100-fifth,p),fifth))
            copies=bids[:,4]==100
            bids[copies,4]-=1;bids[copies,0]+=1
            rows[positions]=bids
        else:
            fifth=rng.integers(81,100,size=n)if style!='rounded'else rng.choice([85,90,95,99],size=n)
            scouts=rng.multinomial(100-fifth,np.ones(4)/4);rows[positions]=np.column_stack((scouts,fifth))
    if np.any(rows<0)or not np.all(rows.sum(axis=1)==100):raise AssertionError('Illegal generated opponent')
    return rows.reshape(size,3,5),types
