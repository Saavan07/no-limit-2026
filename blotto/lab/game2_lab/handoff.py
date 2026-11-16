"""Independently verified theory and explicit handoff-inspired target laws."""
from fractions import Fraction
from itertools import combinations,combinations_with_replacement,product
from math import comb,factorial
import numpy as np
from blotto_lab.core import DEFAULT_GAME,Population,validate_allocation,iter_allocations
from .core import focal_samples


def critical_levels(support,game=DEFAULT_GAME):
    support=[validate_allocation(x,game)for x in support]
    if not support:raise ValueError('Nonempty opponent support required')
    return [sorted({0,*[x[i]for x in support],*[x[i]+1 for x in support if x[i]<game.budget]})for i in range(game.castles)]


def critical_count(support,slack,game=DEFAULT_GAME):
    levels=critical_levels(support,game);counts=[1]+[0]*game.budget
    for i,values in enumerate(levels):
        if i==slack:continue
        nxt=[0]*(game.budget+1)
        for spent,count in enumerate(counts):
            if count:
                for bid in values:
                    if spent+bid<=game.budget:nxt[spent+bid]+=count
        counts=nxt
    return sum(counts)


def critical_actions(support,game=DEFAULT_GAME,slack=None):
    levels=critical_levels(support,game)
    if slack is None:slack=min(range(game.castles),key=lambda j:(critical_count(support,j,game),j))
    if not 0<=slack<game.castles:raise ValueError('Invalid slack coordinate')
    fields=[i for i in range(game.castles)if i!=slack];x=[0]*game.castles
    def visit(pos,spent):
        if pos==len(fields):x[slack]=game.budget-spent;yield tuple(x);return
        i=fields[pos]
        for value in levels[i]:
            if spent+value<=game.budget:x[i]=value;yield from visit(pos+1,spent+value)
    yield from visit(0,0)


def exact_critical_response(population,maximum_actions=100000):
    support=[x for x,n in population.entries];counts=[n for x,n in population.entries];game=population.game
    slack=min(range(game.castles),key=lambda j:(critical_count(support,j,game),j));number=critical_count(support,slack,game)
    if number>maximum_actions:raise ValueError('Exact response exceeds the declared action limit')
    ids=list(combinations_with_replacement(range(len(support)),3));rows=np.array([[support[i]for i in triple]for triple in ids]);weights=[]
    for triple in ids:
        multiplicities=[triple.count(i)for i in set(triple)];m=factorial(3)
        for n in multiplicities:m//=factorial(n)
        for i in triple:m*=counts[i]
        weights.append(m)
    assert sum(weights)==population.total_mass**3;best=-1;chosen=None;checked=0
    for x in critical_actions(support,game,slack):
        samples=focal_samples(x,rows,game)['payout4'];value=sum(int(v)*w for v,w in zip(samples,weights));checked+=1
        if value>best or(value==best and(chosen is None or x<chosen)):best=value;chosen=x
    assert checked==number
    return {'allocation':chosen,'payout':Fraction(best,4*population.total_mass**3),'critical_actions':number,'slack_coordinate':slack,'unordered_opponent_triples':len(rows),
            'scope':'Exact global response value for this finite iid population; representative tie order only'}


def target_population(active,scout,weighting,kappa=25,scale=1000000):
    active=tuple(active)
    if len(active)not in [2,3]or len(set(active))!=len(active)or min(active)<0 or max(active)>4 or not 0<=scout<=10:raise ValueError('Invalid target law')
    remaining=100-(5-len(active))*scout;extra=remaining-8*len(active);entries=[]
    def splits(total,n):
        if n==1:yield(total,);return
        for v in range(total+1):
            for rest in splits(total-v,n-1):yield(v,)+rest
    values=np.array([i+1 for i in active],dtype=float)
    center=np.full(len(active),remaining/len(active))if weighting=='equal'else remaining*values/values.sum()
    if weighting not in ['uniform','equal','value']:raise ValueError('Unknown target weighting')
    for split in splits(extra,len(active)):
        bids=np.array(split)+8;x=[scout]*5
        for i,value in zip(active,bids):x[i]=int(value)
        weight=1 if weighting=='uniform'else max(1,int(round(scale*np.exp(-kappa*float(np.sum(((bids-center)/remaining)**2))))))
        entries.append((tuple(x),weight))
    return Population.from_counts(entries)


def cdf_diagnostic(values,weights,public_scores):
    values=np.asarray(values,dtype=float);weights=np.asarray(weights,dtype=float);weights/=weights.sum();public=np.sort(np.asarray(public_scores,dtype=float));order=np.argsort(values);v=values[order];w=np.cumsum(weights[order])
    def model(t):
        ix=np.searchsorted(v,t,side='right');return np.where(ix>0,w[np.maximum(ix-1,0)],0)
    def observed(t):return np.searchsorted(public,t,side='right')/len(public)
    knots=np.unique(np.r_[v,public]);query=np.r_[knots,np.nextafter(knots,-np.inf)];maximum=float(np.max(abs(model(query)-observed(query))))
    grid=np.linspace(0,9,181);mean=float(np.mean(abs(model(grid)-observed(grid))));threshold=float(public[-1]+.2)
    return {'maximum_cdf_gap':maximum,'mean_grid_cdf_gap':mean,'mass_above_public_max_plus_point_two':float(weights[values>threshold].sum()),'weighted_mean':float(weights@values),'minimum':float(values.min()),'maximum':float(values.max()),
            'quantiles':{str(q):float(v[min(len(v)-1,int(np.searchsorted(w,q)))])for q in [.1,.25,.5,.75,.9,.95,.99]}}
