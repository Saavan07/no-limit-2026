"""Exact four-player scoring, joint-population evaluation, and shared triplets."""
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations_with_replacement
from math import comb,factorial,sqrt,log
import numpy as np
from blotto_lab.core import DEFAULT_GAME,Population,validate_allocation

PRIZES=(9,4,2,0)
PREFIX4=np.array([0,36,52,60,60],dtype=np.int64)


@dataclass(frozen=True)
class GroupOutcome:
    castle_points: tuple[Fraction,...]
    payouts: tuple[Fraction,...]


@dataclass(frozen=True)
class Evaluation:
    expected_payout: Fraction
    expected_castle_points: Fraction
    finish_credits: tuple[Fraction,...]
    outright_first_probability: Fraction
    rank_tie_probability: Fraction
    groups_evaluated: int
    probability_denominator: int
    sampling_scheme: str


def reference_group(actions,game=DEFAULT_GAME):
    rows=tuple(validate_allocation(x,game)for x in actions)
    if len(rows)!=4:raise ValueError('Exactly four players required')
    totals=[Fraction(0)for _ in range(4)]
    for i,value in enumerate(game.values):
        maximum=max(x[i]for x in rows);winners=[j for j,x in enumerate(rows)if x[i]==maximum]
        for j in winners:totals[j]+=Fraction(value,len(winners))
    payouts=[]
    for total in totals:
        above=sum(v>total for v in totals);tied=sum(v==total for v in totals)
        payouts.append(Fraction(sum(PRIZES[above:above+tied]),tied))
    assert sum(totals)==game.total_value and sum(payouts)==15
    return GroupOutcome(tuple(totals),tuple(payouts))


def rank_payouts4(castle12):
    scores=np.asarray(castle12)
    if scores.ndim!=2 or scores.shape[1]!=4 or scores.dtype.kind not in 'iu' or np.any(scores<0):raise ValueError('Four nonnegative integer totals required')
    greater=(scores[:,None,:]>scores[:,:,None]).sum(axis=2)
    tied=(scores[:,None,:]==scores[:,:,None]).sum(axis=2)
    numerator=PREFIX4[greater+tied]-PREFIX4[greater]
    if np.any(numerator%tied):raise AssertionError('Prize vector incompatible with quarter units')
    return numerator//tied


def _rows(array,players,game):
    a=np.asarray(array)
    if a.ndim!=3 or a.shape[1:]!=(players,game.castles)or a.dtype.kind not in 'iu':raise ValueError('Invalid group shape or noninteger bids')
    if game.total_value>np.iinfo(np.int64).max//12:raise ValueError('Castle values exceed safe integer range')
    if np.any(a<0)or np.any(a>game.budget)or not np.all(a.sum(axis=2)==game.budget):raise ValueError('Illegal group allocations')
    return a.astype(np.int64,copy=False)


def group_scores12_4(groups,game=DEFAULT_GAME):
    rows=_rows(groups,4,game);maximum=rows.max(axis=1);wins=rows==maximum[:,None,:];ties=wins.sum(axis=1)
    castle12=(wins*(12//ties)[:,None,:]*np.array(game.values,dtype=np.int64)[None,None,:]).sum(axis=2)
    payouts4=rank_payouts4(castle12)
    if not np.all(castle12.sum(axis=1)==12*game.total_value)or not np.all(payouts4.sum(axis=1)==60):raise AssertionError('Group conservation failed')
    return castle12,payouts4


def focal_samples(candidate,opponent_triplets,game=DEFAULT_GAME):
    x=validate_allocation(candidate,game);opponents=_rows(opponent_triplets,3,game)
    focal=np.broadcast_to(np.array(x),(len(opponents),1,game.castles))
    groups=np.concatenate((focal,opponents),axis=1);castle12,payout4=group_scores12_4(groups,game)
    higher=(castle12>castle12[:,0,None]).sum(axis=1);ties=(castle12==castle12[:,0,None]).sum(axis=1)
    credits12=np.zeros((len(opponents),4),dtype=np.int64)
    for place in range(4):credits12[:,place]=((higher<=place)&(place<higher+ties))*(12//ties)
    return {'payout4':payout4[:,0],'castle12':castle12[:,0],'finish_credits12':credits12,
            'outright_first':(higher==0)&(ties==1),'rank_tie':ties>1}


def _summarize(samples,weights=None,scheme='conditional shared samples'):
    count=len(samples['payout4'])
    if not count:raise ValueError('At least one opponent triple required')
    masses=[1]*count if weights is None else list(weights)
    if len(masses)!=count or any(type(m)is not int or m<=0 for m in masses):raise ValueError('Positive integer triple weights required')
    total=sum(masses)
    weighted=lambda values:sum(int(v)*m for v,m in zip(values,masses))
    payout=Fraction(weighted(samples['payout4']),4*total);castle=Fraction(weighted(samples['castle12']),12*total)
    credits=tuple(Fraction(weighted(samples['finish_credits12'][:,j]),12*total)for j in range(4))
    assert sum(credits)==1 and sum(p*c for p,c in zip(PRIZES,credits))==payout
    return Evaluation(payout,castle,credits,Fraction(weighted(samples['outright_first']),total),
                      Fraction(weighted(samples['rank_tie']),total),count,total,scheme)


def exact_evaluation(candidate,population,scheme='iid'):
    """Finite counts mean distinct opponents; iid integer masses mean probabilities."""
    x=validate_allocation(candidate,population.game)
    if scheme not in ['iid','finite']:raise ValueError('Choose iid or finite')
    total=population.total_mass
    if scheme=='finite'and total<3:raise ValueError('Need three distinct opponent people')
    support=[a for a,m in population.entries];counts=[m for a,m in population.entries];triples=[];weights=[]
    for ids in combinations_with_replacement(range(len(support)),3):
        used=Counter(ids)
        if scheme=='finite':
            weight=1
            for i,k in used.items():weight*=comb(counts[i],k)
        else:
            weight=factorial(3)
            for k in used.values():weight//=factorial(k)
            for i in ids:weight*=counts[i]
        if weight:triples.append([support[i]for i in ids]);weights.append(weight)
    denominator=total**3 if scheme=='iid'else comb(total,3)
    assert sum(weights)==denominator
    result=_summarize(focal_samples(x,np.asarray(triples,dtype=np.int64),population.game),weights,'exact '+scheme)
    assert result.probability_denominator==denominator
    return result


def shared_triplets(population,size,seed,scheme='iid'):
    """Return legal full-vector opponents and distinct person IDs for finite draws."""
    if type(size)is not int or size<=0 or scheme not in ['iid','finite']:raise ValueError('Invalid triplet request')
    rng=np.random.default_rng(seed);support=np.array([x for x,m in population.entries],dtype=np.int64);total=population.total_mass
    if scheme=='iid':
        probabilities=np.array([float(Fraction(m,total))for x,m in population.entries]);probabilities/=probabilities.sum()
        types=rng.choice(len(support),size=(size,3),p=probabilities);people=None
    else:
        if not 3<=total<=np.iinfo(np.int64).max:raise ValueError('Finite opponent count outside supported range')
        a=rng.integers(total,size=size);b=rng.integers(total-1,size=size);b+=b>=a
        lo=np.minimum(a,b);hi=np.maximum(a,b);c=rng.integers(total-2,size=size);c+=c>=lo;c+=c>=hi
        people=np.column_stack((a,b,c));cumulative=np.cumsum([m for x,m in population.entries],dtype=np.int64)
        types=np.searchsorted(cumulative,people,side='right')
        if np.any(people[:,0]==people[:,1])or np.any(people[:,0]==people[:,2])or np.any(people[:,1]==people[:,2]):raise AssertionError('Repeated finite person')
    return {'triplets':support[types],'type_ids':types,'person_ids':people,'seed':seed,'scheme':scheme}


def evaluate_shared(candidates,triplets,game=DEFAULT_GAME,batch_size=8192):
    """Common triplets support paired comparisons; no independence among candidates."""
    if type(batch_size)is not int or batch_size<=0:raise ValueError('Positive batch size required')
    rows=_rows(triplets,3,game);results=[]
    for x in candidates:
        parts=[focal_samples(x,rows[start:start+batch_size],game)for start in range(0,len(rows),batch_size)]
        if not parts:raise ValueError('No triplets')
        samples={k:np.concatenate([p[k]for p in parts])for k in parts[0]}
        results.append(_summarize(samples))
    return results


def fixed_sample_bound(size,comparisons=1,alpha=.05,paired=False):
    if type(size)is not int or size<=0 or type(comparisons)is not int or comparisons<=0 or not 0<alpha<1:raise ValueError('Invalid bound settings')
    width=18 if paired else 9
    return width*sqrt(log(2*comparisons/alpha)/(2*size))
