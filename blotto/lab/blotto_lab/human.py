"""Faithful human inputs, exact priority-coalition hypotheses, and learning tests."""
from fractions import Fraction
from itertools import combinations, product
from math import factorial
import numpy as np
from .core import DEFAULT_GAME, Game, Population, iter_allocations, validate_allocation


def ranged_allocations(bounds,radius=0,game=DEFAULT_GAME):
    if len(bounds)!=game.castles or radius<0:
        raise ValueError('Invalid range definition')
    clipped=[(max(0,a-radius),min(game.budget,b+radius)) for a,b in bounds]
    return [tuple(prefix)+(game.budget-sum(prefix),)
        for prefix in product(*(range(a,b+1) for a,b in clipped[:-1]))
        if clipped[-1][0]<=game.budget-sum(prefix)<=clipped[-1][1]]


def minimal_winning_coalitions(game=DEFAULT_GAME):
    winning=[s for k in range(1,game.castles+1) for s in combinations(range(game.castles),k)
             if 2*sum(game.values[i] for i in s)>game.total_value]
    return [s for s in winning if not any(set(t)<set(s) for t in winning)]


def coalition_population(coalition,scout,law='uniform',heavy_five=None,game=DEFAULT_GAME):
    target=tuple(sorted(coalition))
    if not target or len(set(target))!=len(target) or min(target)<0 or max(target)>=game.castles:
        raise ValueError('Invalid priority coalition')
    if type(scout) is not int or scout<0 or law not in ('uniform','value_multinomial'):
        raise ValueError('Invalid coalition law')
    if heavy_five is not None and 4 not in target:
        raise ValueError('Five-heavy restriction needs castle 5 as a priority')
    free=game.budget-scout*(game.castles-len(target))-len(target)
    if free<0: raise ValueError('Scouts and priority minimum exceed budget')
    factorials=[factorial(i) for i in range(free+1)]
    weights=[]
    for extra in iter_allocations(Game(free,(1,)*len(target))):
        x=[scout]*game.castles
        for i,k in zip(target,extra):x[i]=k+1
        if heavy_five is not None and x[4]<heavy_five:continue
        if law=='uniform':mass=1
        else:
            divisor=1
            for k in extra:divisor*=factorials[k]
            mass=factorials[free]//divisor
            for i,k in zip(target,extra):mass*=game.values[i]**k
        weights.append((tuple(x),mass))
    name='coal_'+''.join(str(i+1) for i in target)+f'_s{scout}_{law}'
    if heavy_five is not None:name+='_heavy5'
    return Population.from_counts(weights,game,name)


def coalition_anchor(coalition,scout=1,game=DEFAULT_GAME):
    target=tuple(sorted(coalition)); budget=game.budget-scout*(game.castles-len(target))
    value=sum(game.values[i] for i in target)
    shares={i:Fraction(budget*game.values[i],value) for i in target}
    bids={i:int(s) for i,s in shares.items()}
    for i in sorted(target,key=lambda i:(-(shares[i]-bids[i]),i))[:budget-sum(bids.values())]:bids[i]+=1
    x=[scout]*game.castles
    for i in target:x[i]=bids[i]
    return validate_allocation(x,game)


def marginal_counts(table,castle):
    row=table.numerators[castle]; denominator=table.game.values[castle]*table.denominator
    counts=[2*row[0]]
    for a,b in zip(row,row[1:]):counts.append(2*(b-a)-counts[-1])
    if min(counts)<0 or sum(counts)!=denominator:
        raise ValueError('Table does not encode a valid marginal distribution')
    return counts,denominator


def score_matrices(actions,game=DEFAULT_GAME):
    rows=np.asarray([validate_allocation(x,game) for x in actions],dtype=np.int64)
    scores=np.zeros((len(rows),len(rows)),dtype=np.int16)
    for i,v in enumerate(game.values):
        scores += 2*v*(rows[:,None,i]>rows[None,:,i])+v*(rows[:,None,i]==rows[None,:,i])
    wins=(2*(scores>game.total_value)+(scores==game.total_value)).astype(np.int16)
    if not np.all(scores+scores.T==2*game.total_value) or not np.all(wins+wins.T==2):
        raise AssertionError('Constant-sum matrix invariant failed')
    return scores,wins


def joint_metrics(actions,population,event_actions=None):
    rows=np.asarray([x for x,m in population.entries],dtype=np.int64)
    masses=np.array([m for x,m in population.entries],dtype=object)
    total=population.total_mass
    output={}
    event_actions=set(event_actions or [])
    for x in actions:
        x=tuple(x)
        twice=np.zeros(len(rows),dtype=np.int16)
        for i,v in enumerate(population.game.values):twice += 2*v*(x[i]>rows[:,i])+v*(x[i]==rows[:,i])
        win=sum(masses[twice>population.game.total_value]);tie=sum(masses[twice==population.game.total_value])
        item={'win_probability':Fraction(int(win),total),'tie_probability':Fraction(int(tie),total),
              'win_tie_adjusted':Fraction(int(2*win+tie),2*total)}
        if x in event_actions:
            item['joint_mean_points']=Fraction(sum(int(s)*m for s,m in zip(twice,masses)),2*total)
            wins234=sum((x[i]>rows[:,i]).astype(np.int8) for i in (1,2,3))
            for label,mask in [('lose5',rows[:,4]>x[4]),('lose5_by20',rows[:,4]>=x[4]+20)]:
                mass=sum(masses[mask]);points=sum(int(s)*m for s,m in zip(twice[mask],masses[mask]))
                item[label+'_probability']=Fraction(int(mass),total)
                item[label+'_points_mass']=Fraction(int(points),2*total)
                item[label+'_win2plus234_mass']=Fraction(int(sum(masses[mask & (wins234>=2)])),total)
        output[x]=item
    return output


def learning_path(payoffs,initial,rule,mutation,steps,record_every=50):
    matrix=np.asarray(payoffs,dtype=np.float64)
    p=np.asarray(initial,dtype=np.float64).copy();p/=p.sum()
    if matrix.shape!=(len(p),len(p)) or np.any(p<0) or not 0<=mutation<1 or not np.isfinite(p).all() or not np.isfinite(matrix).all() or np.any(matrix<0):
        raise ValueError('Invalid learning population')
    if type(steps) is not int or steps<=0 or type(record_every) is not int or record_every<=0:
        raise ValueError('Positive step and recording counts required')
    if rule not in ('replicator','payoff_replacement'):raise ValueError('Unknown revision rule')
    average=np.zeros_like(p);records=[]
    for step in range(1,steps+1):
        fitness=matrix@p
        if rule=='replicator':proposal=p*fitness
        else:proposal=fitness.copy()
        if proposal.sum()<=0:raise ValueError('Undefined nonpositive total fitness')
        proposal/=proposal.sum()
        p=(1-mutation)*proposal+mutation/len(p)
        if abs(p.sum()-1)>1e-10 or np.any(p<0):raise AssertionError('Learning simplex invariant failed')
        p/=p.sum();average+=p
        if step%record_every==0 or step==steps:
            values=matrix@p
            records.append({'step':step,'mean_fitness':float(p@values),'best_available_fitness':float(values.max()),
                'largest_share':float(p.max()),'positive_shares':int((p>1e-12).sum()),
                'replacement_residual_l1':float(np.abs(p-values/values.sum()).sum())})
    return {'final':p,'average':average/steps,'trace':records}
