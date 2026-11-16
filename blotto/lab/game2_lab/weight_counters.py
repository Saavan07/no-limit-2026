"""Exact worst pure clone population for a fixed Game 2 allocation."""
from itertools import product
from fractions import Fraction
import numpy as np
from blotto_lab.core import DEFAULT_GAME,validate_allocation
from .core import group_scores12_4


def worst_clone_population(candidate,game=DEFAULT_GAME):
    x=validate_allocation(candidate,game);actions=[]
    # Every opponent vector in a relation cell gives the same complete group score.
    for state in product(range(3),repeat=game.castles):
        lo=[0 if s==0 else(v if s==1 else v+1)for v,s in zip(x,state)]
        hi=[v-1 if s==0 else(v if s==1 else game.budget)for v,s in zip(x,state)]
        if any(a>b for a,b in zip(lo,hi))or sum(lo)>game.budget or sum(hi)<game.budget:continue
        y=lo.copy();left=game.budget-sum(y)
        for j in range(game.castles-1,-1,-1):take=min(left,hi[j]-y[j]);y[j]+=take;left-=take
        assert not left;actions.append(tuple(y))
    groups=np.array([[x,y,y,y]for y in actions]);castle,rank=group_scores12_4(groups,game)
    chosen=min(range(len(actions)),key=lambda i:(int(rank[i,0]),int(castle[i,0]),actions[i]))
    return {'candidate':x,'opponent':actions[chosen],'payout':Fraction(int(rank[chosen,0]),4),'castle_points':Fraction(int(castle[chosen,0]),12),
            'states_evaluated':len(actions),'scope':'Exact worst iid population concentrated on one pure allocation; not worst general iid mixture'}
