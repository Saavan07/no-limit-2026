"""Broad nonpriority bids, sampled under an explicit legal joint construction."""
import numpy as np
from .historical import validate_rows


def broad_scout_rows(coalition,law,size,seed,heavy_five=None,scout_max=24):
    if law not in ('uniform','value_multinomial') or type(size) is not int or size<=0:
        raise ValueError('Invalid sampling settings')
    target=tuple(sorted(coalition));outside=[i for i in range(5) if i not in target]
    if not target or min(target)<0 or max(target)>=5 or len(set(target))!=len(target):raise ValueError('Invalid coalition')
    if heavy_five is not None and 4 not in target:raise ValueError('Five-heavy needs castle 5')
    if type(scout_max) is not int or not 0<=scout_max<=24 or (heavy_five is not None and (type(heavy_five) is not int or not 1<=heavy_five<=100-(len(target)-1))):raise ValueError('Invalid bid bounds')
    rng=np.random.default_rng(seed);chunks=[];accepted=proposed=0
    probabilities=np.array([i+1 for i in target],dtype=float);probabilities/=probabilities.sum()
    while accepted<size:
        batch=max(4096,min(200000,2*(size-accepted)))
        rows=np.zeros((batch,5),dtype=np.int64)
        rows[:,outside]=rng.integers(0,scout_max+1,size=(batch,len(outside)))
        free=100-rows.sum(axis=1)-len(target)
        if law=='uniform':weights=rng.dirichlet(np.ones(len(target)),size=batch)
        else:weights=probabilities
        rows[:,target]=rng.multinomial(free,weights)+1
        proposed+=batch
        if heavy_five is not None:rows=rows[rows[:,4]>=heavy_five]
        take=min(size-accepted,len(rows));chunks.append(rows[:take]);accepted+=take
    result=np.concatenate(chunks);validate_rows(result)
    return result,{'seed':seed,'accepted':size,'proposals_generated':proposed,'scout_max':scout_max,
        'conditioning':'reject complete proposal unless fifth bid >= minimum' if heavy_five is not None else 'none'}
