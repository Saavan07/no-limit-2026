"""Fresh hypothesis shifts and coupled populations with distinct payoff objectives."""
from fractions import Fraction
from itertools import product
from pathlib import Path
import hashlib
import json
import numpy as np
from .core import Population,best_response,mix_tables,validate_allocation
from .historical import validate_rows
from .human import score_matrices
from .multiwave import atomic_json,simple_json


def mixture_weights(h,l,w,z):
    h,l,w,z=map(Fraction,(h,l,w,z))
    if any(not 0<=x<=1 for x in (h,l,w,z)):raise ValueError('Invalid share')
    return ((1-z)*(1-l)*(1-h),(1-z)*(1-l)*h,z,(1-z)*l*(1-w),(1-z)*l*w)


def fresh_definitions():
    coalitions=[(2,4),(3,4),(0,1,4),(0,2,3),(1,2,3)]
    validation=[];audit=[]
    for c in coalitions:
        label=''.join(str(i+1) for i in c)
        for scout,law in [('geometric','value'),('round','uniform'),('uniform','noisy')]:
            validation.append({'key':f'validation_{label}_{scout}_{law}','coalition':c,'scouts':scout,'law':law,'heavy':None})
        for kind,scout,law,heavy in [('threshold52','geometric','value',52 if 4 in c else None),
                ('inverse','moderate','inverse',None),('correlated','correlated','noisy',None)]:
            audit.append({'key':f'audit_{label}_{kind}','coalition':c,'scouts':scout,'law':law,'heavy':heavy})
    for omitted in range(5):
        validation.append({'key':f'validation_four_skip{omitted+1}','coalition':tuple(i for i in range(5) if i!=omitted),
                           'scouts':'geometric','law':'noisy','heavy':None})
    return validation,audit


def shifted_rows(definition,size,seed):
    if type(size) is not int or size<=0:raise ValueError('Positive sample size required')
    c=tuple(definition['coalition']);outside=[i for i in range(5) if i not in c]
    if not c or len(set(c))!=len(c) or min(c)<0 or max(c)>4:raise ValueError('Invalid priorities')
    rng=np.random.default_rng(seed);chunks=[];accepted=proposals=0
    values=np.array([i+1 for i in c],dtype=float);probabilities=values/values.sum()
    if definition['law']=='inverse':probabilities=(1/values)/(1/values).sum()
    geometric=np.exp(-np.arange(25)/4);geometric/=geometric.sum()
    while accepted<size:
        batch=max(4096,min(200000,2*(size-accepted)))
        rows=np.zeros((batch,5),dtype=np.int64);shape=(batch,len(outside));scout=definition['scouts']
        if scout=='geometric':draws=rng.choice(25,size=shape,p=geometric)
        elif scout=='round':draws=rng.choice([0,5,10,15,20],size=shape,p=np.array([4,3,2,1,1])/11)
        elif scout=='uniform':draws=rng.integers(0,25,size=shape)
        elif scout=='moderate':draws=rng.integers(2,19,size=shape)
        elif scout=='correlated':draws=np.repeat(rng.choice([0,12,24],size=(batch,1)),len(outside),axis=1)
        else:raise ValueError('Unknown scout law')
        rows[:,outside]=draws;free=100-rows.sum(axis=1)-len(c)
        law=definition['law']
        if law=='value':p=probabilities
        elif law=='uniform':p=rng.dirichlet(np.ones(len(c)),size=batch)
        elif law in ('noisy','inverse'):p=rng.dirichlet((4 if law=='noisy' else 9)*probabilities,size=batch)
        else:raise ValueError('Unknown priority law')
        rows[:,c]=rng.multinomial(free,p)+1;proposals+=batch
        if definition['heavy'] is not None:rows=rows[rows[:,4]>=definition['heavy']]
        take=min(size-accepted,len(rows));chunks.append(rows[:take]);accepted+=take
    result=np.concatenate(chunks);validate_rows(result)
    return result,{'seed':seed,'proposals':proposals,'accepted':size}


def quantize(probabilities,total):
    array=np.maximum(np.asarray(probabilities,dtype=float),0);array/=array.sum()
    scaled=array*total;masses=np.floor(scaled).astype(np.int64);missing=total-int(masses.sum())
    if not 0<=missing<=len(masses):raise ValueError('Quantization normalization failed')
    order=np.argsort(-(scaled-masses),kind='stable');masses[order[:missing]]+=1
    return [int(m) for m in masses]


def coupled_path(folder,actions,initial,win_share,rule,mutation,cfg,seed,max_step=None,callback=None):
    """Type shares are fixed. Strategies update within each objective type."""
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    actions=[validate_allocation(x) for x in actions]
    if len(set(actions))!=len(actions) or not 0<=win_share<=1 or not 0<=mutation<1:raise ValueError('Invalid coupled population')
    if rule not in ('replicator','payoff_replacement'):raise ValueError('Unknown update rule')
    initial=np.asarray(initial,dtype=float).copy();initial/=initial.sum()
    if len(initial)!=len(actions) or np.any(initial<0) or not np.isfinite(initial).all():raise ValueError('Invalid initial shares')
    if not 0<cfg['invasion_seed_mass']<1 or cfg['invasion_every']<=0:raise ValueError('Invalid innovation settings')
    settings={k:cfg[k] for k in ['learning_steps','invasion_every','invasion_seed_mass','record_every','quantization_mass']}
    signature={'actions':actions,'initial':initial.tolist(),'win_share':win_share,'rule':rule,'mutation':mutation,
        'settings':settings,'seed':seed,'source_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path(__file__).parent/'core.py',Path(__file__).parent/'human.py']}}
    digest=hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest()
    checkpoint=folder/'checkpoint.json'
    if checkpoint.exists():
        state=json.loads(checkpoint.read_text())
        stored_digest=state.pop('state_sha256')
        if hashlib.sha256(json.dumps(state,sort_keys=True,separators=(',',':')).encode()).hexdigest()!=stored_digest:raise ValueError('Coupled checkpoint content mismatch')
        if state['input_digest']!=digest:raise ValueError('Coupled checkpoint source/input mismatch')
        actions=[tuple(x) for x in state['actions']];point=np.array(state['point']);win=np.array(state['win'])
        cumulative=np.array(state['cumulative']);trace=state['trace'];innovations=state['innovations'];completed=state['step']
    else:
        point=initial.copy();win=initial.copy();cumulative=np.zeros(len(actions));trace=[];innovations=[];completed=0
    target=cfg['learning_steps'] if max_step is None else max_step
    if not max(1,completed)<=target<=cfg['learning_steps']:raise ValueError('Invalid target step')
    if len(point)!=len(actions) or len(win)!=len(actions) or len(cumulative)!=len(actions) or abs(point.sum()-1)>1e-10 or abs(win.sum()-1)>1e-10 or abs(cumulative.sum()-completed)>1e-8:raise ValueError('Invalid retained type masses')
    p2,w2=score_matrices(actions);matrix_point=p2.astype(float)/2;matrix_win=w2.astype(float)/2
    for step in range(completed+1,target+1):
        q=(1-win_share)*point+win_share*win
        fit_point=matrix_point@q;fit_win=matrix_win@q
        a=point*fit_point if rule=='replicator' else fit_point.copy()
        b=win*fit_win if rule=='replicator' else fit_win.copy()
        a/=a.sum();b/=b.sum()
        point=(1-mutation)*a+mutation/len(actions);win=(1-mutation)*b+mutation/len(actions)
        point/=point.sum();win/=win.sum()
        if step%cfg['invasion_every']==0:
            q=(1-win_share)*point+win_share*win;masses=quantize(q,cfg['quantization_mass'])
            population=Population.from_counts([(x,m) for x,m in zip(actions,masses) if m])
            response=best_response(population.table());added=response.allocation not in actions
            innovations.append({'step':step,'pre_invasion_population':population.to_dict(),
                'response':response,'added':added})
            if added:
                eta=cfg['invasion_seed_mass'];actions.append(response.allocation)
                point=np.append((1-eta)*point,eta);win=np.append((1-eta)*win,eta);cumulative=np.append(cumulative,0)
                p2,w2=score_matrices(actions);matrix_point=p2.astype(float)/2;matrix_win=w2.astype(float)/2
        q=(1-win_share)*point+win_share*win;cumulative+=q
        if abs(point.sum()-1)>1e-10 or abs(win.sum()-1)>1e-10 or min(point.min(),win.min())<0:raise AssertionError('Type mass invariant failed')
        if step%cfg['record_every']==0 or step==target:
            official_mean=float(q@(matrix_point@q))
            if abs(official_mean-7.5)>1e-10:raise AssertionError('Official constant-sum invariant failed')
            trace.append({'step':step,'pool_size':len(actions),'point_type_mean_fitness':float(point@(matrix_point@q)),
                'win_type_mean_fitness':float(win@(matrix_win@q)),'official_population_mean':official_mean,
                'fixed_win_share':win_share})
        if step%cfg['invasion_every']==0 or step==target:
            state={'step':step,'input_digest':digest,'actions':actions,'point':point.tolist(),'win':win.tolist(),
                'cumulative':cumulative.tolist(),'trace':trace,'innovations':innovations}
            state=simple_json(state)
            state['state_sha256']=hashlib.sha256(json.dumps(state,sort_keys=True,separators=(',',':')).encode()).hexdigest()
            atomic_json(checkpoint,state)
            if callback:callback(step,len(actions))
    return {'actions':actions,'point':point,'win':win,'final':(1-win_share)*point+win_share*win,
        'average':cumulative/target,'trace':trace,'innovations':innovations,'completed':target,'input_digest':digest}
