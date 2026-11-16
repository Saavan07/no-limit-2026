"""New final-challenge laws and retained-strategy arenas with large transfers."""
from pathlib import Path
import hashlib
import json
import numpy as np
from blotto_lab.core import Population,best_response,validate_allocation
from blotto_lab.historical import validate_rows
from blotto_lab.human import score_matrices
from blotto_lab.merged import quantize
from blotto_lab.multiwave import atomic_json,simple_json

COALITIONS=[(2,4),(3,4),(0,1,4),(0,2,3),(1,2,3)]
LAWS=['soft','heaped','middle5','contrarian']


def final_definitions(phase):
    if phase not in ['challenge','confirmation']:raise ValueError('Unknown phase')
    result=[]
    for c in COALITIONS:
        for law in LAWS:
            result.append({'key':phase+'_'+''.join(str(i+1)for i in c)+'_'+law,
                           'phase':phase,'coalition':list(c),'law':law,'family':law})
    for omitted in range(5):
        result.append({'key':f'{phase}_four_skip{omitted+1}','phase':phase,
                       'coalition':[i for i in range(5)if i!=omitted],'law':'diffuse','family':'four_field'})
    result.append({'key':phase+'_all_diffuse','phase':phase,'coalition':list(range(5)),
                   'law':'diffuse','family':'diffuse'})
    return result


def final_rows(definition,size,seed):
    if type(size)is not int or size<=0:raise ValueError('Positive size required')
    c=tuple(definition['coalition']);law=definition['law'];phase=definition['phase']
    if not c or len(set(c))!=len(c)or min(c)<0 or max(c)>4:raise ValueError('Invalid priorities')
    if phase not in ['challenge','confirmation']or law not in LAWS+['diffuse']:raise ValueError('Invalid law')
    rng=np.random.default_rng(seed);shift=phase=='confirmation';outside=[i for i in range(5)if i not in c]
    values=np.arange(1,6,dtype=float);rows=np.zeros((size,5),dtype=np.int64)
    if law in ['soft','heaped']:
        if law=='soft':scouts=rng.integers(0,29 if shift else 17,size=(size,len(outside)))
        else:scouts=rng.choice([0,2,5,10,15,20,25,30],size=(size,len(outside)))
        rows[:,outside]=scouts;free=100-rows.sum(axis=1)
        priority=values[list(c)]**(.5 if shift else 1.2);priority/=priority.sum()
        p=rng.dirichlet((1.2 if shift else 8)*priority,size=size)
        rows[:,c]=rng.multinomial(free,p)
        if law=='heaped':
            unit=10 if shift else 5;units=rows//unit
            residual=100//unit-units.sum(axis=1)
            ranking=np.argsort(-(rows%unit+rng.random(rows.shape)*.1),axis=1,kind='stable')
            for j in range(5):
                selected=np.flatnonzero(residual>j);units[selected,ranking[selected,j]]+=1
            rows=units*unit
            donors=np.argmax(rows,axis=1);receivers=rng.integers(0,4,size=size);receivers+=receivers>=donors
            amount=rng.integers(0,5 if shift else 3,size=size);index=np.arange(size)
            rows[index,donors]-=amount;rows[index,receivers]+=amount
    elif law=='middle5':
        support=np.arange(12,55)if shift else np.arange(18,47)
        weight=(np.exp(-abs(support-22)/5)+np.exp(-abs(support-42)/5))if shift else np.exp(-abs(support-32)/8)
        fifth=rng.choice(support,size=size,p=weight/weight.sum());rows[:,4]=fifth
        priority=values[:4]**.8*np.array([7 if i in c else .8 for i in range(4)])
        priority/=priority.sum();p=rng.dirichlet((3 if shift else 10)*priority,size=size)
        rows[:,:4]=rng.multinomial(100-fifth,p)
    else:
        profile=values**(-.5 if shift else .7)*np.array([5 if i in c else .7 for i in range(5)])
        profile/=profile.sum();concentration=(.6 if shift else 4)if law=='diffuse'else(1 if shift else 7)
        p=rng.dirichlet(concentration*profile,size=size)
        if law=='contrarian'and outside:
            profile2=values**.3*np.array([6 if i in outside else .7 for i in range(5)])
            profile2/=profile2.sum();mask=rng.random(size)<(.45 if shift else .25)
            p[mask]=rng.dirichlet(concentration*profile2,size=int(mask.sum()))
        rows=rng.multinomial(100,p)
    validate_rows(rows)
    return rows


def large_mutant(action,rng,minimum=5,maximum=25):
    x=list(validate_allocation(action));donors=[i for i,v in enumerate(x)if v>=minimum]
    if not donors:raise ValueError('No eligible donor')
    donor=int(rng.choice(donors));receiver=int(rng.integers(0,4));receiver+=receiver>=donor
    amount=int(rng.integers(minimum,min(maximum,x[donor])+1));x[donor]-=amount;x[receiver]+=amount
    return validate_allocation(x),{'donor':donor,'receiver':receiver,'amount':amount}


def population_from_shares(actions,shares,total):
    return Population.from_counts([(x,m)for x,m in zip(actions,quantize(shares,total))if m])


def enhanced_path(folder,actions,initial,win_share,rule,mutation,cfg,seed,max_step=None):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    actions=[validate_allocation(x)for x in actions];initial=np.asarray(initial,dtype=float).copy()
    if len(set(actions))!=len(actions)or len(initial)!=len(actions)or not np.isfinite(initial).all()or np.any(initial<0)or initial.sum()<=0:raise ValueError('Invalid initial pool')
    if not 0<=win_share<=1 or not 0<=mutation<1 or rule not in ['replicator','payoff_replacement']:raise ValueError('Invalid type settings')
    initial/=initial.sum()
    settings={k:cfg[k]for k in ['learning_steps','invasion_every','invasion_seed_mass','record_every','quantization_mass','large_transfer_min','large_transfer_max']}
    if not 0<cfg['invasion_seed_mass']<1:raise ValueError('Invalid admission mass')
    signature={'actions':actions,'initial':initial.tolist(),'win_share':win_share,'rule':rule,'mutation':mutation,'settings':settings,'seed':seed,
               'sources':{p.name:hashlib.sha256(p.read_bytes()).hexdigest()for p in [Path(__file__),Path(__file__).parent/'blotto_lab/core.py',Path(__file__).parent/'blotto_lab/human.py']}}
    digest=hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest();checkpoint=folder/'checkpoint.json'
    rng=np.random.default_rng(seed)
    if checkpoint.exists():
        state=json.loads(checkpoint.read_text());stored=state.pop('state_sha256')
        if hashlib.sha256(json.dumps(state,sort_keys=True,separators=(',',':')).encode()).hexdigest()!=stored:raise ValueError('Checkpoint content changed')
        if state['input_digest']!=digest:raise ValueError('Checkpoint input changed')
        actions=[tuple(x)for x in state['actions']];point=np.array(state['point']);win=np.array(state['win']);cumulative=np.array(state['cumulative'])
        trace=state['trace'];events=state['events'];completed=state['step'];rng.bit_generator.state=state['rng_state']
    else:point=initial.copy();win=initial.copy();cumulative=np.zeros(len(actions));trace=[];events=[];completed=0
    target=cfg['learning_steps']if max_step is None else max_step
    if not completed<=target<=cfg['learning_steps']or target<=0:raise ValueError('Invalid target')
    if abs(point.sum()-1)>1e-10 or abs(win.sum()-1)>1e-10 or abs(cumulative.sum()-completed)>1e-8:raise ValueError('Checkpoint mass mismatch')
    p2,w2=score_matrices(actions);mp=p2.astype(float)/2;mw=w2.astype(float)/2
    for step in range(completed+1,target+1):
        q=(1-win_share)*point+win_share*win;fp=mp@q;fw=mw@q
        a=point*fp if rule=='replicator'else fp.copy();b=win*fw if rule=='replicator'else fw.copy()
        point=(1-mutation)*a/a.sum()+mutation/len(actions);win=(1-mutation)*b/b.sum()+mutation/len(actions)
        point/=point.sum();win/=win.sum()
        if step%cfg['invasion_every']==0:
            q=(1-win_share)*point+win_share*win
            current=population_from_shares(actions,q,cfg['quantization_mass'])
            historical=population_from_shares(actions,(cumulative+q)/step,cfg['quantization_mass'])
            proposals=[]
            for label,pop in [('current',current),('time_average',historical)]:
                br=best_response(pop.table());proposals.append((br.allocation,{'source':label,'target_population':pop.to_dict(),'response':br}))
            for j in range(2):
                parent=int(rng.choice(len(actions),p=q));x,detail=large_mutant(actions[parent],rng,cfg['large_transfer_min'],cfg['large_transfer_max'])
                proposals.append((x,{'source':'large_transfer','parent':actions[parent],'transfer':detail}))
            for x,detail in proposals:
                added=x not in actions;events.append({'step':step,**detail,'allocation':x,'added':added})
                if added:
                    eta=cfg['invasion_seed_mass'];actions.append(x);point=np.append((1-eta)*point,eta);win=np.append((1-eta)*win,eta);cumulative=np.append(cumulative,0)
            p2,w2=score_matrices(actions);mp=p2.astype(float)/2;mw=w2.astype(float)/2
        q=(1-win_share)*point+win_share*win;cumulative+=q
        if min(point.min(),win.min())<0 or abs(point.sum()-1)>1e-10 or abs(win.sum()-1)>1e-10:raise AssertionError('Type mass invariant')
        if step%cfg['record_every']==0 or step==target:
            mean=float(q@(mp@q));assert abs(mean-7.5)<1e-10
            trace.append({'step':step,'pool_size':len(actions),'official_population_mean':mean,'fixed_win_share':win_share})
        if step%cfg['invasion_every']==0 or step==target:
            state=simple_json({'step':step,'input_digest':digest,'actions':actions,'point':point.tolist(),'win':win.tolist(),
                              'cumulative':cumulative.tolist(),'trace':trace,'events':events,'rng_state':rng.bit_generator.state})
            state['state_sha256']=hashlib.sha256(json.dumps(state,sort_keys=True,separators=(',',':')).encode()).hexdigest();atomic_json(checkpoint,state)
    return {'actions':actions,'initial':initial,'point':point,'win':win,'final':(1-win_share)*point+win_share*win,
            'average':cumulative/target,'trace':trace,'events':simple_json(events),'completed':target,'input_digest':digest}
