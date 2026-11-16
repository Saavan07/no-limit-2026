"""Frozen challenge generators and prior-aware adapting four-player arenas."""
from fractions import Fraction
from pathlib import Path
from itertools import product
import hashlib,json
import numpy as np
from blotto_lab.core import validate_allocation
from blotto_lab.multiwave import atomic_json,simple_json
from .core import shared_triplets,fixed_sample_bound,group_scores12_4
from .human import payoff_arrays
from .arena import (class_name,initial_shares,random_actions,mutate_action,as_population,single_triplet_response)
from .weight_counters import worst_clone_population

A=(0,0,0,0,100)


def prior_weights(actions,kind):
    if kind=='fifth':return initial_shares(actions,'balanced',0)
    if kind!='shape':raise ValueError('Unknown prior')
    groups=[]
    for x in actions:groups.append('fort'if max(x)==100 else('low'if x[4]<=3 else('middle'if x[4]<80 else'near')))
    result=np.zeros(len(actions));masses={'fort':.2,'low':.3,'middle':.3,'near':.2}
    for group,mass in masses.items():
        ix=[i for i,g in enumerate(groups)if g==group]
        if not ix:raise ValueError('Prior class missing')
        result[ix]=mass/len(ix)
    return result/result.sum()


def residual_rows(style,size,rng,mixture,cloud):
    if style not in ['coalition','rounded','edge','counter_aware']:raise ValueError('Unknown challenge style')
    probs=np.asarray(mixture,dtype=float);probs/=probs.sum();types=rng.choice(3,size=size,p=probs);rows=np.empty((size,5),dtype=np.int64)
    profiles=np.array([[.4,.7,5,7],[1,6,.3,8],[3,.4,6,7],[1,5,8,.3],[1,1,1,12],[1,1,12,1]],dtype=float)
    for family in range(3):
        ix=np.flatnonzero(types==family);n=len(ix)
        if not n:continue
        fifth=rng.choice([0,0,0,1,2,4,5],size=n)if family==0 else(rng.integers(6,80,size=n)if family==1 else rng.choice([80,84,88,91,94,96,97,98,99],size=n))
        profile=profiles[rng.integers(len(profiles),size=n)];profile/=profile.sum(axis=1,keepdims=True)
        concentration={'coalition':5,'rounded':12,'edge':.22,'counter_aware':2}[style]
        gamma=rng.gamma(profile*concentration);p=gamma/gamma.sum(axis=1,keepdims=True)
        if style=='rounded':
            fifth=5*(fifth//5);remaining=100-fifth;units=rng.multinomial(remaining//5,p);bids=units*5
        else:bids=rng.multinomial(100-fifth,p)
        sample=np.column_stack((bids,fifth))
        if style=='coalition':
            # Deliberately include exact fort-4/fort-3 crowding, rather than a soft upper cap.
            atoms=(rng.random(n)<.18)&(family==0);forts=rng.choice([2,3],size=int(atoms.sum()));sample[atoms]=0
            sample[np.flatnonzero(atoms),forts]=100
        if style=='counter_aware':
            use=rng.random(n)<.45
            lo,hi=[(0,5),(6,79),(80,99)][family];eligible=[x for x in cloud if lo<=x[4]<=hi]
            for j in np.flatnonzero(use):
                if not eligible:continue
                source=eligible[int(rng.integers(len(eligible)))];opponent=eligible[int(rng.integers(len(eligible)))];mutated=mutate_action(source,opponent,rng,True)
                sample[j]=mutated if lo<=mutated[4]<=hi else source
        # The residual law is conditioned on not A; preserve budget in the rare A event.
        copied=sample[:,4]==100;sample[copied,4]-=1;sample[copied,0]+=1;rows[ix]=sample
    assert np.all(rows>=0)and np.all(rows.sum(axis=1)==100)and np.all(rows[:,4]<100)
    return rows


def core_triplets(definition,size,seed,cloud):
    rng=np.random.default_rng(seed);k=definition['copies'];rows=np.empty((size,3,5),dtype=np.int64);rows[:,:k]=A
    if k<3:rows[:,k:]=residual_rows(definition['style'],size*(3-k),rng,definition['mixture'],cloud).reshape(size,3-k,5)
    return rows


def pressure_rows(family,size,rng,cloud):
    if family=='threshold_four':
        fourth=rng.integers(66,100,size=size);p=rng.dirichlet([1.2,1.8,.6,1.4],size=size);others=rng.multinomial(100-fourth,p)
        return np.column_stack((others[:,:3],fourth,others[:,3]))
    if family=='crowded_four':
        rows=np.zeros((size,5),dtype=np.int64);fort=rng.random(size)<.5;rows[fort,3]=100
        n=int((~fort).sum());fourth=rng.choice([75,80,85,90,95,98,99],size=n);other=rng.multinomial(100-fourth,rng.dirichlet([.7,.7,.7,1.4],size=n));rows[~fort]=np.column_stack((other[:,:3],fourth,other[:,3]));return rows
    if family=='cheap_overbid':
        rows=[]
        for i in rng.integers(len(cloud),size=size):
            x=cloud[int(i)];other=cloud[int(rng.integers(len(cloud)))];rows.append(mutate_action(x,other,rng,True))
        return np.asarray(rows,dtype=np.int64).reshape(-1,5)
    if family=='shifted_near':
        fifth=rng.choice([90,93,95,97,98,99,100],size=size);p=rng.dirichlet([.5,1.5,.7,1.1],size=size);return np.column_stack((rng.multinomial(100-fifth,p),fifth))
    raise ValueError('Unknown pressure family')


def pressure_triplets(definition,size,seed,cloud):
    rng=np.random.default_rng(seed);n=3*size;rows=residual_rows('coalition',n,rng,[.4,.4,.2],cloud)
    # A fixed 10% exact-copy background is one declared stress convention.
    background_copy=rng.random(n)<.1;rows[background_copy]=A;use=rng.random(n)<definition['mass'];rows[use]=pressure_rows(definition['family'],int(use.sum()),rng,cloud)
    assert np.all(rows>=0)and np.all(rows.sum(axis=1)==100)
    return rows.reshape(size,3,5)


def final_arena(folder,seed_actions,archive,cfg,rule,prior_kind,initial_kind,seed,max_step=None):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True);actions=[validate_allocation(x)for x in seed_actions];archive=[validate_allocation(x)for x in archive]
    if len(actions)!=len(set(actions))or rule not in ['replicator','prior_replacement']:raise ValueError('Invalid arena pool or rule')
    signature={'actions':actions,'archive':archive,'configuration':cfg,'rule':rule,'prior':prior_kind,'initial':initial_kind,'seed':seed,
        'sources':{p.name:hashlib.sha256(p.read_bytes()).hexdigest()for p in [Path(__file__),Path(__file__).parent/'core.py',Path(__file__).parent/'human.py',Path(__file__).parent/'arena.py',Path(__file__).parent/'weight_counters.py']}}
    digest=hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest();rng=np.random.default_rng(seed);cp=folder/'checkpoint.json'
    if cp.exists():
        state=json.loads(cp.read_text());stored=state.pop('state_sha256')
        if hashlib.sha256(json.dumps(state,sort_keys=True,separators=(',',':')).encode()).hexdigest()!=stored:raise ValueError('Checkpoint content mismatch')
        if state['input_digest']!=digest:raise ValueError('Checkpoint input/source mismatch')
        actions=[tuple(x)for x in state['actions']];q=np.array(state['q']);cumulative=np.array(state['cumulative']);tail=np.array(state['tail']);previous=np.array(state['previous']);completed=state['step'];tail_count=state['tail_count'];trace=state['trace'];waves=state['waves'];rng.bit_generator.state=state['rng_state']
    else:
        q=initial_shares(actions,initial_kind,0);cumulative=np.zeros(len(actions));tail=np.zeros(len(actions));previous=q.copy();completed=tail_count=0;trace=[];waves=[]
    target=cfg['steps']if max_step is None else max_step
    if not completed<=target<=cfg['steps']or target<=0:raise ValueError('Invalid target step')
    for step in range(completed+1,target+1):
        past=cumulative/completed if completed else q;opponent=(1-cfg['history_mass'])*q+cfg['history_mass']*past;opponent/=opponent.sum()
        triplets=np.asarray(actions)[rng.choice(len(actions),size=(cfg['fitness_triplets'],3),p=opponent)];rank=payoff_arrays(actions,triplets)[0];fitness=rank.mean(axis=1)/4;prior=prior_weights(actions,prior_kind)
        proposal=(q if rule=='replicator'else prior)*fitness
        if proposal.sum()<=0:proposal=q.copy()
        proposal/=proposal.sum();q=(1-cfg['mutation'])*proposal+cfg['mutation']*prior;q/=q.sum()
        if step%cfg['innovation_every']==0 and step<=cfg['steps']-cfg['settling_steps']:
            populations={'current':as_population(actions,q,cfg['quantization_mass']),'history':as_population(actions,(cumulative+q)/step,cfg['quantization_mass'])}
            most=actions[int(q.argmax())];clone=single_triplet_response([most]*3);inverse=worst_clone_population(most)
            sampled=[actions[int(i)]for i in rng.choice(len(actions),size=3,p=q)];group_response=single_triplet_response(sampled)
            mutations=[mutate_action(actions[int(rng.choice(len(actions),p=q))],actions[int(rng.choice(len(actions),p=q))],rng,j%2==0)for j in range(cfg['mutation_proposals'])]
            selected=[archive[int(i)]for i in rng.choice(len(archive),min(cfg['archive_proposals'],len(archive)),replace=False)]
            field=list(dict.fromkeys(actions+selected+mutations+random_actions(rng,cfg['random_proposals'])+[clone['allocation'],inverse['opponent'],group_response['allocation']]))
            wave={'step':step,'field':field,'clone_target':most,'clone_response':clone,'inverse_response':inverse,'sampled_opponents':sampled,'group_response':group_response,'targets':{},'admissions':[]};proposals=[]
            for label,pop in populations.items():
                search_seed=int(rng.integers(2**32));rows=shared_triplets(pop,cfg['search_triplets'],search_seed)['triplets'];sums=payoff_arrays(field,rows)[0].sum(axis=1,dtype=np.int64);best=lambda allowed:min(allowed,key=lambda i:(-int(sums[i]),field[i]))
                winner=best(range(len(field)));incumbent=best(range(len(actions)));gate_seed=int(rng.integers(2**32));gate=shared_triplets(pop,cfg['gate_triplets'],gate_seed)['triplets'];scores=payoff_arrays([field[winner],field[incumbent]],gate)[0].astype(np.int64)
                gain=Fraction(int((scores[0]-scores[1]).sum()),4*len(gate));bound=fixed_sample_bound(len(gate),cfg['maximum_rank_gates'],cfg['gate_alpha'],paired=True)
                wave['targets'][label]={'population':pop.to_dict(),'search_seed':search_seed,'fitness_sums':sums.tolist(),'winner':field[winner],'incumbent':field[incumbent],'gate_seed':gate_seed,'gate_mean_gain':gain,'gate_bound':bound,'gate_certified_improvement':float(gain)>bound}
                proposals.append((field[winner],'rank_'+label))
            proposals.extend([(clone['allocation'],'clone_response'),(inverse['opponent'],'inverse_population_response'),(group_response['allocation'],'sampled_group_response'),(mutations[0],'mutation')]);merged={}
            for x,source in proposals:merged.setdefault(tuple(x),[]).append(source)
            for x,sources in merged.items():
                new=x not in actions
                if new:actions.append(x);q=np.append(q,0);cumulative=np.append(cumulative,0);tail=np.append(tail,0);previous=np.append(previous,0)
                i=actions.index(x);q*=1-cfg['seed_mass'];q[i]+=cfg['seed_mass'];wave['admissions'].append({'allocation':x,'is_new':new,'index':i,'sources':sources})
            waves.append(wave)
        cumulative+=q
        if step>cfg['steps']-cfg['settling_steps']:tail+=q;tail_count+=1
        assert np.all(q>=0)and abs(q.sum()-1)<1e-12
        if step%cfg['record_every']==0 or step==cfg['steps']:
            groups=np.asarray(actions)[rng.choice(len(actions),size=(32,4),p=q)];_,payout=group_scores12_4(groups);assert np.all(payout.sum(axis=1)==60)
            trace.append({'step':step,'pool_size':len(actions),'top_allocation':actions[int(q.argmax())],'top_share':float(q.max()),'l1_from_previous':float(abs(q-previous).sum()),
                'class_shares':{k:float(sum(q[i]for i,x in enumerate(actions)if class_name(x)==k))for k in ['all_in_five','near_all_in_five','low_five','intermediate_five']},'selfplay_group_mean':float(payout.mean()/4)});previous=q.copy()
        completed=step
        if step%cfg['innovation_every']==0 or step==target:
            state=simple_json({'actions':actions,'q':q.tolist(),'cumulative':cumulative.tolist(),'tail':tail.tolist(),'previous':previous.tolist(),'step':completed,'tail_count':tail_count,'trace':trace,'waves':waves,'rng_state':rng.bit_generator.state,'input_digest':digest})
            state['state_sha256']=hashlib.sha256(json.dumps(state,sort_keys=True,separators=(',',':')).encode()).hexdigest();atomic_json(cp,state)
    return {'actions':actions,'final':q,'average':cumulative/target,'tail':tail/tail_count if tail_count else q,'tail_count':tail_count,'trace':simple_json(trace),'waves':simple_json(waves),'input_digest':digest}


def evaluation_weights(design,cfg):
    from math import comb
    refs=design['core_references'];weights={'core':{},'pressure':{},'arenas':{}}
    for ref,ids in refs.items():
        for p in map(lambda x:Fraction(str(x)),cfg['copy_points']):
            for k,key in enumerate(ids):weights['core'][key]=weights['core'].get(key,Fraction(0))+Fraction(comb(3,k),len(refs)*len(cfg['copy_points']))*p**k*(1-p)**(3-k)
    for key in design['pressure']:weights['pressure'][key]=Fraction(1,len(design['pressure']))
    for path in design['arena_paths']:
        for snapshot in ['average','tail']:weights['arenas'][path['name']+'__'+snapshot]=Fraction(1,2*len(design['arena_paths']))
    assert all(sum(w.values())==1 for w in weights.values())
    return weights


def selection_policy(sums,cell_keys,candidates,design,cfg):
    from math import comb
    cids=list(candidates);weights=evaluation_weights(design,cfg);scores={cid:{key:Fraction(int(sums[i,j]),4*cfg['evaluation_triplets'])for j,key in enumerate(cell_keys)}for i,cid in enumerate(cids)}
    blocks={cid:{group:sum(w*scores[cid][key]for key,w in rows.items())for group,rows in weights.items()}for cid in cids}
    maxima={group:max(v[group]for v in blocks.values())for group in weights};regrets={cid:{g:maxima[g]-v[g]for g in weights}for cid,v in blocks.items()};risk={cid:max(v.values())for cid,v in regrets.items()};mean={cid:sum(v.values())/len(v)for cid,v in blocks.items()}
    ranked=sorted(cids,key=lambda cid:(risk[cid],-mean[cid],candidates[cid]['allocation']));models={cid:{}for cid in cids};meta={}
    for ref,ids in design['core_references'].items():
        for p0 in cfg['copy_points']:
            p=Fraction(str(p0));key=ref+'__p'+str(p0);meta[key]={'block':'core','reference':ref,'copy_share':p}
            for cid in cids:models[cid][key]=sum(Fraction(comb(3,k))*p**k*(1-p)**(3-k)*scores[cid][cell]for k,cell in enumerate(ids))
    for key in design['pressure']:
        meta[key]={'block':'pressure'}
        for cid in cids:models[cid][key]=scores[cid][key]
    for path in design['arena_paths']:
        for snapshot in ['average','tail']:
            key=path['name']+'__'+snapshot;meta[key]={'block':'arenas'}
            for cid in cids:models[cid][key]=scores[cid][key]
    minimum={cid:min(v.values())for cid,v in models.items()}
    def choose(value):return min(cids,key=lambda cid:(-value[cid],-mean[cid],candidates[cid]['allocation']))
    low={cid:sum(v for k,v in values.items()if meta[k]['block']=='core'and meta[k]['copy_share']<=Fraction(1,10))/(2*len(design['core_references']))for cid,values in models.items()}
    return {'selected_candidate_id':ranked[0],'ranking':ranked,'block_weights':weights,'block_scores':blocks,'block_benchmarks':maxima,'block_regrets':regrets,'maximum_block_regret':risk,'equal_block_mean':mean,
        'model_scores':models,'model_definitions':meta,'lowest_model_score':minimum,'low_copy_core_mean':low,
        'alternatives':{'equal_block_mean':choose(mean),'lowest_model_score':choose(minimum),'low_copy_core_mean':choose(low)}}
