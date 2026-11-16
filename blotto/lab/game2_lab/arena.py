"""Sampled-fitness four-player arenas, retained counters, and fixed-group oracles."""
from fractions import Fraction
from itertools import product
from pathlib import Path
import hashlib,json
import numpy as np
from blotto_lab.core import DEFAULT_GAME,Population,validate_allocation
from blotto_lab.merged import quantize
from blotto_lab.multiwave import atomic_json,simple_json
from .core import PREFIX4,_rows,group_scores12_4,shared_triplets,fixed_sample_bound


def class_name(x):
    return 'all_in_five'if x[4]==100 else('near_all_in_five'if x[4]>=80 else('low_five'if x[4]<=3 else'intermediate_five'))


def metric_arrays(candidates,triplets,candidate_batch=16,triplet_batch=8192):
    opponents=_rows(triplets,3,DEFAULT_GAME);actions=np.array([validate_allocation(x)for x in candidates],dtype=np.int64)
    if candidate_batch<=0 or triplet_batch<=0:raise ValueError('Positive chunk sizes required')
    shape=(len(actions),len(opponents));rank=np.empty(shape,dtype=np.uint8);castle=np.empty(shape,dtype=np.uint8);first=np.empty(shape,dtype=np.uint8)
    for start in range(0,len(opponents),triplet_batch):
        rows=opponents[start:start+triplet_batch];maximum=rows.max(axis=1);winners=rows==maximum[:,None,:];ties=winners.sum(axis=1)
        for begin in range(0,len(actions),candidate_batch):
            xs=actions[begin:begin+candidate_batch];own=np.zeros((len(xs),len(rows)),dtype=np.int64);enemy=np.zeros((len(xs),len(rows),3),dtype=np.int64)
            for i,value in enumerate(range(1,6)):
                below=xs[:,i,None]<maximum[None,:,i];equal=xs[:,i,None]==maximum[None,:,i]
                own+=value*((~(below|equal))*12+equal*(12//(ties[:,i]+1))[None,:])
                factor=value*(below*(12//ties[:,i])[None,:]+equal*(12//(ties[:,i]+1))[None,:]);enemy+=factor[:,:,None]*winners[None,:,:,i]
            higher=(enemy>own[:,:,None]).sum(axis=2);tied=1+(enemy==own[:,:,None]).sum(axis=2)
            sl=(slice(begin,begin+len(xs)),slice(start,start+len(rows)))
            rank[sl]=((PREFIX4[higher+tied]-PREFIX4[higher])//tied).astype(np.uint8);castle[sl]=own.astype(np.uint8)
            first[sl]=((higher==0)*(12//tied)).astype(np.uint8)
    return rank,castle,first


def single_triplet_response(opponents,game=DEFAULT_GAME):
    rows=np.asarray([validate_allocation(x,game)for x in opponents],dtype=np.int64)
    if len(rows)!=3:raise ValueError('Three opponents required')
    maxima=rows.max(axis=0);representatives=[];states=[]
    for state in product(range(3),repeat=game.castles):
        lo=[];hi=[]
        for m,s in zip(maxima,state):
            m=int(m);lo.append(0 if s==0 else(m if s==1 else m+1));hi.append(m-1 if s==0 else(m if s==1 else game.budget))
        if any(a>b for a,b in zip(lo,hi))or sum(lo)>game.budget or sum(hi)<game.budget:continue
        x=lo.copy();left=game.budget-sum(x)
        for i in range(game.castles-1,-1,-1):
            take=min(left,hi[i]-x[i]);x[i]+=take;left-=take
        assert not left;representatives.append(tuple(x));states.append(state)
    groups=np.concatenate((np.array(representatives)[:,None,:],np.broadcast_to(rows,(len(representatives),3,game.castles))),axis=1)
    castle,rank=group_scores12_4(groups,game);best=int(rank[:,0].max());winner=min(i for i in range(len(representatives))if int(rank[i,0])==best)
    # States are enumerated lexicographically, but interval representatives need their own tie order.
    winner=min((i for i in range(len(representatives))if int(rank[i,0])==best),key=lambda i:representatives[i])
    return {'allocation':representatives[winner],'payout':Fraction(best,4),'castle_points':Fraction(int(castle[winner,0]),12),
            'states_evaluated':len(representatives),'state':states[winner],
            'scope':'Exact best response to this fixed opponent triple only'}


def random_actions(rng,count):
    result=[]
    for j in range(count):
        if j%3==0:
            bars=np.sort(rng.choice(104,4,replace=False));x=tuple(map(int,np.diff(np.r_[-1,bars,104])-1))
        else:x=tuple(map(int,rng.multinomial(100,rng.dirichlet(np.ones(5)*[.25,1,5,20][j%4]))))
        result.append(x)
    return result


def mutate_action(parent,opponent,rng,threshold=False):
    x=list(validate_allocation(parent))
    if threshold:
        i=int(rng.integers(5));target=max(0,min(100,int(opponent[i])+int(rng.choice([-1,0,1]))));difference=target-x[i];x[i]=target
        other=[j for j in rng.permutation(5).tolist()if j!=i]
        if difference>0:
            for j in other:
                take=min(x[j],difference);x[j]-=take;difference-=take
            assert not difference
        else:x[other[0]]-=difference
    else:
        donor=int(rng.choice([i for i,v in enumerate(x)if v]));recipient=int(rng.integers(4));recipient+=recipient>=donor
        amount=int(rng.integers(1,min(40,x[donor])+1));x[donor]-=amount;x[recipient]+=amount
    return validate_allocation(x)


def as_population(actions,shares,total):
    return Population.from_counts([(x,m)for x,m in zip(actions,quantize(shares,total))if m])


def initial_shares(actions,kind,seed):
    rng=np.random.default_rng(seed)
    if kind=='sparse':return rng.dirichlet(np.full(len(actions),.08))
    if kind not in ['balanced','heavy_five']:raise ValueError('Unknown initial condition')
    groups=['all_in_five','near_all_in_five','low_five','intermediate_five'];masses=[.25]*4 if kind=='balanced'else[.45,.35,.1,.1]
    p=np.zeros(len(actions))
    for group,mass in zip(groups,masses):
        positions=[j for j,x in enumerate(actions)if class_name(x)==group]
        if not positions:raise ValueError('Initial pool misses a class')
        p[positions]=mass/len(positions)
    return p/p.sum()


def arena_path(folder,seed_actions,archive,initial,type_shares,rule,history_mass,mutation,cfg,seed,max_step=None):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True);actions=[validate_allocation(x)for x in seed_actions]
    archive=[validate_allocation(x)for x in archive];initial=np.asarray(initial,dtype=float).copy();shares=np.asarray(type_shares,dtype=float)
    if len(set(actions))!=len(actions)or len(initial)!=len(actions)or np.any(initial<0)or not np.isfinite(initial).all()or initial.sum()<=0:raise ValueError('Invalid initial pool')
    if shares.shape!=(3,)or np.any(shares<0)or abs(shares.sum()-1)>1e-12 or rule not in ['replicator','payoff_replacement']or not 0<=history_mass<=1 or not 0<=mutation<1:raise ValueError('Invalid path settings')
    if not 0<cfg['seed_mass']<1 or not 0<=cfg['settling_steps']<cfg['steps'] or min(cfg['fitness_triplets'],cfg['innovation_every'],cfg['mutation_proposals'],cfg['search_triplets'],cfg['gate_triplets'])<=0:raise ValueError('Invalid iteration or admission settings')
    initial/=initial.sum();settings={k:cfg[k]for k in ['steps','fitness_triplets','innovation_every','settling_steps','seed_mass','archive_proposals','mutation_proposals','random_proposals','search_triplets','gate_triplets','maximum_rank_gates','gate_alpha','record_every','quantization_mass']}
    signature={'actions':actions,'archive':archive,'initial':initial.tolist(),'type_shares':shares.tolist(),'rule':rule,'history_mass':history_mass,'mutation':mutation,'settings':settings,'seed':seed,
      'source_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest()for p in [Path(__file__),Path(__file__).parent/'core.py',Path(__file__).parents[1]/'blotto_lab/core.py',Path(__file__).parents[1]/'blotto_lab/merged.py']}}
    digest=hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest();checkpoint=folder/'checkpoint.json';rng=np.random.default_rng(seed)
    if checkpoint.exists():
        state=json.loads(checkpoint.read_text());stored=state.pop('state_sha256')
        if hashlib.sha256(json.dumps(state,sort_keys=True,separators=(',',':')).encode()).hexdigest()!=stored:raise ValueError('Checkpoint content changed')
        if state['input_digest']!=digest:raise ValueError('Checkpoint input/source changed')
        actions=[tuple(x)for x in state['actions']];types=np.array(state['types']);cumulative=np.array(state['cumulative']);tail=np.array(state['tail']);tail_count=state['tail_count']
        waves=state['waves'];trace=state['trace'];completed=state['step'];previous=np.array(state['previous_record']);fallbacks=state['fallbacks'];rng.bit_generator.state=state['rng_state']
    else:
        types=np.tile(initial,(3,1));cumulative=np.zeros(len(actions));tail=np.zeros(len(actions));tail_count=0;waves=[];trace=[];completed=0;previous=initial.copy();fallbacks=0
    target=cfg['steps']if max_step is None else max_step
    if not completed<=target<=cfg['steps']or target<=0:raise ValueError('Invalid target')
    if types.shape!=(3,len(actions))or np.any(types<0)or not np.allclose(types.sum(axis=1),1,atol=1e-12)or abs(cumulative.sum()-completed)>1e-8:raise ValueError('Invalid checkpoint masses')
    last_admission=cfg['steps']-cfg['settling_steps']
    for step in range(completed+1,target+1):
        q=shares@types;history=cumulative/completed if completed else q
        # completed advances each loop, so history always uses the actual preceding trajectory.
        opponent=(1-history_mass)*q+history_mass*history;opponent=np.maximum(opponent,0);opponent/=opponent.sum()
        ids=rng.choice(len(actions),size=(cfg['fitness_triplets'],3),p=opponent);triples=np.asarray(actions,dtype=np.int64)[ids]
        raw=metric_arrays(actions,triples);fitness=np.stack([raw[j].mean(axis=1)/(4 if j==0 else 12)for j in range(3)])
        for j in range(3):
            if not shares[j]:continue
            proposal=types[j]*fitness[j]if rule=='replicator'else fitness[j].copy()
            if proposal.sum()<=0:fallbacks+=1;proposal=types[j].copy()
            proposal/=proposal.sum();types[j]=(1-mutation)*proposal+mutation/len(actions);types[j]/=types[j].sum()
        if step%cfg['innovation_every']==0 and step<=last_admission:
            q=shares@types;past=(cumulative+q)/step;populations={'current':as_population(actions,q,cfg['quantization_mass']),'history':as_population(actions,past,cfg['quantization_mass'])}
            most=actions[int(q.argmax())];cloned=single_triplet_response([most]*3)
            oracle_records=[{'opponents':[most]*3,'response':cloned}]
            for _ in range(2):
                group=[actions[int(i)]for i in rng.choice(len(actions),3,p=q)];oracle_records.append({'opponents':group,'response':single_triplet_response(group)})
            mutations=[]
            for j in range(cfg['mutation_proposals']):
                parent=actions[int(rng.choice(len(actions),p=q))];other=actions[int(rng.choice(len(actions),p=q))]
                mutations.append(mutate_action(parent,other,rng,j%2==0))
            chosen_archive=[archive[int(i)]for i in rng.choice(len(archive),min(cfg['archive_proposals'],len(archive)),replace=False)]
            field=list(dict.fromkeys(actions+chosen_archive+mutations+random_actions(rng,cfg['random_proposals'])+[v['response']['allocation']for v in oracle_records]))
            wave={'step':step,'field':field,'fixed_triplet_oracles':oracle_records,'targets':{},'admissions':[]};proposals=[]
            for label,pop in populations.items():
                search_seed=int(rng.integers(2**32));triplets=shared_triplets(pop,cfg['search_triplets'],search_seed)['triplets'];matrices=metric_arrays(field,triplets)
                sums=[v.sum(axis=1,dtype=np.int64)for v in matrices]
                best=lambda j,allowed:min(allowed,key=lambda i:(-int(sums[j][i]),field[i]))
                rank_winner=best(0,range(len(field)));incumbent=best(0,range(len(actions)))
                gate_seed=int(rng.integers(2**32));gate=shared_triplets(pop,cfg['gate_triplets'],gate_seed)['triplets']
                outcomes=metric_arrays([field[rank_winner],field[incumbent]],gate)[0].astype(np.int64)
                gain=Fraction(int((outcomes[0]-outcomes[1]).sum()),4*len(gate));bound=fixed_sample_bound(len(gate),cfg['maximum_rank_gates'],cfg['gate_alpha'],paired=True)
                wave['targets'][label]={'population':pop.to_dict(),'search_seed':search_seed,'fitness_sums':[v.tolist()for v in sums],
                    'rank_winner':field[rank_winner],'incumbent':field[incumbent],'gate_seed':gate_seed,'gate_mean_gain':gain,'gate_bound':bound,'gate_certified_improvement':float(gain)>bound}
                proposals.append((field[rank_winner],'rank_'+label))
                if label=='current':
                    for j,source in [(1,'castle_current'),(2,'first_current')]:
                        if shares[j]:proposals.append((field[best(j,range(len(field)))],source))
            proposals.extend([(cloned['allocation'],'clone_counter'),(mutations[0],'large_or_threshold_mutation')])
            merged={}
            for x,source in proposals:merged.setdefault(tuple(x),[]).append(source)
            for x,sources in merged.items():
                new=x not in actions
                if new:
                    actions.append(x);types=np.pad(types,((0,0),(0,1)));cumulative=np.append(cumulative,0);tail=np.append(tail,0);previous=np.append(previous,0)
                index=actions.index(x);eta=cfg['seed_mass'];types*=(1-eta);types[:,index]+=eta
                wave['admissions'].append({'allocation':x,'sources':sources,'is_new':new,'index':index})
            waves.append(wave)
        q=shares@types;cumulative+=q
        if step>last_admission:tail+=q;tail_count+=1
        if min(types.min(),q.min())<0 or not np.allclose(types.sum(axis=1),1,atol=1e-12):raise AssertionError('Type mass invariant failed')
        if step%cfg['record_every']==0 or step==cfg['steps']:
            groups=np.asarray(actions)[rng.choice(len(actions),size=(64,4),p=q)];_,gp=group_scores12_4(groups);assert np.all(gp.sum(axis=1)==60)
            class_shares={name:float(sum(q[i]for i,x in enumerate(actions)if class_name(x)==name))for name in ['all_in_five','near_all_in_five','low_five','intermediate_five']}
            active=q[q>0];trace.append({'step':step,'pool_size':len(actions),'class_shares':class_shares,'top_allocation':actions[int(q.argmax())],
                  'top_share':float(q.max()),'effective_strategies':float(np.exp(-sum(active*np.log(active)))),'l1_from_previous_record':float(np.abs(q-previous).sum()),
                  'selfplay_group_mean':float(gp.mean()/4),'type_shares':shares.tolist()});previous=q.copy()
        completed=step
        if step%cfg['innovation_every']==0 or step==target:
            state=simple_json({'step':step,'input_digest':digest,'actions':actions,'types':types.tolist(),'cumulative':cumulative.tolist(),'tail':tail.tolist(),
                'tail_count':tail_count,'waves':waves,'trace':trace,'previous_record':previous.tolist(),'fallbacks':fallbacks,'rng_state':rng.bit_generator.state})
            state['state_sha256']=hashlib.sha256(json.dumps(state,sort_keys=True,separators=(',',':')).encode()).hexdigest();atomic_json(checkpoint,state)
    return {'actions':actions,'types':types,'final':shares@types,'average':cumulative/target,'tail_average':tail/tail_count if tail_count else shares@types,
            'trace':simple_json(trace),'waves':simple_json(waves),'input_digest':digest,'completed':completed,'fallbacks':fallbacks,'tail_count':tail_count}


def static_triplets(family,variant,size,seed,cloud):
    rng=np.random.default_rng(seed);n=size*3;values=np.arange(1,6,dtype=float)
    if family=='rotated':
        choices=[(2,4),(3,4),(0,1,4),(0,2,3),(1,2,3)];labels=rng.integers(len(choices),size=n);rows=np.zeros((n,5),dtype=np.int64)
        for j,c in enumerate(choices):
            ix=np.flatnonzero(labels==j);profile=values*np.array([5 if i in c else .4 for i in range(5)]);profile/=profile.sum()
            rows[ix]=rng.multinomial(100,rng.dirichlet((3 if variant else 12)*profile,size=len(ix)))
    elif family=='wide_abandon':
        fifth=rng.choice([0,1,2,3,5,8,13],size=n);p=rng.dirichlet(np.array([1,2,1,2])*(.5 if variant else 2),size=n)
        rows=np.column_stack((rng.multinomial(100-fifth,p),fifth))
    elif family=='middle_five':
        fifth=rng.integers(0,81,size=n)if variant else rng.choice([10,20,30,40,50,60,70],size=n)
        p=rng.dirichlet(np.ones(4)*(.5 if variant else 4),size=n);rows=np.column_stack((rng.multinomial(100-fifth,p),fifth))
    elif family=='counter_cloud':
        rows=[]
        for i in rng.integers(len(cloud),size=n):
            x=cloud[int(i)];x=mutate_action(x,cloud[int(rng.integers(len(cloud)))],rng,threshold=True)
            if variant:x=mutate_action(x,x,rng,threshold=False)
            rows.append(x)
        rows=np.asarray(rows,dtype=np.int64)
    else:raise ValueError('Unknown static challenge')
    assert np.all(rows>=0)and np.all(rows.sum(axis=1)==100)
    return rows.reshape(size,3,5)
