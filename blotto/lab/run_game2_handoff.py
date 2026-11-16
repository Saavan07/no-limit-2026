"""Audit Parry's supplied claims, public compatibility, and expanded target pool."""
from fractions import Fraction
from itertools import combinations,product
from pathlib import Path
import argparse,json,time,shutil,math,platform
import numpy as np
from blotto_lab.core import Population,validate_allocation
from game2_lab.core import reference_group,exact_evaluation,shared_triplets
from game2_lab.human import payoff_arrays
from game2_lab.arena import random_actions
from game2_lab.final import core_triplets
from game2_lab.handoff import exact_critical_response,target_population,cdf_diagnostic
from run_population_study import ROOT,save,sha,write_csv


def read(p):return json.loads(p.read_text())


NAMED={'our_selected':(15,18,49,12,6),'parry_selected':(0,36,0,64,0),'parry_34':(0,0,39,61,0),'parry_scout':(0,3,35,57,5),'our_lowest_model':(20,24,44,5,7),'all_in_five':(0,0,0,0,100),'all_in_four':(0,0,0,100,0),'parry_mix_235':(0,2,35,0,63),'parry_mix_245':(0,31,0,63,6),'parry_mix_24':(0,34,0,66,0),'parry_mix_15':(26,0,0,2,72),'parry_earlier':(15,4,6,6,69),'parry_checkpoint_response':(14,18,27,16,25)}


def expanded_pool(old,cfg):
    roles={}
    def add(x,role):
        x=validate_allocation(x);roles.setdefault(x,[])
        if role not in roles[x]:roles[x].append(role)
    for v in old['candidates'].values():add(v['allocation'],'previous_frozen')
    for name,x in NAMED.items():add(x,name)
    for split in range(101):add((0,split,0,100-split,0),'all_24_splits');add((0,0,split,100-split,0),'all_34_splits')
    for name in ['our_selected','our_lowest_model']:
        x=NAMED[name]
        for i,j in product(range(5),repeat=2):
            if i==j:continue
            for amount in [1,2]:
                if x[i]>=amount:
                    y=list(x);y[i]-=amount;y[j]+=amount;add(y,'local_'+name)
    for x in random_actions(np.random.default_rng(cfg['seed']),cfg['unrestricted_controls']):add(x,'unrestricted_control')
    for name,response in theory_audit()['exact_small_support_responses'].items():add(response['allocation'],'exact_response_'+name)
    return {f'h{i:04d}':{'allocation':x,'roles':r}for i,(x,r)in enumerate(roles.items())}


def theory_audit():
    P=NAMED['parry_selected'];B=NAMED['parry_34'];S=NAMED['parry_scout'];equal=(20,20,20,20,20);v521=(13,25,0,0,62);v431=(13,0,37,50,0);cases=[]
    for x,expected in [(P,9),(B,Fraction(13,2)),(S,4)]:
        outcome=reference_group([x,equal,v521,v431]);assert outcome.payouts[0]==expected;cases.append({'group':[x,equal,v521,v431],'outcome':outcome,'claim':'Defined heterogeneous example'})
    counters=[((20,20,20,20,20),[9,9,9]),((5,5,5,80,5),[0,0,9]),((1,32,1,65,1),[0,0,9]),((0,0,36,58,6),[9,9,0])]
    for y,expected in counters:
        for x,value in zip([P,B,S],expected):
            outcome=reference_group([x,y,y,y]);assert outcome.payouts[0]==value;cases.append({'group':[x,y,y,y],'outcome':outcome,'claim':'Explicit homogeneous counter table'})
    q1=Population.from_people([(40,10,40,10,0),(10,40,10,40,0)]);q2=Population.from_people([(10,40,40,10,0),(40,10,10,40,0)]);x=(0,0,100,0,0)
    a=exact_evaluation(x,q1);b=exact_evaluation(x,q2);assert a.expected_payout==Fraction(45,8)and b.expected_payout==Fraction(57,8)and a.expected_castle_points==b.expected_castle_points==Fraction(17,4)
    y=(0,0,0,50,50);ty=tuple(4*v//5+4 for v in y);z=(30,30,0,20,20);assert reference_group([z,y,y,y]).payouts[0]==Fraction(15,4)and reference_group([z,ty,ty,ty]).payouts[0]==0
    rng=np.random.default_rng(2026111500);original=rng.multinomial(20,np.ones(5)/5,size=(200,4))*5;preserved=0
    for group in original:
        transformed=4*group//5+4;assert reference_group(group.tolist())==reference_group(transformed.tolist());preserved+=1
    A=(0,0,0,0,100);populations={'two_candidates':Population.from_people([P,NAMED['our_selected']]),'candidates_and_allin':Population.from_people([P,NAMED['our_selected'],A]),'named_heterogeneous':Population.from_people([equal,v521,v431])}
    responses={name:exact_critical_response(pop,100000)for name,pop in populations.items()}
    atom=(35-math.sqrt(1173))/26;bound=lambda p:4-9*p+12*p*p-Fraction(13,4)*p**3
    assert bound(Fraction(2888,100000))>Fraction(15,4)and bound(Fraction(2889,100000))<Fraction(15,4)
    return {'exact_group_cases':cases,'identical_marginal_payoffs':[a,b],'affine_witness':{'entrant':z,'opponent':y,'transformed_opponent':ty,'before':Fraction(15,4),'after':0},'affine_random_group_invariance_checks':preserved,
            'exact_small_support_responses':responses,'necessary_symmetric_equilibrium_allin_mass_lower_bound':atom,'atom_formula':'(35 - sqrt(1173))/26','iid_collision_payout_bound_at_1649':Fraction(9*(3*1649-2),1649**2),
            'scope':'Independently proved/checked mathematical claims; not verification of missing saved searches or equilibrium checkpoints'}


def core_population_rows(style,p,size,seed,cloud):
    rng=np.random.default_rng(seed);n=(size+2)//3;definition={'copies':0,'style':style,'mixture':[1/3]*3};rows=core_triplets(definition,n,int(rng.integers(2**32)),cloud).reshape(-1,5)[:size].copy();rows[rng.random(size)<p]=[0,0,0,0,100];return rows


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,default=ROOT/'config/game2-handoff.json');parser.add_argument('--output',type=Path,default=ROOT/'results/game2-handoff-integration');parser.add_argument('--resume',action='store_true');args=parser.parse_args();cfg=read(args.config);out=args.output.resolve();clock=time.perf_counter()
    sources=[Path(__file__),ROOT/'verify_game2_handoff.py',ROOT/'GAME2-HANDOFF-PROTOCOL.md',ROOT/'game2_lab/handoff.py',ROOT/'game2_lab/core.py',ROOT/'game2_lab/human.py',ROOT/'game2_lab/final.py',ROOT/'blotto_lab/core.py',ROOT/'tests/test_game2_handoff.py']
    inputs=[ROOT/'data/parry-handoff/handoff.txt',ROOT/'data/parry-handoff/public-scores.json',ROOT/'results/game2-final/results.json',ROOT/'results/game2-selected-v1.json',ROOT/'results/game2-final-checkpoint-v1.json']
    sig={'configuration':cfg,'source_hashes':{str(p.relative_to(ROOT)):sha(p)for p in sources},'input_hashes':{str(p.relative_to(ROOT)):sha(p)for p in inputs},'environment':{'python':platform.python_version(),'numpy':np.__version__}}
    if out.exists():
        if not args.resume:raise SystemExit('Output exists; use another folder or --resume.')
        if read(out/'run-inputs.json')!=sig:raise SystemExit('Source/input mismatch')
        if (out/'manifest.json').exists():print('Completed. Verify the audit.');return
    else:
        out.mkdir(parents=True)
        for folder in ['source-snapshot','targets','cdf']:(out/folder).mkdir()
        save(out/'run-inputs.json',sig)
        for p in sources:
            target=out/'source-snapshot'/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
    old=read(inputs[2]);public=read(inputs[1]);observed=np.array(public['scores']);pool=expanded_pool(old,cfg);actions=[v['allocation']for v in pool.values()];cids=list(pool);lookup={tuple(v['allocation']):cid for cid,v in pool.items()};named={name:lookup[x]for name,x in NAMED.items()}
    scenarios={}
    for k in [2,3]:
        for active in combinations(range(5),k):
            for scout,mode in product(cfg['scouts'],cfg['target_weightings']):
                key='target'+''.join(str(i+1)for i in active)+'__s'+str(scout)+'__'+mode;scenarios[key]={'active':active,'scout':scout,'weighting':mode,'seed':cfg['seed']+1000+len(scenarios)}
    if not(out/'pool-freeze.json').exists():save(out/'pool-freeze.json',{'created_unix':time.time(),'pool':pool,'named':named,'scenarios':scenarios,'public_scores_sha256':sha(inputs[1]),'scope':'New development pool and structural proxy stresses; no amended final selection'})
    freeze_hash=sha(out/'pool-freeze.json');results=[];target_meta={}
    for key,d in scenarios.items():
        path=out/'targets'/(key+'.npz');pop=target_population(d['active'],d['scout'],d['weighting'],cfg['kernel_kappa'],cfg['kernel_scale'])
        if path.exists():
            with np.load(path)as a:sums=a['rank_sum4'];castle=a['castle_sum12']
        else:
            rows=shared_triplets(pop,cfg['target_triplets'],d['seed'])['triplets'];rank,points=payoff_arrays(actions,rows);sums=rank.sum(axis=1,dtype=np.int64);castle=points.sum(axis=1,dtype=np.int64);np.savez_compressed(path,triplets=rows.astype(np.uint8),rank_sum4=sums,castle_sum12=castle,named_rank4=rank[[cids.index(named[name])for name in NAMED]])
        results.append(sums);target_meta[key]={**d,'support_size':len(pop.entries),'total_mass':pop.total_mass};print('Target coverage: '+key,flush=True)
    target=np.column_stack(results);np.savez_compressed(out/'target-scores.npz',rank_sum4=target);target_scores={cid:{key:Fraction(int(target[i,j]),4*cfg['target_triplets'])for j,key in enumerate(scenarios)}for i,cid in enumerate(cids)}
    cdf={};cloud=read(ROOT/'results/game2-final/candidate-freeze.json')['counter_cloud'];definitions=[]
    for name,p in old['paths'].items():
        for snapshot in ['average','tail']:definitions.append({'key':name+'__'+snapshot,'kind':'arena','population':p[snapshot]['population'],'seed':cfg['seed']+100000+len(definitions)})
    for style,p in product(['coalition','rounded','edge','counter_aware'],[0,.1,.25,.5]):definitions.append({'key':'core__'+style+'__p'+str(p),'kind':'core','style':style,'copy_share':p,'seed':cfg['seed']+100000+len(definitions)})
    for d in definitions:
        key=d['key'];path=out/'cdf'/(key+'.npz')
        if path.exists():
            with np.load(path)as a:own=a['own_actions'];weights=a['own_weights'];sums=a['rank_sum4'];size=int(a['size'])
        else:
            if d['kind']=='arena':
                pop=Population.from_dict(d['population']);own=np.array([x for x,n in pop.entries]);weights=np.array([n for x,n in pop.entries],dtype=np.int64);size=cfg['arena_cdf_triplets'];rows=shared_triplets(pop,size,d['seed'])['triplets']
            else:
                ownrows=core_population_rows(d['style'],d['copy_share'],cfg['core_cdf_own'],d['seed'],cloud);own,weights=np.unique(ownrows,axis=0,return_counts=True);size=cfg['core_cdf_triplets'];rows=core_population_rows(d['style'],d['copy_share'],3*size,d['seed']+1000000,cloud).reshape(size,3,5)
            sums=payoff_arrays(own.tolist(),rows)[0].sum(axis=1,dtype=np.int64);np.savez_compressed(path,own_actions=own.astype(np.uint8),own_weights=weights,rank_sum4=sums,size=np.array(size),triplets=rows.astype(np.uint8))
        metrics=cdf_diagnostic(sums/(4*size),weights,observed);threshold=cfg['cdf_thresholds'];metrics['passes_nominal_screen']=metrics['maximum_cdf_gap']<=threshold['maximum']and metrics['mean_grid_cdf_gap']<=threshold['mean']and metrics['mass_above_public_max_plus_point_two']<=threshold['tail'];cdf[key]={'definition':d,'metrics':metrics,'own_unique_actions':len(own),'opponent_triplets':size};print('Public CDF diagnostic: '+key,flush=True)
    theory=theory_audit();result={'configuration':cfg,'pool':pool,'named':named,'target_scenarios':target_meta,'target_scores':target_scores,'cdf':cdf,'theory':theory,'public_summary':{'generated_at':public['generated_at'],'number':len(observed),'mean':float(observed.mean()),'minimum':float(observed.min()),'maximum':float(observed.max()),'quantiles':np.quantile(observed,[.1,.25,.5,.75,.9,.95,.99]).tolist()},'freeze_sha256':freeze_hash,
      'missing_external_artifacts':['field_fit_validated.json','fit_catalog.npz','fit_catalog_weights.npz','memory_equilibrium_best.json','scorer and fitting/search code'],'scope':'Independent handoff audit and new development pool; proxy stresses are not actual applicant populations or Parry replicated fits; no new final selection'}
    save(out/'results.json',result);write_csv(out/'public-cdf-diagnostics.csv',['model','kind','maximum_cdf_gap','mean_grid_cdf_gap','mass_above_6_point_697','nominal_screen_pass'],[[key,v['definition']['kind'],v['metrics']['maximum_cdf_gap'],v['metrics']['mean_grid_cdf_gap'],v['metrics']['mass_above_public_max_plus_point_two'],v['metrics']['passes_nominal_screen']]for key,v in cdf.items()])
    write_csv(out/'named-target-summary.csv',['name','allocation','minimum','equal_scenario_mean','maximum'],[[name,str(NAMED[name]),float(min(target_scores[cid].values())),float(sum(target_scores[cid].values())/len(scenarios)),float(max(target_scores[cid].values()))]for name,cid in named.items()])
    for p in sources:assert sha(p)==sig['source_hashes'][str(p.relative_to(ROOT))]
    for p in inputs:assert sha(p)==sig['input_hashes'][str(p.relative_to(ROOT))]
    save(out/'manifest.json',{'signature':sig,'freeze_sha256':freeze_hash,'completed_unix':time.time(),'elapsed_seconds':time.perf_counter()-clock,'artifacts':{str(p.relative_to(out)):sha(p)for p in out.rglob('*')if p.is_file()and p.name!='manifest.json'}});print('Handoff audit and pool integration complete. Run verification.',flush=True)


if __name__=='__main__':main()
