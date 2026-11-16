"""Frozen final Game 2 challenge, adapting opponents, and declared selection."""
from fractions import Fraction
from itertools import product
from pathlib import Path
from math import sqrt,log
import argparse,json,time,platform,shutil
import numpy as np
from blotto_lab.core import Population
from game2_lab.core import shared_triplets,fixed_sample_bound,reference_group
from game2_lab.human import payoff_arrays
from game2_lab.arena import random_actions,as_population,class_name
from game2_lab.weights import cubic_value
from game2_lab.weight_counters import worst_clone_population
from game2_lab.final import final_arena,core_triplets,pressure_triplets,selection_policy
from run_game2_weights import definitions as weight_definitions,criteria
from run_population_study import save,sha,write_csv
from run_multiwave_study import ROOT


def read(p):return json.loads(p.read_text())


def development_field(old,cfg):
    actions=[tuple(v['allocation'])for v in old['candidates'].values()];high=[]
    for a,b,c,e in product([0,1,2,3],[0,1,2,3,5],[0,1,2,5,10,15],[0,1,2,3,5,10]):
        d=100-a-b-c-e
        if d>=70:high.append((a,b,c,d,e))
    actions+=high
    anchors=list(dict.fromkeys(tuple(v['allocation'])for v in old['primary'].values()))
    for x in anchors:
        for i,j in product(range(5),repeat=2):
            if i==j:continue
            for amount in [1,2,4]:
                if x[i]>=amount:
                    y=list(x);y[i]-=amount;y[j]+=amount;actions.append(tuple(y))
    random=random_actions(np.random.default_rng(cfg['seed']),cfg['new_random_candidates']);actions+=random
    return list(dict.fromkeys(actions)),set(high),list(dict.fromkeys(random))


def freeze_field(actions,high,random,old,objectives,cfg):
    roles={};lookup={x:i for i,x in enumerate(actions)}
    def keep(x,role):roles.setdefault(x,[]).append(role)
    for cid in old['shortlist']:keep(tuple(old['candidates'][cid]['allocation']),'previous_precision')
    for name,value in objectives.items():
        limit=cfg['development_top_combined']if name=='combined_cap_0.25'else(cfg['development_top_reference']if name=='references_cap_0.25'else cfg['development_top_family'])
        for j in sorted(range(len(actions)),key=lambda j:(-round(float(value[j]),11),actions[j]))[:limit]:keep(actions[j],'development_'+name)
    value=objectives['combined_cap_0.25']
    for x in sorted(high,key=lambda x:(-round(float(value[lookup[x]]),11),x))[:cfg['development_top_high_four']]:keep(x,'high_four_screen')
    for name in ['combined_cap_0.25','family_cap_0.25']:
        x=tuple(old['primary'][name]['allocation'])
        for i,j in product(range(5),repeat=2):
            if i==j:continue
            for amount in [1,2,4]:
                if x[i]>=amount:
                    y=list(x);y[i]-=amount;y[j]+=amount;keep(tuple(y),'unfitted_local_'+name)
    for j in range(5):keep(tuple(100 if i==j else 0 for i in range(5)),'pure_fort')
    for x in random[:cfg['unfitted_controls']]:keep(x,'unfitted_random')
    candidates={f'f{i:04d}':{'allocation':x,'roles':sorted(set(roles[x]))}for i,x in enumerate(actions)if x in roles}
    chosen={tuple(v['allocation']):cid for cid,v in candidates.items()}
    tags={name:chosen[tuple(v['allocation'])]for name,v in old['primary'].items()}
    for j in range(5):tags['fort_'+str(j+1)]=chosen[tuple(100 if i==j else 0 for i in range(5))]
    return candidates,tags


def challenge_design(cfg):
    cells={'all_A':{'kind':'core','copies':3,'seed':cfg['seed']+100000}};refs={}
    for style in cfg['styles']:
        for name,mix in cfg['residual_mixtures'].items():
            ref=style+'__'+name;ids=[]
            for k in range(3):
                key=ref+'__k'+str(k);ids.append(key);cells[key]={'kind':'core','style':style,'mixture':mix,'copies':k,'seed':cfg['seed']+100000+len(cells)}
            refs[ref]=ids+['all_A']
    pressure={}
    for family,mass in product(cfg['pressure_families'],cfg['pressure_masses']):
        key='pressure__'+family+'__'+str(mass);definition={'kind':'pressure','family':family,'mass':mass,'seed':cfg['seed']+200000+len(pressure)};cells[key]=definition;pressure[key]=definition
    paths=[]
    for rule,prior,initial in product(['replicator','prior_replacement'],['fifth','shape'],['balanced','heavy_five']):
        paths.append({'name':'arena__'+rule+'__'+prior+'__'+initial,'rule':rule,'prior':prior,'initial':initial,'seed':cfg['seed']+300000+len(paths)})
    return {'cells':cells,'core_references':refs,'pressure':pressure,'arena_paths':paths}


def path_record(result,spec,cfg):
    p={**spec,'actions':result['actions'],'trace':result['trace'],'waves':result['waves'],'input_digest':result['input_digest'],'tail_count':result['tail_count']}
    for snapshot in ['final','average','tail']:
        pop=as_population(result['actions'],result[snapshot],cfg['quantization_mass']);p[snapshot]={'population':pop.to_dict(),'class_shares':{k:sum(Fraction(n,pop.total_mass)for x,n in pop.entries if class_name(x)==k)for k in ['all_in_five','near_all_in_five','low_five','intermediate_five']},'top_allocation':max(pop.entries,key=lambda v:v[1])[0]}
    return p


def figures(out,r):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    policy=r['policy'];selected=policy['selected_candidate_id'];labels=[selected,r['comparison_tags']['human_100'],r['comparison_tags']['combined_cap_0.25'],r['comparison_tags']['rank_tail_mean'],r['comparison_tags']['fort_4']]+list(policy['alternatives'].values());labels=list(dict.fromkeys(labels));matrix=[[float(policy['block_scores'][cid][g])for g in ['core','pressure','arenas']]for cid in labels]
    fig,ax=plt.subplots(figsize=(10,5));pos=np.arange(len(labels));width=.25
    for i,g in enumerate(['core','pressure','arenas']):ax.bar(pos+(i-1)*width,[v[i]for v in matrix],width,label=g)
    ax.set_xticks(pos,[str(tuple(r['candidates'][cid]['allocation']))for cid in labels],rotation=18,ha='right',fontsize=8);ax.set_ylabel('Mean official ranking payout');ax.set_ylim(0,9);ax.set_title('Frozen final comparisons: coverage blocks, not applicant weights');ax.legend();ax.grid(axis='y',alpha=.2);fig.tight_layout();fig.savefig(out/'final-block-comparison.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for ax,rule in zip(axes,['replicator','prior_replacement']):
        for name,p in r['paths'].items():
            if p['rule']==rule:ax.plot([v['step']for v in p['trace']],[v['class_shares']['all_in_five']for v in p['trace']],label=p['prior']+' / '+p['initial'])
        ax.set_title(rule.replace('_',' '));ax.set_xlabel('Step');ax.set_ylabel('Exact all-in-five share');ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.suptitle('New adapting arenas retain counters and use explicit mutation priors');fig.tight_layout();fig.savefig(out/'final-arena-trajectories.png',dpi=170);plt.close(fig)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,default=ROOT/'config/game2-final.json');parser.add_argument('--output',type=Path,default=ROOT/'results/game2-final');parser.add_argument('--resume',action='store_true');args=parser.parse_args();cfg=read(args.config);out=args.output.resolve();clock=time.perf_counter()
    sources=[Path(__file__),ROOT/'verify_game2_final.py',ROOT/'GAME2-FINAL-PROTOCOL.md',ROOT/'game2_lab/final.py',ROOT/'game2_lab/core.py',ROOT/'game2_lab/human.py',ROOT/'game2_lab/arena.py',ROOT/'game2_lab/weights.py',ROOT/'game2_lab/weight_counters.py',ROOT/'run_game2_weights.py',ROOT/'blotto_lab/core.py',ROOT/'blotto_lab/merged.py',ROOT/'blotto_lab/multiwave.py',ROOT/'run_population_study.py',ROOT/'tests/test_game2_final.py']
    inputs=[ROOT/'results/game2-weights/results.json',ROOT/'results/game2-weights/development-scores.npz',ROOT/'results/game2-weights-counter-probe/results.json',ROOT/'results/game2-weights-checkpoint-v1.json',ROOT/'results/game2-arena/results.json',ROOT/'results/game2-arena/seed-pool.json']
    inputs+=[ROOT/'results/game2-weights/manifest.json']+sorted((ROOT/'results/game2-weights/development').glob('*.npz'))
    signature={'configuration':cfg,'configuration_sha256':sha(args.config),'source_hashes':{str(p.relative_to(ROOT)):sha(p)for p in sources},'input_hashes':{str(p.relative_to(ROOT)):sha(p)for p in inputs},'environment':{'python':platform.python_version(),'numpy':np.__version__,'platform':platform.platform()}}
    if out.exists():
        if not args.resume:raise SystemExit('Output exists. Use --resume or another folder.')
        if read(out/'run-inputs.json')!=signature:raise SystemExit('Resume input/source mismatch')
        if (out/'manifest.json').exists():print('Completed. Run independent verification.');return
    else:
        if args.resume:raise SystemExit('No run to resume')
        out.mkdir(parents=True)
        for folder in ['development','training','evaluation','source-snapshot']:(out/folder).mkdir()
        save(out/'run-inputs.json',signature)
        for p in sources:
            target=out/'source-snapshot'/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
    old=read(inputs[0]);probe=read(inputs[2]);oldarena=read(inputs[4]);oldcfg=old['configuration'];d=weight_definitions(oldarena,oldcfg);oldkeys=list(d['cells']);actions,high,random=development_field(old,cfg)
    if not(out/'candidate-freeze.json').exists():
        known=len(old['candidates']);reuse=cfg['development_triplets']==oldcfg['development_triplets'];values=[]
        with np.load(inputs[1])as a:known_scores=a['rank_sum4']
        for j,key in enumerate(oldkeys):
            with np.load(ROOT/'results/game2-weights/development'/(key+'.npz'))as a:rows=a['triplets'][:cfg['development_triplets']]
            if reuse:
                extra=payoff_arrays(actions[known:],rows)[0].sum(axis=1,dtype=np.int64)if len(actions)>known else np.array([],dtype=np.int64);sums=np.r_[known_scores[:,j],extra]
            else:sums=payoff_arrays(actions,rows)[0].sum(axis=1,dtype=np.int64)
            values.append(sums);print('Known development screen: '+key,flush=True)
        values=np.column_stack(values);np.savez_compressed(out/'development-scores.npz',rank_sum4=values);objectives=criteria(values/(4*cfg['development_triplets']),d,oldcfg);candidates,tags=freeze_field(actions,high,random,old,objectives,cfg);design=challenge_design(cfg)
        cloud=list(dict.fromkeys([tuple(c['opponent'])for c in probe['cases'].values()]+[tuple(v['allocation'])for v in old['primary'].values()]+[tuple(100 if i==j else 0 for i in range(5))for j in range(5)]))
        seeds=list(dict.fromkeys([tuple(v['allocation'])for v in candidates.values()]+[tuple(x)for x in read(inputs[5])]+cloud+random_actions(np.random.default_rng(cfg['seed']+400000),cfg['arena_random_controls'])))
        save(out/'candidate-freeze.json',{'created_unix':time.time(),'candidates':candidates,'comparison_tags':tags,'design':design,'development_field':actions,'development_scores_sha256':sha(out/'development-scores.npz'),'seed_pool':seeds,'counter_cloud':cloud,'selection_rule':cfg['selection_rule'],'run_inputs_sha256':sha(out/'run-inputs.json')})
    freeze=read(out/'candidate-freeze.json');candidates=freeze['candidates'];cids=list(candidates);xs=[v['allocation']for v in candidates.values()];design=freeze['design'];seeds=freeze['seed_pool'];cloud=freeze['counter_cloud'];freeze_hash=sha(out/'candidate-freeze.json');paths={}
    if not(out/'challenge-start.json').exists():save(out/'challenge-start.json',{'started_unix':time.time(),'freeze_sha256':freeze_hash})
    for spec in design['arena_paths']:
        folder=out/'training'/spec['name'];result=final_arena(folder,seeds,actions,cfg,spec['rule'],spec['prior'],spec['initial'],spec['seed'])
        if not(folder/'result.json').exists():save(folder/'result.json',path_record(result,spec,cfg));np.savez_compressed(folder/'arrays.npz',final=result['final'],average=result['average'],tail=result['tail'])
        paths[spec['name']]=read(folder/'result.json');print('Final arena complete: '+spec['name'],flush=True)
    cells=dict(design['cells'])
    for spec in design['arena_paths']:
        for snapshot in ['final','average','tail']:
            key=spec['name']+'__'+snapshot;cells[key]={'kind':'arena','population':paths[spec['name']][snapshot]['population'],'path':spec['name'],'snapshot':snapshot,'seed':cfg['seed']+500000+len(cells)}
    if not(out/'evaluation-start.json').exists():save(out/'evaluation-start.json',{'started_unix':time.time(),'freeze_sha256':freeze_hash,'path_result_sha256':{name:sha(out/'training'/name/'result.json')for name in paths}})
    sums=[];castles=[]
    for key,cell in cells.items():
        filename=out/'evaluation'/(key+'.npz')
        if filename.exists():
            with np.load(filename)as a:rank_sum=a['rank_sum4'];castle_sum=a['castle_sum12']
        else:
            if cell['kind']=='core':rows=core_triplets(cell,cfg['evaluation_triplets'],cell['seed'],cloud)
            elif cell['kind']=='pressure':rows=pressure_triplets(cell,cfg['evaluation_triplets'],cell['seed'],cloud)
            else:rows=shared_triplets(Population.from_dict(cell['population']),cfg['evaluation_triplets'],cell['seed'])['triplets']
            rank,castle=payoff_arrays(xs,rows);rank_sum=rank.sum(axis=1,dtype=np.int64);castle_sum=castle.sum(axis=1,dtype=np.int64);np.savez_compressed(filename,triplets=rows.astype(np.uint8),rank4=rank,rank_sum4=rank_sum,castle_sum12=castle_sum)
        sums.append(rank_sum);castles.append(castle_sum);print('Frozen final evaluation: '+key,flush=True)
    values=np.column_stack(sums);cell_keys=list(cells);policy=selection_policy(values,cell_keys,candidates,design,cfg);selected=policy['selected_candidate_id'];bounds={};alpha=cfg['sampling_alpha']/2
    for group,w in policy['block_weights'].items():bounds[group]=9*sqrt(log(2*len(cids)*3/alpha)/2*sum(float(v)**2/cfg['evaluation_triplets']for v in w.values()))
    score_bound=fixed_sample_bound(cfg['evaluation_triplets'],len(cids)*len(cells),alpha)
    stresses={};populations=list(dict.fromkeys([tuple(c['opponent'])for c in probe['cases'].values()]+[tuple(100 if i==j else 0 for i in range(5))for j in range(5)]))
    for cid,c in candidates.items():
        x=c['allocation'];scores=[reference_group([x,y,y,y]).payouts[0]for y in populations];stresses[cid]={'known_pure_populations':list(zip(populations,scores)),'exact_worst_clone':worst_clone_population(x)}
    witnesses={}
    for key,cell in cells.items():
        if cell['kind']!='arena':continue
        j=cell_keys.index(key);best=int(values[:,j].max());cid=min((cids[i]for i in np.flatnonzero(values[:,j]==best)),key=lambda c:candidates[c]['allocation']);score=Fraction(best,4*cfg['evaluation_triplets'])
        witnesses[key]={'candidate_id':cid,'score':score,'gap_above_selfplay':score-Fraction(15,4),'lower_gap':max(0,float(score-Fraction(15,4))-score_bound),'scope':'Finite frozen-field response witness, not global exploitability upper bound'}
    result={'configuration':cfg,'candidates':candidates,'comparison_tags':freeze['comparison_tags'],'design':design,'paths':paths,'cells':cells,'cell_keys':cell_keys,'policy':policy,'block_error_bounds':bounds,'cell_error_bound':score_bound,
        'homogeneous_stress':stresses,'response_witnesses':witnesses,'selected_allocation':candidates[selected]['allocation'],'freeze_sha256':freeze_hash,'scope':'Fixed Game 2 recommendation under declared coverage regret; no applicant forecast, Nash certificate, or universal optimum'}
    save(out/'results.json',result);np.savez_compressed(out/'evaluation-sums.npz',rank_sum4=values,castle_sum12=np.column_stack(castles));figures(out,result)
    write_csv(out/'candidate-ranking.csv',['rank','candidate_id','allocation','maximum_block_regret','equal_block_mean','core_mean','pressure_mean','arena_mean','lowest_model_score'],[[i+1,cid,str(tuple(candidates[cid]['allocation'])),float(policy['maximum_block_regret'][cid]),float(policy['equal_block_mean'][cid]),*[float(policy['block_scores'][cid][g])for g in ['core','pressure','arenas']],float(policy['lowest_model_score'][cid])]for i,cid in enumerate(policy['ranking'])])
    shares=[]
    for name,p in paths.items():
        for snapshot in ['final','average','tail']:
            pop=Population.from_dict(p[snapshot]['population']);shares.extend([[name,p['rule'],p['prior'],p['initial'],snapshot,*x,100*n/pop.total_mass,n,pop.total_mass]for x,n in pop.entries])
    write_csv(out/'population-shares.csv',['path','revision','prior','initial','snapshot','castle1','castle2','castle3','castle4','castle5','percentage','mass','total_mass'],shares)
    for source in sources:assert sha(source)==signature['source_hashes'][str(source.relative_to(ROOT))]
    save(out/'manifest.json',{'signature':signature,'freeze_sha256':freeze_hash,'elapsed_seconds':time.perf_counter()-clock,'completed_unix':time.time(),'artifacts':{str(p.relative_to(out)):sha(p)for p in sorted(out.rglob('*'))if p.is_file()and p.name!='manifest.json'}})
    print('Frozen final challenge complete. Selected '+str(tuple(result['selected_allocation']))+'. Run independent verification.',flush=True)


if __name__=='__main__':main()
