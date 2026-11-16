"""Adapting Game 2 populations, finite response search, and frozen robustness tests."""
from fractions import Fraction
from itertools import product
from pathlib import Path
import argparse,json,shutil,time,platform
import numpy as np
from blotto_lab.core import Population
from game2_lab.core import shared_triplets,fixed_sample_bound
from game2_lab.human import ANCHORS
from game2_lab.arena import (arena_path,initial_shares,random_actions,as_population,metric_arrays,
                            single_triplet_response,static_triplets,class_name)
from run_multiwave_study import ROOT
from run_population_study import save,sha,write_csv


def read(p):return json.loads(p.read_text())


def path_definitions(cfg):
    result=[]
    for rule,history,initial in product(cfg['rules'],cfg['history_shares'],cfg['initials']):
        name=f'rank__{rule}__h{history}__{initial}'
        result.append({'name':name,'group':'official_rank','type_shares':[1,0,0],'rule':rule,'history_mass':history,'initial':initial,'mutation':cfg['primary_mutation']})
    for rule,mu in product(cfg['rules'],cfg['mixed_mutations']):
        result.append({'name':f'mixed__{rule}__mu{mu}','group':'mixed_objectives','type_shares':cfg['mixed_type_shares'],'rule':rule,'history_mass':.25,'initial':'sparse','mutation':mu})
    for i,d in enumerate(result):d['initial_seed']=cfg['seed']+100+i;d['path_seed']=cfg['seed']+1000+i
    return result


def controls(old):
    return {'human_89':(2,2,3,4,89),'human_96':(1,1,1,1,96),'human_100':(0,0,0,0,100),
            'previous_low_five':(1,29,3,64,3),'previous_zero_five':(2,2,45,51,0),
            'zero_five_probe':(1,29,3,67,0),'game1_reference':(3,8,4,44,41),'equal_spread':(20,20,20,20,20)}


def snapshots_record(path,definition,cfg):
    result={**definition,'actions':path['actions'],'trace':path['trace'],'waves':path['waves'],'input_digest':path['input_digest'],
            'fallbacks':path['fallbacks'],'tail_count':path['tail_count']}
    for snapshot in ['final','average','tail_average']:
        pop=as_population(path['actions'],path[snapshot],cfg['quantization_mass']);q={x:Fraction(n,pop.total_mass)for x,n in pop.entries}
        result[snapshot]={'population':pop.to_dict(),'class_shares':{k:sum(v for x,v in q.items()if class_name(x)==k)for k in ['all_in_five','near_all_in_five','low_five','intermediate_five']},
                         'top_allocation':max(q,key=q.get),'top_share':max(q.values())}
    return result


def plots(out,r):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for group,ax in zip(['official_rank','mixed_objectives'],axes):
        for name,p in r['paths'].items():
            if p['group']!=group:continue
            ax.plot([t['step']for t in p['trace']],[t['class_shares']['all_in_five']for t in p['trace']],label=name.replace('rank__','').replace('mixed__','').replace('__',' / '),lw=1.3)
        ax.set_xlabel('Training step');ax.set_ylabel('Share using exactly all-in five');ax.set_title(group.replace('_',' ').title());ax.grid(alpha=.2);ax.legend(fontsize=6)
    fig.suptitle('Arena shares are simulation outcomes under stated rules');fig.tight_layout();fig.savefig(out/'all-in-share-trajectories.png',dpi=170);plt.close(fig)
    names=list(r['primary']);matrix=[]
    for name in names:
        cid=r['primary'][name]['candidate_id'];matrix.append([float(r['summary'][cid][g]['minimum'])for g in ['rank_paths','mixed_paths','new_static']])
    fig,ax=plt.subplots(figsize=(9,6));im=ax.imshow(matrix,cmap='YlGnBu',vmin=0,vmax=9,aspect='auto')
    ax.set_yticks(range(len(names)),[n.replace('_',' ')for n in names],fontsize=8);ax.set_xticks(range(3),['Rank-only arena snapshots','Mixed-objective snapshots','New static populations'])
    for i,row in enumerate(matrix):
        for j,v in enumerate(row):ax.text(j,i,f'{v:.2f}',ha='center',va='center',color='white'if v>5 else'black')
    ax.set_title('Frozen comparisons: lowest score in each model group\nIndependent evaluation triples; arena distributions were used for training')
    fig.colorbar(im,ax=ax,label='Official rank payout');fig.tight_layout();fig.savefig(out/'primary-worst-scores.png',dpi=170);plt.close(fig)


def report(out,r):
    fmt=lambda v:f'{float(v):.4f}'
    lines=['**Game 2 adapting arena and robustness**','',
        'Twelve paths retain old actions and admit current/history responses, fixed-group counters, '
        'and substantial mutations. Eight paths optimize official rank payout; four add fixed '
        'wrong-objective types as sensitivity tests. No arena share estimates applicant prevalence.','',
        '**Frozen primary comparisons**','',
        '| Rule | Allocation | Rank-arena mean / minimum | Mixed-arena mean / minimum | New-static mean / minimum |',
        '| --- | --- | ---: | ---: | ---: |']
    for name,p in r['primary'].items():
        cid=p['candidate_id'];cells=[fmt(r['summary'][cid][g]['mean'])+' / '+fmt(r['summary'][cid][g]['minimum'])for g in ['rank_paths','mixed_paths','new_static']]
        lines.append(f"| {name} | {tuple(p['allocation'])} | {' | '.join(cells)} |")
    lines+=['','![Frozen worst scores](primary-worst-scores.png)','',
        '**What the simulations do and do not certify**','',
        'Training fitness uses fresh shared samples at each step. Finite search proposes responses '
        'against current and historical mixtures; independent gates compare selected responses with '
        'a fixed incumbent on new triples. These are finite-search improvements, not population-wide '
        'best-response certificates. Independent fixed-group oracles are exact only for their one '
        'specified opponent triple.','',
        'The last 40 steps admit no new actions. Final, whole-run, and last-40 populations are saved. '
        'A settling period, positive mutation, retained support, or a high final share does not prove '
        'equilibrium convergence. All reported rank-response gaps are lower witnesses found in the '
        'frozen finite evaluation field, not global exploitability upper bounds.','',
        'Candidate selection uses training-tail populations and four known static references. The '
        'separate evaluation triples on those arena distributions test estimation, not population '
        'generalization. Eight newly generated static populations provide a separate broader test. '
        'These categories remain separately labeled.','',
        '**An exact worst-group limit**','',
        'Against opponents all-in on castles 3,4,5 respectively, any legal focal allocation earns '
        'at most one ranking-payout point. A bid below 100 cannot take those castles; taking castles '
        '1 and 2 yields three raw points, at most tying the castle-3 opponent for third. Tying a '
        '100-troop fort consumes the entire budget and also leaves the focal player tied for bottom. '
        'All-in five attains one against that group and has a previously established floor of one '
        'against every group. Thus no fixed or randomized focal strategy can guarantee more against '
        'arbitrarily coordinated opponent triples. This does not determine the optimum against an '
        'iid population or the value of a multiplayer equilibrium. The independent audit scans all '
        '4,598,126 default allocations for this fixed-group bound.','',
        '**Population shares**','',
        'Full final, average, and tail supports are retained per path, with fixed objective-type shares '
        'and disjoint fifth-bid classes. Primary paths are not pooled with sensitivity paths to infer '
        'a human population. The class labels describe bids, not inferred player intentions.','',
        '![All-in share trajectories](all-in-share-trajectories.png)','',
        '**Precision and next work**','',
        f"The simultaneous fixed-sample score-error bound is {r['sampling_bound']:.4f} points, "
        'conditional on each frozen population law. It does not validate those laws, the training '
        'dynamics, or applicant weights. Priors, initial shares, mutation, history, and type shares '
        'are explicit hypotheses. No final Game 2 allocation is selected.','',
        'Run `python3 run_game2_arena.py --output results/game2-arena-new`; resume with the same '
        'folder and `--resume`. Verify with `python3 verify_game2_arena.py --study results/game2-arena-new`. '
        'The next stage is weight/failure review followed by a frozen final challenge and selection.','']
    (out/'report.md').write_text('\n'.join(lines))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,default=ROOT/'config/game2-arena.json');parser.add_argument('--output',type=Path,default=ROOT/'results/game2-arena');parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();cfg=read(args.config);out=args.output.resolve();clock=time.perf_counter()
    sources=[Path(__file__),ROOT/'verify_game2_arena.py',ROOT/'GAME2-ARENA-PROTOCOL.md',ROOT/'game2_lab/arena.py',ROOT/'game2_lab/core.py',ROOT/'game2_lab/human.py',
             ROOT/'blotto_lab/core.py',ROOT/'blotto_lab/merged.py',ROOT/'blotto_lab/multiwave.py',ROOT/'run_population_study.py',ROOT/'tests/test_game2_arena.py']
    inputs=[ROOT/'results/game2-human/results.json',ROOT/'results/game2-human-zero-five-probe/results.json',ROOT/'results/game1-selected-v1.json',ROOT/'results/game2-human-checkpoint-v1.json']
    signature={'configuration':cfg,'configuration_sha256':sha(args.config),'environment':{'python':platform.python_version(),'numpy':np.__version__,'platform':platform.platform()},'source_hashes':{str(p.relative_to(ROOT)):sha(p)for p in sources},'input_hashes':{str(p.relative_to(ROOT)):sha(p)for p in inputs}}
    if out.exists():
        if not args.resume:raise SystemExit('Output exists. Use --resume or another folder.')
        if read(out/'run-inputs.json')!=signature:raise SystemExit('Resume source/input mismatch')
        if (out/'manifest.json').exists():print('Run complete. Use the independent verifier.');return
    else:
        if args.resume:raise SystemExit('No run to resume')
        out.mkdir(parents=True);(out/'evaluation').mkdir();(out/'selection').mkdir();save(out/'run-inputs.json',signature)
        for p in sources:
            target=out/'source-snapshot'/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
    old=read(inputs[0]);probe=read(inputs[1]);archive=[tuple(v['allocation'])for v in old['candidates'].values()]
    fixed=controls(old)
    if not(out/'training-start.json').exists():
        actions=[tuple(old['candidates'][cid]['allocation'])for cid in old['shortlist']]+[tuple(v['allocation'])for v in probe['variants']]
        actions+=list(fixed.values())+[(2,30,4,60,4)]+[tuple(100 if i==j else 0 for i in range(5))for j in range(5)]
        cloud=list(dict.fromkeys(list(fixed.values())+[(2,30,4,60,4)]))
        responses=[single_triplet_response([x]*3)['allocation']for x in cloud];actions+=responses
        actions+=random_actions(np.random.default_rng(cfg['seed']),cfg['seed_pool_random']);actions=list(dict.fromkeys(actions));cloud=list(dict.fromkeys(cloud+responses))
        params=path_definitions(cfg);static_defs=[{'key':family+'_'+str(v),'family':family,'variant':v,'size':cfg['evaluation_triplets'],'seed':cfg['seed']+50000+100*j+v}for j,family in enumerate(cfg['static_families'])for v in [0,1]]
        save(out/'seed-pool.json',actions);save(out/'counter-cloud.json',cloud)
        save(out/'training-start.json',{'started_unix':time.time(),'fixed_comparisons':fixed,'path_definitions':params,'static_definitions':static_defs,'seed_pool_sha256':sha(out/'seed-pool.json'),'cloud_sha256':sha(out/'counter-cloud.json')})
    marker=read(out/'training-start.json');seed_pool=[tuple(x)for x in read(out/'seed-pool.json')];cloud=[tuple(x)for x in read(out/'counter-cloud.json')];params=marker['path_definitions'];paths={}
    for definition in params:
        name=definition['name'];folder=out/'training'/name;initial=initial_shares(seed_pool,definition['initial'],definition['initial_seed'])
        path=arena_path(folder,seed_pool,archive,initial,definition['type_shares'],definition['rule'],definition['history_mass'],definition['mutation'],cfg,definition['path_seed'])
        result_path=folder/'result.json'
        if result_path.exists():record=read(result_path)
        else:
            record=snapshots_record(path,definition,cfg);save(result_path,record)
            np.savez_compressed(folder/'arrays.npz',initial=initial,types=path['types'],final=path['final'],average=path['average'],tail_average=path['tail_average'])
        paths[name]=record;print('Arena path complete: '+name,flush=True)
    if (out/'post-training-freeze.json').exists():freeze=read(out/'post-training-freeze.json');candidates=freeze['candidates'];primary=freeze['primary'];shortlist=freeze['shortlist']
    else:
        actions=list(dict.fromkeys(archive+seed_pool+[tuple(x)for p in paths.values()for x in p['actions']]))
        candidates={f'c{i:04d}':{'allocation':x,'class':class_name(x)}for i,x in enumerate(actions)};cids=list(candidates);lookup={x:cid for cid,x in zip(cids,actions)}
        selection={};rank_names=[]
        for name,p in paths.items():
            key='tail__'+name;selection[key]={'population':p['tail_average']['population'],'group':p['group'],'seed':cfg['seed']+20000+len(selection)}
            if p['group']=='official_rank':rank_names.append(key)
        static_pops={
            'static_center':Population.from_counts([(x,10 if group in ['abandon','broad']else(50 if group=='all_in'else 25))for group,rows in ANCHORS.items()for x in rows]),
            'static_abandon':Population.from_people(ANCHORS['abandon']),
            'static_broad':Population.from_people(ANCHORS['broad']),
            'static_heavy':Population.from_counts([(ANCHORS['all_in'][0],2)]+[(x,1)for x in ANCHORS['near_all_in']])}
        for name,pop in static_pops.items():selection[name]={'population':pop.to_dict(),'group':'known_static','seed':cfg['seed']+20000+len(selection)}
        sums=[];selection_keys=list(selection)
        for key,model in selection.items():
            triplets=shared_triplets(Population.from_dict(model['population']),cfg['selection_triplets'],model['seed'])['triplets']
            mat=metric_arrays(actions,triplets)[0];sums.append(mat.sum(axis=1,dtype=np.int64));np.savez_compressed(out/'selection'/(key+'.npz'),triplets=triplets.astype(np.uint8))
            print('Finite-field selection scenario: '+key,flush=True)
        values=np.column_stack(sums);rank_indices=[selection_keys.index(k)for k in rank_names];known_index=selection_keys.index('static_center')
        objectives={'rank_tail_mean':values[:,rank_indices].sum(axis=1),'rank_tail_minimum':values[:,rank_indices].min(axis=1),
                    'all_selection_regret':(values-values.max(axis=0)[None,:]).min(axis=1),
                    'half_static_half_rank':8*values[:,known_index]+values[:,rank_indices].sum(axis=1)}
        primary={name:{'allocation':x,'candidate_id':lookup[x],'origin':'fixed before arena training'}for name,x in fixed.items()}
        for name,value in objectives.items():
            top=int(value.max());i=min(np.flatnonzero(value==top),key=lambda j:actions[int(j)]);primary[name]={'allocation':actions[i],'candidate_id':cids[i],'origin':'finite sampled post-training selection','objective':name,'objective_numerator':top}
        kept={p['candidate_id']for p in primary.values()};kept.update(lookup[tuple(old['candidates'][cid]['allocation'])]for cid in old['shortlist'])
        for value,limit in [(objectives['rank_tail_mean'],cfg['shortlist_mean']),(objectives['rank_tail_minimum'],cfg['shortlist_robust'])]:kept.update(cids[i]for i in sorted(range(len(actions)),key=lambda j:(-int(value[j]),actions[j]))[:limit])
        for p in paths.values():
            for snapshot in ['final','tail_average']:kept.add(lookup[tuple(p[snapshot]['top_allocation'])])
        shortlist=[cid for cid in cids if cid in kept]
        save(out/'selection-definitions.json',{'models':selection,'keys':selection_keys,'rank_indices':rank_indices,'static_center_index':known_index})
        np.savez_compressed(out/'selection-scores.npz',rank_sum4=values)
        freeze={'created_unix':time.time(),'candidates':candidates,'primary':primary,'shortlist':shortlist,'selection_definitions_sha256':sha(out/'selection-definitions.json'),
                'selection_scores_sha256':sha(out/'selection-scores.npz'),'training_start_sha256':sha(out/'training-start.json'),'path_result_sha256':{name:sha(out/'training'/name/'result.json')for name in paths},
                'run_inputs_sha256':sha(out/'run-inputs.json')}
        save(out/'post-training-freeze.json',freeze)
    freeze_sha=sha(out/'post-training-freeze.json')
    if not(out/'evaluation-start.json').exists():save(out/'evaluation-start.json',{'started_unix':time.time(),'freeze_sha256':freeze_sha})
    models={};scores={cid:{}for cid in shortlist};castles={cid:{}for cid in shortlist};firsts={cid:{}for cid in shortlist}
    for name,p in paths.items():
        for snapshot in ['final','average','tail_average']:
            key=name+'__'+snapshot;models[key]={'kind':'trained_arena','group':'rank_paths'if p['group']=='official_rank'else'mixed_paths','path':name,'snapshot':snapshot,'population':p[snapshot]['population'],'seed':cfg['seed']+40000+len(models)}
    for d in marker['static_definitions']:models[d['key']]={'kind':'new_static','group':'new_static','definition':d}
    for key,model in models.items():
        filename=out/'evaluation'/(key+'.npz')
        if filename.exists():
            with np.load(filename)as a:triplets=a['triplets'];rank=a['rank4'];castle_sum=a['castle_sum12'];first_sum=a['first_sum12']
        else:
            if model['kind']=='trained_arena':triplets=shared_triplets(Population.from_dict(model['population']),cfg['evaluation_triplets'],model['seed'])['triplets']
            else:
                d=model['definition'];triplets=static_triplets(d['family'],d['variant'],d['size'],d['seed'],cloud)
            rank,castle,first=metric_arrays([tuple(candidates[cid]['allocation'])for cid in shortlist],triplets);castle_sum=castle.sum(axis=1,dtype=np.int64);first_sum=first.sum(axis=1,dtype=np.int64)
            np.savez_compressed(filename,triplets=triplets.astype(np.uint8),rank4=rank,castle_sum12=castle_sum,first_sum12=first_sum)
        model['file']=str(filename.relative_to(out))
        for i,cid in enumerate(shortlist):
            scores[cid][key]=Fraction(int(rank[i].sum(dtype=np.int64)),4*len(triplets));castles[cid][key]=Fraction(int(castle_sum[i]),12*len(triplets));firsts[cid][key]=Fraction(int(first_sum[i]),12*len(triplets))
        print('Frozen arena/static evaluation: '+key,flush=True)
    summary={}
    for cid,v in scores.items():
        summary[cid]={}
        for group in ['rank_paths','mixed_paths','new_static']:
            keys=[k for k,m in models.items()if m['group']==group];values=[v[k]for k in keys];summary[cid][group]={'mean':sum(values)/len(values),'minimum':min(values),'maximum':max(values),'minimum_scenarios':[k for k in keys if v[k]==min(values)]}
    bound=fixed_sample_bound(cfg['evaluation_triplets'],len(shortlist)*len(models),cfg['sampling_alpha'])
    witnesses={}
    for key,m in models.items():
        if m['kind']!='trained_arena':continue
        value=max(v[key]for v in scores.values());winners=[cid for cid,v in scores.items()if v[key]==value];witnesses[key]={'candidate_id':min(winners,key=lambda cid:candidates[cid]['allocation']),'estimated_response_score':value,'estimated_gap_above_selfplay':value-Fraction(15,4),'simultaneous_lower_gap':max(0,float(value-Fraction(15,4))-bound),'scope':'Finite frozen-field response witness, not global exploitability upper bound'}
    cap_opponents=[(0,0,100,0,0),(0,0,0,100,0),(0,0,0,0,100)]
    result={'configuration':cfg,'paths':paths,'candidates':candidates,'primary':primary,'shortlist':shortlist,'models':models,'scores':scores,'castle_scores':castles,'first_credits':firsts,
            'summary':summary,'response_witnesses':witnesses,'sampling_bound':bound,'freeze_sha256':freeze_sha,
            'worst_group_benchmark':{'opponents':cap_opponents,'exact_response':single_triplet_response(cap_opponents),'scope':'Coordinated worst-group ceiling, not iid population or equilibrium value'},
            'scope':'Sampled-fitness adapting arenas and frozen robustness comparisons; no final Game 2 selection or applicant forecast'}
    save(out/'results.json',result);plots(out,result);report(out,result)
    write_csv(out/'primary-summary.csv',['rule','allocation','group','mean','minimum','maximum'],[[name,str(tuple(p['allocation'])),g,*[float(summary[p['candidate_id']][g][k])for k in ['mean','minimum','maximum']]]for name,p in primary.items()for g in ['rank_paths','mixed_paths','new_static']])
    share_rows=[]
    for name,p in paths.items():
        for snapshot in ['final','average','tail_average']:
            pop=Population.from_dict(p[snapshot]['population'])
            share_rows.extend([[name,p['group'],snapshot,*x,100*n/pop.total_mass,n,pop.total_mass]for x,n in pop.entries])
    write_csv(out/'population-shares.csv',['path','group','snapshot','castle1','castle2','castle3','castle4','castle5','percentage','integer_mass','total_mass'],share_rows)
    for p in sources:assert sha(p)==signature['source_hashes'][str(p.relative_to(ROOT))]
    save(out/'manifest.json',{'signature':signature,'freeze_sha256':freeze_sha,'elapsed_seconds':time.perf_counter()-clock,'completed_unix':time.time(),
         'artifacts':{str(p.relative_to(out)):sha(p)for p in sorted(out.rglob('*'))if p.is_file()and p.name!='manifest.json'}})
    print('Game 2 adapting arena complete. Run independent verification.',flush=True)


if __name__=='__main__':main()
