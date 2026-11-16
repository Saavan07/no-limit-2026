"""Merged human/machine Game 1 selection followed by frozen fresh challenges."""
from fractions import Fraction
from itertools import product
from pathlib import Path
import argparse
import json
import math
import shutil
import sys
import time
import numpy as np
from blotto_lab.core import Population,best_response,mix_tables,security_of
from blotto_lab.contamination import FrontierObjective,exact_counter,search_frontiers
from blotto_lab.historical import rows_to_table,sampled_metadata
from blotto_lab.merged import coupled_path,fresh_definitions,mixture_weights,quantize,shifted_rows
from blotto_lab.multiwave import table_digest
from run_contamination_study import nominal_winner
from run_multiwave_study import ROOT,unpack_table
from run_population_study import encode,save,sha,write_csv


def read(path):return json.loads(path.read_text())


def render_report(out,r):
    fmt=lambda x:f'{float(x):.4f}'
    lines=['**Merged Game 1 study: frozen human/machine selections**','',
        f"This study merges {len(r['development_keys'])} known individual scenarios, selects fixed "
        'allocations over the full legal action space, then freezes all candidates before constructing '
        'fresh generator and interacting-learner tests. Human inputs are retained alongside the original '
        'pre-human baselines. No actual applicant shares have been inferred.','',
        '**Eight predeclared primary selections**','',
        '| Selection | Allocation | Selected objective value | Pure security |',
        '| --- | --- | ---: | ---: |']
    for name,p in r['primary'].items():
        v=p['value'] if p['mode']=='score' else -p['value']
        lines.append(f"| {name} | {tuple(p['allocation'])} | {fmt(v)} | {fmt(p['pure_security'])} |")
    lines+=['', 'Score objectives maximize the stated population score. Regret objectives minimize the '
        'largest gap from each scenario\'s optimum. Security floors are guarantees against arbitrary '
        'legal opponents, not estimates of competition scores. Objective sets differ across rows.','',
        '**Common-set comparison after the freeze**','',
        '| Candidate | Development min / worst regret | Validation min / worst regret | Audit min / worst regret |',
        '| --- | ---: | ---: | ---: |']
    display=[(name,p['candidate_id']) for name,p in r['primary'].items()]
    display += [('previous known-model max-min',r['label_ids']['M_known_model_max_min']),
                ('human H3',r['label_ids']['H3']),('human range illustration',r['label_ids']['R_illustration'])]
    for name,cid in display:
        cells=[fmt(r['summary'][cid][phase]['minimum'])+' / '+fmt(r['summary'][cid][phase]['worst_regret']) for phase in ['development','validation','audit']]
        lines.append('| '+name+' | '+' | '.join(cells)+' |')
    lines+=['', 'Validation and audit contain different scenario sets. Their minima and regrets are '
        'stress summaries, not estimates of a statistical train/test gap. The entire frozen candidate '
        'set is reported; no primary winner was replaced after seeing these results.','',
        '**Development evidence and population weights**','',
        'The individual scenarios comprise 121 original machine models, 48 exact human-coalition '
        'variants, 16 broader-scout models, and 32 final/time-average learning populations. Static '
        'models, strategic constructions, and numerical learning histories have separate labels. '
        'Each retains its original conditional interpretation.','',
        'Five declared reference components are machine behavior, human static behavior, old strategic '
        'opponents, point learners, and win learners. The human reference balances exact/open, '
        'exact/five-heavy, broad/open, and broad/five-heavy cases. Favorite-coalition assumptions '
        'are not silently given additional weight. Paths and final/average snapshots are balanced '
        'inside each learning component.','',
        'The capped prior family is P(h,l,w,z)=z*S+(1-z)*[(1-l)*((1-h)*M+h*H) '
        '+l*((1-w)*Lp+w*Lw)]. The bounds are h in [0,1], l in [0,0.5], w in [0,1], '
        'and z in [0,0.25]. These limits are stated hypotheses. The map is multi-affine, so each '
        'intermediate population is a convex combination of corners. Worst score and worst regret '
        'over this whole parameter box therefore attain their worst cases at corners. Duplicate '
        'corner aliases are retained, without interpreting their count as prevalence.','',
        '**Fresh static challenges**','',
        'Validation has 20 new 200,000-vector populations: geometric scouts with value allocation, '
        'round-number scouts with uniform allocation, uniform scouts with noisy target attention, '
        'and five four-field-priority variants. Audit has 15 more: a castle-5 threshold of 52, '
        'inverse value attention, and correlated outside budgets. All definitions and seeds were '
        'saved before selection, and their draws were generated after the freeze. These are new '
        'hypothesis challenges, not samples of applicants.','',
        '**Interacting objectives and innovation**','',
        'Each fresh learner path contains point-seeking and match-win-seeking types at fixed stated '
        'shares. They learn against the same combined population, updating strategies within their '
        'own type. The different units of points and wins cannot change the type proportions. '
        'Validation and audit vary these shares, starting conditions, update rules, and mutation.','',
        'Every 100 steps the arena admits an exact full-space official-point best response if it is '
        'new, seeds it in both types, and retains all old actions. Win-seeking optimization remains '
        'restricted to available strategies. The initial 331-vector pool predates this study\'s '
        'candidate search; the newly selected vectors are not deliberately seeded as opponents. '
        'Final and time-average distributions have exact official-point security certificates. '
        'Iteration count is not evidence of convergence, and the resulting frequencies are not '
        'applicant estimates.','',
        '| Phase | Learner paths | Innovations admitted | Largest final point exploitability | Smallest final point exploitability |',
        '| --- | ---: | ---: | ---: | ---: |']
    for phase in ['validation','audit']:
        paths=[p for p in r['learning'].values() if p['phase']==phase]
        gaps=[p['final']['certificate']['exploitability'] for p in paths]
        lines.append(f"| {phase} | {len(paths)} | {sum(sum(i['added'] for i in p['innovations']) for p in paths)} | {fmt(max(gaps))} | {fmt(min(gaps))} |")
    lines+=['', '**Precision and reproducibility**','',
        'All scenario optima, frozen scores, regrets, pure counters, and saved-mixture certificates '
        'are rational. Selection is global conditional on the stated tables and pure-security '
        'classes, using bounded floating screening followed by exact finalist comparisons. The '
        'independent integer-upper-bound full-space audit is saved in acceptance.json.','',
        f"The simultaneous 95% direct-generator score error bound is {r['sampling_bound']:.4f} points "
        'for the 35 fresh sampled models. It conditions on generator definitions. It does not '
        'cover model validity, source-population weights, historical transfer error, or the '
        'populations induced by learning. Numerical trajectories and exact certificates for their '
        'quantized legal mixtures are different claims.','',
        f"The candidate freeze SHA-256 is {r['freeze_sha256']}. Source snapshots, configuration, "
        'development tables, conditional optima, all candidates, exact scores, legal samples, '
        'learning checkpoints, innovations, and artifact hashes are retained.','',
        '- `frontiers.json`: exact optima and tie counts within each pure-security class.',
        '- `references.json`: five components, all corner aliases, and exact mixture weights.',
        '- `frozen-candidates.json`: primary choices and full exploratory set before challenges.',
        '- `samples/`, `learning/`: legal draws, coupled histories, and innovation witnesses.',
        '- `results.json`, `scores.csv`, `regrets.csv`: every frozen candidate and scenario.',
        '- `acceptance.json`: independent accounting, generator, and global-selection audit.',
        '', 'Run from the project folder: `python3 run_merged_game1.py --output results/merged-new`. '
        'Resume an interrupted run with the same output and `--resume`. Verify the delivered run '
        'with `python3 verify_merged_game1.py`. No final application vector or Game 2 is chosen.','',
        '![Frozen primary challenge regrets](primary-challenges.png)','']
    (out/'report.md').write_text('\n'.join(lines))


def plot(out,r):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names=list(r['primary']);matrix=[]
    for name in names:
        cid=r['primary'][name]['candidate_id']
        matrix.append([float(r['summary'][cid][p]['worst_regret']) for p in ['development','validation','audit']])
    fig,ax=plt.subplots(figsize=(9,5.7));im=ax.imshow(matrix,cmap='YlOrRd',vmin=0,aspect='auto')
    ax.set_yticks(range(len(names)),[n.replace('_',' ') for n in names],fontsize=8)
    ax.set_xticks(range(3),['Known development','Fresh validation','Fresh audit'])
    for i,row in enumerate(matrix):
        for j,v in enumerate(row):ax.text(j,i,f'{v:.2f}',ha='center',va='center',color='white' if v>max(map(max,matrix))*.65 else 'black')
    ax.set_title('Eight frozen primary choices: worst regret in each scenario set\nDifferent sets; no statistical train/test-gap interpretation')
    fig.colorbar(im,ax=ax,label='Points below the scenario optimum')
    fig.tight_layout();fig.savefig(out/'primary-challenges.png',dpi=180);plt.close(fig)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,default=ROOT/'config/merged-game1.json')
    parser.add_argument('--output',type=Path,default=ROOT/'results/merged-game1');parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();cfg=read(args.config);out=args.output.resolve();start=time.perf_counter()
    source=[Path(__file__),ROOT/'MERGED-GAME1-PROTOCOL.md',ROOT/'run_population_study.py',ROOT/'run_multiwave_study.py',ROOT/'run_contamination_study.py']+sorted((ROOT/'blotto_lab').glob('*.py'))
    input_paths=[ROOT/'results/contamination-study/results.json',ROOT/'results/human-challenge/results.json',ROOT/'results/human-scout-stress/results.json',
        ROOT/'results/game1-machine-checkpoint-v1.json',ROOT/'results/equilibrium-benchmark/mixture.json',ROOT/'results/human-challenge/learning/replicator_points_balanced_sources_mu0.npz']
    signature={'configuration':cfg,'config_sha256':sha(args.config),'source_hashes':{str(p.relative_to(ROOT)):sha(p) for p in source},'input_hashes':{str(p.relative_to(ROOT)):sha(p) for p in input_paths}}
    if out.exists():
        if not args.resume:raise SystemExit('Output exists. Use --resume or a new output folder.')
        if read(out/'run-inputs.json')!=signature:raise SystemExit('Cannot resume changed sources or inputs.')
        if (out/'manifest.json').exists():print('Run complete. Use the independent verifier.');return
    else:
        if args.resume:raise SystemExit('No existing run to resume')
        out.mkdir(parents=True);(out/'samples').mkdir();save(out/'run-inputs.json',signature)
        for p in source:
            target=out/'source-snapshot'/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
    machine,human,broad,baseline=map(read,input_paths[:4]);models={};tables={}
    def add(key,table,phase,kind,meta=None):
        if key in models:raise ValueError('Duplicate scenario')
        br=best_response(table);tables[key]=table
        models[key]={**(meta or {}),'phase':phase,'kind':kind,'table':{'numerators':table.numerators,'denominator':table.denominator,'evidence':table.evidence},
            'best_response':{'allocation':br.allocation,'score':br.expected_score,'optimum_count':br.optimum_count}}
    print('Merging prior machine, human-coalition, broad-scout, and learning scenarios',flush=True)
    machine_keys=[];static_machine=[];human_keys=[];learned_keys=[];learn_by_type={'points':[],'wins':[]}
    for key in machine['known_model_order']:
        m=machine['known_models'][key];name='machine__'+key
        add(name,unpack_table(m,name),'development','machine_known',{'original_key':key})
        machine_keys.append(name)
        if m['kind'] not in ('adaptive_generation','adaptive_archive') and not (m['kind']=='legacy_model' and m['group']=='strategic'):static_machine.append(name)
    for key,m in human['models'].items():
        name='human_exact__'+key;add(name,unpack_table(m,name),'development','human_exact',{'original_key':key});human_keys.append(name)
    for key,m in broad['models'].items():
        name='human_broad__'+key;add(name,unpack_table(m,name),'development','human_broad',{'original_key':key});human_keys.append(name)
    for path,p in human['learning'].items():
        for snapshot in ['final','average']:
            name='learned__'+path+'__'+snapshot;pop=Population.from_dict(p[snapshot]['population'])
            add(name,pop.table(),'development','previous_learning',{'original_path':path,'snapshot':snapshot,'fitness':p['fitness']})
            learned_keys.append(name);learn_by_type[p['fitness']].append(name)
    development_keys=list(tables);assert len(development_keys)==217
    human_reference_tables=[]
    for name in ['all_open','all_heavy5']:
        human_reference_tables.append(unpack_table(human['references'][name],name))
        keys=broad['references'][name]['model_keys'];human_reference_tables.append(mix_tables([unpack_table(broad['models'][k],k) for k in keys],[1]*len(keys)))
    components={
        'machine_behavior':unpack_table(machine['reference_models']['ref_balanced_s0'],'machine behavior'),
        'human_static':mix_tables(human_reference_tables,[1]*4),
        'old_strategic':unpack_table(machine['reference_models']['ref_balanced_s1'],'old strategic'),
        'point_learners':mix_tables([tables[k] for k in learn_by_type['points']],[1]*len(learn_by_type['points'])),
        'win_learners':mix_tables([tables[k] for k in learn_by_type['wins']],[1]*len(learn_by_type['wins']))}
    search_keys=development_keys.copy();search_tables=[tables[k] for k in search_keys]
    def reference_table(name,table):
        digest=table_digest(table)
        for i,t in enumerate(search_tables):
            if table_digest(t)==digest:return i
        index=len(search_tables);search_tables.append(table);search_keys.append(name);return index
    component_indices={k:reference_table('component__'+k,t) for k,t in components.items()}
    corners={};corner_indices=[]
    axes=[cfg['mixture_box'][p] for p in ['h','l','w','z']]
    for bits in product((0,1),repeat=4):
        params=[axes[i][b] for i,b in enumerate(bits)];weights=mixture_weights(*params)
        name='corner_'+''.join(map(str,bits));table=mix_tables(list(components.values()),weights,name=name)
        index=reference_table(name,table);corners[name]={'parameters':dict(zip(['h','l','w','z'],params)),'weights':weights,'table_index':index}
        if index not in corner_indices:corner_indices.append(index)
    mid=cfg['midpoint_prior'];mid_weights=mixture_weights(*(mid[k] for k in ['h','l','w','z']))
    mid_index=reference_table('midpoint_prior',mix_tables(list(components.values()),mid_weights))
    sets={'machine_all':[development_keys.index(k) for k in machine_keys],
        'human_static':[development_keys.index(k) for k in human_keys],
        'static_behavior':[development_keys.index(k) for k in static_machine+human_keys],
        'learned':[development_keys.index(k) for k in learned_keys],
        'merged_all':list(range(len(development_keys))),
        'reference_means':list(component_indices.values()),'prior_box':corner_indices}
    objectives=[FrontierObjective(name+'_'+mode,tuple(indices),mode) for name,indices in sets.items() for mode in ['score','regret']]
    for index in sorted(set(corner_indices+[mid_index])):objectives.append(FrontierObjective('expected_'+search_keys[index],(index,),'score'))
    definitions={'search_keys':search_keys,'tables':{k:{'table':{'numerators':t.numerators,'denominator':t.denominator,'evidence':t.evidence}} for k,t in zip(search_keys,search_tables)},
        'components':component_indices,'corners':corners,'midpoint':{'parameters':mid,'weights':mid_weights,'table_index':mid_index},'sets':sets,'objectives':objectives}
    save(out/'development-score-tables.json',definitions);save(out/'references.json',{k:v for k,v in definitions.items() if k!='tables'})
    print(f'Complete-action search: {len(search_tables)} tables, {len(objectives)} conditional objectives',flush=True)
    def progress(done,total):
        if done%1000000<cfg['enumeration_batch_size'] or done==total:print(f'Global selection {done:,}/{total:,}',flush=True)
    searched=search_frontiers(search_tables,objectives,cfg['enumeration_batch_size'],progress)
    frontiers=searched['frontiers'];save(out/'frontiers.json',frontiers)
    np.savez_compressed(out/'security-classes.npz',security_twice=searched['security_twice'])
    specifications=[('merged_max_min','merged_all_score',0),('merged_minimax_regret','merged_all_regret',0),
        ('merged_regret_security2','merged_all_regret',2),('merged_regret_security3','merged_all_regret',3),
        ('midpoint_expected','expected_'+search_keys[mid_index],0),('midpoint_expected_security2','expected_'+search_keys[mid_index],2),
        ('static_regret_security2','static_behavior_regret',2),('prior_box_regret_security2','prior_box_regret',2)]
    candidates={}
    def candidate(x,role):
        x=tuple(x)
        for cid,c in candidates.items():
            if tuple(c['allocation'])==x:
                if role not in c['roles']:c['roles'].append(role)
                return cid
        cid=f'c{len(candidates):03}';candidates[cid]={'allocation':x,'roles':[role]};return cid
    for i,c in enumerate(baseline['exploratory_vectors']):candidate(c['allocation'],f'pre_human_{i}')
    labels=human['labels'];label_ids={label:candidate(x,label) for label,x in labels.items()}
    for objective,entries in frontiers.items():
        for c,p in entries.items():candidate(p['allocation'],f'{objective}_security{c}')
    primary={}
    for name,obj,floor in specifications:
        selected=nominal_winner(frontiers[obj],Fraction(floor));cid=candidate(selected['allocation'],name);counter,w=exact_counter(selected['allocation'])
        primary[name]={**selected,'objective':obj,'mode':next(o.mode for o in objectives if o.name==obj),
            'security_floor':floor,'candidate_id':cid,'pure_security':w,'counter':counter}
    seed_actions=[tuple(x) for x in human['pool_actions']];save(out/'seed-pool.json',seed_actions)
    frozen_body=encode({'primary':primary,'candidates':candidates,'label_ids':label_ids,
        'development_keys':development_keys,'selection_inputs_sha256':sha(out/'development-score-tables.json'),
        'run_inputs_sha256':sha(out/'run-inputs.json'),'seed_pool_sha256':sha(out/'seed-pool.json')})
    path=out/'frozen-candidates.json'
    if path.exists():
        frozen=read(path)
        if {k:v for k,v in frozen.items() if k!='created_unix'}!=frozen_body:raise ValueError('Reconstructed candidate selection differs from freeze')
    else:frozen={'created_unix':time.time(),**frozen_body};save(path,frozen)
    freeze_hash=sha(path);print(f'FROZEN {len(candidates)} candidates: {freeze_hash}',flush=True)
    marker=out/'evaluation-start.json'
    if marker.exists():
        evaluation=read(marker)
        if evaluation['freeze_sha256']!=freeze_hash:raise ValueError('Evaluation freeze mismatch')
    else:evaluation={'started_unix':time.time(),'freeze_sha256':freeze_hash};save(marker,evaluation)
    validation,audit=fresh_definitions();sample_models=0
    for phase,definitions_phase,offset in [('validation',validation,10000),('audit',audit,20000)]:
        for j,definition in enumerate(definitions_phase):
            key=definition['key'];seed=cfg['seed']+offset+j;rows,meta=shifted_rows(definition,cfg['evaluation_sample_size'],seed)
            sample_file='samples/'+key+'.npz'
            if (out/sample_file).exists():
                with np.load(out/sample_file) as saved:
                    if not np.array_equal(rows,saved['allocations']):raise ValueError('Saved draw mismatch')
            else:np.savez_compressed(out/sample_file,allocations=rows.astype(np.uint8))
            add(key,rows_to_table(rows,key),phase,'fresh_generator',{'definition':definition,'sample_file':sample_file,**meta,**sampled_metadata(rows,seed)})
            sample_models+=1
            if sample_models%5==0:print(f'Fresh static models {sample_models}/35',flush=True)
    with np.load(input_paths[5]) as stored:balanced=stored['initial'].copy()
    learning={}
    for phase,offset in [('validation',100000),('audit',200000)]:
        for number,(w,rule,mutation,initial_name) in enumerate(product(cfg[phase+'_win_shares'],cfg['learning_rules'],cfg[phase+'_mutations'],cfg[phase+'_initials'])):
            seed=cfg['seed']+offset+number
            if initial_name=='source_balanced':initial=balanced.copy()
            elif initial_name=='geometric_tilt':
                concentration=np.square(np.array(seed_actions)/100).sum(axis=1);initial=balanced*np.exp(4*(concentration-.2));initial/=initial.sum()
            else:initial=np.random.default_rng(seed).dirichlet(np.full(len(seed_actions),.25))
            name=f'{phase}_w{w}_{rule}_mu{mutation}_{initial_name}'
            folder=out/'learning'/name
            def callback(step,size):
                if step==cfg['learning_steps']:print(f'{name}: {size} retained strategies',flush=True)
            result=coupled_path(folder,seed_actions,initial,w,rule,mutation,cfg,seed,callback=callback)
            np.savez_compressed(folder/'arrays.npz',initial=initial,point=result['point'],win=result['win'],final=result['final'],average=result['average'])
            entry={'phase':phase,'win_share':w,'rule':rule,'mutation':mutation,'initial':initial_name,'seed':seed,
                'actions':result['actions'],'trace':result['trace'],'innovations':result['innovations'],'input_digest':result['input_digest']}
            for snapshot in ['final','average']:
                masses=quantize(result[snapshot],cfg['quantization_mass']);population=Population.from_counts([(x,m) for x,m in zip(result['actions'],masses) if m],name=name+'_'+snapshot)
                certificate=security_of(population);entry[snapshot]={'population':population.to_dict(),'certificate':certificate}
                add(name+'_'+snapshot,population.table(),phase,'coupled_learning',{'path':name,'snapshot':snapshot,
                    'population_file':f'learning/{name}/result.json','certificate':certificate})
            save(folder/'result.json',entry);learning[name]=entry
    q=Population.from_dict(read(input_paths[4]));add('certified_equilibrium',q.table(),'diagnostic','control')
    scores={};regrets={};summary={};security={}
    for cid,c in candidates.items():
        x=c['allocation'];scores[cid]={k:t.expected_score(x) for k,t in tables.items()}
        regrets[cid]={k:models[k]['best_response']['score']-v for k,v in scores[cid].items()}
        summary[cid]={}
        for phase in ['development','validation','audit']:
            keys=[k for k,m in models.items() if m['phase']==phase]
            summary[cid][phase]={'minimum':min(scores[cid][k] for k in keys),'worst_regret':max(regrets[cid][k] for k in keys),
                'minimum_model':min(keys,key=lambda k:(scores[cid][k],k)),'worst_regret_model':max(keys,key=lambda k:(regrets[cid][k],k))}
        counter,w=exact_counter(x);security[cid]={'counter':counter,'guarantee':w}
    prior_scores={cid:{name:search_tables[p['table_index']].expected_score(c['allocation']) for name,p in {**corners,'midpoint':definitions['midpoint']}.items()} for cid,c in candidates.items()}
    bound=15*math.sqrt(math.log(10*sample_models/cfg['sampling_alpha'])/(2*cfg['evaluation_sample_size']))
    r={'configuration':cfg,'models':models,'development_keys':development_keys,'primary':primary,'candidates':candidates,
        'label_ids':label_ids,'frontiers':frontiers,'class_counts':searched['class_counts'],'class_first':searched['class_first'],
        'scores':scores,'regrets':regrets,'summary':summary,'security':security,'prior_scores':prior_scores,'learning':learning,
        'freeze_sha256':freeze_hash,'frozen_unix':frozen['created_unix'],'evaluation_started_unix':evaluation['started_unix'],
        'sampling_bound':bound,'sampled_scenarios':sample_models,'scope':'Conditional-model selection and fresh constructed challenge; no applicant prevalence inference'}
    save(out/'models.json',models);save(out/'results.json',r)
    for field in ['scores','regrets']:
        write_csv(out/f'{field}.csv',['id','allocation','roles']+list(tables),
            [[cid,str(c['allocation']),';'.join(c['roles'])]+[float(r[field][cid][k]) for k in tables] for cid,c in candidates.items()])
    write_csv(out/'prior-scores.csv',['id']+list(corners)+['midpoint'],[[cid]+[float(v[name]) for name in list(corners)+['midpoint']] for cid,v in prior_scores.items()])
    render_report(out,r);plot(out,r)
    assert sha(path)==freeze_hash and frozen['created_unix']<evaluation['started_unix']
    for p in source:assert sha(p)==signature['source_hashes'][str(p.relative_to(ROOT))]
    save(out/'manifest.json',{'signature':signature,'freeze_sha256':freeze_hash,'elapsed_seconds_this_invocation':time.perf_counter()-start,
        'completed_unix':time.time(),'python':sys.version,'numpy':np.__version__,
        'artifacts':{str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='manifest.json'}})
    print(f'Completed: {len(models)} scenarios, {len(candidates)} frozen candidates, {len(learning)} coupled paths',flush=True)


if __name__=='__main__':main()
