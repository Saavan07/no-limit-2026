"""Broader Game 1 hypotheses, multiwave adaptation, and a frozen external challenge."""
from collections import Counter
from fractions import Fraction
from pathlib import Path
import argparse
import json
import math
import platform
import shutil
import sys
import time

import numpy as np

from blotto_lab.core import DEFAULT_GAME, Population, ScoreTable, best_response, mix_tables, security_of
from blotto_lab.historical import rows_to_table, sampled_metadata
from blotto_lab.multiwave import hypothesis_rows, run_path
from blotto_lab.robust import Objective, search
from run_population_study import encode, save, sha, write_csv

ROOT = Path(__file__).resolve().parent
GROUP_NAMES = ['analytic','historical','anchors','support_rounding','new_reasoning']


def unpack_table(model,key):
    data = model['table']
    return ScoreTable(DEFAULT_GAME,tuple(tuple(row) for row in data['numerators']),data['denominator'],key,
                      data.get('evidence','saved legal population'))


def draw_plots(out,result,paths):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes = plt.subplots(1,2,figsize=(12,4.3),sharey=True)
    colors = {'latest_only':'#a14628','retain_history':'#24628c'}
    for path in paths:
        if path['phase']!='development':
            continue
        records = path['records']
        for ax,field in zip(axes,['generation_security','archive_security']):
            y = [r[field]['exploitability']['decimal'] for r in records]
            ax.plot(range(1,len(y)+1),y,color=colors[path['policy']],
                linestyle='-' if path['replicate']==0 else '--',marker='o',markersize=3,
                label=f"{path['policy'].replace('_',' ')} / seed {path['replicate']+1}")
    for ax,title in zip(axes,['Current generation','Full retained shadow history']):
        ax.set_title(title)
        ax.set_xlabel('Adaptation wave')
        ax.grid(alpha=.2)
    axes[0].set_ylabel('Full-space exploitability (points; smaller is better)')
    axes[1].legend(fontsize=8)
    fig.suptitle('Development arena: exact security diagnostics, without assuming monotonic progress')
    fig.tight_layout()
    fig.savefig(out/'arena-progress.png',dpi=180)
    plt.close(fig)
    display = [(name,item['candidate_id']) for name,item in result['selected'].items()]
    display += [('previous expanded minimax regret',result['baseline_mapping']['expanded_minimax_regret'])]
    partitions = ['legacy','new_behavior','development_arena','validation_behavior','validation_arena','audit_behavior','audit_arena']
    matrix = [[float(result['summary'][cid][p]['worst_regret']) for p in partitions] for name,cid in display]
    fig,ax = plt.subplots(figsize=(11,4.4))
    im = ax.imshow(matrix,cmap='YlOrRd',aspect='auto',vmin=0)
    ax.set_yticks(range(len(display)),[name.replace('_',' ') for name,cid in display],fontsize=8)
    ax.set_xticks(range(len(partitions)),[p.replace('_','\n') for p in partitions],fontsize=9)
    for i,row in enumerate(matrix):
        for j,value in enumerate(row):
            ax.text(j,i,f'{value:.2f}',ha='center',va='center',fontsize=9,
                    color='white' if value>max(map(max,matrix))*.65 else 'black')
    ax.set_title('Frozen selections: worst scenario regret in each set\nSets differ; this is not a statistical train/test-gap estimate')
    fig.colorbar(im,ax=ax,label='Regret (points; smaller is better)')
    fig.tight_layout()
    fig.savefig(out/'challenge-regret.png',dpi=180)
    plt.close(fig)


def render_report(out,result,paths):
    f = lambda x:f'{float(x):.4f}'
    models,summary,selected = result['models'],result['summary'],result['selected']
    lines = ['**Game 1: broader reasoning models and multiwave adaptation**','',
        f"This study contains {len(models)} scenarios and {len(result['candidates'])} frozen fixed allocations. "
        'All four selection objectives were optimized over every one of the 4,598,126 legal allocations. '
        'The previous study\'s evaluation scenarios are now development evidence. Human strategy ideas '
        'have not been supplied or used.','',
        '**Globally selected fixed vectors**','',
        '| Development objective | Allocation | Exact objective, decimal display | Tied optima |',
        '| --- | --- | ---: | ---: |']
    for name,item in selected.items():
        value = item['maximized_value'] if item['mode']=='max_min' else -item['maximized_value']
        lines.append(f"| {name} | {tuple(item['allocation'])} | {f(value)} | {item['optimum_count']} |")
    lines += ['',f"Behavior objectives use {len(result['behavior_keys'])} hypotheses. Expanded objectives use "
        f"{len(result['development_model_order'])} scenarios, adding old strategic constructions, every "
        'development arena generation, and final shadow archives. Max-min maximizes the smallest '
        'score; minimax regret minimizes the largest gap from a scenario\'s own optimum. '
        'Neither objective estimates applicant proportions. The certified near-equilibrium mixture '
        'is a separate diagnostic and is excluded from selection.','',
        '**Withheld challenge results**','',
        '| Frozen selection | Validation minimum / worst regret | Final audit minimum / worst regret | Pure-vector security |',
        '| --- | ---: | ---: | ---: |']
    display = [(name,item['candidate_id']) for name,item in selected.items()]
    display += [('previous expanded minimax regret',result['baseline_mapping']['expanded_minimax_regret']),
                ('previous uniform-population optimum',result['uniform_baseline_id'])]
    for name,cid in display:
        columns = []
        for phase in ['validation','final_audit']:
            keys = [k for k,m in models.items() if m['phase']==phase]
            columns.append(f"{f(min(result['scores'][cid][k] for k in keys))} / {f(max(result['regrets'][cid][k] for k in keys))}")
        lines.append(f"| {name} | "+' | '.join(columns)+f" | {f(result['security'][cid]['security'])} |")
    lines += ['', 'Both challenge sets include direct behavioral generators and newly constructed adapting '
        'opponents. Their compositions differ, so comparing the two columns is not a statistical '
        'train/test-gap estimate. Security in the last column is the score guaranteed by that single '
        'fixed vector against any legal opponent, with its exact counter saved.','',
        '**The most damaging evaluation scenarios**','',
        '| Selection | Worst validation regret: scenario and score / optimum | Worst audit regret: scenario and score / optimum |',
        '| --- | --- | --- |']
    for name,item in selected.items():
        cid = item['candidate_id']
        cells = []
        for phase in ['validation','final_audit']:
            keys = [k for k,m in models.items() if m['phase']==phase]
            k = max(keys,key=lambda k:(result['regrets'][cid][k],k))
            cells.append(f"{k}: {f(result['scores'][cid][k])} / {f(models[k]['best_response']['score'])}")
        lines.append(f"| {name} | "+' | '.join(cells)+' |')
    lines += ['', '**What changed in the population model**','',
        'Eight new development hypotheses model value-favored subsets, cheap scouts, fifth-field '
        'fortresses, skipping a field, round tens, one-above-anchor attacks, heterogeneous value '
        'sensitivity, and monotone ladders. They are deliberately varied legal constructions, '
        'with no estimated frequency among entrants. The previous 27 scenarios retain their exact '
        'saved marginal tables and original evidence labels. Old human data remains a transfer '
        'hypothesis from a different game.','',
        'Validation uses fresh samples of these eight generators plus inverse value preference, '
        'two-field commitment, and a three-unit grid. Its new arenas incorporate those novel families '
        'into their references. Final audit uses rotating templates, two budget blocks, and even '
        'thresholds. Its arenas change environment after wave four, from the development references '
        'to the three audit-family references. The final definitions were saved before running '
        'selection; their score tables were constructed after the candidate freeze.','',
        '**How adaptation works**','',
        f"There are {result['configuration']['development_replicates']} independent development seeds for each "
        'memory policy. Each path has eight waves, each with 192 exact best responses plus random '
        'mutations. Both policies receive coupled exogenous belief seeds. Their first generations '
        'match; later responses differ because their memories differ. Latest-only responds to its '
        'latest generation. Retain-history responds to seed mass and every previous generation. '
        'The latest-only experiment also saves a full shadow archive for diagnostics.','',
        'Beliefs mix declared behavior-group references with strategic memory at shares 0,25,50,75,100 '
        'percent. Initial jobs in each wave respond to individual behavior models, preserving '
        'specialists. DP counts sample uniformly among all optimal responses, removing fixed '
        'lexicographic tie bias inside the arena. Mutations make two legal transfers of at most '
        'eight soldiers, occur with probability one-half, and are retained without claiming '
        'optimality. All original responses, beliefs, tie counts, mutations, and masses are saved.','',
        '| Development path | Final generation exploitability | Final archive exploitability | Distinct archive vectors |',
        '| --- | ---: | ---: | ---: |']
    for path in paths:
        if path['phase']=='development':
            record = path['records'][-1]
            lines.append(f"| {path['label']} | {record['generation_security']['exploitability']['decimal']:.4f} | "
                f"{record['archive_security']['exploitability']['decimal']:.4f} | {len(record['archive']['entries'])} |")
    lines += ['', 'These are exact certificates for the saved distributions, not convergence claims. '
        'Retaining strategies preserves possible counters, but uniform historical mass need not '
        'be the strongest weighting. The earlier LP-certified near-equilibrium mixture remains '
        'the stronger universal-security control. Population exploitation and adversarial security '
        'are different tasks. Within-pool average score is exactly 7.5 regardless of sophistication.','',
        '**Selection discipline and uncertainty**','',
        'The shortlist includes every old candidate, each development best response, four robust '
        'winners, and a declared solver-share sensitivity sweep. No winner or tied alternative was '
        'changed after evaluation. Every frozen allocation is evaluated against every scenario. '
        'All scores and regrets are rational, conditional on saved model tables. Pure-counter '
        'contamination stress is also saved, without claiming global optimization of that objective.','',
        f"The freeze SHA-256 is {result['freeze_sha256']}. Source snapshots and initial input hashes "
        'are saved. Arena checkpoints verify source/settings signatures and rebuild retained mass '
        'before resuming. The runner preserves completed output and rejects changed inputs.','',
        f"For {result['sampling_bounds']['direct_sampled_scenarios']} direct sampled models, simultaneous "
        f"95% DKW score bounds are {result['sampling_bounds']['development_error']:.4f} points at "
        f"{result['configuration']['development_sample_size']:,} draws and "
        f"{result['sampling_bounds']['evaluation_error']:.4f} at "
        f"{result['configuration']['evaluation_sample_size']:,} draws. These bounds condition on "
        'generator definitions and historical fit parameters. They do not cover model validity, '
        'unknown population weights, or the effect of sampled inputs on which adaptive responses '
        'were generated. Arena distributions are exact finite computational constructions, not IID '
        'samples of applicants.','',
        '**Artifacts**','',
        '- `results.json`, `scores.csv`, `regrets.csv`: all frozen candidates and scenarios.',
        '- `models.json`, `samples/`: exact marginal tables, generator metadata, and legal joint draws.',
        '- `arenas/`: every generation, shadow archive, belief, response, mutation, and checkpoint.',
        '- `frozen-candidates.json`: candidate set and hashes saved before evaluation.',
        '- `solver-sensitivity.csv`, `contamination-stress.csv`: declared-prior and counter sensitivity.',
        '- `source-snapshot/`, `run-inputs.json`, `manifest.json`: source, configuration, inputs, and artifact hashes.',
        '- `acceptance.json`: generated by the independent verifier after the experiment.',
        '', 'Run from the project folder: `python3 run_multiwave_study.py --output results/multiwave-new`. '
        'Use the same output and `--resume` after an interruption. Verify with '
        '`python3 verify_multiwave_study.py`. No final application vector or Game 2 is selected here.','',
        '![Exact adaptation diagnostics](arena-progress.png)','',
        '![Frozen challenge regret](challenge-regret.png)','']
    (out/'report.md').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,default=ROOT/'config/multiwave-study.json')
    parser.add_argument('--output',type=Path,default=ROOT/'results/multiwave-study')
    parser.add_argument('--resume',action='store_true')
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    out = args.output.resolve()
    code = [Path(__file__),ROOT/'blotto_lab/core.py',ROOT/'blotto_lab/historical.py',
            ROOT/'blotto_lab/robust.py',ROOT/'blotto_lab/multiwave.py',ROOT/'run_population_study.py',
            ROOT/'MULTIWAVE-PROTOCOL.md']
    old_inputs = [ROOT/'results/population-study/results.json',ROOT/'results/population-study/frozen-candidates.json',
                  ROOT/'results/equilibrium-benchmark/mixture.json']
    signature = {'configuration':cfg,'configuration_sha256':sha(args.config),
        'implementation':{str(p.relative_to(ROOT)):sha(p) for p in code},
        'prior_inputs':{str(p.relative_to(ROOT)):sha(p) for p in old_inputs}}
    if out.exists():
        if not args.resume:
            raise SystemExit('Output exists. Use --resume or a new output folder.')
        if json.loads((out/'run-inputs.json').read_text())!=signature:
            raise SystemExit('Cannot resume changed code, inputs, or configuration.')
        if (out/'manifest.json').exists():
            print('Run already complete; use the independent verifier.',flush=True)
            return
    else:
        if args.resume:
            raise SystemExit('No run to resume')
        out.mkdir(parents=True)
        save(out/'run-inputs.json',signature)
        (out/'samples').mkdir()
        for p in code:
            dest = out/'source-snapshot'/p.relative_to(ROOT)
            dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(p,dest)
        save(out/'configuration.json',cfg)
    start = time.perf_counter()
    previous = json.loads(old_inputs[0].read_text())
    models,tables,paths = {},{},[]
    def add(key,table,phase,partition,group=None,metadata=None):
        if key in models:
            raise ValueError('Duplicate scenario key')
        br = best_response(table)
        tables[key] = table
        models[key] = {**(metadata or {}),'phase':phase,'partition':partition,'group':group,
            'table':{'numerators':table.numerators,'denominator':table.denominator,'evidence':table.evidence},
            'best_response':{'allocation':br.allocation,'score':br.expected_score,'optimum_count':br.optimum_count}}
    print('Loading all previous scenarios as known development evidence',flush=True)
    for key,model in previous['models'].items():
        kind = model['kind']
        if kind=='analytic':
            group = 'analytic'
        elif kind=='historical_feature_transfer':
            group = 'historical'
        elif kind=='synthetic':
            group = 'anchors' if model['family'] in ('equal_jitter','value_jitter') else 'support_rounding'
        else:
            group = 'strategic'
        new_key = 'legacy__'+key
        add(new_key,unpack_table(model,new_key),'development','legacy',group,
            {'kind':'legacy_model','original_key':key,'original_metadata':{k:v for k,v in model.items() if k not in ('table','best_response')}})
    def sample_family(family,phase,seed):
        size = cfg['development_sample_size'] if phase=='development' else cfg['evaluation_sample_size']
        rows = hypothesis_rows(family,size,seed)
        key = phase+'__'+family
        path = out/'samples'/f'{key}.npz'
        if path.exists():
            with np.load(path) as saved:
                if not np.array_equal(saved['allocations'],rows):
                    raise ValueError('Saved sample differs from deterministic reconstruction')
        else:
            np.savez_compressed(path,allocations=rows.astype(np.uint8))
        group = 'new_reasoning' if phase=='development' else None
        partition = 'new_behavior' if phase=='development' else ('validation_behavior' if phase=='validation' else 'audit_behavior')
        add(key,rows_to_table(rows,key),phase,partition,group,{'kind':'direct_generator','family':family,
            'sample_file':str(path.relative_to(out)),**sampled_metadata(rows,seed),
            'status':'unfitted hypothesis; no applicant prevalence estimate'})
        return key
    seed = cfg['seed']
    for i,family in enumerate(cfg['development_families']):
        sample_family(family,'development',seed+100+i)
    behavior_keys = [k for k,m in models.items() if m['group'] in GROUP_NAMES]
    behavior_tables = [tables[k] for k in behavior_keys]
    group_keys = {g:[k for k in behavior_keys if models[k]['group']==g] for g in GROUP_NAMES}
    groups = [mix_tables([tables[k] for k in group_keys[g]],[1]*len(group_keys[g]),name=g+' reference') for g in GROUP_NAMES]
    seed_actions = {tuple(c['allocation']) for c in previous['candidates'].values()}
    seed_actions.update(tuple(m['best_response']['allocation']) for m in models.values())
    seed_population = Population.from_people(sorted(seed_actions),name='one unit per unique seed strategy')
    save(out/'seed-population.json',seed_population.to_dict())
    def arenas(phase,replicates,environments,base_seed):
        for replicate in range(replicates):
            for policy in cfg['memory_policies']:
                label = f'{phase}_r{replicate}_{policy}'
                def callback(wave,generation,archive,gs,hs):
                    print(f"{label} wave {wave}: generation gap {float(gs['exploitability']):.4f}; history gap {float(hs['exploitability']):.4f}",flush=True)
                records = run_path(out/'arenas'/label,policy,seed_population,environments,cfg,
                                   base_seed+replicate*1000000,callback)
                paths.append({'label':label,'phase':phase,'replicate':replicate,'policy':policy,'records':records})
                partition = 'development_arena' if phase=='development' else ('validation_arena' if phase=='validation' else 'audit_arena')
                for record in records:
                    key = f"{label}_w{record['wave']:02}"
                    population = Population.from_dict(record['generation'])
                    add(key,population.table(),phase,partition,'strategic',
                        {'kind':'adaptive_generation','path':label,'wave':record['wave'],'policy':policy,
                         'population_file':f"arenas/{label}/wave-{record['wave']:02}.json",'population_field':'generation'})
                last = records[-1]
                key = label+'_archive'
                add(key,Population.from_dict(last['archive']).table(),phase,partition,'strategic',
                    {'kind':'adaptive_archive','path':label,'wave':last['wave'],'policy':policy,
                     'population_file':f"arenas/{label}/wave-{last['wave']:02}.json",'population_field':'archive'})
    arenas('development',cfg['development_replicates'],[(1,groups,behavior_tables)],seed+10000)
    development_order = list(tables)
    objectives = [Objective('behavior_max_min',tuple(development_order.index(k) for k in behavior_keys)),
        Objective('behavior_minimax_regret',tuple(development_order.index(k) for k in behavior_keys),'minimax_regret'),
        Objective('expanded_max_min',tuple(range(len(development_order)))),
        Objective('expanded_minimax_regret',tuple(range(len(development_order))),'minimax_regret')]
    print('Searching all fixed allocations on the expanded development set',flush=True)
    def progress(done,total):
        if done%1000000==0 or done==total:
            print(f'Robust search {done:,}/{total:,}',flush=True)
    selected = search([tables[k] for k in development_order],objectives,cfg['enumeration_batch_size'],progress)
    candidates = {}
    def candidate(action,role):
        x = tuple(action)
        for cid,item in candidates.items():
            if tuple(item['allocation'])==x:
                if role not in item['roles']:
                    item['roles'].append(role)
                return cid
        cid = f'c{len(candidates):03}'
        candidates[cid] = {'allocation':x,'roles':[role]}
        return cid
    old_mapping = {}
    for cid,item in previous['candidates'].items():
        old_mapping[cid] = candidate(item['allocation'],'previous_'+cid)
    baseline_mapping = {name:old_mapping[item['candidate_id']] for name,item in previous['selected'].items()}
    for name,item in selected.items():
        item['candidate_id'] = candidate(item['allocation'],name)
    for key in development_order:
        candidate(models[key]['best_response']['allocation'],'specialist_'+key)
    reference = mix_tables(groups,[1]*len(groups),name='declared group-balanced behavior reference')
    final_generations = [tables[p['label']+f"_w{cfg['waves']:02}"] for p in paths]
    strategic_reference = mix_tables(final_generations,[1]*len(final_generations),name='declared final-generation reference')
    sensitivity = []
    for raw in cfg['solver_sensitivity_shares']:
        weight = Fraction(str(raw))
        br = best_response(mix_tables([reference,strategic_reference],[1-weight,weight]))
        cid = candidate(br.allocation,f'solver_share_{raw}')
        sensitivity.append({'share':weight,'candidate_id':cid,'allocation':br.allocation,'score':br.expected_score})
    save(out/'development-models.json',models)
    freeze_body = encode({'selected':selected,'candidates':candidates,'development_model_order':development_order,
        'behavior_keys':behavior_keys,'group_keys':group_keys,'baseline_mapping':baseline_mapping,
        'sensitivity':sensitivity,'inputs_sha256':sha(out/'run-inputs.json'),
        'development_models_sha256':sha(out/'development-models.json'),'seed_population_sha256':sha(out/'seed-population.json')})
    freeze_path = out/'frozen-candidates.json'
    if freeze_path.exists():
        frozen = json.loads(freeze_path.read_text())
        if {k:v for k,v in frozen.items() if k!='created_unix'}!=freeze_body:
            raise ValueError('Pre-freeze reconstruction differs from the saved selection')
    else:
        frozen = {'created_unix':time.time(),**freeze_body}
        save(freeze_path,frozen)
    freeze_hash = sha(freeze_path)
    print(f'FROZEN {len(candidates)} candidates: {freeze_hash}',flush=True)
    evaluation_marker = out/'evaluation-start.json'
    if evaluation_marker.exists():
        marker = json.loads(evaluation_marker.read_text())
        if marker['freeze_sha256']!=freeze_hash:
            raise ValueError('Evaluation marker references a different candidate freeze')
        evaluation_started = marker['started_unix']
    else:
        evaluation_started = time.time()
        save(evaluation_marker,{'started_unix':evaluation_started,'freeze_sha256':freeze_hash})
    fresh_keys = [sample_family(f,'validation',seed+200000+i) for i,f in enumerate(cfg['development_families'])]
    validation_novel = [sample_family(f,'validation',seed+210000+i) for i,f in enumerate(cfg['validation_novel_families'])]
    validation_groups = groups[:-1]+[mix_tables([tables[k] for k in fresh_keys+validation_novel],[1]*(len(fresh_keys)+len(validation_novel)))]
    validation_specialists = behavior_tables+[tables[k] for k in fresh_keys+validation_novel]
    arenas('validation',1,[(1,validation_groups,validation_specialists)],seed+3000000)
    audit_keys = [sample_family(f,'final_audit',seed+400000+i) for i,f in enumerate(cfg['audit_novel_families'])]
    audit_groups = [tables[k] for k in audit_keys]
    arenas('final_audit',1,[(1,groups,behavior_tables),(cfg['waves']//2+1,audit_groups,audit_groups)],seed+5000000)
    q = Population.from_dict(json.loads((ROOT/'results/equilibrium-benchmark/mixture.json').read_text()))
    add('certified_equilibrium_control',q.table(),'diagnostic','diagnostic','diagnostic',
        {'kind':'certified_control','population_input':'results/equilibrium-benchmark/mixture.json'})
    scores,regrets,summary,security = {},{},{},{}
    partitions = sorted(set(m['partition'] for m in models.values()))
    for cid,item in candidates.items():
        x = item['allocation']
        scores[cid] = {k:t.expected_score(x) for k,t in tables.items()}
        regrets[cid] = {k:models[k]['best_response']['score']-value for k,value in scores[cid].items()}
        summary[cid] = {}
        for partition in partitions:
            keys = [k for k,m in models.items() if m['partition']==partition]
            summary[cid][partition] = {'minimum':min(scores[cid][k] for k in keys),
                'worst_regret':max(regrets[cid][k] for k in keys),
                'minimum_model':min(keys,key=lambda k:(scores[cid][k],k)),
                'worst_regret_model':max(keys,key=lambda k:(regrets[cid][k],k))}
        security[cid] = security_of(Population.from_people([x]))
    samples = sum('sample_size' in m for m in previous['models'].values())+sum(m['kind']=='direct_generator' for m in models.values())
    error = lambda n:15*math.sqrt(math.log(10*samples/cfg['sampling_alpha'])/(2*n))
    result = {'configuration':cfg,'models':models,'selected':selected,'candidates':candidates,
        'scores':scores,'regrets':regrets,'summary':summary,'security':security,
        'freeze_sha256':freeze_hash,'baseline_mapping':baseline_mapping,'uniform_baseline_id':old_mapping['c004'],
        'behavior_keys':behavior_keys,'development_model_order':development_order,'group_keys':group_keys,
        'sensitivity':sensitivity,'sampling_bounds':{'alpha':cfg['sampling_alpha'],
            'direct_sampled_scenarios':samples,'development_error':error(cfg['development_sample_size']),
            'evaluation_error':error(cfg['evaluation_sample_size'])},
        'arena_paths':[{k:v for k,v in p.items() if k!='records'} for p in paths]}
    save(out/'models.json',models)
    save(out/'results.json',result)
    for field in ['scores','regrets']:
        write_csv(out/f'{field}.csv',['id','roles','allocation']+list(tables),
            [[cid,';'.join(item['roles']),str(tuple(item['allocation']))]+[float(result[field][cid][k]) for k in tables]
             for cid,item in candidates.items()])
    write_csv(out/'solver-sensitivity.csv',['strategic_share','id','allocation','score'],
        [[float(x['share']),x['candidate_id'],str(tuple(x['allocation'])),float(x['score'])] for x in sensitivity])
    write_csv(out/'contamination-stress.csv',['id','scenario','adversarial_share','stressed_score'],
        [[cid,k,raw,float((1-e)*scores[cid][k]+e*security[cid]['security'])]
         for cid in candidates for k in tables for raw in cfg['contamination_shares'] for e in [Fraction(str(raw))]])
    write_csv(out/'arena-progress.csv',['phase','replicate','policy','wave','generation_mass','archive_mass','generation_exploitability','archive_exploitability'],
        [[p['phase'],p['replicate'],p['policy'],r['wave'],sum(m for x,m in r['generation']['entries']),
          sum(m for x,m in r['archive']['entries']),r['generation_security']['exploitability']['decimal'],
          r['archive_security']['exploitability']['decimal']] for p in paths for r in p['records']])
    draw_plots(out,result,paths)
    render_report(out,result,paths)
    assert sha(freeze_path)==freeze_hash and frozen['created_unix']<evaluation_started
    assert json.loads((out/'run-inputs.json').read_text())==signature
    for p in code:
        assert sha(p)==signature['implementation'][str(p.relative_to(ROOT))]
    save(out/'manifest.json',{'freeze_sha256':freeze_hash,'frozen_unix':frozen['created_unix'],
        'evaluation_started_unix':evaluation_started,'completed_unix':time.time(),
        'elapsed_seconds_this_invocation':time.perf_counter()-start,'python':sys.version,
        'numpy':np.__version__,'platform':platform.platform(),'signature':signature,
        'artifacts':{str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='manifest.json'}})
    print(f'Completed {len(models)} models and {len(candidates)} frozen candidates.',flush=True)


if __name__=='__main__':
    main()
