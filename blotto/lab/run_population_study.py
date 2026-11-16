"""Protocol-frozen population robustness study. Run from any working directory."""
from collections import Counter
from dataclasses import asdict, is_dataclass
from fractions import Fraction
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math
import platform
import sys
import time

import numpy as np

from blotto_lab.core import (DEFAULT_GAME, Population, best_response, mix_tables,
                            multinomial, security_of, uniform_composition)
from blotto_lab.historical import (fit_features, load_round, rows_to_table,
    sample_synthetic, sample_transferred, sampled_metadata, transfer_parameters)
from blotto_lab.robust import Objective, search

ROOT = Path(__file__).resolve().parent


def encode(obj):
    if is_dataclass(obj):
        return encode(asdict(obj))
    if isinstance(obj,Fraction):
        return {'numerator':obj.numerator,'denominator':obj.denominator,'decimal':float(obj)}
    if isinstance(obj,dict):
        return {str(k):encode(v) for k,v in obj.items()}
    if isinstance(obj,(list,tuple)):
        return [encode(v) for v in obj]
    return obj


def save(path,obj):
    path.write_text(json.dumps(encode(obj),indent=2,sort_keys=True,ensure_ascii=True)+'\n')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fingerprints():
    paths = [Path(__file__),ROOT/'POPULATION-PROTOCOL.md',
             ROOT/'results/first-pass/arena-mixture.json']+sorted((ROOT/'blotto_lab').glob('*.py'))
    return {str(p.relative_to(ROOT)):sha(p) for p in paths}


def make_cloud(tables,size,seed,name):
    """Finite response population, not an estimated distribution of human beliefs."""
    rng = np.random.default_rng(seed)
    counts, beliefs = Counter(),[]
    for j in range(size):
        if j<len(tables):
            weights = [int(i==j) for i in range(len(tables))]
        else:
            concentration = float(rng.choice([.05,.2,1.,5.]))
            weights = rng.multinomial(1000,rng.dirichlet(np.full(len(tables),concentration))).tolist()
        response = best_response(mix_tables(tables,weights,name='declared belief')).allocation
        counts[response] += 1
        beliefs.append({'weights':weights,'response':response})
    return Population.from_counts(counts,name=name),beliefs


def write_csv(path,header,rows):
    with path.open('w',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def render_report(out,result):
    candidates = result['candidates']
    models = result['models']
    selected = result['selected']
    summary = result['summary']
    def fmt(x):
        return f'{float(x):.4f}'
    lines = ['**Game 1: population robustness and independent challenges**','',
        'Four objectives were optimized over all 4,598,126 legal fixed allocations. '
        'The shortlist was saved and hashed before the validation and final-audit models were constructed. '
        'These are controlled population experiments, not observations of Sequoia applicants.','',
        '**The four selected allocations**','',
        '| Selection objective | Allocation | Development objective | Tied optima |',
        '| --- | --- | ---: | ---: |']
    for name,item in selected.items():
        value = item['maximized_value'] if item['mode']=='max_min' else -item['maximized_value']
        lines.append(f"| {name} | {tuple(item['allocation'])} | {fmt(value)} | {item['optimum_count']} |")
    lines += ['', 'For max-min, the objective is the smallest expected score. For minimax regret, '
        'it is the largest gap from each scenario\'s own optimum, so smaller is better. '
        'Behavior objectives use ten development hypotheses. Expanded objectives add the response '
        'cloud and the earlier full-history arena mixture. No prevalence weights are inferred.','',
        '**Performance after the selection was frozen**','',
        '| Selection | Development minimum / worst regret | Validation minimum / worst regret | Final audit minimum / worst regret |',
        '| --- | ---: | ---: | ---: |']
    for name,item in selected.items():
        cid = item['candidate_id']
        columns = [f"{fmt(summary[cid][part]['minimum'])} / {fmt(summary[cid][part]['worst_regret'])}" for part in ('development','validation','final_audit')]
        lines.append(f"| {name} | "+' | '.join(columns)+' |')
    lines += ['', 'Development columns include both strategic models for a common comparison, even when '
        'a behavior-only objective did not optimize them. Scenario sets differ across columns; changes '
        'in their minima do not measure a statistical train/test gap. Regret is relative to an exact '
        'best response to the same saved population.','',
        '**Where each candidate fails most sharply**','',
        '| Selection | Worst validation scenario | Score | That scenario\'s optimum | Worst final-audit scenario | Score |',
        '| --- | --- | ---: | ---: | --- | ---: |']
    for name,item in selected.items():
        cid = item['candidate_id']
        v = summary[cid]['validation']['worst_regret_model']
        a = summary[cid]['final_audit']['worst_regret_model']
        lines.append(f"| {name} | {v} | {fmt(result['scores'][cid][v])} | {fmt(models[v]['best_response']['score'])} | {a} | {fmt(result['scores'][cid][a])} |")
    lines += ['', '**How the population models were built**','',
        'Three analytic controls represent uniform legal vectors, independent soldier placement, and '
        'uniform multiples-of-five vectors. Three synthetic models represent perturbations of equal '
        'allocation, perturbations of a value-proportional allocation, and sparse support. Development '
        'samples contain 100,000 legal vectors each.','',
        'Riddler rounds 1/2 contribute four transferred models. A Dirichlet-multinomial model matches '
        'mean attention and allocation concentration, with power-law or interpolated-rank attention. '
        'A divisibility-by-five rate supplies a declared rounding proxy. The original game used ten '
        'fields and match-win ranking. These assumptions create five-field hypotheses; they do not '
        'recover the current applicant population. Duplicates retain their original frequency.','',
        'Validation uses fresh 200,000-vector samples, rounds 3/4, middle-focused and one-above-grid '
        'families, and a separately seeded response cloud. Final audit uses 13-field round 5 under '
        'two attention transfers and two concentration transfers, plus a top-two-field family. '
        'Raw historical data had previously been inspected for data quality. Only their transferred '
        'scenarios were withheld from selection. Repeated entrants and shared round history prevent '
        'treating historical rounds as independent human cohorts.','',
        '**Strategic adaptation**','',
        'The development response cloud contains 512 exact best responses to varied, declared beliefs '
        'about the ten development models. It preserves duplicate response mass. Its belief generator '
        'is a search convention, not an estimate of human reasoning levels. The second strategic '
        'model is the previous 5,001-mass archive, which retains every historical response.','',
        '| Strategic distribution | Distinct allocations | Exact mixture security | Full-space exploitability |',
        '| --- | ---: | ---: | ---: |']
    for key,info in result['strategic_models'].items():
        lines.append(f"| {key} | {info['distinct']} | {fmt(info['security'])} | {fmt(info['exploitability'])} |")
    lines += ['', 'A mixture certificate describes drawing from that distribution. Every selected fixed '
        'vector has its own legal counter. Security against arbitrary opponents is shown in '
        'candidate-security.csv; contamination-stress.csv combines that counter with each modeled '
        'population. Those stress results evaluate the frozen shortlist and are not a globally '
        'optimized contamination strategy.','',
        '**Sensitivity to assumed applicant mix**','',
        'solver-sensitivity.csv varies strategic-opponent mass from 0% to 100%. Its behavior reference '
        'gives equal weight to four declared groups: analytic controls, anchor perturbations, sparse '
        'allocations, and historical transfers. Equal weighting inside those groups is a comparison '
        'convention. The strategic reference weights the response cloud and archive equally. '
        'The exact best response is recomputed at each share, and its scores against every model are '
        'saved in results.json. There is no data-supported preferred share.','',
        '**What is exact, and what remains uncertain**','',
        'Individual model optima, shortlist scores, regrets, pure counters, and mixture certificates '
        'use rational arithmetic. The four robust searches scan the complete legal action space, '
        'use a rigorously bounded floating screen, and rescore every possible maximizer exactly. '
        'The exact tie counts and screening bounds are retained in results.json.','',
        'For each independently generated sampled model, a simultaneous DKW bound covers all '
        'candidate vectors, including candidates selected from that sample. At familywise alpha '
        f"{result['sampling_bounds']['alpha']}, the 100,000-draw score error bound is "
        f"{result['sampling_bounds']['development_score_error']:.4f} points and the 200,000-draw bound is "
        f"{result['sampling_bounds']['evaluation_score_error']:.4f} points. Corresponding optimization "
        'regret or estimated-regret errors can be bounded by twice the score bound. These conservative '
        'bounds condition on fitted generator parameters. They cover neither the fit itself nor '
        'transfer assumptions, population weights, or real applicants. The listed response-cloud '
        'scores are exact for finite constructed populations, not IID estimates of applicants.','',
        'The complete shortlist was frozen under SHA-256 '+result['freeze_sha256']+'. '
        'All results were reported without changing the selected vectors after evaluation. '
        'A withheld failure is evidence to revise the model in a later study, not permission to '
        'quietly tune this run. Your independently developed strategies remain an unused challenge.','',
        '**Files and reproducibility**','',
        '- `scores.csv` and `regrets.csv`: every frozen candidate against every scenario.',
        '- `models.json`: parameters, sample seeds, raw-file hashes, fitted features, and exact marginal tables.',
        '- `samples/`: compressed legal joint draws, preserving construction and multiplicities.',
        '- `frozen-candidates.json`: selection-time vectors and hashes of development inputs.',
        '- `results.json`: rational scores, full-space winners, exact optima, uncertainty, and security.',
        '- `manifest.json`: configuration, source hashes, versions, chronology, and artifact hashes.',
        '- `response-cloud-*.json`: legal supports and exact integer belief weights.',
        '', 'Run `python3 run_population_study.py --output results/population-study-new` for a new '
        'reproduction. Existing output folders are rejected to preserve the freeze boundary. '
        'Game 2 and a final application submission are outside this study.','',
        'Historical source: [FiveThirtyEight Riddler Castles](https://github.com/fivethirtyeight/data/tree/master/riddler-castles), '
        'CC BY 4.0, raw files and attribution retained in `data/riddler`. The sample-confidence '
        'inequality follows the DKW-Massart form, applied by a union bound to five marginal CDFs '
        'per sampled scenario. See the project protocol for scope.','',
        '![Frozen-candidate scenario regret](regret-heatmap.png)','']
    (out/'report.md').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,default=ROOT/'config/population-study.json')
    parser.add_argument('--output',type=Path,default=ROOT/'results/population-study')
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    out = args.output.resolve()
    if out.exists():
        raise SystemExit('Output exists; use a new output folder to preserve the frozen study.')
    out.mkdir(parents=True)
    (out/'samples').mkdir()
    save(out/'configuration.json',cfg)
    (out/'protocol.md').write_bytes((ROOT/'POPULATION-PROTOCOL.md').read_bytes())
    started,clock = time.time(),time.perf_counter()
    initial_sources = fingerprints()
    models,tables,features,source_metadata = {},{}, {},{}
    phases = {'development':[],'validation':[],'final_audit':[]}
    sample_count = 0
    def add(key,table,phase,metadata):
        tables[key] = table
        response = best_response(table)
        models[key] = {**metadata,'phase':phase,'table':{'denominator':table.denominator,
            'numerators':table.numerators,'evidence':table.evidence},
            'best_response':{'allocation':response.allocation,'score':response.expected_score,
                             'optimum_count':response.optimum_count}}
        phases[phase].append(key)
    def sampled(key,rows,seed,phase,metadata):
        nonlocal sample_count
        sample_count += 1
        np.savez_compressed(out/'samples'/f'{key}.npz',allocations=rows.astype(np.uint8))
        add(key,rows_to_table(rows,key),phase,{**metadata,**sampled_metadata(rows,seed),
            'sample_file':f'samples/{key}.npz','iid_given_generator':True})
    def synthetic(family,phase,seed):
        key = phase+'_'+family
        size = cfg['development_sample_size'] if phase=='development' else cfg['evaluation_sample_size']
        sampled(key,sample_synthetic(family,size,seed),seed,phase,
                {'kind':'synthetic','family':family,'status':'unfitted hypothesis'})
    def historical(rounds,phase,base_seed,concentrations):
        offset = 0
        for number in rounds:
            rows,source = load_round(ROOT/'data/riddler',number)
            fitted = fit_features(rows)
            features[number],source_metadata[number] = fitted,source
            for profile in cfg['profiles']:
                for concentration in concentrations:
                    offset += 1
                    seed = base_seed+offset
                    parameters = transfer_parameters(fitted,profile,concentration)
                    size = cfg['development_sample_size'] if phase=='development' else cfg['evaluation_sample_size']
                    key = f'{phase}_r{number}_{profile}_{concentration}'
                    sampled(key,sample_transferred(parameters,size,seed),seed,phase,
                        {'kind':'historical_feature_transfer','source':source,'features':fitted,
                         'parameters':parameters,'status':'uncertain transfer hypothesis'})
    seed = cfg['seed']
    print('Constructing development models',flush=True)
    for key,table in [('uniform_legal',uniform_composition()),('soldier_random',multinomial()),
                      ('round_grid',uniform_composition(unit=5))]:
        add(key,table,'development',{'kind':'analytic','status':'exact mathematical control'})
    for i,family in enumerate(cfg['development_families']):
        synthetic(family,'development',seed+10+i)
    historical(cfg['development_rounds'],'development',seed+100,[cfg['development_concentration']])
    behavior_keys = list(phases['development'])
    behavior_tables = [tables[k] for k in behavior_keys]
    print('Constructing response cloud from exact population best responses',flush=True)
    cloud,beliefs = make_cloud(behavior_tables,cfg['response_cloud_beliefs'],seed+200,'development_cloud')
    save(out/'response-cloud-development.json',{'population':cloud.to_dict(),'beliefs':beliefs,
        'input_model_order':behavior_keys,'seed':seed+200,'concentrations':[.05,.2,1,5]})
    archive = Population.from_dict(json.loads((ROOT/'results/first-pass/arena-mixture.json').read_text()))
    add('response_cloud',cloud.table(),'development',{'kind':'constructed_strategic','mass':cloud.total_mass})
    add('history_archive',archive.table(),'development',{'kind':'constructed_strategic','mass':archive.total_mass})
    dev_keys = list(phases['development'])
    dev_tables = [tables[k] for k in dev_keys]
    objectives = [Objective('behavior_max_min',tuple(range(len(behavior_keys)))),
        Objective('behavior_minimax_regret',tuple(range(len(behavior_keys))),'minimax_regret'),
        Objective('expanded_max_min',tuple(range(len(dev_keys)))),
        Objective('expanded_minimax_regret',tuple(range(len(dev_keys))),'minimax_regret')]
    def progress(done,total):
        if done%1000000==0 or done==total:
            print(f'Full-space robust search: {done:,}/{total:,}',flush=True)
    selected = search(dev_tables,objectives,cfg['enumeration_batch_size'],progress)
    candidates = {}
    def candidate(action,label):
        action = tuple(action)
        for cid,item in candidates.items():
            if tuple(item['allocation'])==action:
                item['roles'].append(label)
                return cid
        cid = f'c{len(candidates):03d}'
        candidates[cid] = {'allocation':action,'roles':[label]}
        return cid
    for name,item in selected.items():
        item['candidate_id'] = candidate(item['allocation'],name)
    for key in dev_keys:
        candidate(models[key]['best_response']['allocation'],'best_response_'+key)
    candidate((20,20,20,20,20),'equal_anchor')
    candidate((7,13,20,27,33),'value_anchor')
    # Group-balanced reference, specified before evaluation. No fitted prior.
    groups = [behavior_keys[:3],behavior_keys[3:5],[behavior_keys[5]],behavior_keys[6:]]
    group_tables = [mix_tables([tables[k] for k in group],[1]*len(group)) for group in groups]
    reference = mix_tables(group_tables,[1]*len(groups),name='declared grouped behavior reference')
    strategic_reference = mix_tables([cloud.table(),archive.table()],[1,1],name='declared strategic reference')
    sensitivity = []
    for share in cfg['solver_shares']:
        s = Fraction(str(share))
        response = best_response(mix_tables([reference,strategic_reference],[1-s,s]))
        cid = candidate(response.allocation,f'solver_share_{share}')
        sensitivity.append({'solver_share':s,'candidate_id':cid,'allocation':response.allocation,
                            'mixture_score':response.expected_score,'optimum_count':response.optimum_count})
    save(out/'development-models.json',{k:models[k] for k in dev_keys})
    freeze = {'created_unix':time.time(),'configuration_sha256':sha(out/'configuration.json'),
        'protocol_sha256':sha(out/'protocol.md'),'implementation':initial_sources,
        'development_models_sha256':sha(out/'development-models.json'),
        'development_cloud_sha256':sha(out/'response-cloud-development.json'),
        'selected':selected,'candidates':candidates,'sensitivity':sensitivity}
    save(out/'frozen-candidates.json',freeze)
    freeze_hash = sha(out/'frozen-candidates.json')
    print(f'FROZEN {len(candidates)} candidates: {freeze_hash}',flush=True)
    evaluation_started = time.time()
    print('Constructing withheld validation models',flush=True)
    for i,family in enumerate(cfg['development_families']+cfg['validation_novel_families']):
        synthetic(family,'validation',seed+1000+i)
    historical(cfg['validation_rounds'],'validation',seed+1100,[cfg['development_concentration']])
    eval_cloud,eval_beliefs = make_cloud(behavior_tables,cfg['evaluation_response_beliefs'],seed+1200,'validation_cloud')
    save(out/'response-cloud-validation.json',{'population':eval_cloud.to_dict(),'beliefs':eval_beliefs,
        'input_model_order':behavior_keys,'seed':seed+1200,'concentrations':[.05,.2,1,5]})
    add('validation_response_cloud',eval_cloud.table(),'validation',{'kind':'constructed_strategic','mass':eval_cloud.total_mass})
    print('Constructing withheld final-audit models',flush=True)
    historical(cfg['final_audit_rounds'],'final_audit',seed+2100,cfg['audit_concentrations'])
    for i,family in enumerate(cfg['audit_novel_families']):
        synthetic(family,'final_audit',seed+2200+i)
    scores,regrets,summary,security = {},{}, {},{}
    for cid,item in candidates.items():
        x = item['allocation']
        scores[cid] = {k:t.expected_score(x) for k,t in tables.items()}
        regrets[cid] = {k:models[k]['best_response']['score']-v for k,v in scores[cid].items()}
        summary[cid] = {}
        for phase,keys in phases.items():
            summary[cid][phase] = {'minimum':min(scores[cid][k] for k in keys),
                'worst_regret':max(regrets[cid][k] for k in keys),
                'minimum_model':min(keys,key=lambda k:(scores[cid][k],k)),
                'worst_regret_model':max(keys,key=lambda k:(regrets[cid][k],k))}
        security[cid] = security_of(Population.from_people([x]))
    ndev,neval = cfg['development_sample_size'],cfg['evaluation_sample_size']
    alpha = cfg['simultaneous_sampling_alpha']
    error = lambda n:15*math.sqrt(math.log(10*sample_count/alpha)/(2*n))
    bounds = {'alpha':alpha,'sampled_scenarios':sample_count,'development_score_error':error(ndev),
              'evaluation_score_error':error(neval),'scope':'conditional on specified fitted generators only'}
    strategic = {}
    for key,pop in [('response_cloud',cloud),('history_archive',archive),('validation_response_cloud',eval_cloud)]:
        info = security_of(pop)
        strategic[key] = {**info,'distinct':len(pop.entries),'mass':pop.total_mass}
    result = {'configuration':cfg,'selected':selected,'candidates':candidates,'models':models,
        'scores':scores,'regrets':regrets,'summary':summary,'candidate_security':security,
        'strategic_models':strategic,'sensitivity':sensitivity,'sampling_bounds':bounds,
        'phases':phases,'freeze_sha256':freeze_hash,'reference_groups':groups,
        'source_data':source_metadata,'fitted_features':features}
    save(out/'results.json',result)
    save(out/'models.json',models)
    score_rows = [[cid,';'.join(item['roles']),str(tuple(item['allocation']))]+[float(scores[cid][k]) for k in tables] for cid,item in candidates.items()]
    regret_rows = [[cid,';'.join(item['roles']),str(tuple(item['allocation']))]+[float(regrets[cid][k]) for k in tables] for cid,item in candidates.items()]
    write_csv(out/'scores.csv',['id','roles','allocation']+list(tables),score_rows)
    write_csv(out/'regrets.csv',['id','roles','allocation']+list(tables),regret_rows)
    write_csv(out/'candidate-security.csv',['id','allocation','pure_security','counter','counter_score'],
        [[cid,str(tuple(candidates[cid]['allocation'])),float(info['security']),
          str(tuple(info['best_response'].allocation)),float(info['best_response'].expected_score)] for cid,info in security.items()])
    write_csv(out/'solver-sensitivity.csv',['solver_share','id','allocation','mixture_score','optimum_count'],
        [[float(s['solver_share']),s['candidate_id'],str(tuple(s['allocation'])),float(s['mixture_score']),s['optimum_count']] for s in sensitivity])
    write_csv(out/'contamination-stress.csv',['id','scenario','adversarial_share','stressed_score'],
        [[cid,k,float(e),float((1-e)*scores[cid][k]+e*security[cid]['security'])]
         for cid in candidates for k in tables for raw in cfg['contamination_shares'] for e in [Fraction(str(raw))]])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax = plt.subplots(figsize=(15,5.8))
    selected_names = list(selected)
    matrix = [[float(regrets[selected[n]['candidate_id']][k]) for k in tables] for n in selected_names]
    image = ax.imshow(matrix,aspect='auto',cmap='magma_r',vmin=0)
    ax.set_yticks(range(4),[n.replace('_',' ') for n in selected_names])
    short = [k.replace('development_','dev_').replace('validation_','val_').replace('final_audit_','audit_').replace('_fixed_total','').replace('_fixed_per_field','_scaled') for k in tables]
    ax.set_xticks(range(len(tables)),short,rotation=65,ha='right',fontsize=8)
    ax.set_title('Frozen selections: gap from each scenario optimum (points)\nSmaller is better; scenario frequencies among applicants are unknown',pad=15)
    for boundary in (len(phases['development'])-.5,len(phases['development'])+len(phases['validation'])-.5):
        ax.axvline(boundary,color='white',linewidth=2)
    fig.colorbar(image,ax=ax,label='Scenario regret (points)',shrink=.75)
    fig.tight_layout()
    fig.savefig(out/'regret-heatmap.png',dpi=180)
    plt.close(fig)
    render_report(out,result)
    # An immutable freeze boundary is part of acceptance, not just report text.
    assert sha(out/'frozen-candidates.json')==freeze_hash
    assert fingerprints()==initial_sources
    assert freeze['created_unix']<evaluation_started
    manifest = {'started_unix':started,'frozen_unix':freeze['created_unix'],
        'evaluation_started_unix':evaluation_started,'completed_unix':time.time(),
        'elapsed_seconds':time.perf_counter()-clock,'implementation':initial_sources,
        'python':sys.version,'numpy':np.__version__,'platform':platform.platform(),
        'config_sha256':sha(args.config),'freeze_sha256':freeze_hash,
        'artifacts':{str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file()}}
    save(out/'manifest.json',manifest)
    print(f"Completed: {out}; {manifest['elapsed_seconds']:.1f} seconds",flush=True)


if __name__=='__main__':
    main()
