"""Faithful human-strategy comparison, exact coalition models, and learning tests."""
from fractions import Fraction
from itertools import product
from pathlib import Path
import argparse
import json
import shutil
import sys
import time
import numpy as np
from blotto_lab.core import Population,best_response,mix_tables,score,security_of
from blotto_lab.contamination import exact_counter
from blotto_lab.human import (coalition_anchor,coalition_population,joint_metrics,learning_path,
    marginal_counts,minimal_winning_coalitions,ranged_allocations,score_matrices)
from run_equilibrium_benchmark import integer_masses
from run_multiwave_study import ROOT,unpack_table
from run_population_study import encode,save,sha,write_csv


def value(record):return Fraction(record['numerator'],record['denominator'])


def choose_ranges(rows,tables,optima,reference,half):
    records=[]
    for x in rows:
        scores=[t.expected_score(x) for t in tables]
        records.append({'allocation':x,'max_min':min(scores),'minimax_regret':-max(m-s for m,s in zip(optima,scores)),
            'balanced':reference.expected_score(x),'half_strategic':half.expected_score(x)})
    chosen={}
    for criterion in ['max_min','minimax_regret','balanced','half_strategic']:
        maximum=max(row[criterion] for row in records)
        winners=[row['allocation'] for row in records if row[criterion]==maximum]
        chosen[criterion]={'allocation':min(winners),'maximized_value':maximum,'optimum_count':len(winners)}
    return {'count':len(rows),'chosen':chosen,'records':records}


def render_report(out,r):
    fmt=lambda x:f'{float(x):.4f}'
    lines=['**Game 1: human-strategy challenge**','',
        'The five submitted human allocations and the literal 114-vector range were evaluated against '
        'the fixed machine baseline before introducing new human-informed populations. The four-player '
        'ideas remain reserved. New coalition populations and learning paths are declared hypotheses, '
        'not observed applicant frequencies or new independent applicant holdouts.','',
        '**Five literal allocations on previously known models**','',
        '| Human vector | Allocation | Balanced-reference points | Half-strategic-reference points | Minimum across 121 known models | Worst regret | Pure security |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for label in r['human_labels']:
        item=r['known_summary'][label]
        lines.append(f"| {label} | {tuple(r['labels'][label])} | {fmt(item['balanced'])} | {fmt(item['half_strategic'])} | {fmt(item['minimum'])} | {fmt(item['worst_regret'])} | {fmt(item['security'])} |")
    lines += ['', '**Range fitting on known models**','',
        '| Region | Criterion | Chosen allocation | Objective value | Number of tied optima |',
        '| --- | --- | --- | ---: | ---: |']
    for name,region in r['range_results'].items():
        for criterion,chosen in region['chosen'].items():
            v=chosen['maximized_value'] if criterion!='minimax_regret' else -chosen['maximized_value']
            lines.append(f"| {name}, {region['count']} vectors | {criterion} | {tuple(chosen['allocation'])} | {fmt(v)} | {chosen['optimum_count']} |")
    lines += ['', 'The literal ranges are [2,3],[5,7],[22,25],[36,42],[25,30], with exact total 100. '
        'The relaxed region extends every endpoint by two, an analyst construction. These choices '
        'were fitted only to known models, frozen before the new coalition and learning tests, '
        'and retained as exploratory comparisons. They are not globally optimal unrestricted vectors.','',
        '**Your priority coalitions**','',
        'The five named sets {3,5}, {4,5}, {1,2,5}, {1,3,4}, {2,3,4} are exactly all inclusion-minimal '
        'sets whose values exceed 7.5. Coalition membership does not fix bid sizes or establish '
        'dominance over another coalition. Scouting bids and ties at otherwise abandoned fields '
        'can add points outside the priority set.','',
        'The later literal range scores exactly five points against (4,19,27,48,2): all four first '
        'fields lose, and castle 5 wins. Its 22-25 troops at castle 3 do not beat 27. The earlier '
        '(0,10,30,30,30) instead wins castles 3 and 5 for eight points. Every vector in the literal '
        'range has pure security exactly two, with a saved exact counter.','',
        '**Exact new population interpretations**','',
        'There are 48 variants: five priority sets; scouts 0,1,3; uniform positive composition or '
        'value-weighted multinomial allocation; and separate five-heavy conditioning for the three '
        'sets containing castle 5. Target bids are positive and all legal joint supports are saved '
        'with exact integer masses. Requiring 50 or more troops at castle 5 is tested as a separate '
        'condition, not assumed for every priority-coalition player.','',
        '| Human vector | Open-coalition reference: points / win credit | Five-heavy reference: points / win credit | Favorite-pair reference: points / win credit |',
        '| --- | ---: | ---: | ---: |']
    for label in r['human_labels']:
        cid=r['label_ids'][label];cells=[]
        for name in ['all_open','all_heavy5','favorites_open']:
            s=r['reference_scores'][name][cid]
            cells.append(fmt(s['points'])+' / '+fmt(s['win_tie_adjusted']))
        lines.append(f'| {label} | '+' | '.join(cells)+' |')
    lines += ['', 'Win credit is win probability plus half the tie probability. Each reference gives '
        'equal weight to its specified priority sets, laws, and scout levels. These are transparent '
        'comparison conventions, not inferred prevalence. The two-favorite reference includes '
        '{3,5} and {1,3,4} equally. Full primary-machine comparisons and all exploratory candidates '
        'appear in the saved tables.','',
        '**Castle-5 bid-gap and conditional tests**','',
        '| Reference | Probability of bid 26-40 at castle 5 | Probability of bid at least 50 | Illustration: probability of losing castle 5 by at least 20 | Points conditional on that loss |',
        '| --- | ---: | ---: | ---: | ---: |']
    for name,ref in r['references'].items():
        event=r['illustration_events'][name]
        conditional=event['lose5_by20_conditional_points']
        lines.append(f"| {name} | {fmt(ref['fifth_bins']['26_to40'])} | {fmt(ref['fifth_bins']['50_plus'])} | {fmt(event['lose5_by20_probability'])} | {fmt(conditional) if conditional is not None else 'undefined'} |")
    lines += ['', 'The illustration is (2,6,24,40,28), not an allocation explicitly chosen by the users. '
        'Losing castle 5 by a large margin spends opponents\' troops elsewhere, but the remaining '
        'points depend on their legal joint allocations. No conditional victory follows solely '
        'from the margin. Exact conditional statistics are saved. Marginal bid masses alone suffice '
        'for official mean points; win rates and conditional outcomes need joint supports.','',
        '**Population frequencies and learning**','',
        'The replicator mechanism makes a strategy grow relative to its current share when it '
        'outperforms the population average. It does not set its eventual share proportional to '
        'its current payoff. In the unmutated replicator model, used strategies at a rest point '
        'have equal payoff. That is not enough for full Nash equilibrium if a missing strategy can '
        'invade. With mutation, the stationarity condition also includes mutation flows. '
        'See [Cressman and Tao, equation 1](https://pmc.ncbi.nlm.nih.gov/articles/PMC4113915/).','',
        'The discrete replicator update tested here is p_next(i)=p(i)*fitness(i)/mean_fitness. '
        'The literal replacement hypothesis is p_next(i)=fitness(i)/sum(fitness). They coincide '
        'on the first update from uniform shares, then generally differ. For constant equal '
        'payoffs the replicator preserves unequal shares while replacement makes them uniform. '
        'These are separate learning assumptions, not estimates of how applicants revise choices.','',
        'The same finite legal pool and starting conditions are tested with points fitness or '
        'binary-win fitness, both update rules, and zero or 1% mutation: 16 paths of 1,000 updates. '
        'Final and time-average mixtures are quantized to integer masses and checked by the full-space '
        'official-points oracle. Restricted win exploitability is measured only within this pool. '
        'No global win-probability best response or convergence is claimed.','',
        '| Learning rule | Fitness | Initial state | Mutation | Final official-points exploitability | Average official-points exploitability |',
        '| --- | --- | --- | ---: | ---: | ---: |']
    for name,p in r['learning'].items():
        lines.append(f"| {p['rule']} | {p['fitness']} | {p['initial']} | {p['mutation']:.0%} | {fmt(p['final']['certificate']['exploitability'])} | {fmt(p['average']['certificate']['exploitability'])} |")
    lines += ['', 'Learning weights are numerical simulation results; the subsequent saved-mixture point '
        'certificates are rational. The pool contains human and machine hypotheses, so the paths '
        'are adaptive stress constructions, not independent applicant samples. Population-average '
        'points remain 7.5 and population-average win credit remains 0.5.','',
        '**Artifacts and reproduction**','',
        '- `phase1-freeze.json`: human range choices and candidate pool, saved before new models.',
        '- `known-summary.json`, `primary-head-to-head.csv`: literal and fixed-primary comparisons.',
        '- `populations/`, `models.json`, `joint-metrics.json`: exact legal supports, tables, and win/conditional outcomes.',
        '- `reference-scores.csv`, `local-transfers.csv`, `fifth-bins.csv`: point/win comparisons and marginal hypotheses.',
        '- `learning/`: 16 paths, exact quantized supports, float trajectories, and certificates.',
        '- `results.json`, `manifest.json`, `source-snapshot/`: all definitions, hashes, and exact results.',
        '- `acceptance.json`: independent verification after the experiment.',
        '', 'Run `python3 run_human_challenge.py --output results/human-challenge-new` from the project '
        'folder. Existing output is preserved. Verify with `python3 verify_human_challenge.py`. '
        'A future redesign informed by this stage needs a subsequent fresh challenge.','',
        '![Points and match-win credit](points-versus-wins.png)','',
        '![Learning-rule sensitivity](learning-sensitivity.png)','']
    (out/'report.md').write_text('\n'.join(lines))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,default=ROOT/'config/human-challenge.json')
    parser.add_argument('--output',type=Path,default=ROOT/'results/human-challenge')
    args=parser.parse_args();cfg=json.loads(args.config.read_text());out=args.output.resolve()
    if out.exists():raise SystemExit('Output exists. Use a new output folder.')
    out.mkdir(parents=True);start=time.perf_counter()
    sources=[Path(__file__),ROOT/'blotto_lab/human.py',ROOT/'blotto_lab/core.py',ROOT/'blotto_lab/contamination.py',
        ROOT/'run_equilibrium_benchmark.py',ROOT/'run_population_study.py',ROOT/'run_multiwave_study.py',ROOT/'HUMAN-CHALLENGE-PROTOCOL.md']
    input_paths=[ROOT/'results/game1-machine-checkpoint-v1.json',ROOT/'results/contamination-study/results.json',ROOT/'results/equilibrium-benchmark/mixture.json']
    signature={'configuration':cfg,'config_sha256':sha(args.config),
        'source_hashes':{str(p.relative_to(ROOT)):sha(p) for p in sources},'input_hashes':{str(p.relative_to(ROOT)):sha(p) for p in input_paths}}
    save(out/'run-inputs.json',signature)
    for p in sources:
        dest=out/'source-snapshot'/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
    baseline=json.loads(input_paths[0].read_text())
    for path,digest in baseline['source_artifact_sha256'].items():assert sha(ROOT/path)==digest,path
    old=json.loads(input_paths[1].read_text());known_order=old['known_model_order']
    known_tables=[unpack_table(old['known_models'][k],k) for k in known_order]
    optima=[best_response(t).expected_score for t in known_tables]
    reference=unpack_table(old['reference_models']['ref_balanced_s0'],'balanced known reference')
    half=unpack_table(old['reference_models']['ref_balanced_s0.50'],'half strategic known reference')
    literal={k:tuple(x) for k,x in cfg['literal_vectors'].items()}
    labels=dict(literal);labels.update({'M_'+k:tuple(x) for k,x in baseline['primary_comparison_vectors'].items()})
    labels['R_illustration']=tuple(cfg['illustration'])
    print('Testing literal human inputs and fitting ranges only on known models',flush=True)
    region_rows={'literal':ranged_allocations(cfg['literal_ranges']),
                 'relaxed_by_two':ranged_allocations(cfg['literal_ranges'],cfg['relaxation_radius'])}
    ranges={name:choose_ranges(rows,known_tables,optima,reference,half) for name,rows in region_rows.items()}
    for name,r in ranges.items():
        for criterion,p in r['chosen'].items():labels[f'{name}_{criterion}']=p['allocation']
    known_summary={}
    for label,x in labels.items():
        scores=[t.expected_score(x) for t in known_tables];counter,security=exact_counter(x)
        known_summary[label]={'minimum':min(scores),'worst_regret':max(m-s for m,s in zip(optima,scores)),
            'balanced':reference.expected_score(x),'half_strategic':half.expected_score(x),'counter':counter,'security':security}
    save(out/'known-summary.json',known_summary)
    write_csv(out/'primary-head-to-head.csv',['candidate']+list(labels),
        [[a]+[float(score(x,y)) for y in labels.values()] for a,x in labels.items()])
    coalitions=[tuple(i-1 for i in c) for c in cfg['coalitions']]
    assert set(coalitions)==set(minimal_winning_coalitions())
    source_actions={'machine':[tuple(c['allocation']) for c in baseline['exploratory_vectors']],
        'human_literals':list(literal.values()),'human_ranges':list(region_rows['literal'])+
            [p['allocation'] for r in ranges.values() for p in r['chosen'].values()],
        'coalition_anchors':[coalition_anchor(c,s) for c in coalitions for s in cfg['scout_levels']]}
    actions=sorted(set(x for group in source_actions.values() for x in group)|set(labels.values()))
    ids={x:f'v{i:03}' for i,x in enumerate(actions)};label_ids={k:ids[x] for k,x in labels.items()}
    freeze={'created_unix':time.time(),'labels':labels,'range_results':ranges,'pool_actions':actions,
        'source_action_groups':source_actions,'baseline_sha256':sha(input_paths[0]),'run_inputs_sha256':sha(out/'run-inputs.json')}
    save(out/'phase1-freeze.json',freeze);freeze_hash=sha(out/'phase1-freeze.json')
    new_models_started=time.time();print(f'FROZEN {len(actions)} candidate pool; {len(region_rows["literal"])} literal range vectors',flush=True)
    models,metrics,populations={},{},{}
    (out/'populations').mkdir()
    lookup={}
    for coalition in coalitions:
        for scout in cfg['scout_levels']:
            for law in cfg['allocation_laws']:
                for heavy in ([False,True] if 4 in coalition else [False]):
                    p=coalition_population(coalition,scout,law,cfg['heavy_five_minimum'] if heavy else None)
                    key=p.name;populations[key]=p;lookup[(coalition,scout,law,heavy)]=key
                    table=p.table();br=best_response(table)
                    model={'coalition':[i+1 for i in coalition],'scout':scout,'law':law,'heavy5':heavy,
                        'support_size':len(p.entries),'total_mass':p.total_mass,'population_file':f'populations/{key}.json',
                        'table':{'numerators':table.numerators,'denominator':table.denominator,'evidence':table.evidence},
                        'best_response':{'allocation':br.allocation,'score':br.expected_score,'optimum_count':br.optimum_count}}
                    save(out/model['population_file'],p.to_dict());models[key]=model
                    measured=joint_metrics(actions,p,set(labels.values()))
                    metrics[key]={ids[x]:{'points':table.expected_score(x),**m} for x,m in measured.items()}
                    for x in set(labels.values()):assert measured[x]['joint_mean_points']==table.expected_score(x)
                    if len(models)%8==0:print(f'Exact coalition models: {len(models)}/48',flush=True)
    favorites={tuple(i-1 for i in c) for c in cfg['favorite_coalitions']}
    ref_keys={}
    for name,chosen_coalitions,heavy in [('all_open',coalitions,False),('all_heavy5',coalitions,True),
            ('favorites_open',[c for c in coalitions if c in favorites],False),('favorites_heavy5',[c for c in coalitions if c in favorites],True)]:
        ref_keys[name]=[lookup[(c,s,law,heavy and 4 in c)] for c in chosen_coalitions for s in cfg['scout_levels'] for law in cfg['allocation_laws']]
    references,reference_scores,illustration_events={},{},{}
    for name,keys in ref_keys.items():
        table=mix_tables([populations[k].table() for k in keys],[1]*len(keys),name=name)
        counts,den=marginal_counts(table,4)
        bins={'under25':(0,24),'at25':(25,25),'26_to40':(26,40),'41_to49':(41,49),'50_plus':(50,100)}
        references[name]={'model_keys':keys,'fifth_bins':{k:Fraction(sum(counts[a:b+1]),den) for k,(a,b) in bins.items()},
            'table':{'numerators':table.numerators,'denominator':table.denominator,'evidence':table.evidence}}
        reference_scores[name]={cid:{'points':table.expected_score(x),
            'win_tie_adjusted':sum(metrics[k][cid]['win_tie_adjusted'] for k in keys)/len(keys),
            'win_probability':sum(metrics[k][cid]['win_probability'] for k in keys)/len(keys)} for x,cid in ids.items()}
        illustration=ids[tuple(cfg['illustration'])];event={}
        for label in ['lose5','lose5_by20']:
            probability=sum(metrics[k][illustration][label+'_probability'] for k in keys)/len(keys)
            point_mass=sum(metrics[k][illustration][label+'_points_mass'] for k in keys)/len(keys)
            wins_mass=sum(metrics[k][illustration][label+'_win2plus234_mass'] for k in keys)/len(keys)
            event.update({label+'_probability':probability,label+'_conditional_points':point_mass/probability if probability else None,
                          label+'_conditional_win2plus234':wins_mass/probability if probability else None})
        illustration_events[name]=event
    write_csv(out/'reference-scores.csv',['reference','candidate_id','labels','allocation','mean_points','win_credit','win_probability'],
        [[name,cid,';'.join(k for k,v in label_ids.items() if v==cid),str(actions[int(cid[1:])]),float(s['points']),float(s['win_tie_adjusted']),float(s['win_probability'])]
         for name,scores in reference_scores.items() for cid,s in scores.items()])
    write_csv(out/'fifth-bins.csv',['reference','under25','at25','26_to40','41_to49','50_plus'],
        [[name]+[float(r['fifth_bins'][k]) for k in ['under25','at25','26_to40','41_to49','50_plus']] for name,r in references.items()])
    transfer_rows=[]
    for name,ref in references.items():
        table=unpack_table(ref,name)
        for label,x in labels.items():
            for donor in range(5):
                for receiver in range(5):
                    if donor==receiver:continue
                    for amount in [1,3]:
                        if x[donor]<amount:continue
                        y=list(x);y[donor]-=amount;y[receiver]+=amount
                        delta=table.expected_score(y)-table.expected_score(x)
                        transfer_rows.append([name,label,donor+1,receiver+1,amount,float(delta)])
    write_csv(out/'local-transfers.csv',['reference','candidate','from_castle','to_castle','troops','exact_delta_decimal'],transfer_rows)
    points2,wins2=score_matrices(actions)
    initial={};initial['uniform_pool']=np.full(len(actions),1/len(actions))
    source_prior=np.zeros(len(actions))
    for group in source_actions.values():
        unique=sorted(set(group))
        for x in unique:source_prior[int(ids[x][1:])]+=1/(len(source_actions)*len(unique))
    # The illustration is explanatory rather than a source prior. Give any otherwise
    # uncovered label a small declared seed mass, then normalize.
    missing=np.flatnonzero(source_prior==0)
    source_prior[missing]=1/(1000*len(actions));source_prior/=source_prior.sum()
    initial['balanced_sources']=source_prior
    learning={};(out/'learning').mkdir()
    print('Comparing point learning, mistaken win learning, and two revision rules',flush=True)
    for rule,fitness,initial_name,mutation in product(cfg['learning_rules'],cfg['fitness_objectives'],cfg['initial_conditions'],cfg['mutation_rates']):
        key=f'{rule}_{fitness}_{initial_name}_mu{mutation}'
        matrix=points2/2 if fitness=='points' else wins2/2
        path=learning_path(matrix,initial[initial_name],rule,mutation,cfg['learning_steps'],cfg['record_every'])
        np.savez_compressed(out/'learning'/f'{key}.npz',initial=initial[initial_name],final=path['final'],average=path['average'])
        entry={'rule':rule,'fitness':fitness,'initial':initial_name,'mutation':mutation,'trace':path['trace']}
        for phase in ['final','average']:
            masses=integer_masses(path[phase],cfg['quantization_mass'])
            population=Population.from_counts([(x,m) for x,m in zip(actions,masses) if m],name=key+'_'+phase)
            info=security_of(population)
            win_values=wins2.astype(np.int64)@np.array(masses,dtype=np.int64)
            point_values=points2.astype(np.int64)@np.array(masses,dtype=np.int64)
            entry[phase]={'population':population.to_dict(),'certificate':info,
                'restricted_point_exploitability':Fraction(int(point_values.max()),2*population.total_mass)-Fraction(15,2),
                'restricted_win_exploitability':Fraction(int(win_values.max()),2*population.total_mass)-Fraction(1,2),
                'human_points':{label:population.table().expected_score(x) for label,x in literal.items()},
                'quantization_l1':float(np.abs(np.array(masses)/sum(masses)-path[phase]).sum())}
        save(out/'learning'/f'{key}.json',entry);learning[key]=entry
        print(f"{key}: exact final point gap {float(entry['final']['certificate']['exploitability']):.4f}",flush=True)
    # A concrete payoff-ranking reversal on the five literal example opponents.
    five=Population.from_people(literal.values(),name='five human iterations, illustrative only')
    comparisons=joint_metrics(list(labels.values()),five)
    reversal=[]
    for a,x in labels.items():
        if a.startswith(('literal_','relaxed_','R_')):continue
        for b,y in labels.items():
            if b.startswith(('literal_','relaxed_','R_')):continue
            pa,pb=five.table().expected_score(x),five.table().expected_score(y)
            wa,wb=comparisons[x]['win_tie_adjusted'],comparisons[y]['win_tie_adjusted']
            if pa<pb and wa>wb:reversal.append({'higher_win_label':a,'higher_points_label':b,'win_side_points':pa,'points_side_points':pb,'win_side_credit':wa,'points_side_credit':wb})
    reversal=sorted(reversal,key=lambda r:(r['points_side_points']-r['win_side_points'],r['win_side_credit']-r['points_side_credit'],r['higher_win_label'],r['higher_points_label']),reverse=True)
    q=Population.from_dict(json.loads(input_paths[2].read_text()));qt=q.table()
    equilibrium={'support_size':len(q.entries),'smallest_share':Fraction(min(m for x,m in q.entries),q.total_mass),
        'largest_share':Fraction(max(m for x,m in q.entries),q.total_mass),
        'smallest_active_points':min(qt.expected_score(x) for x,m in q.entries),'largest_active_points':max(qt.expected_score(x) for x,m in q.entries)}
    result={'configuration':cfg,'human_labels':list(literal),'labels':labels,'label_ids':label_ids,
        'pool_actions':actions,'source_action_groups':source_actions,'known_summary':known_summary,'range_results':ranges,
        'models':models,'references':references,'reference_scores':reference_scores,'illustration_events':illustration_events,
        'learning':learning,'ranking_reversal':reversal[0] if reversal else None,'equilibrium_example':equilibrium,
        'phase1_freeze_sha256':freeze_hash,'phase1_frozen_unix':freeze['created_unix'],'new_models_started_unix':new_models_started,
        'baseline_sha256':sha(input_paths[0]),'scope':'Human-informed exact hypotheses and adaptive stress tests, not applicant observations or final submissions'}
    save(out/'models.json',models);save(out/'joint-metrics.json',metrics);save(out/'results.json',result)
    plots(out,result);render_report(out,result)
    assert sha(out/'phase1-freeze.json')==freeze_hash and freeze['created_unix']<new_models_started
    for p in sources:assert sha(p)==signature['source_hashes'][str(p.relative_to(ROOT))]
    save(out/'manifest.json',{'signature':signature,'phase1_freeze_sha256':freeze_hash,
        'completed_unix':time.time(),'elapsed_seconds':time.perf_counter()-start,'python':sys.version,'numpy':np.__version__,
        'artifacts':{str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='manifest.json'}})
    print(f'Completed {len(models)} exact coalition variants and {len(learning)} learning paths.',flush=True)


def plots(out,r):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels=list(r['human_labels'])+['M_known_model_max_min','M_known_model_minimax_regret','M_minimax_regret_security_floor_two']
    fig,axes=plt.subplots(1,2,figsize=(11,4.7))
    for ax,name in zip(axes,['all_open','all_heavy5']):
        for label in labels:
            item=r['reference_scores'][name][r['label_ids'][label]]
            ax.scatter(float(item['win_tie_adjusted']),float(item['points']),color='#286182' if label.startswith('H') else '#ad542c')
            short=label if label.startswith('H') else {'M_known_model_max_min':'M score','M_known_model_minimax_regret':'M regret','M_minimax_regret_security_floor_two':'M protected'}[label]
            offset=(-6,-12) if label=='M_minimax_regret_security_floor_two' else ((-6,7) if label.startswith('M_') else (4,5))
            ax.annotate(short,(float(item['win_tie_adjusted']),float(item['points'])),xytext=offset,textcoords='offset points',fontsize=8,ha='right' if label.startswith('M_') else 'left')
        ax.margins(x=.20,y=.15)
        ax.set_title(name.replace('_',' '));ax.set_xlabel('Match-win credit (win + half tie probability)');ax.grid(alpha=.2)
    axes[0].set_ylabel('Mean official points')
    fig.suptitle('Declared coalition populations: winning more often and earning more points differ')
    fig.tight_layout();fig.savefig(out/'points-versus-wins.png',dpi=180);plt.close(fig)
    fig=plt.figure(figsize=(12,4.8))
    grid=fig.add_gridspec(1,3,width_ratios=[1,1,.055])
    axes=[fig.add_subplot(grid[0,0]),fig.add_subplot(grid[0,1])]
    cax=fig.add_subplot(grid[0,2])
    rows=[(rule,f) for rule in r['configuration']['learning_rules'] for f in r['configuration']['fitness_objectives']]
    columns=[(i,m) for i in r['configuration']['initial_conditions'] for m in r['configuration']['mutation_rates']]
    maximum=max(float(p[phase]['certificate']['exploitability']) for p in r['learning'].values() for phase in ['final','average'])
    for ax,phase in zip(axes,['final','average']):
        matrix=[]
        for rule,f in rows:
            matrix.append([float(next(p for p in r['learning'].values() if p['rule']==rule and p['fitness']==f and p['initial']==i and p['mutation']==m)[phase]['certificate']['exploitability']) for i,m in columns])
        im=ax.imshow(matrix,cmap='YlOrRd',vmin=0,vmax=maximum,aspect='auto')
        ax.set_xticks(range(4),[('uniform' if i=='uniform_pool' else 'source-balanced')+'\n'+f'mutation {m:.0%}' for i,m in columns],fontsize=7)
        ax.set_yticks(range(4),[rule.replace('_',' ')+' / '+f for rule,f in rows] if phase=='final' else ['']*4,fontsize=8)
        for i,row in enumerate(matrix):
            for j,v in enumerate(row):ax.text(j,i,f'{v:.2f}',ha='center',va='center',color='white' if v>maximum*.65 else 'black')
        ax.set_title(phase+' mixture')
    fig.suptitle('Exact official-point exploitability after learning; smaller is better')
    fig.colorbar(im,cax=cax,label='Full-space point exploitability')
    fig.subplots_adjust(left=.20,right=.92,bottom=.22,wspace=.15,top=.84)
    fig.savefig(out/'learning-sensitivity.png',dpi=180);plt.close(fig)


if __name__=='__main__':main()
