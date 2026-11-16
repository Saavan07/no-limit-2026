"""Complete-action weighting ablations and exact adversarial-security tradeoffs."""
from fractions import Fraction
from pathlib import Path
import argparse
import json
import platform
import shutil
import sys
import time
import numpy as np
from blotto_lab.core import Population, best_response, mix_tables
from blotto_lab.contamination import (FrontierObjective, continuous_envelope, exact_counter,
    search_frontiers, select_epsilon)
from run_multiwave_study import ROOT, unpack_table
from run_population_study import save, sha, write_csv

BEHAVIOR_GROUPS = ['analytic','historical','anchors','support_rounding','new_reasoning']
STRATEGIC_GROUPS = ['legacy_strategic','adaptive_generations','adaptive_archives']


def nominal_winner(frontier,minimum_security=Fraction(0)):
    allowed = [p for p in frontier.values() if p['security']>=minimum_security]
    value = max(p['nominal_value'] for p in allowed)
    winners = [p for p in allowed if p['nominal_value']==value]
    return {'allocation':min(p['allocation'] for p in winners),'value':value,
            'optimum_count':sum(p['optimum_count'] for p in winners)}


def render_report(out,result):
    f = lambda x:f'{float(x):.4f}'
    lines = ['**Game 1: exact weighting and adversarial-contamination frontiers**','',
        f"This controlled sensitivity study uses all {len(result['known_models'])} previously examined "
        f"non-equilibrium scenarios, {len(result['reference_models'])} declared mixture references, "
        f"and {len(result['objectives'])} objectives. It searches every legal allocation conditional "
        'on its exact pure-vector security. Previous evaluation sets are known evidence here. '
        'There is no new holdout or applicant-population validation claim. Human ideas remain unused.','',
        '**Exact security classes across all legal allocations**','',
        '| Guaranteed score against any legal opponent | Number of fixed vectors | Lexicographically first vector |',
        '| --- | ---: | --- |']
    for c,count in sorted(result['class_counts'].items()):
        lines.append(f"| {c/2:g} | {count:,} | {tuple(result['class_first'][c])} |")
    lines += ['', 'Guarantees are exact for the integer, five-castle game, including ties. The largest '
        'pure-vector guarantee is five points. A randomized mixture can guarantee essentially 7.5; '
        'that protection does not transfer to a fixed member.','',
        '**Ablations on a common comparison set**','',
        '| Nominal set | Objective | Chosen vector | Objective on that set | Minimum score on all known models | Worst regret on all known models | Pure security |',
        '| --- | --- | --- | ---: | ---: | ---: | ---: |']
    for name,item in result['ablations'].items():
        value = item['nominal_value'] if item['mode']=='score' else -item['nominal_value']
        lines.append(f"| {item['set']} | {item['mode']} | {tuple(item['allocation'])} | {f(value)} | "
            f"{f(item['common_minimum'])} | {f(item['common_worst_regret'])} | {f(item['security'])} |")
    lines += ['', 'Every row is globally optimal for its stated set and objective. Removing scenarios '
        'changes the problem, so the common-set columns show what that choice gives up. Repeated '
        'copies of a scenario do not change a max-min or minimax-regret objective. Frequency weights '
        'do change a declared expected-score reference. Neither operation estimates real prevalence.','',
        '**Protection against a specified adversarial share**','',
        'Let W(x) be a fixed vector\'s minimum score against any legal opponent and m(x) its smallest '
        'score across the chosen nominal scenarios. If at most epsilon of the population is arbitrary '
        'and the rest is any mixture of those nominal scenarios, the exact worst-case modeled score is '
        '(1-epsilon)*m(x)+epsilon*W(x). The arbitrary part can be selected against the fixed vector. '
        'This is a deliberately severe uncertainty model, not a forecast that applicants will know '
        'our submission. The nominal component must still lie in the stated modeled set.','',
        '| Adversarial share | Globally optimal vector, all-known nominal set | Guaranteed contaminated-model score | Pure security | Number of tied optima |',
        '| --- | --- | ---: | ---: | ---: |']
    for row in result['epsilon_grid']['all_known_score']:
        lines.append(f"| {float(row['epsilon']):.0%} | {tuple(row['allocation'])} | {f(row['value'])} | {f(row['security'])} | {row['optimum_count']:,} |")
    lines += ['', 'The result covers every epsilon in [0,1], not only these displayed settings. Exact '
        'rational transition points, open-interval winners, and all boundary ties are in '
        '`continuous-envelopes.json`. At epsilon=1, nominal performance has zero weight, so every '
        'maximum-security vector is counted as a tie.','',
        '**Declared priors and security floors**','',
        'Five behavior groups form four comparison conventions: balanced, historical-emphasis, '
        'anchor-emphasis, and no analytic controls. Strategic mass varies from zero to one. Its '
        'reference balances legacy strategic models, adaptive generations, and adaptive archives, '
        'then weights models equally inside each group. The complete definitions are saved. '
        'These are hypotheses for sensitivity, not estimates of applicant composition. Each has '
        'its own complete adversarial-contamination frontier.','',
        'Nominal minimax regret under a pure-security floor is saved separately in '
        '`security-floors.csv`. We do not label a blend of nominal regret and worst-case score '
        'as minimax regret for a contaminated game: that game would have a different benchmark.','',
        '**Why the global frontier is computable**','',
        'If the opponent loses a set L, replacing those bids with zero releases sum_L x_i soldiers. '
        'Setting every other field to a tie then costs B-sum_L x_i. Each tie can become a win for '
        'one more soldier, so the opponent wins the most valuable remaining fields first. This '
        'enumerates the opponent\'s exact best outcome across feasible losing sets, with all-tie '
        'as a fallback. Counter witnesses are retained.','',
        'For every attainable security class c, the study finds the exact largest nominal score '
        'F(c). The contaminated objective then becomes the largest of finitely many straight lines '
        '(1-epsilon)*F(c)+epsilon*c. This proves global pure-vector optimality under the stated '
        'model and gives the continuous switch points. Nominal selection uses bounded floating '
        'screening followed by rational rescoring. An independent verifier uses integer upper '
        'bounds and a separate loss/tie/win pattern enumeration.','',
        '**Limits and reproducibility**','',
        'All results condition on the existing saved scenario tables. Earlier sample bounds do '
        'not validate transfer assumptions or applicant frequencies. There is no new sampling '
        'claim in this study. The original reports and artifacts remain intact. No final application '
        'vector has been chosen.','',
        '- `frontiers.json`: exact optimum, vector, and tie count in every security class.',
        '- `continuous-envelopes.json`: exact epsilon intervals and boundary ties.',
        '- `ablations.csv`: nominal choices and their common-set costs.',
        '- `security-floors.csv`: optimal nominal objectives subject to a security floor.',
        '- `epsilon-grid.csv`, `scores.csv`, `regrets.csv`: all displayed choices and comparisons.',
        '- `security-classes.npz`: ordered exact security class for every legal allocation.',
        '- `results.json`, `source-snapshot/`, `manifest.json`: complete definitions, rational data, source, and hashes.',
        '- `acceptance.json`: independent all-action and counter verification after the study.',
        '', 'Run from the project folder: `python3 run_contamination_study.py --output results/contamination-new`. '
        'Existing folders are rejected. Verify the delivered study with '
        '`python3 verify_contamination_study.py`.','',
        '![Nominal score versus pure security](security-frontier.png)','',
        '![Exact all-known contamination choices](contamination-switches.png)','']
    (out/'report.md').write_text('\n'.join(lines))


def plots(out,result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax = plt.subplots(figsize=(9,5))
    for name,label in [('all_known_score','Worst score: all known models'),('behavior_only_score','Worst score: behavior models'),
                       ('ref_balanced_s0','Expected score: balanced behavior reference'),('ref_balanced_s0.50','Expected score: half strategic reference')]:
        frontier = result['frontiers'][name]
        xs = [float(p['security']) for c,p in sorted(frontier.items())]
        ys = [float(p['nominal_value']) for c,p in sorted(frontier.items())]
        ax.plot(xs,ys,marker='o',label=label)
    ax.set_xlabel('Exact score guaranteed against any legal opponent')
    ax.set_ylabel('Best nominal score within that exact security class')
    ax.set_title('Fixed-vector performance and protection\nEach curve defines a different nominal population objective')
    ax.grid(alpha=.2)
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out/'security-frontier.png',dpi=180); plt.close(fig)
    fig,axes = plt.subplots(1,2,figsize=(11,4.5))
    envelope = result['envelopes']['all_known_score']
    grid = np.linspace(0,1,501)
    frontier = result['frontiers']['all_known_score']
    values,security = [],[]
    for e in grid:
        chosen = select_epsilon(frontier,Fraction(str(float(e))),result['class_counts'],result['class_first'])
        values.append(float(chosen['value'])); security.append(float(chosen['security']))
    axes[0].plot(grid,values,color='#285b81')
    axes[0].set_ylabel('Optimal guaranteed contaminated-model score')
    axes[1].step(grid,security,where='post',color='#a04d2a')
    axes[1].set_ylabel('Pure security of selected fixed vector')
    for ax in axes:
        ax.set_xlabel('Allowed arbitrary-opponent share')
        ax.grid(alpha=.2)
    fig.suptitle('All-known nominal set: exact global contamination tradeoff')
    fig.tight_layout(); fig.savefig(out/'contamination-switches.png',dpi=180); plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,default=ROOT/'config/contamination-study.json')
    parser.add_argument('--output',type=Path,default=ROOT/'results/contamination-study')
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    out = args.output.resolve()
    if out.exists():
        raise SystemExit('Output exists. Choose a new output folder.')
    out.mkdir(parents=True)
    start = time.perf_counter()
    source = [Path(__file__),ROOT/'blotto_lab/contamination.py',ROOT/'blotto_lab/core.py',ROOT/'blotto_lab/robust.py',
              ROOT/'run_multiwave_study.py',ROOT/'run_population_study.py',ROOT/'CONTAMINATION-PROTOCOL.md']
    inputs = [ROOT/'results/multiwave-study/results.json',ROOT/'results/multiwave-study/frozen-candidates.json']
    signature = {'configuration':cfg,'configuration_sha256':sha(args.config),
        'source_hashes':{str(p.relative_to(ROOT)):sha(p) for p in source},
        'input_hashes':{str(p.relative_to(ROOT)):sha(p) for p in inputs}}
    save(out/'run-inputs.json',signature)
    for p in source:
        target = out/'source-snapshot'/p.relative_to(ROOT)
        target.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(p,target)
    old = json.loads(inputs[0].read_text())
    known = {k:m for k,m in old['models'].items() if m['phase']!='diagnostic'}
    tables = [unpack_table(m,k) for k,m in known.items()]
    known_order = list(known)
    group_keys = {g:[] for g in BEHAVIOR_GROUPS+STRATEGIC_GROUPS}
    for key,m in known.items():
        if m['kind']=='legacy_model':
            group = m['group'] if m['group']!='strategic' else 'legacy_strategic'
        elif m['kind']=='direct_generator': group = 'new_reasoning'
        elif m['kind']=='adaptive_generation': group = 'adaptive_generations'
        elif m['kind']=='adaptive_archive': group = 'adaptive_archives'
        else: raise ValueError('Unexpected model kind')
        group_keys[group].append(key)
    behavior_keys = [k for g in BEHAVIOR_GROUPS for k in group_keys[g]]
    old_archive = 'legacy__history_archive'
    sets = {'all_known':known_order,'all_without_old_archive':[k for k in known_order if k!=old_archive],
        'behavior_only':behavior_keys,'previous_development':old['development_model_order'],
        'previous_without_old_archive':[k for k in old['development_model_order'] if k!=old_archive]}
    for group,keys in group_keys.items():
        sets['without_'+group] = [k for k in known_order if k not in set(keys)]
    objectives = [FrontierObjective(name+'_'+mode,tuple(known_order.index(k) for k in keys),mode)
                  for name,keys in sets.items() for mode in ['score','regret']]
    group_tables = {g:mix_tables([tables[known_order.index(k)] for k in keys],[1]*len(keys),name=g) for g,keys in group_keys.items()}
    strategic = mix_tables([group_tables[g] for g in STRATEGIC_GROUPS],[1]*len(STRATEGIC_GROUPS))
    references = {}
    for ref,weights in cfg['behavior_reference_weights'].items():
        behavior = mix_tables([group_tables[g] for g in BEHAVIOR_GROUPS],weights)
        for raw in cfg['solver_shares']:
            s = Fraction(raw)
            name = 'ref_'+ref+'_s'+raw
            table = mix_tables([behavior,strategic],[1-s,s],name=name)
            references[name] = {'behavior_group_weights':weights,'strategic_share':s,
                'table':{'numerators':table.numerators,'denominator':table.denominator,'evidence':table.evidence}}
            index = len(tables); tables.append(table)
            objectives.append(FrontierObjective(name,(index,),'score'))
    save(out/'objective-definitions.json',{'known_model_order':known_order,'groups':group_keys,'sets':sets,
        'reference_models':references,'objectives':objectives})
    print(f'Searching {len(objectives)} conditional objectives over all legal allocations',flush=True)
    def progress(done,total):
        if done%1000000==0 or done==total: print(f'{done:,}/{total:,}',flush=True)
    searched = search_frontiers(tables,objectives,cfg['batch_size'],progress)
    frontiers,counts,first = searched['frontiers'],searched['class_counts'],searched['class_first']
    assert sum(counts.values())==4598126 and max(counts)==10
    np.savez_compressed(out/'security-classes.npz',security_twice=searched['security_twice'])
    envelopes = {obj.name:continuous_envelope(frontiers[obj.name],counts,first) for obj in objectives if obj.mode=='score'}
    epsilon_grid = {obj.name:[{'epsilon':Fraction(e),**select_epsilon(frontiers[obj.name],Fraction(e),counts,first)}
                             for e in cfg['epsilon_grid']] for obj in objectives if obj.mode=='score'}
    candidates = {}
    def candidate(x,role):
        x = tuple(x)
        for cid,item in candidates.items():
            if item['allocation']==x:
                if role not in item['roles']: item['roles'].append(role)
                return cid
        cid = f'c{len(candidates):03}'
        candidates[cid] = {'allocation':x,'roles':[role]}; return cid
    for cid,item in old['candidates'].items(): candidate(item['allocation'],'previous_'+cid)
    for name,entries in frontiers.items():
        for c,p in entries.items(): candidate(p['allocation'],f'{name}_security_{c}/2')
    for name,entries in epsilon_grid.items():
        for row in entries: row['candidate_id'] = candidate(row['allocation'],f'{name}_epsilon_{row["epsilon"]}')
    optima = [best_response(t).expected_score for t in tables]
    scores = {cid:{k:tables[i].expected_score(item['allocation']) for i,k in enumerate(known_order)} for cid,item in candidates.items()}
    regrets = {cid:{k:optima[i]-scores[cid][k] for i,k in enumerate(known_order)} for cid in candidates}
    security = {cid:dict(zip(['counter','guarantee'],exact_counter(item['allocation']))) for cid,item in candidates.items()}
    ablations = {}
    floors = []
    for name,keys in sets.items():
        for mode in ['score','regret']:
            objective = name+'_'+mode
            best = nominal_winner(frontiers[objective])
            x = best['allocation']; cid = candidate(x,'nominal_'+objective)
            ablations[objective] = {'set':name,'mode':mode,'allocation':x,'candidate_id':cid,'nominal_value':best['value'],
                'optimum_count':best['optimum_count'],'common_minimum':min(scores[cid].values()),
                'common_worst_regret':max(regrets[cid].values()),'security':security[cid]['guarantee']}
    for obj in objectives:
        for c in sorted(counts):
            best = nominal_winner(frontiers[obj.name],Fraction(c,2))
            floors.append({'objective':obj.name,'mode':obj.mode,'minimum_security':Fraction(c,2),**best})
    # These two objectives exactly reproduce the previous predeclared selection.
    for mode,previous_name in [('score','expanded_max_min'),('regret','expanded_minimax_regret')]:
        best = nominal_winner(frontiers['previous_development_'+mode])
        saved = old['selected'][previous_name]
        assert list(best['allocation'])==saved['allocation']
        assert best['value']==Fraction(saved['maximized_value']['numerator'],saved['maximized_value']['denominator'])
        assert best['optimum_count']==saved['optimum_count']
    result = {'configuration':cfg,'known_models':known,'reference_models':references,'objectives':objectives,
        'known_model_order':known_order,'groups':group_keys,'sets':sets,'frontiers':frontiers,'class_counts':counts,
        'class_first':first,'envelopes':envelopes,'epsilon_grid':epsilon_grid,'ablations':ablations,
        'candidates':candidates,'scores':scores,'regrets':regrets,'security':security,'security_floors':floors,
        'actions_scanned':searched['evaluated'],'scope':'controlled known-model sensitivity; no new holdout or human input'}
    save(out/'results.json',result); save(out/'frontiers.json',frontiers); save(out/'continuous-envelopes.json',envelopes)
    write_csv(out/'ablations.csv',['set','objective','allocation','nominal_value','common_minimum','common_worst_regret','pure_security','optimum_count'],
        [[a['set'],a['mode'],str(a['allocation']),float(a['nominal_value']),float(a['common_minimum']),float(a['common_worst_regret']),float(a['security']),a['optimum_count']] for a in ablations.values()])
    write_csv(out/'security-floors.csv',['objective','mode','minimum_security','allocation','nominal_value','optimum_count'],
        [[a['objective'],a['mode'],float(a['minimum_security']),str(a['allocation']),float(a['value']),a['optimum_count']] for a in floors])
    write_csv(out/'epsilon-grid.csv',['objective','epsilon','allocation','guaranteed_score','pure_security','optimum_count'],
        [[name,float(a['epsilon']),str(a['allocation']),float(a['value']),float(a['security']),a['optimum_count']] for name,rows in epsilon_grid.items() for a in rows])
    for field in ['scores','regrets']:
        write_csv(out/f'{field}.csv',['id','allocation']+known_order,
            [[cid,str(item['allocation'])]+[float(result[field][cid][k]) for k in known_order] for cid,item in candidates.items()])
    plots(out,result); render_report(out,result)
    for p in source: assert sha(p)==signature['source_hashes'][str(p.relative_to(ROOT))]
    save(out/'manifest.json',{'signature':signature,'started_unix':time.time()-(time.perf_counter()-start),
        'completed_unix':time.time(),'elapsed_seconds':time.perf_counter()-start,'python':sys.version,
        'numpy':np.__version__,'platform':platform.platform(),
        'artifacts':{str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='manifest.json'}})
    print(f'Completed: {len(counts)} security classes, {len(objectives)} objectives, {len(candidates)} audit vectors.',flush=True)


if __name__=='__main__': main()
