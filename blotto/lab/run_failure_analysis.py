"""Known-evidence Game 1 failure atlas and exact replacement-mass maps."""
from fractions import Fraction
from pathlib import Path
import argparse
import json
import shutil
import time
import numpy as np
from blotto_lab.core import best_response
from failure_tools import attribute,capped_response,first_loss_share,frozen_envelope,marginal_probabilities,share_map
from run_multiwave_study import ROOT,unpack_table
from run_population_study import save,sha,write_csv
from verify_population_study import rational


def read(path):return json.loads(path.read_text())


def selected_cases(r):
    focus={name:p['candidate_id'] for name,p in r['primary'].items()}
    focus['previous_machine_max_min']=r['label_ids']['M_known_model_max_min']
    keys=[k for k,m in r['models'].items() if m['phase']!='control']
    selected={};extremes={}
    for name,cid in focus.items():
        result={}
        for label,field,subset,minimum in [('all_minimum','scores',keys,True),('all_regret','regrets',keys,False),
                ('audit_regret','regrets',[k for k in keys if r['models'][k]['phase']=='audit'],False)]:
            values={k:rational(r[field][cid][k]) for k in subset}
            value=(min if minimum else max)(values.values());winners=[k for k in subset if values[k]==value]
            result[label]={'value':value,'scenarios':winners}
            for k in winners:selected.setdefault(k,[]).append({'candidate':name,'criterion':label})
        extremes[name]=result
    return focus,dict(sorted(selected.items())),extremes


def plots(out,r):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names=list(r['focus']);cases=list(r['critical_scenarios'])
    matrix=np.array([[float(r['attributions'][name][k]['regret']) for k in cases]for name in names])
    labels=[r['critical_scenarios'][k]['short_name'] for k in cases]
    fig,ax=plt.subplots(figsize=(max(10,len(cases)*.63),6.8))
    im=ax.imshow(matrix,cmap='YlOrRd',vmin=0,aspect='auto')
    ax.set_xticks(range(len(cases)),labels,rotation=65,ha='right',fontsize=7)
    ax.set_yticks(range(len(names)),[n.replace('_',' ')for n in names],fontsize=8)
    for i in range(len(names)):
        for j in range(len(cases)):
            ax.text(j,i,f'{matrix[i,j]:.1f}',ha='center',va='center',fontsize=7,
                    color='white' if matrix[i,j]>matrix.max()*.65 else 'black')
    ax.set_title('Known failure cases: points below the scenario best response\nSelected from observed extremes; column counts do not estimate prevalence')
    fig.colorbar(im,ax=ax,label='Regret in official points');fig.tight_layout()
    fig.savefig(out/'failure-atlas.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,5.8))
    ordered=sorted(cases,key=lambda k:float(r['maps'][k]['security2']['loss_thresholds']['0.25'] or Fraction(2)))
    ypos=np.arange(len(ordered))
    for j,kind in enumerate(['unrestricted','security2']):
        vals=[r['maps'][k][kind]['loss_thresholds']['0.25'] for k in ordered]
        axes[j].barh(ypos,[float(v)if v is not None else 1 for v in vals],color=['#2878a7'if v is not None else '#bbbbbb'for v in vals])
        axes[j].set_yticks(ypos,[r['critical_scenarios'][k]['short_name']for k in ordered],fontsize=7)
        axes[j].invert_yaxis();axes[j].set_xlim(0,1)
        axes[j].set_xlabel('Mass replaced with the selected scenario')
        axes[j].set_title(('All legal allocations'if j==0 else 'Pure security at least 2')+'\nWhen baseline choice first loses 0.25 points')
        for i,v in enumerate(vals):axes[j].text(min((float(v)if v is not None else 1)+.015,.96),i,'never'if v is None else f'{float(v):.1%}',va='center',fontsize=7)
    fig.tight_layout();fig.savefig(out/'weight-thresholds.png',dpi=170);plt.close(fig)


def report(out,r):
    rows=['**Game 1: failure attribution and exact population-weight sensitivity**','',
        'This is a diagnostic analysis of known merged-study outcomes. The previous validation and audit '
        'sets are now development evidence. No new holdout or applicant-frequency evidence is claimed. '
        'All nine frozen focus candidates remain unchanged.','',
        f"The declared extreme-selection rule identifies {len(r['critical_scenarios'])} distinct scenarios. "
        'For each candidate it retains all ties for lowest score and highest regret overall, and highest '
        'regret in the previous audit set. Overlapping selections are deduplicated. Their count does not '
        'set their population weight.','',
        '**Where the frozen candidates fail**','',
        '| Candidate | Largest known regret | Cause | Worst score | Cause |',
        '| --- | ---: | --- | ---: | --- |']
    for name,e in r['extremes'].items():
        cause=lambda label:', '.join(r['critical_scenarios'][k]['short_name']for k in e[label]['scenarios'])
        rows.append(f"| {name} | {float(e['all_regret']['value']):.3f} | {cause('all_regret')} | {float(e['all_minimum']['value']):.3f} | {cause('all_minimum')} |")
    rows+=['','![Known failure atlas](failure-atlas.png)','',
        '**How much additional replacement changes the answer**','',
        'P(alpha)=(1-alpha)*P0+alpha*Q, where P0 is exactly the previously declared midpoint prior and Q '
        'is one selected failure population. Q can overlap with P0. Alpha is an additional replacement '
        'parameter, not the total real-world frequency of a strategy type. All other proportions are '
        'scaled down together. Each row has its own conditional mixture experiment.','',
        'The first line change can concern a tiny rounding preference. The 0.25-point threshold asks '
        'when retaining the baseline-optimal vector becomes materially worse than adapting to that '
        'mixture. Both benchmarks use the same allowed action class. Exact thresholds of 0.10 and 0.50 '
        'are also saved in results.json.','',
        '| Failure scenario | First change, all actions | 0.25-point loss, all actions | First change, security 2 | 0.25-point loss, security 2 |',
        '| --- | ---: | ---: | ---: | ---: |']
    def pct(v):return 'not reached'if v is None else f'{float(v):.2%}'
    for k,c in r['critical_scenarios'].items():
        a,b=r['maps'][k]['unrestricted'],r['maps'][k]['security2']
        rows.append(f"| {c['short_name']} | {pct(a['first_line_change'])} | {pct(a['loss_thresholds']['0.25'])} | {pct(b['first_line_change'])} | {pct(b['loss_thresholds']['0.25'])} |")
    rows+=['','![Replacement-mass thresholds](weight-thresholds.png)','',
        '**A useful exact security simplification**','',
        'For this 100-troop game with values (1,2,3,4,5), a fixed allocation guarantees at least two '
        'points if and only if its castle-1 bid is at most three. If x1>=4, the opponent bids zero at '
        'castle 1 and at least one more troop at each remaining castle, using the released budget. '
        'That leaves us exactly one point. Conversely, a score below two requires us to win only '
        'castle 1 and lose all four others: any tie elsewhere is already worth at least one point, '
        'and any other win is worth at least two. The opponent needs four extra troops across those '
        'castles, which is impossible when x1<=3. An all-tie outcome is 7.5 and cannot be the exception. '
        'The verifier checks this equivalence on every legal allocation using independent outcome '
        'patterns. This statement is specific to this game.','',
        '**What the attribution says**','',
        'Every saved candidate/scenario attribution lists its castle win/tie/loss probabilities, expected '
        'castle score, the scenario best-response bid, and the signed difference in castle score. The '
        'signed differences sum to total regret; a negative contribution means our current vector '
        'actually does better on that particular castle. Transfers of one troop are evaluated exactly. '
        'A small or zero local gain does not mean the strategy is globally strong.','',
        'The marginals are recovered from score-table increments and checked to be nonnegative with '
        'unit mass. They describe each castle separately; they do not identify correlations between '
        'castle bids. Learning snapshots keep their original path and snapshot evidence labels.','',
        '**Interpreting model weights**','',
        'This study measures the consequence of different weights. It cannot determine whether a '
        'particular case is common among applicants. We should retain concentrated coalition cases '
        'and strategic snapshots as sensitivity tests, without treating their frequency in our test '
        'library as a population estimate. A final choice needs an explicit preference between mean '
        'performance under declared priors, protection against individual extremes, and pure security.','',
        'The finite nine-candidate envelope answers which frozen candidate is best at each weight. '
        'The full-space envelope answers what could be achieved if we adapted perfectly to that '
        'weight. They are reported separately. New envelope vectors are exploratory and have not '
        'passed an additional fresh challenge. No final primary is replaced here.','',
        '**Verification and next-stage boundary**','',
        'Exact rational directional oracles discover every response interval. A separately implemented '
        'constrained DP certifies global optimality at both ends of every interval and verifies '
        'interior/boundary tie counts. Score differences are linear in alpha, so endpoint certificates '
        'cover the entire closed intervals. Small-game enumeration tests both implementations, cap '
        'restrictions, threshold calculations, and tie handling.','',
        'The next challenge must use new populations and locked selection rules, since the previous '
        'challenges have now informed this analysis. This diagnostic does not itself require another '
        'unbounded search cycle. The remaining Game 1 stages are a final frozen challenge and a '
        'selection review, unless those reveal a material flaw. Game 2 remains separate.','',
        'Run `python3 run_failure_analysis.py --output results/failure-new`, then '
        '`python3 verify_failure_analysis.py --study results/failure-new`. Detailed attributions, all '
        'continuous maps, exact fractions, finite-candidate comparisons, input hashes, and source '
        'snapshots are saved beside this report.','']
    (out/'report.md').write_text('\n'.join(rows))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=ROOT/'results/failure-analysis')
    out=parser.parse_args().output.resolve()
    if out.exists():raise SystemExit('Output exists. Choose a new output folder.')
    clock=time.perf_counter();out.mkdir(parents=True)
    source=[Path(__file__),ROOT/'failure_tools.py',ROOT/'verify_failure_analysis.py',ROOT/'FAILURE-ANALYSIS-PROTOCOL.md',
            ROOT/'blotto_lab/core.py',ROOT/'blotto_lab/contamination.py',ROOT/'run_population_study.py',ROOT/'run_multiwave_study.py']
    inputs=[ROOT/'results/merged-game1/results.json',ROOT/'results/merged-game1/development-score-tables.json',
            ROOT/'results/merged-game1/security-classes.npz',ROOT/'results/game1-merged-checkpoint-v2.json']
    signature={'source_hashes':{str(p.relative_to(ROOT)):sha(p)for p in source},'input_hashes':{str(p.relative_to(ROOT)):sha(p)for p in inputs}}
    for p in source:
        target=out/'source-snapshot'/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
    save(out/'run-inputs.json',signature)
    prior=read(inputs[0]);defs=read(inputs[1]);focus,critical,extremes=selected_cases(prior)
    base=unpack_table(defs['tables'][defs['search_keys'][defs['midpoint']['table_index']]],'midpoint prior')
    actions={name:tuple(prior['candidates'][cid]['allocation'])for name,cid in focus.items()}
    tables={k:unpack_table(prior['models'][k],k)for k in critical};descriptions={};attributions={name:{}for name in focus};maps={}
    for i,(key,reasons)in enumerate(critical.items(),1):
        model=prior['models'][key];table=tables[key];br=best_response(table)
        descriptions[key]={'short_name':f'F{i:02d}','reasons':reasons,'phase':model['phase'],'kind':model['kind'],
             'source_metadata':{k:v for k,v in model.items()if k not in ['table','best_response']},'best_response':br,
             'best_response_advantage_over_value':br.expected_score-Fraction(15,2),
             'marginal_atoms':[[{'bid':j,'probability':v}for j,v in enumerate(row)if v]for row in marginal_probabilities(table)]}
        for name,x in actions.items():attributions[name][key]=attribute(table,x,br.allocation)
        maps[key]={}
        for label,cap in [('unrestricted',None),('security2',3)]:
            mapped=share_map(base,table,cap);start=capped_response(base,cap).allocation
            first=mapped['segments'][0]
            line_same=base.expected_score(start)==first['baseline_score']and table.expected_score(start)==first['scenario_score']
            mapped['baseline_allocation']=start
            mapped['first_line_change']=first['upper_share']if line_same and len(mapped['segments'])>1 else (None if line_same else Fraction(0))
            mapped['loss_thresholds']={str(t):first_loss_share(mapped,base,table,start,Fraction(t))for t in ['0.1','0.25','0.5']}
            maps[key][label]=mapped
            print(f"{i}/{len(critical)} {key} {label}: {len(mapped['segments'])} exact intervals",flush=True)
        maps[key]['frozen_candidates']=frozen_envelope(base,table,actions)
    r={'scope':'Known-evidence diagnostics; no new holdout or applicant proportions','focus':focus,'allocations':actions,
       'extremes':extremes,'critical_scenarios':descriptions,'attributions':attributions,'maps':maps,
       'baseline':{'source':'merged midpoint prior','table':{'numerators':base.numerators,'denominator':base.denominator}},
       'security_equivalence':'pure security >=2 iff first castle bid <=3, for exact application game only'}
    save(out/'results.json',r)
    write_csv(out/'thresholds.csv',['scenario','case','phase','kind','class','first_line_change','loss_0.1','loss_0.25','loss_0.5','intervals'],
        [[k,c['short_name'],c['phase'],c['kind'],label,
          None if maps[k][label]['first_line_change']is None else float(maps[k][label]['first_line_change']),
          *[None if maps[k][label]['loss_thresholds'][t]is None else float(maps[k][label]['loss_thresholds'][t])for t in ['0.1','0.25','0.5']],
          len(maps[k][label]['segments'])]for k,c in descriptions.items()for label in ['unrestricted','security2']])
    plots(out,r);report(out,r)
    for p in source:assert sha(p)==signature['source_hashes'][str(p.relative_to(ROOT))]
    save(out/'manifest.json',{'signature':signature,'completed_unix':time.time(),'elapsed_seconds':time.perf_counter()-clock,
       'artifacts':{str(p.relative_to(out)):sha(p)for p in sorted(out.rglob('*'))if p.is_file()and p.name!='manifest.json'}})
    print(f'Failure study complete: {len(critical)} critical scenarios. Run the independent verifier.',flush=True)


if __name__=='__main__':main()
