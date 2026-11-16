"""Retrospective provenance, model-influence and class-share audit.

No new population is fitted, sampled, evolved or admitted by this audit.
"""
from pathlib import Path
import numpy as np
from game2_lab.cross_arena import read
from game2_lab.core import reference_group
from run_population_study import ROOT,save,sha,write_csv


def category_shares(folder,names):
    features=[]
    with np.load(folder/'rows.npz')as rows,np.load(folder/'weights.npz')as mass:
        for name in names:
            x=rows[name];w=mass[name].astype(float);w/=w.sum()
            outside=x[:,[0,2,4]].sum(axis=1);active=(x[:,1]>=16)&(x[:,3]>=16)
            masks=[np.all(x==[0,36,0,64,0],axis=1),active&(outside==0),active&(outside<=8),active&(outside<=15),active&(outside<=24)]
            features.append([float(w[m].sum())for m in masks])
    return np.array(features)


def main():
    study=ROOT/'results/game2-catalog-stress';out=ROOT/'results/game2-overfit-audit'
    out.mkdir(exist_ok=True)
    models=[r for r in read(study/'model-validation.json')['models']if r['validation']['passes']]
    freeze=read(study/'freeze.json')
    for rel,digest in freeze['source_and_input_sha256'].items():assert sha(ROOT/rel)==digest,rel
    with np.load(study/'split-scores.npz')as d:v24=d['values'];field_names=d['names'].tolist()
    with np.load(study/'refinement/scores.npz')as d:v34=d['values']
    assert field_names==[r['name']for r in models]
    subsets={
        'all_87':np.ones(len(models),dtype=bool),
        'supplied_models_only':np.array([r['constraint']=='original_supplied'for r in models]),
        'unchanged_catalog_only':np.array([r['catalog']=='original'for r in models]),
        'exclude_two_targeted_threshold_catalogs':np.array([r['catalog']not in ['c24_threshold_family','c4_threshold_upweight']for r in models]),
        'exclude_equal_centered_catalog':np.array([r['catalog']!='equal_active_targets'for r in models])}
    for kind in sorted({r['catalog']for r in models}):subsets['omit_catalog__'+kind]=np.array([r['catalog']!=kind for r in models])
    rows=[]
    for name,mask in subsets.items():
        a=v24[:,mask].min(axis=1);b=v34[:,mask].min(axis=1)
        rows.append({'subset':name,'fields':int(mask.sum()),'best_C2':int(a.argmax()),'best_C2_C4_floor':float(a.max()),
            'best_C3':int(b.argmax()),'best_C3_C4_floor':float(b.max()),'fixed_C2_36_floor':float(a[36]),'fixed_C3_36_floor':float(b[36]),
            'preferred_pair_by_minimum':'C3/C4'if b.max()>a.max()else 'C2/C4'})
    influence=[]
    for j,model in enumerate(models):
        mask=np.arange(len(models))!=j;a=v24[:,mask].min(axis=1);b=v34[:,mask].min(axis=1)
        influence.append({'removed_field':model['name'],'best_C2':int(a.argmax()),'C2_C4_floor':float(a.max()),
            'best_C3':int(b.argmax()),'C3_C4_floor':float(b.max()),'preferred_pair':'C3/C4'if b.max()>a.max()else 'C2/C4'})
    single_switches=[r for r in influence if r['preferred_pair']=='C2/C4']
    assert len(single_switches)==1 and single_switches[0]['removed_field']=='equal_active_targets__unrestricted__2'
    names=read(ROOT/'data/parry-replay/results/field_fit_validated.json')['catalog']
    features={kind:category_shares(study/'catalogs'/kind,names)for kind in {r['catalog']for r in models}}
    share_names=['exact_old_allocation','pure_24_both_at_least16','outside_at_most8_both_at_least16','outside_at_most15_both_at_least16','outside_at_most24_both_at_least16']
    shares={r['name']:dict(zip(share_names,(np.array(r['weights'])@features[r['catalog']]).tolist()))for r in models}
    old=ROOT/'results/game2-cross-arena'
    with np.load(old/'fresh-scores.npz')as d:
        old_values=d['values'];old_actions=d['allocations'].tolist();old_fields=d['names'].tolist()
    i24=old_actions.index([0,36,0,64,0]);i34=old_actions.index([0,0,36,64,0])
    old_freeze=read(old/'freeze.json')
    original17={'C2_C4_allocation':[0,36,0,64,0],'C3_C4_allocation':[0,0,36,64,0],
        'C2_C4_range':[float(old_values[i24].min()),float(old_values[i24].max())],
        'C3_C4_range':[float(old_values[i34].min()),float(old_values[i34].max())],
        'fields_C2_C4_higher':int(np.sum(old_values[i24]>old_values[i34])),
        'fields_C3_C4_higher':int(np.sum(old_values[i34]>old_values[i24])),
        'new_candidate_source_tags_in_pre_evaluation_pool':old_freeze['pool'][i34]['sources']}
    legacy=read(old/'old-stress-summary.json')
    original_arenas={str(tuple(x)):{g:legacy['block_values'][g][old_actions.index(x)]for g in ['core','pressure','arenas']}for x in [[0,0,36,64,0],[0,36,0,64,0],[15,18,49,12,6]]}
    # Exact symmetric four-person controls: either entry can win as the rare type.
    a=(0,36,0,64,0);b=(0,0,36,64,0)
    clone_cases=[{'C3_C4_people':k,'actions':[b]*k+[a]*(4-k),'outcome':reference_group([b]*k+[a]*(4-k))}for k in range(5)]
    assert clone_cases[1]['outcome'].payouts[0]==9
    assert clone_cases[3]['outcome'].payouts[-1]==9
    for row in clone_cases:
        assert sum(row['outcome'].castle_points)==15 and sum(row['outcome'].payouts)==15
    inputs=[study/'freeze.json',study/'model-freeze.json',study/'model-validation.json',study/'split-scores.npz',study/'refinement/scores.npz',old/'freeze.json',old/'fresh-scores.npz',old/'old-stress-summary.json']
    for kind in features:inputs += [study/'catalogs'/kind/'rows.npz',study/'catalogs'/kind/'weights.npz']
    result={'scope':'Retrospective influence/provenance audit on existing scores, not a fresh holdout or new fitted population. No earlier records changed.',
        'source_sha256':sha(Path(__file__)),'input_sha256':{str(p.relative_to(ROOT)):sha(p)for p in inputs},
        'candidate_identity':[0,0,36,64,0],'population_subsets':rows,'single_model_removals_reversing_pair_choice':single_switches,
        'class_definitions':{'outside':'Troops on castles 1,3,5 for the C2/C4-focused class','active':'At least 16 on each of castles 2 and 4; cutoffs are descriptive, not observed labels.'},
        'all_87_class_shares':shares,'original_17':original17,'original_stress_block_means':original_arenas,'exact_clone_cases':clone_cases,
        'mechanical_feedback_audit':{'latest_stage_runs_evolution':False,'fit_objective':'Public incumbent score-CDF discrepancy plus fixed small weight regularizer',
            'model_retention_uses_entry_payoff':False,'model_retention_uses':'Independent score-CDF screen',
            'candidate_frozen_before_new_populations_scored':True,'candidate_in_model_fitting_loss':False,
            'candidate_informed_model_design_exists':True,'candidate_informed_design_examples':['c4_threshold_upweight','c24_threshold_family'],
            'exclusion_of_those_two_catalogs_changes_winner':False},
        'lock_readiness':{'ready_to_lock':False,'working_candidate':[0,0,36,64,0],
            'reason':'The specific frequency-feedback loop is absent, but the pair switch depends on one additional equal-centered fitted field and the 760-vector candidate set is not a full allocation search.',
            'remaining_checks':['Independently rebuild/refit the influential equal-centered population assumptions; assess whether the pair switch survives reasonable independent realizations.',
                'Search beyond the frozen 760-vector set, including small scout allocations around the C3/C4 leaders, then qualify any challengers independently.']}}
    save(out/'results.json',result)
    write_csv(out/'population-subset-comparisons.csv',list(rows[0]),[[r[k]for k in rows[0]]for r in rows])
    write_csv(out/'all-87-single-model-removals.csv',list(influence[0]),[[r[k]for k in influence[0]]for r in influence])
    write_csv(out/'actual-C2-C4-class-shares.csv',['field',*share_names],[[name,*[row[k]for k in share_names]]for name,row in shares.items()])
    new=shares['equal_active_targets__unrestricted__2'];original=shares['supplied_fit_13']
    report=ROOT.parent/'28-game2-overfitting-and-lock-readiness-audit.md'
    text=f'''**Game 2: checking the feedback-loop concern before locking**

The specific winner-frequency feedback loop did not occur in the latest study. That does not establish that model overfitting or misspecification has been ruled out. **Keep (0,0,36,64,0) as the working candidate; do not treat Game 2 as ready to lock yet.**

The latest stage did not run a reproducing evolutionary arena. It froze the 760 candidates, fitted hypothetical population weights to the published incumbent-score distribution, screened those fits on independent numerical samples, then evaluated the frozen entries. The fitting loss and retention screen did not reward a model for defeating either candidate. All 101 exact C3/C4 splits were already in the frozen pool before that comparison. The C3/C4 candidate was not introduced after seeing the result of the 87-field test.

There was candidate-informed model design: two catalog variants deliberately emphasized castle-4 threshold bids and a C2/C4-targeting family. Those were deliberate stress assumptions. Removing both variants leaves 81 accepted fields and **does not change the C3/C4 recommendation**. Thus those targeted additions are not the cause of the switch.

**The population choice still matters substantially**

| Existing population set | Minimum for (0,36,0,64,0) | Minimum for (0,0,36,64,0) |
|---|---:|---:|
| Original 17 supplied fitted fields | {original17['C2_C4_range'][0]:.3f} | {original17['C3_C4_range'][0]:.3f} |
| All 87 accepted expanded fields | {v24[36].min():.3f} | {v34[36].min():.3f} |
| Expanded set excluding equal_active_targets__unrestricted__2 | {single_switches[0]['C2_C4_floor']:.3f} | {single_switches[0]['C3_C4_floor']:.3f} |

The original 17 are the unchanged source models on the earlier replay's fresh draws. The latter two rows use the expanded study's million-group draws. Differences between draws should not be confused with model changes. Within the 87-field comparison, the best C2/C4 split is 37/63, with minimum 4.891. Removing the influential field restores 36/64 as that family's best split.

Only **one** of the 87 individual model removals reverses the preferred castle pair: removing **equal_active_targets__unrestricted__2**. Removing the full equal-centered catalog also reverses the decision. Keeping only the unchanged catalog and its omissions leaves 59 fields and again favors C2/C4. This is an important qualification to the previous headline.

A minimum-across-populations rule is designed to respond to one sufficiently adverse plausible field. Dependence on that field is not, by itself, proof of overfitting. But the public score distribution does not tell us how likely that field is. The confidence bounds in the previous report measure simulation error within the supplied hypotheses; they do not validate the hypotheses as actual applicant populations.

**Was the influential population filled with the previous winner?**

No. Its exact (0,36,0,64,0) share is **{100*new['exact_old_allocation']:.4f}%**. For a broader C2/C4-focused class, defined as at least 16 troops on each target and at most 15 total on the other castles, the share is **{100*new['outside_at_most15_both_at_least16']:.2f}%**. The same class has share **{100*original['outside_at_most15_both_at_least16']:.2f}%** in the original supplied fit_13 population, which instead favors C2/C4. A large new prevalence of this class does not explain that particular change.

Those are explicit numerical class definitions, not observed human archetype labels. The [all-field class-share table]({out/'actual-C2-C4-class-shares.csv'}) also includes outside-troop cutoffs of zero, eight and twenty-four. Changes in allocation shape, other families and their joint bids still matter even if the broad C2/C4 percentage is lower.

The new entry also is not an unconditional counter that dominates the old one in a four-person game. With one C3/C4 entrant and three copies of the C2/C4 entry, the C3/C4 entrant earns nine. With one C2/C4 entrant and three copies of the C3/C4 entry, the C2/C4 entrant earns nine. The exact clone controls are saved in the audit result. Directly defeating one allocation is insufficient evidence of population performance.

**How it performed before the expanded models**

(0,0,36,64,0) already earned **{original17['C3_C4_range'][0]:.3f} to {original17['C3_C4_range'][1]:.3f}** on the original seventeen supplied population hypotheses. It was a strong entry there, but C2/C4 earned more in **{original17['fields_C2_C4_higher']} of 17** fields; C3/C4 earned more in the other {original17['fields_C3_C4_higher']}. It did not win the original seventeen-field maximin comparison.

On our older adapting arenas, the new C3/C4 entry averaged **{original_arenas[str(tuple(b))]['arenas']:.3f}**, compared with **{original_arenas[str(tuple(a))]['arenas']:.3f}** for the previous C2/C4 entry and **{original_arenas[str((15,18,49,12,6))]['arenas']:.3f}** for our historical balanced choice. Those constructed arena populations were not themselves good public-score matches. They remain stress evidence, and the new candidate was not their overall winner either.

**Lock readiness**

The result supports a provisional C3/C4 choice under the expanded worst-population criterion. It does not justify saying that all forms of overfitting are ruled out or that final selection is complete. Before locking, independently rebuild and refit the influential equal-centered assumptions, and expand allocation discovery beyond the 760 frozen entries. In particular, small scout bids around the C3/C4 leaders have not received a complete legal-action search under the broadened model set.

This audit used existing saved results and exact rational controls. It did not evolve the arena, add copies of a successful allocation, fit new populations, draw a new holdout, or alter earlier freezes and recommendations. The [population-subset comparisons]({out/'population-subset-comparisons.csv'}) and [all 87 single-model removals]({out/'all-87-single-model-removals.csv'}) make the sensitivity reviewable.
'''
    assert '\u2014'not in text;report.write_text(text.rstrip()+'\n')
    save(ROOT/'results/game2-lock-readiness-v1.json',{'ready_to_lock':False,'working_candidate':[0,0,36,64,0],
        'audit_results_sha256':sha(out/'results.json'),'report_sha256':sha(report),**result['lock_readiness']})
    print('Saved',report,flush=True)


if __name__=='__main__':main()
