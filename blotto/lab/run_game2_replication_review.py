"""Review the accepted population replication and complete-action study."""
from pathlib import Path
import json
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from game2_lab.cross_arena import read
from game2_lab.core import focal_samples
from game2_lab.catalog_stress import empirical_bernstein_error
from run_population_study import ROOT,save,sha,write_csv


def vector(x):return '('+', '.join(map(str,x))+')'


def main():
    population=ROOT/'results/game2-population-replication';out=ROOT/'results/game2-fullspace-search'
    accepts=[population/'acceptance-population.json']+[out/('acceptance-'+p+'.json')for p in ['discovery','selection','confirmation']]
    for p in accepts:assert read(p)['passed'],p
    freeze=read(out/'freeze.json');models=freeze['models'];rules={k:np.array(v,bool)for k,v in freeze['rules'].items()}
    selected=read(out/'selection-results.json');discovery=read(out/'discovery-results.json');confirmation=read(out/'confirmation-results.json');panel=read(out/'confirmation-freeze.json')
    with np.load(out/'confirmation-scores.npz')as d:v=d['values'];errors=d['mean_error'];cross=d['cross_sum16']
    chosen=panel['indices'].index(panel['chosen_before_confirmation_index']);actions=panel['allocations'];n=freeze['configuration']['confirmation_groups'];c=len(actions)
    pointmin={k:v[:,mask].min(axis=1)for k,mask in rules.items()}
    lb={k:(v[:,mask]-errors[mask].T).min(axis=1)for k,mask in rules.items()}
    ub={k:(v[:,mask]+errors[mask].T).min(axis=1)for k,mask in rules.items()}
    def weak(i,mask):return int(np.flatnonzero(mask)[np.argmin(v[i,mask])])
    def pindex(x):return actions.index(list(x))
    old3=pindex([0,0,36,64,0]);old2=pindex([0,36,0,64,0]);qual=rules['qualified']
    candidate_rows=[]
    for i,x in enumerate(actions):
        for rule,mask in rules.items():candidate_rows.append([vector(x),rule,pointmin[rule][i],lb[rule][i],ub[rule][i],models[weak(i,mask)]['name']])
    write_csv(out/'confirmation-by-rule.csv',['allocation','rule','minimum','simultaneous_lower','simultaneous_upper','weakest_field'],candidate_rows)
    write_csv(out/'confirmation-by-field.csv',['model','qualified','legacy87','original17','allocation','expected_payout','simultaneous_error'],
        [[m['name'],m['qualified'],'legacy87'in m['tags'],'original17'in m['tags'],vector(x),v[i,j],errors[j,i]]for j,m in enumerate(models)for i,x in enumerate(actions)])
    allmodels=read(population/'validation.json')['models'];family_names=read(ROOT/'data/parry-replay/results/field_fit_validated.json')['catalog']
    write_csv(out/'model-family-weights.csv',['model','qualified','catalog','family','percent_of_conditional_population'],
        [[m['name'],m['qualified'],m['catalog_key'],name,100*w]for m in allmodels for name,w in zip(family_names,m['weights'])])
    anchors=[]
    for m in allmodels:
        if 'anchor_transplant'not in m['tags']:continue
        rows=[]
        for phase in range(2):
            with np.load(population/('cdf-pass-'+str(phase))/(m['name']+'.npz'))as d:u=d['control_rank_sum4']/(4*int(d['size']))
            rows.append({'pass':phase,'cdf_gap':m['cdf_checks'][str(phase)]['maximum_cdf_gap'],'C2_C4':float(u[0]),'C3_C4':float(u[2])})
        anchors.append({'model':m['name'],'qualified':m['qualified'],'passes':rows})
    save(out/'replicated-anchor-controls.json',anchors)
    paired=[];comparisons=len(models)*(c+c*(c-1)//2);alpha=.025
    for label,i in [('previous_C3_C4',old3),('previous_C2_C4',old2)]:
        delta=v[chosen]-v[i];variance=np.maximum(((cross[:,chosen,chosen].astype(float)+cross[:,i,i]-2*cross[:,chosen,i])/16-n*delta**2)/(n-1),0)
        err=empirical_bernstein_error(variance,n,comparisons,alpha,18)
        write_csv(out/('paired-'+label+'.csv'),['model','qualified','selected_minus_baseline','simultaneous_lower','simultaneous_upper'],
            [[m['name'],m['qualified'],delta[j],delta[j]-err[j],delta[j]+err[j]]for j,m in enumerate(models)])
        paired.append({'baseline':label,'robust_point_difference':float(pointmin['qualified'][chosen]-pointmin['qualified'][i]),
            'robust_difference_lower':float(lb['qualified'][chosen]-ub['qualified'][i]),'robust_difference_upper':float(ub['qualified'][chosen]-lb['qualified'][i]),
            'individual_fields_higher':int(np.sum(delta[qual]>0)),'individual_fields_lower':int(np.sum(delta[qual]<0))})
    mechanisms=[]
    for j in sorted({weak(chosen,qual),weak(old3,qual)}):
        with np.load(out/'confirmation'/(models[j]['name']+'.npz'))as d:rows=d['triplets']
        samples=[focal_samples(actions[i],rows)for i in [chosen,old3]]
        maximum=rows.max(axis=1);raw=[]
        for i in [chosen,old3]:
            x=np.array(actions[i]);wins=x>=maximum;tie_count=1+(rows==x[None,None,:]).sum(axis=1)
            raw.append((wins*(12//tie_count)*np.arange(1,6)).mean(axis=0)/12)
        diff=(samples[0]['payout4']-samples[1]['payout4'])/4
        assert abs(diff.mean()-(v[chosen,j]-v[old3,j]))<1e-12
        mechanisms.append({'model':models[j]['name'],'selected_raw_points_by_castle':raw[0].tolist(),'previous_raw_points_by_castle':raw[1].tolist(),
            'raw_point_change_by_castle':(raw[0]-raw[1]).tolist(),'payout_difference':float(diff.mean()),'higher_payout_probability':float(np.mean(diff>0)),
            'lower_payout_probability':float(np.mean(diff<0)),'same_payout_probability':float(np.mean(diff==0)),
            'castle2_opponent_max_0_1_2_probability':[float(np.mean(maximum[:,1]==b))for b in range(3)]})
    save(out/'scout-mechanism.json',mechanisms)
    removal=read(out/'model-removal-diagnostics.json')
    conditional_ready=all(read(p)['passed']for p in accepts) and panel['chosen_allocation']==selected['leaders']['qualified']['allocation']
    review={'selected_allocation':actions[chosen],'selection_and_discovery_agree':selected['leaders']['qualified']['allocation']==discovery['leaders']['qualified']['allocation'],
        'selected_before_confirmation':True,'candidate_conditional_floor':float(pointmin['qualified'][chosen]),'candidate_conditional_floor_interval':[float(lb['qualified'][chosen]),float(ub['qualified'][chosen])],
        'primary_confirmation_point_leader':actions[int(pointmin['qualified'].argmax())],
        'conditional_ready_to_use':conditional_ready,'actual_applicant_optimality_established':False,'pair_comparisons':paired,'anchor_controls':anchors,
        'post_selection_model_removal_diagnostics':removal,
        'review_source_sha256':sha(Path(__file__)),'scope':'Maximin expected placement payout under alternative fixed population laws. Catalog-family percentages are conditional hypotheses, not observed applicant frequencies.'}
    save(out/'review.json',review)
    # Readiness records completion of the requested checks, not a performance gate or model truth.
    working={'allocation':actions[chosen],'status':'Ready as a model-conditional maximin choice after independent population replication, complete action search and fixed-candidate confirmation',
        'criterion':freeze['configuration']['primary_rule'],'previous_working_record':'results/game2-working-candidate-v3.json','previous_allocation':[0,0,36,64,0],
        'confirmation':confirmation,'review_sha256':sha(out/'review.json'),'game1_unchanged':[3,8,4,44,41],
        'limitations':['Actual applicant allocations and family shares remain unknown','Public score-CDF compatibility does not identify joint bids','Equal-centered assumptions affect the preferred castle pair','Removing equal_rebuild_3__refit_0 restores the zero-scout C3/C4 selection','Adjacent scout bids remain close under simulation error','Maximin and minimax regret are different objectives'],
        'no_application_submitted':True}
    save(ROOT/'results/game2-working-candidate-v4.json',working)
    save(ROOT/'results/game2-lock-readiness-v2.json',{'working_candidate':actions[chosen],'ready_to_use_under_declared_model_criterion':conditional_ready,
        'unconditional_optimality_proven':False,'previous_required_numerical_checks_completed':True,'remaining_required_numerical_checks':[],
        'irreducible_choice':'Accept the explicit minimum-across-plausible-models criterion; actual applicant frequencies are unobserved.',
        'working_record_sha256':sha(ROOT/'results/game2-working-candidate-v4.json'),'no_application_submitted':True})
    figure=out/'figures';figure.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})
    fig,ax=plt.subplots(figsize=(9.4,4.8));shown=[old2,pindex([0,37,0,63,0]),old3,pindex([0,0,37,63,0]),chosen]
    for y,i in enumerate(shown):
        value=pointmin['qualified'][i];ax.errorbar(value,y,xerr=[[value-lb['qualified'][i]],[ub['qualified'][i]-value]],fmt='o',color='#167a65'if i==chosen else'#47617e',capsize=4)
    ax.set_yticks(range(len(shown)),[vector(actions[i])for i in shown]);ax.set_xlabel('Minimum expected placement payout across 99 qualified models');ax.set_title('Fresh confirmation of candidates fixed before testing');ax.grid(axis='x',alpha=.2);fig.tight_layout();fig.savefig(figure/'confirmed-floors.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11.6,4.7));labels=['Control','Rebuild 1','Rebuild 2','Rebuild 3']
    for j,a in enumerate(anchors):
        for k,p in enumerate(a['passes']):
            x=j+(k-.5)*.14;axes[0].scatter(x,p['cdf_gap'],color=['#167a65','#47617e'][k],s=35)
            axes[1].scatter(x,p['C2_C4'],color='#d88b39',s=35);axes[1].scatter(x,p['C3_C4'],color='#167a65',s=35)
    axes[0].axhline(.085,color='#a54d4d',linestyle=':',label='Maximum-gap tolerance');axes[0].legend(fontsize=8);axes[0].set_ylabel('Maximum score-CDF gap');axes[0].set_title('Both independent checks pass')
    axes[1].scatter([],[],color='#d88b39',label='(0, 36, 0, 64, 0)');axes[1].scatter([],[],color='#167a65',label='(0, 0, 36, 64, 0)');axes[1].legend(fontsize=8);axes[1].set_ylabel('Expected placement payout');axes[1].set_title('Critical model effect repeats')
    for ax in axes:ax.set_xticks(range(4),labels);ax.grid(axis='y',alpha=.2)
    fig.tight_layout();fig.savefig(figure/'independent-replication.png',dpi=180);plt.close(fig)
    table=[]
    for rule,label in [('qualified','99 qualified models'),('legacy87','Previous 87 models'),('legacy_union','Union: 105 models'),('original17','Original 17 models'),('qualified_non_equal','79 qualified, excluding equal-centered')]:
        l=selected['leaders'][rule];i=pindex(l['allocation']);table.append(f"| {label} | {vector(l['allocation'])} | {pointmin[rule][i]:.3f} |")
    ctable=[]
    for i in shown:ctable.append(f"| {vector(actions[i])} | {pointmin['qualified'][i]:.4f} | {lb['qualified'][i]:.4f} to {ub['qualified'][i]:.4f} |")
    atable=[]
    for label,a in zip(labels,anchors):
        p=a['passes'];atable.append(f"| {label} | {min(r['cdf_gap']for r in p):.4f} to {max(r['cdf_gap']for r in p):.4f} | {min(r['C2_C4']for r in p):.3f} to {max(r['C2_C4']for r in p):.3f} | {min(r['C3_C4']for r in p):.3f} to {max(r['C3_C4']for r in p):.3f} |")
    weakname=models[weak(chosen,qual)]['name'];near=[actions[i]for i in range(c)if ub['qualified'][i]>=lb['qualified'][chosen]]
    report=f'''**Game 2: independent replication and the complete allocation search**

The new choice is **{vector(actions[chosen])}**. It won both complete-action discovery and the larger independent selection under our primary rule: maximize the minimum expected placement payout across the qualified population hypotheses. It also won when we retained all 87 old models alongside the new qualified ones. It was fixed before fresh confirmation, and no candidate was substituted afterward.

The two outstanding numerical checks from the previous audit are complete. We can use this as our model-based submission choice. Actual applicant frequencies remain unknown, so this does not establish the best allocation against the real field. Game 1 remains **(3, 8, 4, 44, 41)**. No application has been submitted.

**The influential population effect reproduced**

We rebuilt the equal-centered target-family catalog three times with independent seeds and 16,384 joint vectors per target family. Each retained the same family definitions. Transplanting the original influential weights isolates catalog-sampling sensitivity; twelve new fitting starts per catalog separately explored alternative weights. Retention used public incumbent-score fit and diversity, with no entrant payoff in its objective.

The original control and all three rebuilt anchors passed both fresh validation screens. Their older C3/C4 entry continued to outperform the older C2/C4 entry in every check:

| Fixed-weight population | Maximum CDF gap, two checks | C2/C4 payout | C3/C4 payout |
|---|---:|---:|---:|
{chr(10).join(atable)}

The full screen scored every stored incumbent allocation at its actual catalog weight against 262,144 fresh opponent trios in each of two passes. This removes own-allocation sampling noise from the stored-law CDF; opponent sampling and uncertainty about human behavior remain. Of 110 unique frozen laws, 99 passed both checks: 81 of the old 87, 15 of 16 new refits, and three rebuilt anchors. The control anchor is already counted among the old models. Failed fields remain visible, and the old 87 are also compared without deleting their six newly failing fields.

These replications show that the earlier effect was reproducible within that population family. They are not independent observations that applicants actually follow it. The public score distribution still permits substantially different joint-bid populations.

**Every legal submission was searched**

We scored all **4,598,126** legal allocations against 65,536 discovery trios per field, then against a separate 524,288 selection trios per field. There were 109 evaluated fields, including reference fields outside the 99-model primary set. The candidate universe was fixed before scoring, and all exact C2/C4 and C3/C4 splits and every possible scout allocation are included.

For each opponent trio, the exact placement payout is a table of five lose/tie/win comparisons, preserving opponent identities and all castle/rank ties. Finite differences express it as signed threshold contributions. Five prefix passes on the budget simplex recover every action's integer sample score. This avoids billions of separate action-versus-trio calculations without replacing the joint game with castle marginals. The mathematical sample score is exact; its expectation is estimated from the draws.

**Fresh confirmation**

The selection phase fixed 22 comparison entries and the primary choice before 1,048,576 new trios per field. The following rows all use the same 99 qualified fields and new samples:

| Allocation | Minimum expected payout | Simultaneous numerical interval for the minimum |
|---|---:|---:|
{chr(10).join(ctable)}

The selected entry's payouts range from **{confirmation['confirmation_minimum']:.4f} to {confirmation['confirmation_maximum']:.4f}** across those models. Its weakest confirmation field is `{weakname}`. A floor difference compares two minima; it does not imply the selected allocation beats the other one in every model.

![Fresh confirmation]({figure/'confirmed-floors.png'})

The new scout bid buys additional castle-2 wins and shares, funded by one fewer troop on each of castles 3 and 4. In `equal_rebuild_3__refit_0`, it raises placement payout by 0.051 even though average raw castle points fall slightly. In the selected entry's weakest threshold-family field, it instead costs 0.140 placement points relative to the old C3/C4 entry. The improvement is in the minimum across fields. The [scout mechanism table]({out/'scout-mechanism.json'}) records its actual castle-point changes and payout gains and losses. Adjacent scout splits remain close; the exact two-troop choice is the prespecified empirical winner, rather than a claim that every nearby split is statistically inferior. There are {len(near)} frozen entries, including the selected entry, whose minimum-payoff intervals overlap the selected entry's lower bound.

**Changing the population assumptions still changes the choice**

Each row below reuses the complete-action selection under a rule fixed before testing. Payouts are from fresh confirmation on that row's own population set, so they should not be compared as if all rows used the same opponents.

| Population set | Complete-space selection winner | Confirmed minimum |
|---|---|---:|
{chr(10).join(table)}

The pure C3/C4 entry remains the winner on the previous 87. The new scout improves protection against the broader refits. Removing every equal-centered law returns the C2/C4 choice, as does keeping only the original 17. Numerical replication substantially reduces the concern about an accidental catalog draw, while preserving the substantive uncertainty about this family of human behavior.

A post-selection removal diagnostic, using the saved complete-action selection scores, retained the exact two-scout choice after **{removal['single_model_omissions_preserving_exact_choice']} of {removal['single_model_omissions']}** individual model deletions. Removing `equal_rebuild_3__refit_0` restores (0, 0, 36, 64, 0); removing `legacy__supplied_fit_27` instead favors (0, 2, 47, 51, 0). The exact scout refinement is therefore model-sensitive too. Removing both deliberately targeted catalog variants together leaves (0, 2, 35, 63, 0) as the complete-space winner. The [117 removal diagnostics]({out/'model-removal-diagnostics.csv'}) are retrospective sensitivity evidence, with no new holdout or change to the frozen candidate choice.

We did not assign a percentage chance to each model. A model is an alternative complete population hypothesis. Taking the minimum makes duplicates irrelevant, but adding a genuinely new adverse plausible law can change the result. The [family-weight table]({out/'model-family-weights.csv'}) gives all forty conditional family percentages for every law; those are model assumptions, not measured applicant shares. The complete-space minimax-regret choice is {vector(selected['leaders']['qualified']['minimax_regret_allocation'])}, reflecting a different objective from our chosen maximin rule.

![Population replication]({figure/'independent-replication.png'})

**Numerical confidence and limits**

Half of the 5% numerical error budget covers every action and field in the complete selection scan with a Hoeffding union bound. The other half covers the fixed confirmation panel's means and paired comparisons with empirical Bernstein bounds. Conditional on the frozen laws, the joint calculation bounds the selected entry's maximin shortfall from the best of all legal allocations by **{confirmation['conditional_full_space_optimality_shortfall_bound']:.4f} placement-payout points**, at at least 95% confidence. The complete-space selection error allowance is {confirmation['full_space_selection_cell_error_bound']:.4f}; the selected minimum's confirmation interval is {confirmation['confirmation_robust_lower']:.4f} to {confirmation['confirmation_robust_upper']:.4f}.

This bound includes action search uncertainty. It excludes error in the population assumptions, which more simulation from the same generators cannot remove. Score-CDF tolerances are diagnostic acceptance rules, not probabilities that a model is true. Earlier adapting arenas remain separate stress evidence and are preserved.

**Validation and saved evidence**

All 122 tests passed, including exhaustive reduced-budget checks, full-budget indexing and direct-score comparisons. Independent acceptance regenerated every catalog and opponent bank, checked all CDF decisions, reconstructed every complete-action objective, checked representative whole-space entries against direct lookup and the independent foundation scorer on all groups, and verified the confirmation moments and confidence calculations. The audits document exactly which score cells were checked; they do not claim a second cell-by-cell replay of the complete action-by-trio Cartesian product.

The [protocol]({ROOT/'GAME2-REPLICATION-SEARCH-PROTOCOL.md'}), [confirmation results]({out/'confirmation-results.json'}), [all candidate/model scores]({out/'confirmation-by-field.csv'}), [population-subset comparisons]({out/'confirmation-by-rule.csv'}) and [versioned working record]({ROOT/'results/game2-working-candidate-v4.json'}) preserve the decision. Earlier freezes, source inputs, results and recommendations remain unchanged. The prior two requested numerical tasks are finished; another numerical stage is not required before using this conditional choice.
'''
    assert '\u2014'not in report
    report_path=ROOT.parent/'29-game2-independent-replication-and-full-allocation-search.md';report_path.write_text(report)
    proof=[out/'freeze.json',out/'selection-results.json',out/'confirmation-freeze.json',out/'confirmation-results.json',out/'review.json',out/'model-removal-diagnostics.json',population/'freeze.json',population/'model-freeze.json',population/'validation.json',*accepts]
    checkpoint={'allocation':actions[chosen],'report_sha256':sha(report_path),'evidence_sha256':{str(p.relative_to(ROOT)):sha(p)for p in proof},'previous_records_preserved':True}
    save(ROOT/'results/game2-replication-checkpoint-v1.json',checkpoint)
    evidence=ROOT/'results/game2-replication-checkpoint-v1-evidence'
    for p in proof:
        target=evidence/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    print(json.dumps(review,indent=2))


if __name__=='__main__':main()
