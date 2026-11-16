"""Explain verified catalog stresses and complete C2/C4 curves."""
import argparse
from pathlib import Path
import json
import math
import numpy as np
from game2_lab.cross_arena import read,native_module,counts_from_groups,load_catalog
from game2_lab.catalog_stress import CATALOG_VARIANTS,empirical_bernstein_error
from run_population_study import ROOT,save,sha,write_csv


LABELS={'original':'Original catalog, new fits','zero_scout_targets':'Zero scouts on inactive castles',
    'broad_scout_targets':'Inactive scouts 0 to 24','high_scout_targets':'Inactive scouts 5 to 16',
    'uniform_active_targets':'Uniform active splits','equal_active_targets':'Equal-centered active splits',
    'value_active_targets':'Value-centered active splits','corner_active_targets':'Asymmetric active splits',
    'c4_threshold_upweight':'More mass on C4 bids 61 to 72','c24_threshold_family':'C2/C4 family near C4 60 to 75',
    'nonatom_transfer1':'One troop moved within non-atom families','nonatom_transfer5':'Five troops moved within non-atom families',
    'supplied':'Supplied 17 fields, freshly checked','omissions':'All accepted family omissions','overall':'All accepted fields'}


def category(row):
    if row['constraint']=='original_supplied':return 'supplied'
    if row['constraint'].startswith('omit_'):return 'omissions'
    return row['catalog']


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--prepare',action='store_true');args=parser.parse_args()
    study=ROOT/'results/game2-catalog-stress';out=study/'review';out.mkdir(exist_ok=True)
    freeze=read(study/'freeze.json');cfg=freeze['configuration'];validation=read(study/'model-validation.json');selection=read(study/'selection.json')
    models=[r for r in validation['models']if r['validation']['passes']]
    with np.load(study/'split-scores.npz')as d:v=d['values'];cross=d['cross_sum16'];field_names=d['names'].tolist()
    with np.load(study/'candidate-scores.npz')as d:cv=d['values'];actions=d['allocations']
    with np.load(study/'split-confidence.npz')as d:errors=d['mean_error'];pair_errors=d['pair_error']
    # Reallocate the joint sampling budget across the two complete split families
    # and the full 760-vector prefix comparison, as declared in the refinement.
    n=cfg['split_groups'];comparisons=len(models)*(101+5050);alpha=.05/3
    variance=(cross[:,np.arange(101),np.arange(101)]/16-n*v.T**2)/(n-1)
    errors=empirical_bernstein_error(variance,n,comparisons,alpha,9)
    pair_errors=[]
    for j,matrix in enumerate(cross):
        square=(np.diag(matrix)[:,None]+np.diag(matrix)[None,:]-2*matrix.astype(float))/16
        var=np.maximum((square-n*(v[:,j,None]-v[None,:,j])**2)/(n-1),0)
        pair_errors.append(empirical_bernstein_error(var,n,comparisons,alpha,18))
    pair_errors=np.array(pair_errors)
    names=list(load_catalog(ROOT/'data/parry-replay')[1]);groups={key:[j for j,r in enumerate(models)if category(r)==key]for key in CATALOG_VARIANTS+['supplied','omissions']};groups['overall']=list(range(len(models)))
    group_rows=[]
    for key in CATALOG_VARIANTS+['supplied','omissions','overall']:
        test=[r for r in validation['models']if category(r)==key]if key!='overall'else validation['models']
        ids=groups[key]
        if ids:
            floor=v[:,ids].min(axis=1);b=int(floor.argmax())
            group_rows.append({'group':key,'label':LABELS[key],'tested':len(test),'accepted':len(ids),'best_C2':b,'best_C4':100-b,
                'best_floor':float(floor[b]),'floor_36':float(floor[36]),'floor_37':float(floor[37]),'floor_38':float(floor[38])})
        else:group_rows.append({'group':key,'label':LABELS[key],'tested':len(test),'accepted':0,'best_C2':None,'best_C4':None,'best_floor':None,'floor_36':None,'floor_37':None,'floor_38':None})
    save(out/'population-group-results.json',group_rows)
    write_csv(out/'population-group-results.csv',list(group_rows[0]),[[r[k]for k in group_rows[0]]for r in group_rows])
    omission_rows=[]
    for name in names:
        test=[r for r in validation['models']if r['constraint']=='omit_'+name];ids=[j for j,r in enumerate(models)if r['constraint']=='omit_'+name]
        if ids:
            floor=v[:,ids].min(axis=1);b=int(floor.argmax());worst=field_names[ids[int(v[36,ids].argmin())]]
            omission_rows.append([name,len(test),len(ids),b,100-b,float(floor[b]),float(floor[36]),worst])
        else:omission_rows.append([name,len(test),0,'','','','',''])
    write_csv(out/'all-40-family-omissions.csv',['omitted_generator','models_tested','models_accepted','best_C2','best_C4','best_floor','floor_for_36','worst_field_for_36'],omission_rows)
    write_csv(out/'all-model-family-shares.csv',['model','catalog','constraint','passes',*names],[[r['name'],r['catalog'],r['constraint'],r['validation']['passes'],*[100*w for w in r['weights']]]for r in validation['models']])
    write_csv(out/'split-by-field.csv',['C2','C4',*field_names],[[b,100-b,*v[b]]for b in range(101)])
    # Attribute every one-troop or larger split change against the predeclared 36/64 reference.
    transition_path=out/'split-transitions.npz';native=native_module(ROOT/'data/parry-replay')
    if not transition_path.exists():
        gain=[];loss=[];gain_points=[];loss_points=[]
        xs=[[0,b,0,100-b,0]for b in range(101)]
        for j,row in enumerate(models):
            with np.load(study/'evaluation'/(row['name']+'.npz'))as d:rows=d['triplets'];n=int(d['size'])
            g=native.compress_groups(rows);counts=counts_from_groups(g,n).astype(float)
            q=(native.payoff_matrix(xs,g)//3).astype(np.int16);delta=q-q[36]
            gain.append((delta>0)@counts/n);loss.append((delta<0)@counts/n)
            gain_points.append(np.maximum(delta,0)@counts/(4*n));loss_points.append(np.maximum(-delta,0)@counts/(4*n))
            np.testing.assert_allclose((gain_points[-1]-loss_points[-1]),v[:,j]-v[36,j],rtol=0,atol=1e-12)
        np.savez_compressed(transition_path,gain_probability=np.array(gain).T,loss_probability=np.array(loss).T,gain_points=np.array(gain_points).T,loss_points=np.array(loss_points).T,names=field_names)
    with np.load(transition_path)as d:gain=d['gain_probability'];loss=d['loss_probability'];gain_points=d['gain_points'];loss_points=d['loss_points']
    write_csv(out/'split-payout-changes.csv',['C2','C4','field','probability_higher_payout_than_36','probability_lower_payout_than_36','expected_positive_gain','expected_loss','net_payout_change'],
        [[b,100-b,name,float(gain[b,j]),float(loss[b,j]),float(gain_points[b,j]),float(loss_points[b,j]),float(v[b,j]-v[36,j])]for b in range(101)for j,name in enumerate(field_names)])
    # Detailed family stresses remain separate from all fitted-field selection rules.
    dest=study/'family-stress';mapping=read(dest/'family-map.json');shock_summary=read(dest/'results.json')
    with np.load(dest/'scores.npz')as d:pure=d['pure_values'];keys=d['pure_keys'].tolist();coeff=d['shock_coefficients'];shock_ids=d['shock_action_indices']
    key_index={key:i for i,key in enumerate(keys)};action_index={tuple(x):i for i,x in enumerate(actions.tolist())}
    split_ids=[action_index[(0,b,0,100-b,0)]for b in range(101)]
    family_rows=[]
    for f,name in enumerate(names):
        pp=pure[:,key_index[mapping['original'][name]]];b=int(pp[split_ids].argmax())
        w=np.array([.9**3,3*.1*.9**2,3*.1**2*.9,.1**3]);changed=coeff[:,f]@w
        family_rows.append([name,float(pp[split_ids[36]]),float(pp[split_ids[38]]),b,100-b,float(pp[split_ids[b]]),float(changed[:,36].min()),float(changed[:,38].min()),float((coeff[:,f,36,0]-changed[:,36]).max())])
    write_csv(out/'all-40-family-stresses.csv',['family','pure_36','pure_38','pure_best_C2','pure_best_C4','pure_best_payout','worst_36_at_10pct_replacement','worst_38_at_10pct_replacement','largest_36_loss_at_10pct_replacement'],family_rows)
    write_csv(out/'all-catalog-pure-families.csv',['catalog','family','law_key',*[str(b)for b in range(101)]],[[kind,name,key,*pure[split_ids,key_index[key]]]for kind,rows in mapping.items()for name,key in rows.items()])
    old_study=ROOT/'results/game2-cross-arena';old=read(old_study/'old-stress-summary.json')
    with np.load(old_study/'old-stress-scores.npz')as d:
        np.testing.assert_array_equal(d['allocations'],actions);old_values=d['values']
    write_csv(out/'all-splits-prior-stress-banks.csv',['C2','C4','old_core_mean','old_pressure_mean','old_arena_mean','lowest_old_conditional_cell'],
        [[bid,100-bid,*[old['block_values'][g][split_ids[bid]]for g in ['core','pressure','arenas']],float(old_values[split_ids[bid]].min())]for bid in range(101)])
    b=selection['selected_split_C2'];floor=v.min(axis=1);upper=(v.T+errors).min(axis=0);lower=(v.T-errors).min(axis=0)
    per_field_best=np.argmax(v,axis=0);regret=(v.max(axis=0)-v).max(axis=1);regret_best=int(regret.argmin())
    best_field=int(v[b].argmin());near=[r for r in sorted(set([b,35,36,37,38]))]
    write_csv(out/'nearby-paired-comparisons.csv',['reference_C2','other_C2','field','mean_other_minus_reference','joint_95pct_pair_error','lower','upper'],
        [[ref,other,name,float(v[other,j]-v[ref,j]),float(pair_errors[j,other,ref]),float(v[other,j]-v[ref,j]-pair_errors[j,other,ref]),float(v[other,j]-v[ref,j]+pair_errors[j,other,ref])]for ref in [36,b]for other in near for j,name in enumerate(field_names)])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,(ax,bx)=plt.subplots(1,2,figsize=(12,5),gridspec_kw={'width_ratios':[1.1,1]})
    for a in [ax,bx]:
        a.fill_between(np.arange(101),lower,upper,color='#167b80',alpha=.2,label='Joint 95% sampling bounds')
        a.plot(np.arange(101),floor,color='#126d75',lw=2,label='Lowest mean across accepted fields')
        a.axvline(b,color='#bd6726',lw=1.5,ls='--',label=f'Point leader: {b}/{100-b}')
        a.grid(alpha=.2);a.set_xlabel('Troops on castle 2; castle 4 receives the rest')
    ax.set(xlim=(0,100),ylim=(0,9),ylabel='Expected placement payout',title='Every legal C2/C4 split')
    low=max(0,b-12);high=min(100,b+12);bx.set(xlim=(low,high),ylim=(max(0,lower[low:high+1].min()-.05),min(9,upper[low:high+1].max()+.05)),title='Closer view around the point leader');bx.legend(fontsize=8)
    fig.tight_layout();fig.savefig(out/'all-split-curve.png',dpi=170);plt.close(fig)
    visible=[r for r in group_rows if r['accepted']]
    heat=np.array([v[:,groups[r['group']]].min(axis=1)for r in visible])
    fig,ax=plt.subplots(figsize=(13,max(5,.4*len(visible)+1.2)));im=ax.imshow(heat,aspect='auto',origin='upper',vmin=0,vmax=9,cmap='viridis')
    ax.set_yticks(range(len(visible)),[f"{r['label']} ({r['accepted']})"for r in visible]);ax.set_xticks(np.arange(0,101,10));ax.set(xlabel='Troops on castle 2; castle 4 receives the rest',title='Population assumptions can change the preferred split')
    for j,row in enumerate(visible):
        x=row['best_C2'];ax.scatter(x,j,s=50,facecolors='white',edgecolors='black');ax.text(min(95,x+2),j,str(x),ha='left',va='center',color='white',fontsize=8)
    fig.colorbar(im,ax=ax,label='Lowest mean placement payout in the group');fig.tight_layout();fig.savefig(out/'split-by-population-group.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,5));shares=[r['maximum_replacement_share']for r in shock_summary['continuous_share_point_estimate_summary']]
    for bid in sorted(set([b,35,36,38])):
        yy=[r['all_action_floors'][bid]for r in shock_summary['continuous_share_point_estimate_summary']]
        ax.plot(shares,yy,marker='o',label=f'{bid}/{100-bid}')
    ax.set(xlabel='Maximum independent single-family replacement share',ylabel='Lowest estimated payout over fields, families and shares',title='Deliberate family shocks, not applicant forecasts',xlim=(0,.5));ax.grid(alpha=.2);ax.legend();fig.tight_layout();fig.savefig(out/'family-shock-curves.png',dpi=170);plt.close(fig)
    summary={'groups':group_rows,'selected_split_C2':b,'minimax_regret_split_C2':regret_best,'per_field_best_C2_range':[int(per_field_best.min()),int(per_field_best.max())],
        'nearby':{str(bid):{'minimum':float(floor[bid]),'maximum':float(v[bid].max()),'max_regret':float(regret[bid]),'lower':float(lower[bid]),'upper':float(upper[bid])}for bid in sorted(set([b,35,36,37,38]))},
        'omissions_with_no_retained_model':[r[0]for r in omission_rows if r[2]==0],
        'worst_field_for_selected_split':models[best_field],'all_split_worst_field':field_names[best_field],
        'source_sha256':sha(Path(__file__)),'scope':'Review of fixed comparisons; model counts are not population probabilities.'}
    save(out/'summary.json',summary)
    if args.prepare:
        print('Prepared review assets',flush=True);return
    refined=read(study/'refinement/results.json');refined_audit=read(study/'refinement/acceptance.json')
    assert refined_audit['passed']and refined_audit['results_sha256']==sha(study/'refinement/results.json')
    with np.load(study/'refinement/scores.npz')as d:v34=d['values'];error34=d['mean_error']
    b34=refined['C3'];allocation34=refined['allocation'];lower34=np.array(refined['c34_robust_lower']);upper34=np.array(refined['c34_robust_upper'])
    certified=bool(lower34[b34]>refined['best_non_c34_prefix_upper'])
    cases=read(out/'two-compatible-populations.json')
    case_table='\n'.join(f"| {row['name']} | {row['full_cdf_gap']:.3f} | {row['C2_36_C4_64']:.3f} | {row['C3_36_C4_64']:.3f} |"for row in cases)
    write_csv(out/'all-101-C2-C4-final-joint-bounds.csv',['C2','C4','minimum','maximum','lower','upper','worst_field'],
        [[bid,100-bid,float(v[bid].min()),float(v[bid].max()),float(lower[bid]),float(upper[bid]),field_names[int(v[bid].argmin())]]for bid in range(101)])
    fig,(ax,bx)=plt.subplots(1,2,figsize=(12,5))
    for a in [ax,bx]:
        a.plot(np.arange(101),v.min(axis=1),lw=2,color='#167b80',label='Castles 2 and 4')
        a.fill_between(np.arange(101),lower,upper,color='#167b80',alpha=.15)
        a.plot(np.arange(101),v34.min(axis=1),lw=2,color='#bd6726',label='Castles 3 and 4')
        a.fill_between(np.arange(101),lower34,upper34,color='#bd6726',alpha=.15)
        a.grid(alpha=.2);a.set_xlabel('Troops on castle 2 or 3; castle 4 receives the rest')
    ax.set(xlim=(0,100),ylim=(0,9),ylabel='Lowest mean payout across the 87 accepted fields',title='Complete comparison of both 101-split families')
    bx.set(xlim=(max(0,min(b,b34)-10),min(100,max(b,b34)+10)),ylim=(min(floor[b],v34[b34].min())-.5,max(floor[b],v34[b34].min())+.2),title='Closer view of the robust leaders');bx.legend()
    fig.tight_layout();fig.savefig(out/'C2-C4-versus-C3-C4.png',dpi=170);plt.close(fig)
    write_csv(out/'both-complete-split-families.csv',['lower_target_troops','C4_troops','C2_C4_minimum','C3_C4_minimum','C2_C4_lower','C2_C4_upper','C3_C4_lower','C3_C4_upper'],
        [[bid,100-bid,float(v[bid].min()),float(v34[bid].min()),float(lower[bid]),float(upper[bid]),float(lower34[bid]),float(upper34[bid])]for bid in range(101)])
    new_id=action_index[tuple(allocation34)]
    write_csv(out/'C3-C4-prior-stress-banks.csv',['C3','C4','old_core_mean','old_pressure_mean','old_arena_mean','lowest_old_conditional_cell'],
        [[bid,100-bid,*[old['block_values'][g][action_index[(0,0,bid,100-bid,0)]]for g in ['core','pressure','arenas']],float(old_values[action_index[(0,0,bid,100-bid,0)]].min())]for bid in range(101)])
    audits=[read(study/('acceptance-'+part+'.json'))for part in ['calibration','evaluation','stresses']]
    assert all(r['passed']for r in audits)
    count=len(models);tested=len(validation['models']);omitted=sum(r[2]>0 for r in omission_rows)
    order=np.argsort(-floor);top=[]
    for bid in order[:8]:
        top.append(f"| {bid} / {100-bid} | {floor[bid]:.3f} | {v[bid].max():.3f} | {lower[bid]:.3f} to {upper[bid]:.3f} |")
    category_table=[]
    for row in group_rows:
        if row['group']in ['overall','omissions']:continue
        score='No model passed'if not row['accepted']else f"{row['best_C2']} / {row['best_C4']}, floor {row['best_floor']:.3f}"
        category_table.append(f"| {row['label']} | {row['accepted']} / {row['tested']} | {score} |")
    worst=models[best_field];same_as_before=(b==36)
    headline=f"The broader population test changes the working lead to **{tuple(allocation34)}**, targeting castles 3 and 4. Within the complete C2/C4 family, **(0,{b},0,{100-b},0)** is the point leader."
    pool_choice=selection['selected_allocation'];pool_floor=selection['selected_minimum_mean'];pool_is_split=pool_choice[0]==pool_choice[2]==pool_choice[4]==0
    plausible=refined['sampling_plausible_C2_values_joint_with_c34']
    same_statement='The all-760 comparison also selects a C2/C4 allocation.'if pool_is_split else 'The all-760 comparison selects an allocation outside the exact C2/C4 family, so this study does not justify restricting the final submission to C2/C4.'
    report=ROOT.parent/'27-game2-population-family-and-split-challenge.md'
    text=f'''**Game 2: family stress changes the leading castle pair**

{headline}

The new C3/C4 lead has a lowest modeled mean of **{v34[b34].min():.3f} placement points** among the {count} accepted fields. The best C2/C4 floor is **{floor[b]:.3f}**. The floors for C2/C4 36/64 and 38/62 are **{floor[36]:.3f}** and **{floor[38]:.3f}**. These are conditional model results, not guarantees against actual applicants.

The complete 760-vector comparison, evaluated on 262,144 groups per field, selects **{tuple(pool_choice)}**, with a lowest modeled mean of **{pool_floor:.3f}**. The 101 exact C2/C4 splits use **1,048,576 groups per field**. {same_statement} Comparisons across those two numerical precision levels should allow for sampling error.

That outside-family result triggered an equal-precision check of **every one of the 101 C3/C4 splits**, using the same million-group banks. All those allocations were already in the initial 760-vector freeze. The C3/C4 splits not excluded by sampling bounds have C3 = **{refined['sampling_plausible_C3_values']}**. The selected split's robust lower bound is {lower34[b34]:.3f}; the largest robust upper bound for any frozen allocation outside C3/C4 is {refined['best_non_c34_prefix_upper']:.3f}. {'This separates the selected C3/C4 allocation from every frozen non-C3/C4 allocation for the worst-population decision rule at joint 95% sampling confidence, conditional on these fields.'if certified else 'The numerical bounds do not certify separation from every frozen non-C3/C4 allocation.'} It is not pointwise superiority in every field, a real-population guarantee, or a full-action optimum.

![Complete comparison of the two castle-pair families]({out/'C2-C4-versus-C3-C4.png'})

The change is driven by population assumptions. A field with more equal-centered active allocations still passes the public-score check but lowers the C2/C4 floor. The C3/C4 candidate retains a higher worst-field score. Adding that credible alternative changes the robust decision even though the original seventeen-field comparison favored C2/C4.

Two concrete examples show why the hidden allocations matter:

| Accepted population | Full-CDF gap | C2/C4 36/64 payout | C3/C4 36/64 payout |
|---|---:|---:|---:|
{case_table}

![Two compatible score distributions, different entrant preferences]({out/'two-score-compatible-populations.png'})

Both match the public scores within our tolerance, but they prefer different entries. Their [full comparison table]({out/'two-compatible-populations.csv'}) contains the nearby splits too. A model's incumbent tail screen also does not cap a new entrant's predicted payout at the published leaderboard maximum.

The [two-family table]({out/'both-complete-split-families.csv'}) contains both complete curves and their bounds. The [full C3/C4 table]({study/'refinement/all-101-C3-C4-splits.csv'}) identifies its weakest field for every split.

**Coverage and what was held fixed**

The 760-vector pool was fixed before this stage and includes all 101 legal integer C2/C4 splits, including both extremes. We refitted twelve predefined catalog variants and all forty individual family omissions. This produced {tested} frozen fitted/supplied models, of which {count} passed the new independent score-CDF screen. At least one model passed for {omitted} of the forty omissions. We tested every original family as a pure opponent population and as a replacement direction inside each of the seventeen supplied fields.

Across twelve catalogs, 480 family slots reduce to **268 distinct frozen pure-family laws** after deduplication. All 760 candidates were tested against each. The replacement study contains **1,360 conditional group banks**, covering one or two replaced opponents, with the zero- and three-replacement cases supplied separately. It tests all 101 exact splits plus six fixed outside-family references.

Every opponent vector remains a legal full joint allocation of 100 troops. No castle-marginal approximation or set of pairwise games replaces simultaneous four-player scoring. Every existing study and recommendation record is preserved.

**All 101 split results**

Rankings maximize the lowest mean across accepted fields. Model counts do not assign probabilities to population hypotheses. The interval is a joint sampling interval for the split's minimum across these fixed laws, not an uncertainty interval for the hidden real population.

| C2 / C4 troops | Lowest modeled mean | Highest modeled mean | Joint sampling bounds on the minimum |
|---|---:|---:|---:|
{chr(10).join(top)}

The [complete 101-row table]({out/'all-101-C2-C4-final-joint-bounds.csv'}) includes every allocation, its weakest field, and the final joint sampling bounds. The [split-by-field table]({out/'split-by-field.csv'}) shows each individual population rather than an invented weighted average.

![Every split and sampling uncertainty]({out/'all-split-curve.png'})

The split values not excluded by the simultaneous sampling bounds are C2 = **{plausible}**, with C4 receiving the rest. The minimax-regret sensitivity rule selects **{regret_best}/{100-regret_best}**, measuring shortfall from the best exact split in each field. Across individual accepted fields, the best C2 point estimate ranges from {summary['per_field_best_C2_range'][0]} through {summary['per_field_best_C2_range'][1]}. A single universally optimal split is not established.

**Which population assumptions change the result**

The modified target laws cover zero scouts, substantially wider scouts, uniform/equal/value-centered/asymmetric target bids, more mass near the castle-4 threshold, and small legal troop transfers. The five named exact point families remain intact. An omission removes one generator family, not every matching allocation that may also appear in another family.

| Population model group | Accepted / tested | Best split within that group |
|---|---:|---|
{chr(10).join(category_table)}

The forty omission results are in [their full table]({out/'all-40-family-omissions.csv'}). No passing model was found for these omissions: {', '.join(summary['omissions_with_no_retained_model'])or 'none'}. That is a finite search outcome, not proof those families must occur. We keep optimizer statuses and failed validation results, rather than silently treating unsuccessful fits as impossible populations.

![Population-group split curves]({out/'split-by-population-group.png'})

The field setting the selected split's floor is **{worst['name']}**, from catalog **{worst['catalog']}** under constraint **{worst['constraint']}**. Its full-CDF gap is {worst['validation']['maximum_cdf_gap']:.4f}, with 181-grid mean gap {worst['validation']['mean_grid_cdf_gap']:.4f}. The [complete family-weight table]({out/'all-model-family-shares.csv'}) lists all forty percentages in every tested model, with its acceptance flag. These are fitted hypotheses, not observed applicant shares or posterior probabilities.

The screen requires a full-CDF maximum gap at most 0.085, both mean-gap diagnostics at most 0.035, and no more than 0.015 modeled incumbent mass above the published maximum plus 0.2. Own-action sampling, opponent sampling, finite public reshuffles and the unobserved allocation catalog all limit interpretation. Multiple fits to the same public scores are not independent evidence that the underlying bids are correct.

**What a one-troop change actually does**

Every split used the same opponents within each model, and we saved all 101-by-101 payout cross-products. This lets us distinguish small paired numerical differences from changes caused by a different population.

For example, moving one troop from castle 4 to castle 2 can improve the castle-2 result against bids at the threshold while weakening the castle-4 result against its threshold bids. The final payout also depends on how the other three players divide the remaining castle points. We therefore measure higher and lower *placement payouts* directly.

The [payout-change table]({out/'split-payout-changes.csv'}) reports, for every split and accepted field, its probability of improving or hurting the payout relative to 36/64, the expected positive gain, the expected loss, and the net change. The [nearby paired intervals]({out/'nearby-paired-comparisons.csv'}) use the same complete groups.

Sampling bounds use a two-sided finite union of the empirical Bernstein bound in [Maurer and Pontil, Theorem 4](https://www.cs.mcgill.ca/~colt2009/papers/012.pdf). They account jointly for all 101 means and 5,050 unordered pairs in every retained field. Lower variance can give nearby split comparisons much narrower intervals than generic payoff-range bounds. They exclude population-model error and provide no full-action optimality certificate.

**Deliberate counterfamily stresses**

These tests remain separate from the public-score-compatible selection. Each opponent independently switches from a supplied base field to one chosen family. Conditional cases with zero, one, two or three replacements define a cubic in the replacement share. We check its endpoints and interior stationary points, so the point-estimate minima cover every share up to the stated limit, rather than just a handful of tested shares.

{chr(10).join(['| Maximum replacement share | Best split | Its lowest estimated mean | 36/64 lowest mean |','|---|---:|---:|---:|']+[f"| {100*r['maximum_replacement_share']:g}% | {r['best_split_C2']}/{r['best_split_C4']} | {r['best_split_floor']:.3f} | {r['C2_36_floor']:.3f} |"for r in shock_summary['continuous_share_point_estimate_summary']])}

The low-share winners can be separated by only a few thousandths of a point, which is too small to treat as a firm change in strategy with these conditional sample sizes. Larger hostile shares show how the allocation can move substantially toward castle 4. These shares are hypothetical changes, not predicted applicant percentages. Conditional estimates and their cubic interpolation still have sampling error. Single-family replacement directions also do not exhaust simultaneous changes involving several different families.

![Single-family shock curves]({out/'family-shock-curves.png'})

The [forty-family stress table]({out/'all-40-family-stresses.csv'}) gives pure-family results and 10% replacement diagnostics. The [complete twelve-catalog pure-family table]({out/'all-catalog-pure-families.csv'}) contains all 101 split scores for every catalog/family slot, with shared laws explicitly identified.

The previous synthetic populations and adapting arenas also remain in the comparison. The [all-split prior-stress table]({out/'all-splits-prior-stress-banks.csv'}) reuses their independently verified eighty-five saved banks, with all 101 exact splits present in the old 760-vector score table. For {b}/{100-b}, the old core/pressure/arena block means are {old['block_values']['core'][split_ids[b]]:.3f}, {old['block_values']['pressure'][split_ids[b]]:.3f}, and {old['block_values']['arenas'][split_ids[b]]:.3f}. These are retrospective stresses, not newly accepted applicant forecasts.

The new C3/C4 lead's old core/pressure/arena means are {old['block_values']['core'][new_id]:.3f}, {old['block_values']['pressure'][new_id]:.3f}, and {old['block_values']['arenas'][new_id]:.3f}. Its [complete C3/C4 prior-stress table]({out/'C3-C4-prior-stress-banks.csv'}) makes those tradeoffs visible too.

**What this establishes**

We have now tested every exact C2/C4 split across the declared family stresses and freshly checked population variants. We can choose a split under an explicit population model and quantify simulation error. The hidden applicant allocation distribution remains unknown, so neither this stage nor additional runs of the same models can establish a 100% real-world optimum.

There is also a concrete zero-payout counter to the new C3/C4 lead: three copies of **(1,1,0,65,33)** give **(0,0,36,64,0)** only three castle points while each opponent gets four. The entrant finishes last. The [exact rational check]({out/'new-lead-counterexample.json'}) records that outcome. The new recommendation is a model-conditional decision, not a universal security guarantee.

The decision is recorded as provisional in the new working record; earlier records remain intact. Game 1 remains **(3,8,4,44,41)**. No submission has been made. All three original audits, the C3/C4 refinement audit, and all 117 unit tests passed. The final confidence allocation covers the 760-vector prefix class, the predeclared C2/C4 class, and all 760 potential full-group means and pairs for the C3/C4 refinement. That last, wider class accounts for discovering the C3/C4 family from the pool. Reproduction commands and source/input hashes are in [the project]({ROOT/'README.md'}).
'''
    assert '\u2014'not in text;report.write_text(text.rstrip()+'\n')
    working={'status':'Provisional model-conditional choice after catalog and all-split challenge; no submission',
        'allocation':allocation34,'all_760_prefix_point_leader':pool_choice,'best_exact_C2_C4_split':[0,b,0,100-b,0],
        'best_exact_C3_C4_split':allocation34,'C3_C4_minimum':float(v34[b34].min()),'sampling_plausible_C3_values':refined['sampling_plausible_C3_values'],
        'selected_C3_C4_separates_non_C3_C4_frozen_pool':certified,
        'conditional_separation_metric':'Minimum expected payout across the 87 fixed accepted models, not superiority in each individual model or the hidden real population',
        'candidate_pool_minimum':pool_floor,'exact_split_minimum':float(floor[b]),'accepted_fields':count,'tested_fields':tested,
        'sampling_plausible_split_C2_values':plausible,'previous_working_record':'results/game2-working-candidate-v2.json',
        'previous_working_allocation':[0,36,0,64,0],'game1_unchanged':[3,8,4,44,41],
        'freeze_sha256':sha(study/'freeze.json'),'selection_sha256':sha(study/'selection.json'),'report_sha256':sha(report),
        'limitations':['Unknown real population','Finite frozen candidate pool','Finite local model fits and sampled catalog supports','Diagnostic public-CDF tolerances do not identify hidden bids','Deliberate stresses are not applicant forecasts']}
    save(ROOT/'results/game2-working-candidate-v3.json',working)
    save(out/'completion.json',{'passed':True,'report_sha256':sha(report),'source_sha256':sha(Path(__file__)),
        'working_record_sha256':sha(ROOT/'results/game2-working-candidate-v3.json'),'audit_sha256':{part:sha(study/('acceptance-'+part+'.json'))for part in ['calibration','evaluation','stresses']},'refinement_acceptance_sha256':sha(study/'refinement/acceptance.json')})
    print('Saved',report,flush=True)


if __name__=='__main__':main()
