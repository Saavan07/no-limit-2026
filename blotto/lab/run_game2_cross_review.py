"""Summarize verified cross-arena evidence without changing frozen studies."""
from pathlib import Path
import json
import numpy as np
from game2_lab.cross_arena import read,load_catalog
from run_population_study import ROOT,save,sha,write_csv


def main():
    study=ROOT/'results/game2-cross-arena';out=study/'review';out.mkdir(exist_ok=True)
    result=read(study/'results.json');audit=read(study/'acceptance.json')
    sensitivity=read(study/'sensitivity/results.json');sa=read(study/'sensitivity/acceptance.json')
    assert audit['passed']and sa['passed']and audit['results_sha256']==sha(study/'results.json')
    assert sa['results_sha256']==sha(study/'sensitivity/results.json')
    freeze=read(study/'freeze.json');vendor=ROOT/'data/parry-replay';fit,catalog,fields=load_catalog(vendor)
    with np.load(study/'fresh-scores.npz')as data:xs=data['allocations'];values=data['values']
    with np.load(study/'sensitivity/scores.npz')as data:shift=data['values'];shift_names=data['names'];strategic=data['strategic_exact']
    ids={tuple(x):i for i,x in enumerate(xs.tolist())}
    rank=sorted(range(len(xs)),key=lambda i:(-values[i].min(),tuple(xs[i])))
    rank_number={i:j+1 for j,i in enumerate(rank)}
    cdf=read(study/'cdf-comparison.json');family_weights=np.array([f['weights']for f in fields])
    exact_a=np.array([w[np.all(x==[0,0,0,0,100],axis=1)].sum()/w.sum()for x,w in catalog.values()])
    exact_a_shares=family_weights@exact_a
    write_csv(out/'population-family-shares.csv',['family',*[f['name']for f in fields],'minimum_percent','median_percent','maximum_percent'],
        [[name,*[100*w for w in family_weights[:,j]],100*family_weights[:,j].min(),100*np.median(family_weights[:,j]),100*family_weights[:,j].max()]for j,name in enumerate(catalog)])
    write_csv(out/'public-cdf-diagnostics.csv',['field','strict_original','original_coarse_gap','original_full_gap','fresh_coarse_gap','fresh_full_gap','fresh_coarse_pass','fresh_full_pass'],
        [[f['name'],f['strict'],cdf['original'][f['name']]['original_26_threshold_max_gap'],cdf['original'][f['name']]['maximum_cdf_gap'],cdf['fresh'][f['name']]['original_26_threshold_max_gap'],cdf['fresh'][f['name']]['maximum_cdf_gap'],cdf['fresh'][f['name']]['passes_original_26_threshold_screen'],cdf['fresh'][f['name']]['passes_full_curve_screen']]for f in fields])
    # Explain the mechanism directly on each saved full legal group bank.
    mechanism={};x=np.array([0,36,0,64,0])
    for f in fields:
        with np.load(study/'fresh-fields'/(f['name']+'.npz'))as data:rows=data['triplets']
        maxima=rows.max(axis=1);wins=x>maxima;ties=x==maxima
        mechanism[f['name']]={'outright_castle_win_probabilities':wins.mean(axis=0).tolist(),
            'castle_top_tie_probabilities':ties.mean(axis=0).tolist(),
            'probability_outright_wins_both_2_and_4':float(np.mean(wins[:,1]&wins[:,3])),
            'actual_exact_allin5_share':float(exact_a_shares[[v['name']for v in fields].index(f['name'])])}
    save(out/'castle-mechanism.json',mechanism)
    # Contrast whole populations; retain separate panels instead of inventing pooled population weights.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10})
    show=[('36/64 on castles 2/4',[0,36,0,64,0]),('37/63 on castles 2/4',[0,37,0,63,0]),('38/62 on castles 2/4',[0,38,0,62,0]),
          ('39/61 on castles 3/4',[0,0,39,61,0]),('Earlier scout',[0,3,35,57,5]),('All-in castle 5',[0,0,0,0,100]),('Our earlier choice',[15,18,49,12,6])]
    fig,(ax,bx)=plt.subplots(1,2,figsize=(12,5),gridspec_kw={'width_ratios':[1.55,1]})
    for j,(label,allocation)in enumerate(show):
        i=ids[tuple(allocation)];y=len(show)-1-j
        ax.plot([values[i].min(),values[i].max()],[y,y],lw=6,solid_capstyle='round',color='#167b80'if j<3 else '#78919c')
        ax.scatter([values[i].min(),values[i].max()],[y,y],s=32,c='white',edgecolors='#167b80',zorder=3)
        bx.scatter(strategic[i],y,s=65,color='#bd6726',zorder=3)
        bx.text(strategic[i]+.025,y,f'{strategic[i]:.3f}',va='center',fontsize=9)
    ax.set_yticks(range(len(show)),[s[0]for s in reversed(show)])
    ax.set(xlim=(0,9),xlabel='Mean placement payout',title='Range across 17 fitted catalog fields')
    bx.set_yticks(range(len(show)),['']*len(show));bx.set(xlim=(3.35,4.1),xlabel='Mean placement payout',title='Exact supplied strategic checkpoint')
    bx.axvline(3.75,lw=1,ls='--',color='gray')
    for a in [ax,bx]:a.grid(axis='x',alpha=.2);a.set_ylim(-.7,len(show)-.3)
    fig.suptitle('A strong fitted-field choice still has a strategic tradeoff',fontsize=14)
    fig.tight_layout();fig.savefig(out/'candidate-comparison.png',dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,5.5));public=np.sort(read(vendor/'public-scores.json')['scores']);xx=np.linspace(0,9,2001)
    ax.plot(xx,np.searchsorted(public,xx,side='right')/len(public),color='black',lw=2.6,label='Published scores')
    for name in ['fit_3','fit_13','fit_14']:
        f=next(f for f in fields if f['name']==name)
        with np.load(study/'cdf-fresh'/(name+'.npz'))as data:u=data['values'];families=data['own_families']
        w=np.array(f['weights'])[families]/freeze['configuration']['cdf_fresh_actions_per_family'];order=np.argsort(u);u=u[order];cw=np.cumsum(w[order]);ix=np.searchsorted(u,xx,side='right');yy=np.r_[0.,cw][ix]
        ax.plot(xx,yy,lw=1.8,label=name+', full gap '+f"{cdf['fresh'][name]['maximum_cdf_gap']:.3f}")
    old_path=ROOT/'results/game2-handoff-integration/cdf/arena__replicator__fifth__balanced__tail.npz'
    with np.load(old_path)as data:u=data['rank_sum4']/(4*int(data['size']));w=data['own_weights'].astype(float)
    w/=w.sum();order=np.argsort(u);u=u[order];cw=np.cumsum(w[order]);ix=np.searchsorted(u,xx,side='right');ax.plot(xx,np.r_[0.,cw][ix],lw=1.7,ls='--',color='#ba4f58',label='Our earlier replicator tail stress')
    ax.set(xlabel='Incumbent mean placement payout',ylabel='Cumulative population mass',xlim=(0,9),ylim=(0,1),title='Fitted fields explain more of the public score distribution')
    ax.grid(alpha=.2);ax.legend(fontsize=9,loc='lower right');fig.tight_layout();fig.savefig(out/'public-score-comparison.png',dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,5))
    for label,allocation in [show[0],show[2],show[-1]]:
        i=ids[tuple(allocation)];yy=[values[i].min()]
        for share in [10,25,50]:
            cols=[j for j,name in enumerate(shift_names)if name.endswith('_share'+str(share))]
            yy.append(shift[i,cols].min())
        yy.append(strategic[i]);ax.plot([0,.1,.25,.5,1],yy,marker='o',lw=2,label=label)
    ax.axhline(3.75,ls='--',color='gray',lw=1,label='Symmetric self-play mean')
    ax.set(xlabel='Share replaced by supplied strategic checkpoint',ylabel='Lowest modeled mean among 17 base-field variants',xlim=(0,1),ylim=(2.5,6),title='Five tested response shares; lines connect estimates')
    ax.grid(alpha=.2);ax.legend(fontsize=9);fig.tight_layout();fig.savefig(out/'response-share-sensitivity.png',dpi=170);plt.close(fig)
    table=[]
    for label,allocation in show:
        i=ids[tuple(allocation)]
        table.append(f"| `{tuple(allocation)}` | {rank_number[i]} | {values[i].min():.3f} to {values[i].max():.3f} | {strategic[i]:.3f} |")
    ours=freeze['named_indices']['our_v1'];parry=freeze['named_indices']['parry_36_64'];delta=values[parry]-values[ours]
    epsilon=result['conditional_simultaneous_95_percent_cell_error_bound']
    selected_min=values.min(axis=1)
    surviving=[j for j,f in enumerate(fields)if cdf['fresh'][f['name']]['passes_full_curve_screen']]
    subset_winner=min(range(len(xs)),key=lambda i:(-values[i,surviving].min(),tuple(xs[i])))
    assert xs[subset_winner].tolist()==[0,36,0,64,0]
    report=ROOT.parent/'26-game2-cross-arena-replay.md'
    text=f'''**Game 2 cross-arena replay: 36/64 becomes the working lead**

I would now carry **(0,36,0,64,0)** forward as our leading Game 2 candidate. It wins the declared comparison on Parry's fitted population fields and clearly beats our earlier **(15,18,49,12,6)** there. Our earlier arenas remain useful stress tests, but their public-score mismatch means we should give their applicant predictions less weight.

This is a provisional working choice. **(0,38,0,62,0)** is the main nearby challenger because it gives up a little fitted-field performance for more strength against the supplied strategic checkpoint. No strategy has been submitted or locked. Game 1 remains **(3,8,4,44,41)**.

**What reproduced and what was tested**

The supplied original replay matched **all 68 available saved candidate-by-field results exactly**, including the original 36/64 range of 5.365043 to 6.992648. The original public-CDF validation also reproduced to floating-point precision. We used Python {freeze['versions']['python']} and NumPy {freeze['versions']['numpy']}; the package records NumPy 2.5.4. This is verified numerical agreement with our actual environment, not a claim that we installed the recorded version.

Before new candidate evaluation, we froze **760 distinct legal allocations**: our previous 498, plus Parry's final-audit entries and strategic checkpoint candidates and remembered deviations. All seventeen retained fields then received **524,288 fresh opponent trios**, a total of 8,912,896 groups. Both engines agreed cell by cell for the named candidates on every group. All original 191 candidate sums also matched on all eighty-five old banks.

The primary comparison maximizes the lowest modeled expected 9/4/2/0 placement payout across the seventeen retained fields. The table's rank uses that rule within this 760-vector pool. A range spans different population hypotheses, not a confidence interval or individual-game outcomes. The strategic column integrates every iid triple from the supplied seventy-allocation checkpoint.

| Allocation, castles 1 through 5 | Rank / 760 | Range across 17 fitted fields | Strategic checkpoint |
|---|---:|---:|---:|
{chr(10).join(table)}

36/64 leads under all four declared rules: maximin over seventeen fields, maximin over the original nine stricter fields, and minimax regret on each set. The regret benchmark is the best of these 760 frozen candidates in each field, not a certified global best response.

It beats our earlier vector in **every one of the seventeen fields**, by **{delta.min():.3f} to {delta.max():.3f} payout points**. This gap is far larger than numerical sampling error. Our earlier vector ranks {rank_number[ours]}th here; its old winning rank applied to our old populations and decision rule.

![Candidate comparison]({out/'candidate-comparison.png'})

**How much trust to put in the population fits**

The original check compares incumbent-score CDFs at twenty-six selected thresholds. All seventeen fields pass that same screen again with independently sampled own allocations and opponent groups. A stricter check uses every CDF jump and a separate 181-point mean-gap grid. Thirteen fields pass on the original draws and fourteen on the fresh draws. The fresh full-curve failures are fit_9, fit_14, and fit_28.

These pass counts are diagnostics with own-action and opponent-sampling error. They are not formal identification of the hidden applicant population, confidence regions, or proof that a field failing one tolerance is impossible. The seventeen-field primary comparison stays unchanged. Descriptively removing the three fresh full-curve failures still leaves 36/64 as the leader.

This resolves the apparent inconsistency with our earlier review. We previously checked our arenas using the full curve, whereas Parry used twenty-six thresholds. Both checks now run on the same supplied fields, and those fields are substantially closer to the published score distribution than our previous arena models.

![Public score comparison]({out/'public-score-comparison.png'})

There is **no single inferred population percentage table**. The seventeen retained fits contain different forty-family weight vectors. They are alternative hypotheses, without assigned probabilities over fits. For example, the actual exact-all-in-five allocation share ranges from **{100*exact_a_shares.min():.2f}% to {100*exact_a_shares.max():.2f}%** after counting its presence inside every family, not just the family named exact_allin5. Family labels can overlap in their allocations.

The [complete family shares]({out/'population-family-shares.csv'}) list percentages in every field, plus descriptive minima, medians and maxima. A median across fits is a summary, not our estimated applicant share. The [CDF diagnostics]({out/'public-cdf-diagnostics.csv'}) expose both screen definitions and every field's results. The original public snapshot is dated {fit['source_generated_at']}, and its delivered copy contains scores only.

**What happens when opponents become more strategic**

We separately replaced 10%, 25%, and 50% of each of the seventeen base fields with Parry's saved strategic population, generating fifty-one new stress fields. These shifts are sensitivity tests, not fresh applicant forecasts, and were not refitted to the public CDF. 36/64 beats our earlier vector in all fifty-one, and also in the pure strategic checkpoint.

36/64's lowest payout among the fifty-one shifts is {shift[parry].min():.3f}; its exact pure-checkpoint payout is {strategic[parry]:.3f}. The checkpoint is approximate: its stored best-response allocation earns {sensitivity['stored_best_response_payoff_reproduced']:.6f}, about 0.275 above 3.75. We reproduced that allocation's payoff, but did not repeat its full 4.6-million-action global search. This is not a certified Nash equilibrium.

![Response share sensitivity]({out/'response-share-sensitivity.png'})

The nearby 37/63 split has a fitted-field floor of 5.372 and checkpoint payout of 3.738. The 38/62 split has a fitted-field floor of 5.326 and checkpoint payout of 3.829. Thus, a preference for stronger performance against near-strategic opponents can reasonably change the last two troops. In an illustrative rule requiring at least 3.75 in every tested shift and the exact checkpoint, 38/62 is the best fitted-field choice in the frozen pool. That is an additional preference, not the primary rule or a guarantee outside those tested laws.

The connecting lines in the plot show tested shares only. We have not certified the full continuous share interval. We also retain the old stress losses: 36/64's old core/pressure/arena block means are 5.552, 3.852, and 3.787, versus 5.969, 5.561, and 4.180 for our earlier vector. Those constructed extremes did not disappear when the candidate changed.

**Why this allocation can work, and its limits**

Winning castles 2 and 4 outright secures six raw castle points. In these fitted populations, that often puts the entrant above several opponents who split their effort and points across other castles. The goal is placement payout, so six raw points can be very valuable without winning a majority of the fifteen castle points. This supports your broader idea of giving up castle 5, while the exact 2/4 split depends on opponent behavior. The per-field castle win and tie probabilities are saved in [the mechanism table]({out/'castle-mechanism.json'}).

It still has a zero-payout counter: three copies of **(1,0,1,65,33)** leave 36/64 with only castle 2's two points, while each opponent gets 13/3 points. The fitted-field result is not a universal guarantee. Score-compatible models can also hide different underlying allocations, and changing the family catalog or within-family scout distribution may change the preferred entrant.

The simultaneous 95% sampling allowance is **+/-{epsilon:.5f} points per candidate/field cell**, conditional on this frozen pool and these seventeen laws. This implies at most {2*epsilon:.5f} maximin shortfall relative to the best frozen candidate under those same laws. It does not cover model error or omitted allocations. The 36/64 versus 37/63 floor gap is only about 0.004, so the broad simultaneous bound does not certify a uniquely best split.

**What comes next**

The important remaining test is the population catalog itself: broaden scout amounts and target-split assumptions, remove influential families in turn, refit each variant against the same public-score evidence, and compare 36/64, 37/63, 38/62 and any independently discovered challengers. Freeze the resulting candidates before a fresh final comparison. More rounds against unchanged fitted fields would mostly reduce numerical error; they would not resolve this modeling uncertainty.

The replay and response-share verifiers passed, and all 110 unit tests passed. The prior recommendation files and previous studies remain intact. A small interrupted-run repair handled absent old-bank weights as numeric zero; the pre-repair freeze is preserved, and a verifier confirms that the candidate pool, seeds, populations, scorer and selection rule did not change.

All candidate rows are in [the comparison table]({study/'candidate-comparison.csv'}). The reproducible code, source provenance, saved group banks, two verifiers and figures are in [the project]({ROOT/'README.md'}). Run instructions are in the project README.
'''
    assert '\u2014'not in text
    report.write_text(text.rstrip()+'\n')
    working={'status':'Provisional working lead; catalog robustness challenge pending; not submitted or locked',
        'allocation':[0,36,0,64,0],'nearby_challengers':[[0,37,0,63,0],[0,38,0,62,0]],
        'historical_v1_allocation_unchanged':[15,18,49,12,6],'game1_allocation_unchanged':[3,8,4,44,41],
        'reason':'Leads all four declared fitted-catalog comparisons, clear advantage over historical choice, and stronger public-score compatibility than prior synthetic arena populations.',
        'fresh_fitted_field_minimum':float(values[parry].min()),'fresh_fitted_field_maximum':float(values[parry].max()),
        'strategic_checkpoint_payout':float(strategic[parry]),'report_sha256':sha(report),
        'freeze_sha256':sha(study/'freeze.json'),'acceptance_sha256':sha(study/'acceptance.json'),
        'sensitivity_acceptance_sha256':sha(study/'sensitivity/acceptance.json'),
        'limitations':['Hypothetical fields, not identified applicant allocations','No global full-action optimality certificate','No universal positive payout floor for this vector','Nearby split not uniquely certified']}
    path=ROOT/'results/game2-working-candidate-v2.json'
    if path.exists():assert read(path)==working
    else:save(path,working)
    save(out/'summary.json',{'source_sha256':sha(Path(__file__)),'report_sha256':sha(report),
        'working_record_sha256':sha(path),'fresh_full_curve_surviving_fields':len(surviving),
        'survivor_subset_leader':xs[subset_winner].tolist(),'actual_exact_allin5_shares':dict(zip([f['name']for f in fields],exact_a_shares.tolist())),
        'named_primary_ranks':{name:rank_number[i]for name,i in result['named_indices'].items()},
        'scope':'Review only; original candidate freeze and selection records unchanged.'})
    print('Saved',report)


if __name__=='__main__':main()
