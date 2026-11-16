"""Game 2 exact scorer foundation and illustrative structural controls."""
from fractions import Fraction
from pathlib import Path
import argparse,json,platform,shutil,sys,time
import numpy as np
from blotto_lab.core import DEFAULT_GAME,Population
from game2_lab.core import (reference_group,exact_evaluation,shared_triplets,evaluate_shared,fixed_sample_bound)
from run_multiwave_study import ROOT
from run_population_study import save,sha,write_csv


def controls():
    candidates={'equal':(20,20,20,20,20),'value_proportional':(7,13,20,27,33),'game1_selected':(3,8,4,44,41),
                'all_five':(0,0,0,0,100),'four_five':(0,0,0,50,50),'three_five':(0,0,38,0,62),
                'spread_without_one':(0,15,25,30,30),'four_focus':(3,9,15,65,8)}
    populations={
        'equal_clones':Population.from_counts({(20,20,20,20,20):4}),
        'proportional_clones':Population.from_counts({(7,13,20,27,33):4}),
        'single_castle_forts':Population.from_counts({tuple(100 if i==j else 0 for i in range(5)):2 for j in range(5)}),
        'five_priority_anchors':Population.from_counts({(0,0,37,0,63):2,(0,0,0,44,56):2,(12,25,0,0,63):2,(12,0,38,50,0):2,(0,22,33,45,0):2})}
    return candidates,populations


def joint_counterexample():
    p=Population.from_people([(0,0,50,50,0),(25,25,0,0,50)])
    q=Population.from_people([(0,0,50,0,50),(25,25,0,50,0)])
    x=(0,0,50,50,0);a=exact_evaluation(x,p);b=exact_evaluation(x,q)
    assert p.histograms()==q.histograms()and p.table().expected_score(x)==q.table().expected_score(x)
    assert a.expected_castle_points==b.expected_castle_points and a.expected_payout!=b.expected_payout
    return {'candidate':x,'population_p':p.to_dict(),'population_q':q.to_dict(),'game1_score':p.table().expected_score(x),'game2_p':a,'game2_q':b}


def objective_reversal():
    rng=np.random.default_rng(7202631)
    rows=[]
    for j in range(128):
        bars=np.sort(rng.choice(24,4,replace=False));rows.append(tuple(map(int,5*(np.diff(np.r_[-1,bars,24])-1))))
    pop=Population.from_counts([(rows[0],4),(rows[1],3),(rows[2],2)])
    evaluations=[exact_evaluation(x,pop)for x in rows[3:]]
    pairs=[(a,b)for a in range(len(evaluations))for b in range(len(evaluations))
           if evaluations[a].expected_castle_points>evaluations[b].expected_castle_points and evaluations[a].expected_payout<evaluations[b].expected_payout]
    if not pairs:raise AssertionError('Illustrative fixture did not expose an objective reversal')
    a,b=max(pairs,key=lambda p:(evaluations[p[1]].expected_payout-evaluations[p[0]].expected_payout,
                               evaluations[p[0]].expected_castle_points-evaluations[p[1]].expected_castle_points))
    return {'population':pop.to_dict(),'higher_castle_candidate':rows[a+3],'higher_payout_candidate':rows[b+3],
            'higher_castle_evaluation':evaluations[a],'higher_payout_evaluation':evaluations[b],
            'construction':'Exact comparison within 125 seeded coarse fixtures; illustrative, not strategy-search evidence'}


def report(out,r):
    j=r['joint_counterexample'];v=r['objective_reversal']
    lines=['**Game 2 foundation: exact four-player rank scoring**','',
        'A new broad deep-research round is not necessary before this foundation. The supplied research '
        'already identifies rank-based scoring, full-vector correlations, tied places, duplicate applicants, '
        'shared triplets, and reduced-game audits. A targeted primary-source check confirms that standard '
        'multiplayer additive Blotto results do not directly solve this discrete rank-prize objective.','',
        'The official application awards castles among maximum bidders, ranks the four total castle '
        'scores, and pays 9,4,2,0, sharing occupied prizes for ties. The scoring implementation therefore '
        'preserves both the castle layer and the final payout layer. The exact number and implementation '
        'of regroupings are not published.','',
        '**Five castle points can mean different payouts**','',
        'An all-in castle-5 player can face opponents concentrating separately on castle 4, castle 3, '
        'and castles 1+2. The castle totals are (5,4,3,3), giving the focal player first place and '
        'a payout of 9. With opponents focusing instead on castles 3+4, castle 2, and castle 1, the '
        'totals become (5,7,2,1), and the same focal allocation earns 4 for second place.','',
        '**Castle marginals are insufficient for rank payout**','',
        f"The saved example uses candidate {j['candidate']}. Its two opponent populations have exactly "
        'the same marginal histogram at each castle and legal 100-troop vectors. Every Game 1 candidate '
        'therefore has identical expected score under those marginals. For this candidate, even its '
        'expected Game 2 castle points are identical, but its rank payouts differ:','',
        '| Quantity | Population P | Population Q |', '| --- | ---: | ---: |',
        f"| Game 1 expected points | {float(j['game1_score']):.5f} | {float(j['game1_score']):.5f} |",
        f"| Game 2 expected castle points | {float(j['game2_p'].expected_castle_points):.5f} | {float(j['game2_q'].expected_castle_points):.5f} |",
        f"| Game 2 expected rank payout | {float(j['game2_p'].expected_payout):.5f} | {float(j['game2_q'].expected_payout):.5f} |",'',
        'The difference comes from which bids occur together inside a legal allocation. This is a '
        'direct reason to retain whole opponent vectors throughout Game 2 evaluation.','',
        '**The objective can reverse a candidate comparison**','',
        '| Illustrative candidate | Expected castle points | Expected rank payout |','| --- | ---: | ---: |',
        f"| {v['higher_castle_candidate']} | {float(v['higher_castle_evaluation'].expected_castle_points):.4f} | {float(v['higher_castle_evaluation'].expected_payout):.4f} |",
        f"| {v['higher_payout_candidate']} | {float(v['higher_payout_evaluation'].expected_castle_points):.4f} | {float(v['higher_payout_evaluation'].expected_payout):.4f} |",'',
        'These are exact evaluations against one small synthetic population, chosen to demonstrate '
        'the difference between objectives. They are not finalists or applicant predictions.','',
        '**Engine and sampling models**','',
        'A direct Fraction scorer is checked against a separate vectorized integer scorer. Castle '
        'scores use twelfths, and final payouts use quarter units for this particular prize vector. '
        'Each group conserves 15 castle points and 15 rank-payout points. Identical strategies yield '
        '3.75 per player, and symmetric population self-play calibrates to 3.75. That calibration '
        'does not supply a four-player equilibrium or a minimax guarantee for one fixed allocation.','',
        'Exact evaluation compresses unordered opponent triples. The iid model uses probability '
        'weights and permits repeated types. The finite model samples three distinct people and '
        'retains multiplicities when several people choose the same allocation. Candidate exclusion '
        'is explicit when evaluating an applicant already in a finite cohort. Artificial mixture '
        'masses are not silently interpreted as a real applicant count.','',
        'Shared sampled triplets give candidate comparisons the same opponents. Fixed-sample '
        'Hoeffding bounds distinguish sampling error from model error; paired differences use their '
        'own wider possible range. Adaptive stopping will require a separate valid design.','',
        '**Small controls, not a population model**','',
        'Eight baseline allocations are evaluated against four small declared populations under '
        'both iid and finite grouping. These include clone controls, single-castle forts, and five '
        'priority anchors. Person counts in the finite fixtures are demonstration counts. Full exact '
        'evaluations, conditional sample means, and saved triplets are in results.json. They are '
        'sanity controls and do not choose a Game 2 strategy.','',
        '**Verification and next work**','',
        'The independent audit checks every one of the 50,625 four-player profiles in a reduced-budget '
        'five-castle game against rational arithmetic, every tie pattern, player permutations, '
        'sample IDs, compressed exact enumeration, symmetric self-play, and all illustrative claims. '
        'Game 1 files and its selected allocation remain unchanged.','',
        'The next stage is to incorporate the reserved four-player reasoning, define explicit joint '
        'population hypotheses, and build a search and arena that optimize expected rank payout. '
        'No global Game 2 best response, equilibrium, actual applicant-frequency forecast, or final '
        'allocation is claimed. Further deep research should be a narrow question driven by a '
        'specific gap in that stage.','',
        '- Official rule: [Sequoia application](https://pokerbot.sequoiacap.com/).',
        '- Scope check: [The Multiplayer Colonel Blotto Game](https://arxiv.org/html/2002.05240v3).',
        '', 'Run `python3 run_game2_foundation.py --output results/game2-foundation-new`; verify with '
        '`python3 verify_game2_foundation.py --study results/game2-foundation-new`.','']
    (out/'report.md').write_text('\n'.join(lines))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=ROOT/'results/game2-foundation')
    out=parser.parse_args().output.resolve()
    if out.exists():raise SystemExit('Output exists. Choose a new folder.')
    out.mkdir(parents=True);(out/'samples').mkdir();clock=time.perf_counter()
    sources=[Path(__file__),ROOT/'verify_game2_foundation.py',ROOT/'GAME2-FOUNDATION-PROTOCOL.md',ROOT/'game2_lab/core.py',ROOT/'game2_lab/__init__.py',
             ROOT/'tests/test_game2.py',ROOT/'blotto_lab/core.py',ROOT/'run_population_study.py']
    inputs=[ROOT/'results/game1-selected-v1.json',ROOT/'results/selection-review/selected-game1.txt']
    signature={'source_hashes':{str(p.relative_to(ROOT)):sha(p)for p in sources},'input_hashes':{str(p.relative_to(ROOT)):sha(p)for p in inputs},
               'environment':{'python':sys.version,'numpy':np.__version__,'platform':platform.platform()}}
    save(out/'run-inputs.json',signature)
    for p in sources:
        target=out/'source-snapshot'/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
    candidates,populations=controls();exact={};sampled={};sample_size=20000;timing=None
    for pindex,(name,pop)in enumerate(populations.items()):
        exact[name]={};sampled[name]={}
        for sindex,scheme in enumerate(['iid','finite']):
            exact[name][scheme]={label:exact_evaluation(x,pop,scheme)for label,x in candidates.items()}
            seed=2026111000+100*pindex+sindex;draw=shared_triplets(pop,sample_size,seed,scheme)
            arrays={'triplets':draw['triplets'].astype(np.uint8),'type_ids':draw['type_ids']}
            if draw['person_ids']is not None:arrays['person_ids']=draw['person_ids']
            filename=name+'_'+scheme+'.npz';np.savez_compressed(out/'samples'/filename,**arrays)
            start=time.perf_counter();evaluations=evaluate_shared(list(candidates.values()),draw['triplets']);elapsed=time.perf_counter()-start
            if timing is None:timing={'candidate_triplet_evaluations':len(candidates)*sample_size,'seconds':elapsed,'evaluations_per_second':len(candidates)*sample_size/elapsed}
            sampled[name][scheme]={'seed':seed,'size':sample_size,'sample_file':'samples/'+filename,'evaluations':dict(zip(candidates,evaluations))}
            print('Game 2 control completed: '+name+' '+scheme,flush=True)
    focal=(0,0,0,0,100)
    first=[focal,(0,0,0,100,0),(0,0,100,0,0),(50,50,0,0,0)]
    second=[focal,(0,0,50,50,0),(0,100,0,0,0),(100,0,0,0,0)]
    r={'scope':'Game 2 foundation and controls; no finalists or applicant forecast',
       'official_rules':{'source':'https://pokerbot.sequoiacap.com/','budget':100,'values':[1,2,3,4,5],'prizes':[9,4,2,0],
                         'unpublished':['exact number of regroupings','sampling implementation','combined-game priority cutoff']},
       'candidates':candidates,'populations':{name:p.to_dict()for name,p in populations.items()},'exact':exact,'sampled':sampled,
       'five_points_examples':{'first':{'group':first,'outcome':reference_group(first)},'second':{'group':second,'outcome':reference_group(second)}},
       'joint_counterexample':joint_counterexample(),'objective_reversal':objective_reversal(),
       'fixed_sample_uniform_bound':fixed_sample_bound(sample_size,len(candidates)*len(populations)*2),'benchmark_timing':timing,
       'research_recommendation':'No additional broad deep research now; use targeted rank-payoff search/statistics research only for a concrete gap',
       'research_scope_source':'https://arxiv.org/html/2002.05240v3','human_game2_ideas':'reserved; not incorporated'}
    save(out/'results.json',r);report(out,r)
    write_csv(out/'control-scores.csv',['population','sampling','candidate','allocation','exact_rank_payout','exact_castle_points','sample_rank_payout'],
      [[name,scheme,label,str(x),float(exact[name][scheme][label].expected_payout),float(exact[name][scheme][label].expected_castle_points),
         float(sampled[name][scheme]['evaluations'][label].expected_payout)]for name in populations for scheme in ['iid','finite']for label,x in candidates.items()])
    for p in sources:assert sha(p)==signature['source_hashes'][str(p.relative_to(ROOT))]
    save(out/'manifest.json',{'signature':signature,'elapsed_seconds':time.perf_counter()-clock,'completed_unix':time.time(),
         'artifacts':{str(p.relative_to(out)):sha(p)for p in sorted(out.rglob('*'))if p.is_file()and p.name!='manifest.json'}})
    print('Game 2 foundation complete. Run the independent verifier.',flush=True)


if __name__=='__main__':main()
