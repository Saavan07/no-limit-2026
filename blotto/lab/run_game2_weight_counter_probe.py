"""Separate retrospective analytic counter-population probe, preserving the freeze."""
from pathlib import Path
from itertools import product
import argparse,json,time
from blotto_lab.core import Game,iter_allocations
from game2_lab.core import reference_group
from game2_lab.weight_counters import worst_clone_population
from run_multiwave_study import ROOT
from run_population_study import save,sha,write_csv


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=ROOT/'results/game2-weights-counter-probe');out=parser.parse_args().output.resolve()
    if out.exists():raise SystemExit('Use a new output folder.')
    small_checks=0
    for game in [Game(4,(1,2,3)),Game(2,(1,2,3,4,5))]:
        actions=list(iter_allocations(game))
        for x in actions:
            scored=[]
            for y in actions:
                r=reference_group([x,y,y,y],game);scored.append((r.payouts[0],r.castle_points[0],y));small_checks+=1
            actual=worst_clone_population(x,game);assert (actual['payout'],actual['castle_points'],actual['opponent'])==min(scored)
    study=ROOT/'results/game2-weights/results.json';r=json.loads(study.read_text());freeze=ROOT/'results/game2-weights/candidate-freeze.json';freeze_hash=sha(freeze);cases={}
    for cid in r['primary_ids']:
        x=r['candidates'][cid]['allocation'];case=worst_clone_population(x);group=reference_group([x]+[case['opponent']]*3)
        assert group.payouts[0]==case['payout']and group.castle_points[0]==case['castle_points'];case['group_outcome']=group;cases[cid]=case
    out.mkdir(parents=True);sources=[Path(__file__),ROOT/'game2_lab/weight_counters.py',ROOT/'tests/test_game2_weight_counters.py']
    save(out/'results.json',{'created_unix':time.time(),'cases':cases,'rules':r['primary'],'exhaustive_reduced_game_group_checks':small_checks,'source_sha256':{str(p.relative_to(ROOT)):sha(p)for p in sources},
       'inputs':{'study_sha256':sha(study),'freeze_sha256':freeze_hash},'scope':'Retrospective exact pure-clone population counterexamples, not a new holdout or final selection'})
    write_csv(out/'counter-populations.csv',['candidate','opponent_population','candidate_castle_points','candidate_payout'],[[str(tuple(c['candidate'])),str(tuple(c['opponent'])),float(c['castle_points']),float(c['payout'])]for c in cases.values()])
    assert sha(freeze)==freeze_hash;print('Exact pure-clone population diagnostic saved; frozen study unchanged.')


if __name__=='__main__':main()
