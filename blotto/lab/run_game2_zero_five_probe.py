"""Post-challenge diagnostic: isolate zero-five variants of the emerging candidate."""
from fractions import Fraction
from pathlib import Path
import json,time
import numpy as np
from game2_lab.core import focal_samples
from game2_lab.human import payoff_arrays
from run_multiwave_study import ROOT
from run_population_study import save,sha


def main():
    out=ROOT/'results/game2-human-zero-five-probe'
    if out.exists():raise SystemExit('Probe output exists')
    out.mkdir(parents=True);study=ROOT/'results/game2-human';r=json.loads((study/'results.json').read_text())
    variants=[(1,29,3,64,3),(4,29,3,64,0),(1,32,3,64,0),(1,29,6,64,0),(1,29,3,67,0)]
    scores={x:{}for x in variants};spotchecks=0
    for key,model in r['fresh'].items():
        with np.load(study/model['file'])as a:rows=a['triplets']
        p,_=payoff_arrays(variants,rows)
        for i,x in enumerate(variants):
            np.testing.assert_array_equal(p[i,:256],focal_samples(x,rows[:256])['payout4']);spotchecks+=256
            scores[x][key]=Fraction(int(p[i].sum(dtype=np.int64)),4*len(rows))
    result={'status':'Retrospective diagnostic after the frozen challenge; no new primary or selection',
       'reason':'Changing the fifth bid also changes another castle; compare nearby zero-five reallocations before attributing the full fitted-strategy gap to a small fifth bid.',
       'created_unix':time.time(),'freeze_sha256':r['freeze_sha256'],
       'inputs':{'study_results_sha256':sha(study/'results.json'),'study_manifest_sha256':sha(study/'manifest.json'),'probe_source_sha256':sha(Path(__file__))},
       'variants':[{'allocation':x,'scores':v,'mean':sum(v.values())/len(v),'minimum':min(v.values()),'maximum':max(v.values())}for x,v in scores.items()],
       'foundation_spotcheck_cells':spotchecks,'interpretation':'Both zero and small-positive fifth bids remain plausible. These probes require a later fresh test before final selection.'}
    save(out/'results.json',result);print('Retrospective zero-five probe saved; original freeze unchanged.')


if __name__=='__main__':main()
