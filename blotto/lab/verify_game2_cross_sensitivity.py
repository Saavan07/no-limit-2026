"""Check response-share draws, all-candidate subsets and exact alternatives."""
from pathlib import Path
import itertools
import time
import numpy as np
from game2_lab.cross_arena import read,load_catalog,native_module,draw_population
from game2_lab.core import focal_samples
from game2_lab.human import payoff_arrays
from run_population_study import ROOT,save,sha


def main():
    out=ROOT/'results/game2-cross-arena';dest=out/'sensitivity';start=time.perf_counter()
    freeze=read(out/'freeze.json');protocol=read(dest/'protocol.json');result=read(dest/'results.json')
    assert protocol['candidate_freeze_sha256']==sha(out/'freeze.json')
    assert result['protocol_sha256']==sha(dest/'protocol.json')
    vendor=ROOT/'data/parry-replay';_,p,fields=load_catalog(vendor);native=native_module(vendor)
    xs=np.array([r['allocation']for r in freeze['pool']],np.uint8)
    strat=np.zeros(len(p));strat[list(p).index('strategic_mixture')]=1
    values=[];checked=0;draws=0
    for j,field in enumerate(fields):
        for k,share in enumerate(protocol['shares']):
            name=field['name']+'_strategic_share'+str(int(100*share))
            with np.load(dest/(name+'.npz'))as data:rows=data['triplets'];sums=data['rank_sum4'];n=int(data['size']);seed=int(data['seed'])
            assert seed==protocol['base_seed']+3*j+k
            weights=(1-share)*np.array(field['weights'])+share*strat
            np.testing.assert_array_equal(draw_population(p,weights,3*n,seed).reshape(n,3,5),rows)
            ix=np.random.default_rng(879+j*3+k).choice(n,512,replace=False)
            direct,_=payoff_arrays(xs.tolist(),rows[ix]);g=native.compress_groups(rows[ix]);pay=native.payoff_matrix(xs,g)
            counts=np.rint(g[2]*len(ix)).astype(np.uint64)
            np.testing.assert_array_equal(direct.sum(axis=1),np.einsum('ij,j->i',pay,counts,dtype=np.uint64)//3)
            checked+=len(xs)*len(ix);draws+=n;values.append(sums/(4*n))
    values=np.column_stack(values)
    with np.load(dest/'scores.npz')as data:
        np.testing.assert_array_equal(values,data['values']);strategic=data['strategic_exact']
    support,w=p['strategic_mixture'];w=w/w.sum()
    ids=np.array(list(itertools.product(range(len(support)),repeat=3)))
    rows=support[ids];mass=np.prod(w[ids],axis=1);exact_cells=0
    with np.load(out/'fresh-scores.npz')as data:calibrated=data['values'].min(axis=1)
    stress=np.minimum(values.min(axis=1),strategic)
    # Check every alternative that appears on the illustrative stress-floor frontier.
    for r in result['illustrative_safety_frontier']:
        floor=r['stress_floor'];allowed=np.where(stress>=floor)[0]
        if not len(allowed):assert r['feasible']is False;continue
        i=min(allowed,key=lambda j:(-calibrated[j],tuple(xs[j])))
        assert i==r['index']and xs[i].tolist()==r['allocation']
        direct=focal_samples(xs[i].tolist(),rows)['payout4']/4
        assert abs(direct@mass-strategic[i])<1e-10
        exact_cells+=len(rows)
    # The implementation repair preserved the actual pre-draw candidate/model freeze.
    before=read(out/'freeze-pre-decoder-fix.json');after=read(out/'freeze.json')
    for key in ['pool','configuration','named_indices','field_names','versions']:assert before[key]==after[key]
    differences=[k for k in before['input_and_source_sha256']if before['input_and_source_sha256'][k]!=after['input_and_source_sha256'][k]]
    assert differences==['run_game2_cross_arena.py']
    save(dest/'acceptance.json',{'passed':True,'results_sha256':sha(dest/'results.json'),'protocol_sha256':sha(dest/'protocol.json'),
        'source_sha256':sha(Path(__file__)),'runner_sha256':sha(ROOT/'run_game2_cross_sensitivity.py'),
        'regenerated_response_shift_triplets':draws,'all_candidate_subset_cells_checked':checked,
        'full_frontier_strategic_cells_checked':exact_cells,'runner_full_named_cells_checked':result['independent_candidate_group_cells'],
        'pre_draw_pool_and_design_preserved_across_decoder_fix':True,'elapsed_seconds':time.perf_counter()-start,
        'scope':'Specified response shifts and supplied strategic checkpoint only. No actual-population or global-equilibrium claim.'})
    print('Accepted response-share sensitivity',flush=True)


if __name__=='__main__':main()
