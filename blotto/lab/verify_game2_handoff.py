"""Independent scoring and provenance checks for handoff integration."""
from pathlib import Path
from fractions import Fraction
import argparse,json,time
import numpy as np
from blotto_lab.core import Population,validate_allocation
from blotto_lab.multiwave import simple_json
from game2_lab.core import focal_samples,shared_triplets
from game2_lab.human import payoff_arrays
from game2_lab.handoff import target_population,cdf_diagnostic
from run_game2_handoff import expanded_pool,theory_audit,core_population_rows,NAMED
from run_population_study import ROOT,save,sha
from verify_population_study import rational


def read(p):return json.loads(p.read_text())


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--study',type=Path,default=ROOT/'results/game2-handoff-integration');out=parser.parse_args().study.resolve();clock=time.perf_counter();r=read(out/'results.json');cfg=r['configuration'];m=read(out/'manifest.json')
    for key,digest in m['artifacts'].items():assert sha(out/key)==digest,key
    for key,digest in m['signature']['source_hashes'].items():assert sha(ROOT/key)==sha(out/'source-snapshot'/key)==digest,key
    for key,digest in m['signature']['input_hashes'].items():assert sha(ROOT/key)==digest,key
    freeze=read(out/'pool-freeze.json');assert sha(out/'pool-freeze.json')==r['freeze_sha256']==m['freeze_sha256']and freeze['pool']==r['pool']and freeze['named']==r['named']
    old=read(ROOT/'results/game2-final/results.json');assert simple_json(expanded_pool(old,cfg))==r['pool'];cids=list(r['pool']);actions=[v['allocation']for v in r['pool'].values()]
    for x in actions:validate_allocation(x)
    assert len(set(map(tuple,actions)))==len(actions)
    public=read(ROOT/'data/parry-handoff/public-scores.json');scores=np.array(public['scores']);assert len(scores)==r['public_summary']['number']and scores.mean()==r['public_summary']['mean']and scores.min()==r['public_summary']['minimum']and scores.max()==r['public_summary']['maximum']
    assert simple_json(theory_audit())==r['theory'];count=primary_cells=all_spot=0;columns=[]
    for key,d in freeze['scenarios'].items():
        pop=target_population(d['active'],d['scout'],d['weighting'],cfg['kernel_kappa'],cfg['kernel_scale']);expected=shared_triplets(pop,cfg['target_triplets'],d['seed'])['triplets']
        with np.load(out/'targets'/(key+'.npz'))as a:rows=a['triplets'];sums=a['rank_sum4'];castles=a['castle_sum12'];named=a['named_rank4']
        np.testing.assert_array_equal(rows,expected);rank,points=payoff_arrays(actions,rows,candidate_batch=11,triplet_batch=3001);np.testing.assert_array_equal(sums,rank.sum(axis=1,dtype=np.int64));np.testing.assert_array_equal(castles,points.sum(axis=1,dtype=np.int64))
        for j,(name,x)in enumerate(NAMED.items()):
            direct=focal_samples(x,rows);np.testing.assert_array_equal(named[j],direct['payout4']);cid=r['named'][name];i=cids.index(cid);assert int(direct['payout4'].sum())==int(sums[i]);primary_cells+=len(rows)
        for i,x in enumerate(actions):
            direct=focal_samples(x,rows[:32]);np.testing.assert_array_equal(rank[i,:32],direct['payout4']);all_spot+=len(direct['payout4'])
            assert rational(r['target_scores'][cids[i]][key])==Fraction(int(sums[i]),4*cfg['target_triplets'])
        assert len(pop.entries)==r['target_scenarios'][key]['support_size'];columns.append(sums);count+=1
        if count%30==0:print('Verified target laws: '+str(count),flush=True)
    # Saved JSON sorts scenario keys; use the runner's declaration order for its matrix.
    ordered=[]
    from itertools import combinations,product
    for k in [2,3]:
        for active in combinations(range(5),k):
            for scout,mode in product(cfg['scouts'],cfg['target_weightings']):ordered.append('target'+''.join(str(i+1)for i in active)+'__s'+str(scout)+'__'+mode)
    with np.load(out/'target-scores.npz')as a:matrix=a['rank_sum4']
    for j,key in enumerate(ordered):np.testing.assert_array_equal(matrix[:,j],[int(rational(r['target_scores'][cid][key])*4*cfg['target_triplets'])for cid in cids])
    cdf_count=cdf_spot=0;cloud=read(ROOT/'results/game2-final/candidate-freeze.json')['counter_cloud']
    for key,v in r['cdf'].items():
        d=v['definition']
        with np.load(out/'cdf'/(key+'.npz'))as a:own=a['own_actions'];weights=a['own_weights'];sums=a['rank_sum4'];size=int(a['size']);rows=a['triplets']
        if d['kind']=='arena':
            pop=Population.from_dict(d['population']);expected=shared_triplets(pop,size,d['seed'])['triplets'];np.testing.assert_array_equal(own,np.array([x for x,n in pop.entries]));np.testing.assert_array_equal(weights,[n for x,n in pop.entries])
        else:
            ownrows=core_population_rows(d['style'],d['copy_share'],cfg['core_cdf_own'],d['seed'],cloud);eo,ew=np.unique(ownrows,axis=0,return_counts=True);np.testing.assert_array_equal(own,eo);np.testing.assert_array_equal(weights,ew);expected=core_population_rows(d['style'],d['copy_share'],3*size,d['seed']+1000000,cloud).reshape(size,3,5)
        np.testing.assert_array_equal(rows,expected);rank=payoff_arrays(own.tolist(),rows,candidate_batch=13,triplet_batch=7001)[0];np.testing.assert_array_equal(sums,rank.sum(axis=1,dtype=np.int64))
        for j,x in enumerate(own):np.testing.assert_array_equal(focal_samples(x.tolist(),rows[:64])['payout4'],rank[j,:64]);cdf_spot+=len(rows[:64])
        metrics=cdf_diagnostic(sums/(4*size),weights,scores);threshold=cfg['cdf_thresholds'];metrics['passes_nominal_screen']=metrics['maximum_cdf_gap']<=threshold['maximum']and metrics['mean_grid_cdf_gap']<=threshold['mean']and metrics['mass_above_public_max_plus_point_two']<=threshold['tail'];assert metrics==v['metrics'];cdf_count+=1;print('Verified public compatibility diagnostic: '+key,flush=True)
    for filename in ['game1-selected-v1.json','game2-selected-v1.json','game2-final-checkpoint-v1.json']:
        for key,digest in read(ROOT/'results'/filename)['source_artifact_sha256'].items():assert sha(ROOT/key)==digest,key
    acceptance={'passed':True,'verified_unix':time.time(),'elapsed_seconds':time.perf_counter()-clock,'verifier_sha256':sha(Path(__file__)),'expanded_candidates':len(cids),'target_laws':count,'named_target_candidate_group_cells_foundation_checked':primary_cells,'all_pool_target_spot_cells_foundation_checked':all_spot,'public_cdf_models':cdf_count,'all_cdf_action_spot_cells_foundation_checked':cdf_spot,'public_snapshot_size':len(scores),'exact_group_fixtures':len(r['theory']['exact_group_cases']),'affine_invariance_groups':r['theory']['affine_random_group_invariance_checks'],'original_selections_unchanged':True,'freeze_sha256':r['freeze_sha256'],'scope':'Verified theory and independent proxy development audit; Parry actual fitted fields and code remain unavailable'}
    save(out/'acceptance.json',acceptance);print(json.dumps(acceptance,indent=2))


if __name__=='__main__':main()
