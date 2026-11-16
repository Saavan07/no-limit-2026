"""Separate response-share stresses for the unchanged cross-arena pool."""
from pathlib import Path
import itertools
import time
import numpy as np
from game2_lab.cross_arena import read,load_catalog,native_module,draw_population,exact_sample_sums
from game2_lab.core import focal_samples
from run_population_study import ROOT,save,sha,write_csv


def main():
    out=ROOT/'results/game2-cross-arena';freeze=read(out/'freeze.json');base=read(out/'results.json')
    dest=out/'sensitivity';dest.mkdir(exist_ok=True)
    vendor=ROOT/'data/parry-replay';fit,p,fields=load_catalog(vendor);native=native_module(vendor)
    xs=np.array([r['allocation']for r in freeze['pool']],np.uint8)
    protocol={'candidate_freeze_sha256':sha(out/'freeze.json'),'shares':[.1,.25,.5],
        'groups':65536,'base_seed':91511001,'scope':'Response-share stresses, not newly calibrated applicant populations or probabilities over fields. Pure strategic support evaluated exactly. Same frozen candidate pool; no new admissions.'}
    path=dest/'protocol.json'
    if path.exists():assert read(path)==protocol
    else:save(path,protocol)
    strategic=np.zeros(len(p));strategic[list(p).index('strategic_mixture')]=1
    values=[];names=[];count=0;checked=0;start=time.perf_counter()
    for j,field in enumerate(fields):
        for k,share in enumerate(protocol['shares']):
            name=field['name']+'_strategic_share'+str(int(100*share));path=dest/(name+'.npz')
            if not path.exists():
                weights=(1-share)*np.array(field['weights'])+share*strategic
                n=protocol['groups'];seed=protocol['base_seed']+3*j+k
                rows=draw_population(p,weights,3*n,seed).reshape(n,3,5)
                sums=exact_sample_sums(native,xs,native.compress_groups(rows),n)
                np.savez_compressed(path,triplets=rows,rank_sum4=sums,size=n,seed=seed)
            with np.load(path)as data:
                rows=data['triplets'];sums=data['rank_sum4'];n=int(data['size'])
            # Complete independent mean for the two submitted-work baselines.
            for label in ['our_v1','parry_36_64']:
                i=freeze['named_indices'][label];q=focal_samples(xs[i].tolist(),rows)
                assert int(q['payout4'].sum())==int(sums[i]);checked+=n
            values.append(sums/(4*n));names.append(name);count+=1
        print('Response-share stresses',j+1,'/17',round(time.perf_counter()-start,1),'seconds',flush=True)
    values=np.column_stack(values)
    # Full ordered triples retain every non-negligible strategic allocation and its actual mass.
    support,weights=p['strategic_mixture'];weights=weights/weights.sum()
    groups=native.exact_iid(support,weights)
    strategic_scores=native.evaluate(xs,groups)
    self_scores=native.evaluate(support,groups)
    assert abs(weights@self_scores-3.75)<1e-10
    triple_ids=np.array(list(itertools.product(range(len(support)),repeat=3)))
    rows=support[triple_ids];triple_mass=np.prod(weights[triple_ids],axis=1)
    exact_checks={}
    for name,i in base['named_indices'].items():
        direct=focal_samples(xs[i].tolist(),rows)['payout4']/4
        score=float(direct@triple_mass)
        assert abs(score-strategic_scores[i])<1e-10
        exact_checks[name]=score;checked+=len(rows)
    memory=read(vendor/'results/memory_equilibrium_best.json')
    response=next(i for i,x in enumerate(xs.tolist())if x==memory['global_best_response'])
    claimed=memory['global_best_response_payout']
    assert abs(strategic_scores[response]-claimed)<1e-8
    np.savez_compressed(dest/'scores.npz',values=values,names=names,allocations=xs,strategic_exact=strategic_scores)
    with np.load(out/'fresh-scores.npz')as data:calibrated=data['values'].min(axis=1)
    stress=np.minimum(values.min(axis=1),strategic_scores)
    frontier=[]
    for floor in [0,3,3.25,3.5,3.75,4]:
        allowed=np.where(stress>=floor)[0]
        if len(allowed):
            i=min(allowed,key=lambda j:(-calibrated[j],tuple(xs[j])))
            frontier.append({'stress_floor':floor,'index':int(i),'allocation':xs[i].tolist(),
                'minimum_calibrated_mean':float(calibrated[i]),'minimum_stress_mean':float(stress[i])})
        else:frontier.append({'stress_floor':floor,'feasible':False})
    summary={'protocol_sha256':sha(dest/'protocol.json'),'response_shift_field_count':len(names),
        'exact_strategic_support_size':len(support),'exact_strategic_selfplay_mean':float(weights@self_scores),
        'stored_best_response_payoff_reproduced':float(strategic_scores[response]),
        'global_best_response_claim_recertified':False,'all_named_exact_checks':exact_checks,
        'independent_candidate_group_cells':checked,'illustrative_safety_frontier':frontier,
        'named':{name:{'allocation':xs[i].tolist(),'minimum_calibrated_mean':float(calibrated[i]),
            'minimum_response_shift_mean':float(values[i].min()),'strategic_exact_mean':float(strategic_scores[i]),
            'all_response_shift_means':dict(zip(names,values[i].tolist()))}for name,i in base['named_indices'].items()},
        'elapsed_seconds':time.perf_counter()-start,'scope':protocol['scope']}
    save(dest/'results.json',summary)
    write_csv(dest/'response-shift-comparison.csv',['index','allocation','minimum_calibrated','minimum_response_shifts','exact_strategic','minimum_stress'],
        [[i,str(x),float(calibrated[i]),float(values[i].min()),float(strategic_scores[i]),float(stress[i])]for i,x in enumerate(xs.tolist())])
    print('Sensitivity complete',summary['illustrative_safety_frontier'],flush=True)


if __name__=='__main__':main()
