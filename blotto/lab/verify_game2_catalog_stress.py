"""Independent numerical audit of the family/catalog challenge."""
import argparse
import itertools
from pathlib import Path
import time
import numpy as np
from game2_lab.cross_arena import read,load_catalog,native_module,draw_population,sample_actions,uncompressed_groups
from game2_lab.catalog_stress import CATALOG_VARIANTS,make_catalog,fingerprint,choose_diverse
from game2_lab.core import focal_samples
from game2_lab.human import payoff_arrays
from verify_game2_cross_arena import independent_cdf_gap
from run_population_study import ROOT,save,sha


def load_saved_catalog(out,kind,names):
    with np.load(out/'catalogs'/kind/'rows.npz')as x,np.load(out/'catalogs'/kind/'weights.npz')as w:
        return {name:(x[name].copy(),w[name].copy())for name in names}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--study',type=Path,default=ROOT/'results/game2-catalog-stress')
    parser.add_argument('--part',choices=['all','calibration','evaluation','stresses'],default='all');args=parser.parse_args()
    out=args.study.resolve();freeze=read(out/'freeze.json');cfg=freeze['configuration'];start=time.perf_counter()
    for rel,digest in freeze['source_and_input_sha256'].items():assert sha(ROOT/rel)==digest,rel
    vendor=ROOT/'data/parry-replay';fit,base,fields=load_catalog(vendor);native=native_module(vendor)
    names=list(base);actions=np.array([r['allocation']for r in freeze['pool']],np.uint8)
    catalogs={};counts={};public=np.array(read(vendor/'public-scores.json')['scores'])
    for j,kind in enumerate(CATALOG_VARIANTS):
        actual=load_saved_catalog(out,kind,names)
        expected=make_catalog(base,kind,cfg['variant_family_rows'],cfg['seed']+j*100)
        for name in names:
            np.testing.assert_array_equal(actual[name][0],expected[name][0]);np.testing.assert_array_equal(actual[name][1],expected[name][1])
            assert np.all(actual[name][0].sum(axis=1)==100)
        catalogs[kind]=actual
    if args.part in ['all','calibration']:
        coefficient_cells=0
        for index,kind in enumerate(CATALOG_VARIANTS):
            p=catalogs[kind]
            with np.load(out/'training'/kind/'coefficients.npz')as d:
                own=d['own'];families=d['families'];ids=d['ids'];sums=d['rank_sum4'];rows=d['conditional_triplets'];m=int(d['size'])
            expected_own,expected_family=sample_actions(p,cfg['training_own_per_family'],cfg['seed']+1000+index)
            np.testing.assert_array_equal(own,expected_own);np.testing.assert_array_equal(families,expected_family)
            np.testing.assert_array_equal(ids,np.array(list(itertools.combinations_with_replacement(range(40),3))))
            rng=np.random.default_rng(cfg['seed']+2000+index)
            for j,combo in enumerate(ids):
                for pos,family in enumerate(combo):
                    x,w=p[names[family]];np.testing.assert_array_equal(rows[j,:,pos],x[rng.choice(len(x),m,p=w/w.sum())])
            rng=np.random.default_rng(17800+index)
            own_ids=np.unique(np.r_[np.arange(0,len(own),cfg['training_own_per_family']),rng.choice(len(own),8,replace=False)])
            # Full conditional opponent sample for these actions, including every family triple.
            quick,_=payoff_arrays(own[own_ids].tolist(),rows.reshape(-1,3,5))
            np.testing.assert_array_equal(quick.reshape(len(own_ids),len(ids),m).sum(axis=2),sums[own_ids])
            coefficient_cells+=len(own_ids)*len(ids)*m
            records=sorted((out/'training'/kind).glob('*.json'))
            for path in records:
                r=read(path);selected=choose_diverse(r['runs'],cfg['models_per_catalog']if r['constraint']=='unrestricted'else cfg['models_per_leave_one_out'],cfg['model_l1_separation'])
                assert r['selected']==selected
                for run in r['runs']:
                    w=np.array(run['weights']);assert np.all(w>=0)and abs(w.sum()-1)<1e-10
                    if r['omitted_family']is not None:assert w[names.index(r['omitted_family'])]==0
            print('Verified training',kind,flush=True)
        assert len(list((out/'training/original').glob('omit_*.json')))==40
        model_freeze=read(out/'model-freeze.json');assert model_freeze['candidate_freeze_sha256']==sha(out/'freeze.json')
        expected_models={}
        for kind in CATALOG_VARIANTS:
            for path in (out/'training'/kind).glob('*.json'):
                r=read(path)
                for j,row in enumerate(r['selected']):expected_models[kind+'__'+r['constraint']+'__'+str(j)]=(kind,r['constraint'],row['weights'])
        for row in fields:expected_models['supplied_'+row['name']]=('original','original_supplied',row['weights'])
        assert set(expected_models)=={r['name']for r in model_freeze['models']}
        for row in model_freeze['models']:
            kind,constraint,weights=expected_models[row['name']]
            assert row['catalog']==kind and row['constraint']==constraint and row['weights']==weights
        validation=read(out/'model-validation.json');assert [r['name']for r in validation['models']]==[r['name']for r in model_freeze['models']]
        cdf_cells=0
        for j,row in enumerate(validation['models']):
            assert (row['catalog'],row['constraint'],row['weights'])==expected_models[row['name']]
            p=catalogs[row['catalog']]
            with np.load(out/'validation'/(row['name']+'.npz'))as d:
                rows=d['opponent_triplets'];u=d['values'];own=d['own'];family=d['families'];n=int(d['size']);seed=int(d['seed'])
            assert seed==cfg['seed']+60000+j
            np.testing.assert_array_equal(rows,draw_population(p,row['weights'],3*n,seed).reshape(n,3,5))
            own_indices=np.arange(0,len(own),cfg['validation_own_per_family'])
            direct,_=payoff_arrays(own[own_indices].tolist(),rows)
            np.testing.assert_allclose(direct.mean(axis=1)/4,u[own_indices],rtol=0,atol=1e-10);cdf_cells+=len(own_indices)*n
            mass=np.array(row['weights'])[family]/cfg['validation_own_per_family']
            gap=independent_cdf_gap(u,mass,public)
            assert abs(gap-row['validation']['maximum_cdf_gap'])<1e-11
            def model_cdf(t):return np.array([mass[u<=v].sum()for v in t])
            dense=np.linspace(0,9,181);gridgap=abs(model_cdf(dense)-np.array([(public<=t).mean()for t in dense])).mean()
            originalgap=abs(model_cdf(fit['thresholds'])-np.array([(public<=t).mean()for t in fit['thresholds']])).mean()
            tail=mass[u>public.max()+.2].sum()
            assert abs(gridgap-row['validation']['mean_grid_cdf_gap'])<1e-11
            assert abs(originalgap-row['validation']['original_threshold_mean_gap'])<1e-11
            assert abs(tail-row['validation']['mass_above_public_max_plus_point_two'])<1e-11
            passes=gap<=.085 and gridgap<=.035 and originalgap<=.035 and tail<=.015
            assert passes==row['validation']['passes']
            print('Verified CDF',j+1,'/',len(validation['models']),flush=True)
        counts['coefficient_foundation_cells']=coefficient_cells;counts['cdf_full_opponent_foundation_cells']=cdf_cells
    if args.part in ['all','evaluation']:
        models=[r for r in read(out/'model-validation.json')['models']if r['validation']['passes']]
        full_cells=0;subset_cells=0;candidate_values=[];split_values=[];matrices=[]
        split_x=[[0,b,0,100-b,0]for b in range(101)];selection=read(out/'selection.json')
        chosen=selection['selected_split_C2'];check_splits=sorted({36,37,38,chosen})
        for j,row in enumerate(models):
            with np.load(out/'evaluation'/(row['name']+'.npz'))as d:
                rows=d['triplets'];n=int(d['size']);prefix=int(d['candidate_size']);candidate_sum=d['candidate_rank_sum4'];split_sum=d['split_rank_sum4'];cross=d['split_cross_sum16'];seed=int(d['seed'])
            assert seed==cfg['seed']+100000+j
            np.testing.assert_array_equal(rows,draw_population(catalogs[row['catalog']],row['weights'],3*n,seed).reshape(n,3,5))
            ix=np.random.default_rng(781+j).choice(prefix,1024,replace=False)
            quick,_=payoff_arrays(actions.tolist(),rows[ix]);q=native.payoff_matrix(actions,uncompressed_groups(rows[ix]))//3
            np.testing.assert_array_equal(q,quick);subset_cells+=len(actions)*len(ix)
            full,_=payoff_arrays([split_x[b]for b in check_splits],rows)
            np.testing.assert_array_equal(full.sum(axis=1),split_sum[check_splits])
            full64=full.astype(np.float64)
            np.testing.assert_array_equal(full64@full64.T,cross[np.ix_(check_splits,check_splits)])
            full_cells+=len(check_splits)*n
            candidate_values.append(candidate_sum/(4*prefix));split_values.append(split_sum/(4*n));matrices.append(cross)
            print('Verified evaluation',j+1,'/',len(models),flush=True)
        cv=np.array(candidate_values).T;sv=np.array(split_values).T;matrices=np.array(matrices)
        selected=min(range(len(actions)),key=lambda i:(-cv[i].min(),tuple(actions[i])))
        b=int(sv.min(axis=1).argmax())
        assert selection['selected_index']==selected and selection['selected_allocation']==actions[selected].tolist()
        assert selection['selected_split_C2']==b
        with np.load(out/'candidate-scores.npz')as d:np.testing.assert_array_equal(cv,d['values'])
        with np.load(out/'split-scores.npz')as d:np.testing.assert_array_equal(sv,d['values']);np.testing.assert_array_equal(matrices,d['cross_sum16'])
        # Independently reconstruct the two-sided finite-union confidence formula.
        comparisons=len(models)*(101+5050);log=np.log(4*comparisons/.05)
        mean_var=(matrices[:,np.arange(101),np.arange(101)]/16-n*sv.T**2)/(n-1)
        mean_error=np.sqrt(2*np.maximum(mean_var,0)*log/n)+7*9*log/(3*(n-1))
        pair_error=[]
        for j,matrix in enumerate(matrices):
            sum_square=(np.diag(matrix)[:,None]+np.diag(matrix)[None,:]-2*matrix.astype(float))/16
            var=np.maximum((sum_square-n*(sv[:,j,None]-sv[None,:,j])**2)/(n-1),0)
            pair_error.append(np.sqrt(2*var*log/n)+7*18*log/(3*(n-1)))
        with np.load(out/'split-confidence.npz')as d:
            np.testing.assert_allclose(mean_error,d['mean_error'],rtol=0,atol=1e-12)
            np.testing.assert_allclose(pair_error,d['pair_error'],rtol=0,atol=1e-12)
        lower=(sv.T-mean_error).min(axis=0);upper=(sv.T+mean_error).min(axis=0)
        assert np.flatnonzero(upper>=lower.max()).tolist()==selection['sampling_plausible_split_C2_values']
        counts['full_split_foundation_cells']=full_cells;counts['all_candidate_subset_cells']=subset_cells
        counts['regenerated_evaluation_groups']=len(models)*n
    if args.part in ['all','stresses']:
        dest=out/'family-stress';mapping=read(dest/'family-map.json');pure={};unique={};cells=0;regenerated=0
        for kind,p in catalogs.items():
            for name,(x,w)in p.items():
                key=fingerprint(x,w);assert mapping[kind][name]==key
                if key not in unique:unique[key]=(x,w)
        with np.load(dest/'scores.npz')as d:
            saved_pure=d['pure_values'];pure_keys=d['pure_keys'].tolist();coeff=d['shock_coefficients'];shock_ids=d['shock_action_indices'];
        assert list(unique)==pure_keys
        for j,(key,(x,w))in enumerate(unique.items()):
            with np.load(dest/(key+'.npz'))as d:rows=d['triplets'];sums=d['rank_sum4'];n=int(d['size']);seed=int(d['seed'])
            rng=np.random.default_rng(seed);np.testing.assert_array_equal(rows,x[rng.choice(len(x),(n,3),p=w/w.sum())])
            direct,_=payoff_arrays(actions.tolist(),rows[:256]);native_pay=native.payoff_matrix(actions,uncompressed_groups(rows[:256]))//3
            np.testing.assert_array_equal(direct,native_pay);cells+=len(actions)*256;regenerated+=n
            pure[key]=sums/(4*n);np.testing.assert_array_equal(pure[key],saved_pure[:,j])
        shock_actions=actions[shock_ids];n=cfg['family_shock_conditional_groups']
        for j,field in enumerate(fields):
            base_rows=draw_population(base,field['weights'],3*n,cfg['seed']+300000+j).reshape(n,3,5)
            base_value=np.array([focal_samples(x.tolist(),base_rows)['payout4'].mean()/4 for x in shock_actions])
            np.testing.assert_array_equal(base_value,coeff[j,0,:,0])
            for f,(name,(x,w))in enumerate(base.items()):
                np.testing.assert_array_equal(coeff[j,f,:,0],base_value)
                np.testing.assert_array_equal(coeff[j,f,:,3],pure[mapping['original'][name]][shock_ids])
                for k in [1,2]:
                    with np.load(dest/(field['name']+'__'+name+'__k'+str(k)+'.npz'))as d:rows=d['triplets'];sums=d['rank_sum4'];seed=int(d['seed'])
                    assert seed==cfg['seed']+400000+j*1000+f*10+k
                    rng=np.random.default_rng(seed);expected=base_rows.copy();expected[:,:k]=x[rng.choice(len(x),(n,k),p=w/w.sum())]
                    np.testing.assert_array_equal(rows,expected);regenerated+=n
                    direct,_=payoff_arrays(shock_actions.tolist(),rows[:128]);native_pay=native.payoff_matrix(shock_actions,uncompressed_groups(rows[:128]))//3
                    np.testing.assert_array_equal(direct,native_pay);cells+=len(shock_ids)*128
                    np.testing.assert_array_equal(coeff[j,f,:,k],sums/(4*n))
            print('Verified shocks',j+1,'/17',flush=True)
        # Verify every reported minimum with an independent polynomial-root calculation.
        report=read(dest/'results.json')
        for row in report['continuous_share_point_estimate_summary']:
            q=row['maximum_replacement_share'];floor=np.full(len(shock_ids),np.inf)
            for j in range(17):
                for f in range(40):
                    for i,c in enumerate(coeff[j,f]):
                        polynomial=np.array([c[3]-3*c[2]+3*c[1]-c[0],3*(c[2]-2*c[1]+c[0]),3*(c[1]-c[0]),c[0]])
                        roots=np.roots(np.polyder(polynomial));points=[0.,q]+[float(r.real)for r in roots if abs(r.imag)<1e-9 and 0<r.real<q]
                        floor[i]=min(floor[i],min(np.polyval(polynomial,t)for t in points))
            np.testing.assert_allclose(floor,row['all_action_floors'],rtol=0,atol=1e-9)
            assert int(floor[:101].argmax())==row['best_split_C2']
        counts['family_stress_subset_cells']=cells;counts['regenerated_family_stress_groups']=regenerated
    for rel,digest in freeze['source_and_input_sha256'].items():assert sha(ROOT/rel)==digest,rel
    save(out/('acceptance-'+args.part+'.json'),{'passed':True,'part':args.part,'freeze_sha256':sha(out/'freeze.json'),
        'source_sha256':sha(Path(__file__)),'counts':counts,'elapsed_seconds':time.perf_counter()-start,
        'scope':'Independent numerical audit of the declared finite laws, fitting constraints, score screens and frozen comparisons. No hidden-field or global optimum certificate.'})
    print('Accepted',args.part,flush=True)


if __name__=='__main__':main()
