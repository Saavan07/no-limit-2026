"""Independent numerical acceptance for replication and the complete search."""
import argparse
from pathlib import Path
import time
import numpy as np
from game2_lab.cross_arena import read,load_catalog,native_module,draw_population,counts_from_groups,uncompressed_groups
from game2_lab.catalog_stress import make_catalog,choose_diverse
from game2_lab.fullscan import allocations,allocation_indices,direct_sums
from game2_lab.human import payoff_arrays
from verify_game2_cross_arena import independent_cdf_gap
from run_game2_catalog_stress import get_catalog
from run_game2_replicate_populations import model_key
from run_population_study import ROOT,save,sha


def verify_hashes(record,key='source_input_sha256'):
    for rel,digest in record[key].items():assert sha(ROOT/rel)==digest,rel


def population_part(out,cfg):
    freeze=read(out/'freeze.json');verify_hashes(freeze);vendor=ROOT/'data/parry-replay'
    original,base,oldfields=load_catalog(vendor);names=list(base)
    old=ROOT/'results/game2-catalog-stress';catalogs={};own_rows={}
    for kind in sorted({m['catalog']for m in read(old/'model-validation.json')['models']}):catalogs[kind]=get_catalog(old/'catalogs'/kind,names)
    for j in range(cfg['independent_rebuilds']+1):
        key='equal_control'if j==0 else'equal_rebuild_'+str(j);actual=get_catalog(out/'catalogs'/key,names)
        expected=catalogs['equal_active_targets']if j==0 else make_catalog(base,'equal_active_targets',cfg['rebuild_rows_per_target_family'],cfg['catalog_seed']+j*1000)
        for name in names:
            np.testing.assert_array_equal(actual[name][0],expected[name][0]);np.testing.assert_array_equal(actual[name][1],expected[name][1]);assert np.all(actual[name][0].sum(axis=1)==100)
        catalogs[key]=actual
        fits=read(out/'training'/key/'fits.json')
        assert len(fits['runs'])==cfg['fit_starts_per_catalog']
        assert fits['selected']==choose_diverse(fits['runs'],cfg['retained_fits_per_catalog'],cfg['minimum_fit_l1_distance'])
        for r in fits['runs']:assert min(r['weights'])>=0 and abs(sum(r['weights'])-1)<1e-10
    model_freeze=read(out/'model-freeze.json');assert model_freeze['replication_freeze_sha256']==sha(out/'freeze.json')
    validated=read(out/'validation.json');assert len(validated['models'])==len(model_freeze['models'])
    public=np.sort(read(vendor/'public-scores.json')['scores']);native=native_module(vendor);checked=0;lookup_checked=0;generated=0
    for j,m in enumerate(validated['models']):
        initial=model_freeze['models'][j]
        assert {k:v for k,v in m.items()if k not in ['qualified','cdf_checks']}==initial
        p=catalogs[m['catalog_key']];assert model_key(p,m['weights'])==m['law_fingerprint']
        x=np.vstack([a for a,w in p.values()]);mass=np.concatenate([np.asarray(w)/np.sum(w)*m['weights'][f]for f,(a,w)in enumerate(p.values())]);mass/=mass.sum()
        pindices=allocation_indices(x)
        positives=np.flatnonzero(mass>0);rng=np.random.default_rng(11799+j);pick=rng.choice(positives,min(32,len(positives)),replace=False)
        for phase in range(cfg['cdf_independent_passes']):
            path=out/('cdf-pass-'+str(phase))/(m['name']+'.npz')
            with np.load(path)as d:
                rows=d['triplets'];sums=d['entry_rank_sum4'];controls=d['control_allocations'];csums=d['control_rank_sum4'];n=int(d['size']);seed=int(d['seed'])
            assert seed==cfg['validation_seed']+phase*1000+j
            np.testing.assert_array_equal(rows,draw_population(p,m['weights'],3*n,seed).reshape(n,3,5));generated+=n
            g=native.compress_groups(rows);counts=counts_from_groups(g,n)
            selected=np.vstack([x[pick],controls]);actual=direct_sums(selected,g,counts,4)
            np.testing.assert_array_equal(actual[:len(pick)],sums[pick]);np.testing.assert_array_equal(actual[len(pick):],csums)
            lookup_checked+=len(selected)*n
            check=np.vstack([x[pick[:2]],controls[[0,2]]]);quick,_=payoff_arrays(check.tolist(),rows)
            np.testing.assert_array_equal(quick.sum(axis=1),np.r_[sums[pick[:2]],csums[[0,2]]]);checked+=len(check)*n
            u=sums/(4*n);gap=independent_cdf_gap(u,mass,public);metrics=m['cdf_checks'][str(phase)]
            assert abs(gap-metrics['maximum_cdf_gap'])<1e-10
            order=np.argsort(u);v=u[order];cw=np.cumsum(mass[order])
            def cdf(t):return np.r_[0.,cw][np.searchsorted(v,t,side='right')]
            grid=np.linspace(0,9,181);mean=float(abs(cdf(grid)-np.searchsorted(public,grid,side='right')/len(public)).mean())
            thresholds=np.array(original['thresholds']);original_mean=float(abs(cdf(thresholds)-np.searchsorted(public,thresholds,side='right')/len(public)).mean())
            tail=float(mass[u>public.max()+.2].sum())
            assert abs(mean-metrics['mean_grid_cdf_gap'])<1e-10 and abs(original_mean-metrics['original_threshold_mean_gap'])<1e-10
            assert abs(tail-metrics['mass_above_public_max_plus_point_two'])<1e-10
            passed=gap<=.085 and mean<=.035 and original_mean<=.035 and tail<=.015
            assert passed==metrics['passes']and metrics['own_action_sampling']is False
        assert m['qualified']==all(r['passes']for r in m['cdf_checks'].values())
        print('Verified population',j+1,'/',len(validated['models']),flush=True)
    assert validated['qualified_count']==sum(m['qualified']for m in validated['models'])
    verify_hashes(freeze)
    return {'models':len(validated['models']),'qualified':validated['qualified_count'],'regenerated_opponent_groups':generated,
        'full_opponent_direct_lookup_cells':lookup_checked,'full_opponent_foundation_cells':checked,'cdf_checks_cover_complete_stored_laws':True}


def search_part(out,cfg,phase):
    freeze=read(out/'freeze.json');verify_hashes(freeze);models=freeze['models'];rules={k:np.array(v,bool)for k,v in freeze['rules'].items()};xs=allocations()
    for name,mask in rules.items():
        if name=='qualified':assert mask.tolist()==[m['qualified']for m in models]
        if name=='legacy87':assert mask.tolist()==['legacy87'in m['tags']for m in models]
        if name=='original17':assert mask.tolist()==['original17'in m['tags']for m in models]
    names=read(ROOT/'data/parry-replay/results/field_fit_validated.json')['catalog'];catalogs={};native=native_module(ROOT/'data/parry-replay')
    n=cfg[phase+'_groups'];minima={k:np.full(len(xs),np.inf)for k in rules};regrets={k:np.zeros(len(xs))for k in rules}
    old_ids=allocation_indices([[0,36,0,64,0],[0,0,36,64,0],[15,18,49,12,6],[0,0,0,0,100]])
    checked=0;lookup_checked=0;generated=0
    final=read(out/(phase+'-results.json'));leader_ids=[v['index']for v in final['leaders'].values()]
    for j,m in enumerate(models):
        folder=m['catalog_folder']
        if folder not in catalogs:catalogs[folder]=get_catalog(ROOT/folder,names)
        p=catalogs[folder];assert model_key(p,m['weights'])==m['law_fingerprint']
        with np.load(out/phase/(m['name']+'.npz'))as d:rows=d['triplets'];sums=d['rank_sum4'];seed=int(d['seed']);assert int(d['size'])==n
        assert len(sums)==4598126 and np.all(sums<=36*n)
        assert seed==cfg[phase+'_seed']+j
        np.testing.assert_array_equal(rows,draw_population(p,m['weights'],3*n,seed).reshape(n,3,5));generated+=n
        rng=np.random.default_rng(12773+j);ids=np.unique(np.r_[old_ids,leader_ids,rng.choice(len(xs),64,replace=False)])
        groups=native.compress_groups(rows);counts=counts_from_groups(groups,n)
        np.testing.assert_array_equal(direct_sums(xs[ids],groups,counts,4),sums[ids]);lookup_checked+=len(ids)*n
        check=np.unique(np.r_[old_ids,leader_ids[0]]);direct,_=payoff_arrays(xs[check].tolist(),rows)
        np.testing.assert_array_equal(direct.sum(axis=1),sums[check]);checked+=len(check)*n
        u=sums.astype(float)/(4*n);best=float(u.max())
        for name,mask in rules.items():
            if mask[j]:np.minimum(minima[name],u,out=minima[name]);np.maximum(regrets[name],best-u,out=regrets[name])
        print('Verified full space',phase,j+1,'/',len(models),flush=True)
    with np.load(out/(phase+'-objectives.npz'))as d:
        for name in rules:
            np.testing.assert_array_equal(minima[name],d['minimum__'+name]);np.testing.assert_array_equal(regrets[name],d['regret__'+name])
            i=int(np.argmax(minima[name]));r=int(np.argmin(regrets[name]));record=final['leaders'][name]
            assert record['index']==i and record['allocation']==xs[i].tolist()and record['minimum']==minima[name][i]
            assert record['minimax_regret_index']==r
    verify_hashes(freeze)
    return {'legal_actions':len(xs),'fields':len(models),'regenerated_groups':generated,'full_opponent_lookup_cells':lookup_checked,
        'full_opponent_foundation_cells':checked,'all_full_space_aggregate_entries_reconstructed':True}


def confirmation_part(out,cfg):
    freeze=read(out/'freeze.json');verify_hashes(freeze);models=freeze['models'];frozen=read(out/'confirmation-freeze.json');result=read(out/'confirmation-results.json')
    assert frozen['selection_results_sha256']==sha(out/'selection-results.json')
    assert frozen['chosen_before_confirmation_index']==read(out/'selection-results.json')['leaders']['qualified']['index']
    np.testing.assert_array_equal(allocations()[frozen['indices']],frozen['allocations'])
    c=len(frozen['indices']);n=cfg['confirmation_groups'];values=[];matrices=[];checked=0;subset=0
    names=read(ROOT/'data/parry-replay/results/field_fit_validated.json')['catalog'];catalogs={};native=native_module(ROOT/'data/parry-replay')
    chosen=frozen['indices'].index(frozen['chosen_before_confirmation_index'])
    named=frozen['named_indices'];full_ids=sorted({chosen,*[frozen['indices'].index(named[key])for key in ['previous_C3_C4','previous_C2_C4','historical_balanced']]})
    for j,m in enumerate(models):
        folder=m['catalog_folder']
        if folder not in catalogs:catalogs[folder]=get_catalog(ROOT/folder,names)
        with np.load(out/'confirmation'/(m['name']+'.npz'))as d:rows=d['triplets'];sums=d['rank_sum4'];cross=d['cross_sum16'];seed=int(d['seed']);assert int(d['size'])==n
        assert seed==cfg['confirmation_seed']+j
        np.testing.assert_array_equal(rows,draw_population(catalogs[folder],m['weights'],3*n,seed).reshape(n,3,5))
        full,_=payoff_arrays([frozen['allocations'][i]for i in full_ids],rows)
        np.testing.assert_array_equal(full.sum(axis=1),sums[full_ids]);q=full.astype(float);np.testing.assert_array_equal(q@q.T,cross[np.ix_(full_ids,full_ids)])
        checked+=len(full_ids)*n
        ix=np.random.default_rng(1979+j).choice(n,1024,replace=False);quick,_=payoff_arrays(frozen['allocations'],rows[ix]);original=native.payoff_matrix(frozen['allocations'],uncompressed_groups(rows[ix]))//3
        np.testing.assert_array_equal(quick,original);subset+=c*len(ix)
        values.append(sums/(4*n));matrices.append(cross)
        print('Verified confirmation',j+1,'/',len(models),flush=True)
    v=np.array(values).T;cross=np.array(matrices);alpha=.05/2;comparisons=len(models)*(c+c*(c-1)//2);log=np.log(4*comparisons/alpha)
    var=np.maximum((cross[:,np.arange(c),np.arange(c)]/16-n*v.T**2)/(n-1),0)
    error=np.sqrt(2*var*log/n)+7*9*log/(3*(n-1));mask=np.array(freeze['rules']['qualified'],bool)
    lower=float((v[chosen,mask]-error[mask,chosen]).min());upper=float((v[chosen,mask]+error[mask,chosen]).min())
    eps=9*np.sqrt(np.log(2*4598126*len(models)/alpha)/(2*cfg['selection_groups']))
    global_upper=read(out/'selection-results.json')['leaders']['qualified']['minimum']+eps
    assert abs(lower-result['confirmation_robust_lower'])<1e-12 and abs(upper-result['confirmation_robust_upper'])<1e-12
    assert abs(max(0,global_upper-lower)-result['conditional_full_space_optimality_shortfall_bound'])<1e-12
    with np.load(out/'confirmation-scores.npz')as d:
        np.testing.assert_array_equal(v,d['values']);np.testing.assert_array_equal(cross,d['cross_sum16']);np.testing.assert_allclose(error,d['mean_error'],rtol=0,atol=1e-12)
    verify_hashes(freeze)
    return {'confirmation_candidates':c,'fields':len(models),'full_opponent_foundation_cells':checked,'all_candidate_subset_cells':subset,
        'confidence_bounds_and_global_shortfall_reconstructed':True}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--part',choices=['population','discovery','selection','confirmation'],required=True);args=parser.parse_args();start=time.perf_counter()
    cfg=read(ROOT/'config/game2-replication-search.json');out=ROOT/'results'/('game2-population-replication'if args.part=='population'else'game2-fullspace-search')
    if args.part=='population':counts=population_part(out,cfg)
    elif args.part=='confirmation':counts=confirmation_part(out,cfg)
    else:counts=search_part(out,cfg,args.part)
    save(out/('acceptance-'+args.part+'.json'),{'passed':True,'part':args.part,'source_sha256':sha(Path(__file__)),'freeze_sha256':sha(out/'freeze.json'),
        'counts':counts,'elapsed_seconds':time.perf_counter()-start,'scope':'Numerical/source validation under the declared finite catalog laws. Does not validate actual applicant proportions.'})
    print('Accepted',args.part,flush=True)


if __name__=='__main__':main()
