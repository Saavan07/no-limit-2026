"""Independent equal-centered population replication and full-catalog CDF checks."""
import argparse
import hashlib
from pathlib import Path
import shutil
import platform
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from scipy.optimize import minimize
from game2_lab.cross_arena import read,load_catalog,native_module,draw_population,counts_from_groups
from game2_lab.catalog_stress import make_catalog,fingerprint,CubicCDF,choose_diverse
from game2_lab.fullscan import all_sample_sums,allocation_indices,library
from game2_lab.handoff import cdf_diagnostic
from run_game2_catalog_stress import prepare,store_catalog,get_catalog
from run_population_study import ROOT,save,sha


def fit_rebuild(catalog,folder,index,cfg,original_fields,influential):
    folder.mkdir(parents=True,exist_ok=True)
    prepare(catalog,folder/'coefficients.npz',native_module(ROOT/'data/parry-replay'),cfg,index)
    with np.load(folder/'coefficients.npz')as d:
        coeff=d['rank_sum4'].astype(float)/(4*int(d['size']));ids=d['ids'];families=d['families']
    public=np.array(read(ROOT/'data/parry-replay/public-scores.json')['scores'])
    thresholds=np.array(read(ROOT/'data/parry-replay/results/field_fit_validated.json')['thresholds'])
    query=np.r_[thresholds,np.linspace(0,9,181)];observed=np.array([(public<=t).mean()for t in query])
    importance=np.r_[np.full(len(thresholds),.5/len(thresholds)),np.full(181,.5/181)]
    objective=CubicCDF(coeff,ids,families,query,observed,importance,.06)
    path=folder/'fits.json';record=read(path)if path.exists()else {'runs':[]}
    warm=[influential['weights'],original_fields[0]['weights'],original_fields[12]['weights'],original_fields[15]['weights']]
    rng=np.random.default_rng(cfg['optimizer_seed']+index*1000)
    initial=[np.asarray(warm[r],float).copy()if r<len(warm)else rng.dirichlet(np.full(40,.7))for r in range(cfg['fit_starts_per_catalog'])]
    for run in range(len(record['runs']),len(initial)):
        objective.bandwidth=[.06,.10,.16][run%3];objective.last=None
        res=minimize(lambda w:objective.stats(w)[0],initial[run],jac=lambda w:objective.stats(w)[1],method='SLSQP',bounds=[(0,1)]*40,
            constraints={'type':'eq','fun':lambda w:w.sum()-1,'jac':lambda w:np.ones(40)},
            options={'maxiter':cfg['maximum_optimizer_iterations'],'ftol':cfg['optimizer_ftol']})
        w=np.maximum(res.x,0);w/=w.sum();loss,gradient,u,cdf=objective.stats(w)
        record['runs'].append({'run':run,'weights':w.tolist(),'bandwidth':objective.bandwidth,'loss':loss,'iterations':int(res.nit),
            'solver_success':bool(res.success),'solver_message':str(res.message),'smoothed_max_cdf_gap':float(abs(cdf-observed).max())})
        save(path,record);print('REPLICATE FIT',folder.name,run+1,'/',len(initial),'loss',round(loss,7),flush=True)
    record['selected']=choose_diverse(record['runs'],cfg['retained_fits_per_catalog'],cfg['minimum_fit_l1_distance'])
    save(path,record)
    return record['selected']


def model_key(catalog,weights):
    h=hashlib.sha256()
    for name,(x,w)in catalog.items():h.update(name.encode());h.update(fingerprint(x,w).encode())
    w=np.array(weights,float);w/=w.sum();h.update(w.tobytes())
    return h.hexdigest()


def collect_models(out,catalogs,cfg,old_models,original_fields,influential):
    rows=[];seen={}
    def add(name,catkey,weights,tags,origin):
        p=catalogs[catkey];key=model_key(p,weights)
        if key in seen:
            row=rows[seen[key]];row['aliases'].append(name);row['tags']=sorted(set(row['tags']+tags));return
        seen[key]=len(rows)
        rows.append({'name':name,'catalog_key':catkey,'catalog_folder':str(out/'catalogs'/catkey)if (catkey=='equal_control'or catkey.startswith('equal_rebuild_'))else str(ROOT/'results/game2-catalog-stress/catalogs'/catkey),
            'weights':weights,'tags':tags,'aliases':[name],'law_fingerprint':key,'origin':origin,'equal_centered':catkey.startswith('equal_')or catkey=='equal_active_targets'})
    for m in old_models:
        tags=['legacy87']+(['original17']if m['constraint']=='original_supplied'else[])
        add('legacy__'+m['name'],m['catalog'],m['weights'],tags,{'stage':'catalog-stress','model':m['name']})
    present={m['name']for m in old_models if m['constraint']=='original_supplied'}
    for m in original_fields:
        if 'supplied_'+m['name']not in present:add('reference__'+m['name'],'original',m['weights'],['original17'],{'stage':'original supplied field','model':m['name']})
    for j in range(cfg['independent_rebuilds']+1):
        key='equal_control'if j==0 else'equal_rebuild_'+str(j)
        fits=read(out/'training'/key/'fits.json')['selected']
        for k,row in enumerate(fits):add(key+'__refit_'+str(k),key,row['weights'],['independent_refit',key],{'kind':'refit','training_run':row['run']})
        add(key+'__transplanted_anchor',key,influential['weights'],['anchor_transplant',key],{'kind':'unchanged influential weights on independent catalog'if j else'original influential field recheck'})
    # Store relative portable paths, rather than machine-owned absolute paths.
    for row in rows:row['catalog_folder']=str(Path(row['catalog_folder']).relative_to(ROOT))
    save(out/'model-freeze.json',{'models':rows,'replication_freeze_sha256':sha(out/'freeze.json'),
        'scope':'All models fixed before full-catalog validation and before new entrant evaluation. Duplicate identical law/weight inputs merged with aliases.'})
    return rows


def validate_one(model,index,phase,out,cfg,catalogs,native,public,thresholds):
    p=catalogs[model['catalog_key']];n=cfg['cdf_validation_groups'];seed=cfg['validation_seed']+phase*1000+index
    path=out/('cdf-pass-'+str(phase))/(model['name']+'.npz')
    x=np.vstack([a for a,w in p.values()]);indices=allocation_indices(x)
    mass=np.concatenate([np.asarray(w)/np.sum(w)*model['weights'][j]for j,(a,w)in enumerate(p.values())]);mass/=mass.sum()
    if not path.exists():
        rows=draw_population(p,model['weights'],3*n,seed).reshape(n,3,5);g=native.compress_groups(rows)
        sums,timing=all_sample_sums(g,counts_from_groups(g,n))
        path.parent.mkdir(exist_ok=True)
        controls=np.array([[0,36,0,64,0],[0,37,0,63,0],[0,0,36,64,0],[0,0,37,63,0],[15,18,49,12,6],[0,0,0,0,100]],np.uint8)
        np.savez_compressed(path,triplets=rows,entry_rank_sum4=sums[indices].astype(np.uint32),control_allocations=controls,control_rank_sum4=sums[allocation_indices(controls)],size=n,seed=seed)
        save(path.with_suffix('.timing.json'),timing)
    with np.load(path)as d:u=d['entry_rank_sum4']/(4*n)
    metrics=cdf_diagnostic(u,mass,public)
    observed=np.array([(public<=t).mean()for t in thresholds]);predicted=np.array([mass[u<=t].sum()for t in thresholds])
    metrics['original_threshold_mean_gap']=float(abs(predicted-observed).mean())
    metrics['passes']=bool(metrics['maximum_cdf_gap']<=cfg['cdf_maximum_tolerance']and metrics['mean_grid_cdf_gap']<=cfg['cdf_mean_tolerance']and metrics['original_threshold_mean_gap']<=cfg['cdf_original_threshold_mean_tolerance']and metrics['mass_above_public_max_plus_point_two']<=cfg['cdf_tail_tolerance'])
    metrics['full_catalog_rows']=len(x);metrics['positive_mass_unique_allocations']=int(len(np.unique(indices[mass>0])))
    metrics['own_action_sampling']=False;metrics['opponent_sampling']=True
    print('FULL CATALOG CDF',phase,index+1,model['name'],'gap',round(metrics['maximum_cdf_gap'],4),'pass',metrics['passes'],flush=True)
    return metrics


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--phase',choices=['all','fit','validate'],default='all')
    parser.add_argument('--output',type=Path,default=ROOT/'results/game2-population-replication');args=parser.parse_args()
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True);cfg=read(ROOT/'config/game2-replication-search.json')
    old=ROOT/'results/game2-catalog-stress';old_models=[m for m in read(old/'model-validation.json')['models']if m['validation']['passes']]
    influential=next(m for m in old_models if m['name']=='equal_active_targets__unrestricted__2')
    original_fit,base,original_fields=load_catalog(ROOT/'data/parry-replay');names=list(base)
    sources=[Path(__file__),ROOT/'game2_lab/fullscan.cpp',ROOT/'game2_lab/fullscan.py',ROOT/'config/game2-replication-search.json',ROOT/'tests/test_game2_fullscan.py']
    inputs=[old/'freeze.json',old/'model-validation.json',old/'catalogs/equal_active_targets/rows.npz',old/'catalogs/equal_active_targets/weights.npz',
        ROOT/'data/parry-replay/public-scores.json',ROOT/'data/parry-replay/lookup.cpp',ROOT/'run_game2_catalog_stress.py',ROOT/'game2_lab/catalog_stress.py',ROOT/'game2_lab/cross_arena.py']
    inputs += [ROOT/'data/parry-replay/results'/name for name in ['field_fit_validated.json','fit_catalog.npz','fit_catalog_weights.npz']]
    inputs += sorted((old/'catalogs').glob('*/rows.npz'))+sorted((old/'catalogs').glob('*/weights.npz'))
    frozen={'configuration':cfg,'legal_action_universe':4598126,'source_input_sha256':{str(p.relative_to(ROOT)):sha(p)for p in sources+inputs},
        'versions':{'python':platform.python_version(),'numpy':np.__version__},'scope':cfg['scope']}
    path=out/'freeze.json'
    if path.exists():assert read(path)==frozen,'Changed replication source or inputs'
    else:
        save(path,frozen)
        for p in sources:
            dest=out/'source-snapshot'/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    catalogs={}
    for kind in sorted({r['catalog']for r in old_models}|{'original'}):catalogs[kind]=get_catalog(old/'catalogs'/kind,names)
    for j in range(cfg['independent_rebuilds']+1):
        key='equal_control'if j==0 else'equal_rebuild_'+str(j);folder=out/'catalogs'/key
        if not(folder/'rows.npz').exists():
            p=catalogs['equal_active_targets']if j==0 else make_catalog(base,'equal_active_targets',cfg['rebuild_rows_per_target_family'],cfg['catalog_seed']+j*1000)
            store_catalog(folder,p)
        catalogs[key]=get_catalog(folder,names)
    if args.phase in ['all','fit']:
        for j in range(cfg['independent_rebuilds']+1):
            key='equal_control'if j==0 else'equal_rebuild_'+str(j)
            fit_rebuild(catalogs[key],out/'training'/key,j,cfg,original_fields,influential)
        collect_models(out,catalogs,cfg,old_models,original_fields,influential)
    if args.phase in ['all','validate']:
        model_freeze=read(out/'model-freeze.json');assert model_freeze['replication_freeze_sha256']==sha(out/'freeze.json')
        models=model_freeze['models']
        for m in models:assert model_key(catalogs[m['catalog_key']],m['weights'])==m['law_fingerprint']
        native=native_module(ROOT/'data/parry-replay');native._library();library()
        public=np.array(read(ROOT/'data/parry-replay/public-scores.json')['scores']);thresholds=original_fit['thresholds'];metrics=[{}for _ in models]
        for phase in range(cfg['cdf_independent_passes']):
            with ThreadPoolExecutor(max_workers=cfg['workers'])as pool:
                jobs=[pool.submit(validate_one,m,j,phase,out,cfg,catalogs,native,public,thresholds)for j,m in enumerate(models)]
                for j,job in enumerate(jobs):metrics[j][str(phase)]=job.result()
        records=[{**m,'cdf_checks':metrics[j],'qualified':all(v['passes']for v in metrics[j].values())}for j,m in enumerate(models)]
        save(out/'validation.json',{'models':records,'qualified_count':sum(m['qualified']for m in records),
            'scope':'Two full-stored-catalog CDF screens; opponents still sampled. Passing is diagnostic compatibility, not identification or a probability that the population is real.'})
        print('QUALIFIED',sum(m['qualified']for m in records),'/',len(records),flush=True)
    for rel,digest in frozen['source_input_sha256'].items():assert sha(ROOT/rel)==digest,rel
    print('Population phase complete',args.phase,flush=True)


if __name__=='__main__':main()
