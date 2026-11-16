"""All-allocation discovery/selection and fresh fixed-candidate confirmation."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import shutil
import numpy as np
from game2_lab.cross_arena import read,native_module,draw_population,counts_from_groups
from game2_lab.fullscan import allocations,allocation_indices,all_sample_sums,direct_matrix4,library
from game2_lab.catalog_stress import paired_moments,empirical_bernstein_error
from run_game2_catalog_stress import get_catalog
from run_game2_replicate_populations import model_key
from run_population_study import ROOT,save,sha,write_csv


def top_indices(values,count,largest=True):
    z=np.asarray(values);count=min(count,len(z))
    if not largest:z=-z
    threshold=np.partition(z,len(z)-count)[len(z)-count]
    better=np.flatnonzero(z>threshold);tied=np.flatnonzero(z==threshold)[:count-len(better)]
    ids=np.r_[better,tied]
    return ids[np.lexsort((ids,-z[ids]))]


def masks_for(models):
    qualified=np.array([r['qualified']for r in models]);legacy=np.array(['legacy87'in r['tags']for r in models]);original=np.array(['original17'in r['tags']for r in models])
    rules={'qualified':qualified,'legacy87':legacy,'legacy_union':legacy|qualified,'original17':original,'qualified_non_equal':qualified&~np.array([r['equal_centered']for r in models])}
    if not qualified.any():raise ValueError('No populations passed both full-catalog checks')
    return {k:v for k,v in rules.items()if v.any()}


def score_field(model,j,phase,out,cfg,catalogs,native):
    n=cfg['discovery_groups']if phase=='discovery'else cfg['selection_groups']
    seed=cfg[phase+'_seed']+j;path=out/phase/(model['name']+'.npz')
    if not path.exists():
        p=catalogs[model['catalog_folder']];rows=draw_population(p,model['weights'],3*n,seed).reshape(n,3,5)
        groups=native.compress_groups(rows);sums,timing=all_sample_sums(groups,counts_from_groups(groups,n))
        assert np.all((sums>=0)&(sums<=36*n))and 36*n<2**32
        path.parent.mkdir(exist_ok=True)
        np.savez_compressed(path,triplets=rows,rank_sum4=sums.astype(np.uint32),size=n,seed=seed)
        save(path.with_suffix('.timing.json'),timing)
    print('ALL ACTIONS',phase,j+1,model['name'],flush=True)
    return path


def complete_phase(phase,models,catalogs,rules,out,cfg,native):
    xs=allocations();size=len(xs);minima={k:np.full(size,np.inf)for k in rules};regrets={k:np.zeros(size)for k in rules}
    tops={};n=cfg['discovery_groups']if phase=='discovery'else cfg['selection_groups']
    with ThreadPoolExecutor(max_workers=cfg['workers'])as executor:
        jobs=[executor.submit(score_field,r,j,phase,out,cfg,catalogs,native)for j,r in enumerate(models)]
        for j,job in enumerate(jobs):
            path=job.result()
            with np.load(path)as d:u=d['rank_sum4'].astype(float)/(4*n)
            best=float(u.max());tops[models[j]['name']]=top_indices(u,cfg['discovery_top_per_field']).tolist()
            for name,mask in rules.items():
                if mask[j]:np.minimum(minima[name],u,out=minima[name]);np.maximum(regrets[name],best-u,out=regrets[name])
    np.savez_compressed(out/(phase+'-objectives.npz'),**{'minimum__'+k:v for k,v in minima.items()},**{'regret__'+k:v for k,v in regrets.items()})
    leaders={};short=set()
    for name in rules:
        i=int(minima[name].argmax());r=int(regrets[name].argmin())
        leaders[name]={'index':i,'allocation':xs[i].tolist(),'minimum':float(minima[name][i]),'minimax_regret_index':r,'minimax_regret_allocation':xs[r].tolist(),'maximum_regret':float(regrets[name][r]),'fields':int(rules[name].sum())}
        short.update(top_indices(minima[name],cfg['discovery_shortlist_per_rule']).tolist());short.update(top_indices(regrets[name],cfg['discovery_shortlist_per_rule'],False).tolist())
    for v in tops.values():short.update(v)
    save(out/(phase+'-results.json'),{'universe_size':size,'groups_per_model':n,'leaders':leaders,'shortlist_indices':sorted(short),
        'field_top_indices':tops,'scope':'Complete action enumeration, exact integer sample scores; population payoffs remain Monte Carlo estimates under fixed model laws.'})
    print(phase.upper(),'LEADERS',leaders,flush=True)
    return minima,regrets,leaders


def freeze_confirmation(out,cfg,rules,minima,regrets,leaders):
    xs=allocations();chosen=set()
    for name in rules:
        chosen.update(top_indices(minima[name],cfg['confirmation_candidates_per_rule']).tolist())
        chosen.add(int(regrets[name].argmin()))
    baseline={'previous_C3_C4':[0,0,36,64,0],'adjacent_C3_C4':[0,0,37,63,0],'previous_C2_C4':[0,36,0,64,0],
        'best_C2_C4_last_stage':[0,37,0,63,0],'historical_balanced':[15,18,49,12,6],
        'all_in_five':[0,0,0,0,100],'all_in_four':[0,0,0,100,0]}
    ids=allocation_indices(list(baseline.values()));chosen.update(ids.tolist())
    chosen.add(read(out/'discovery-results.json')['leaders']['qualified']['index'])
    assert len(chosen)<=cfg['maximum_confirmation_candidates']
    indices=sorted(chosen);frozen={'indices':indices,'allocations':xs[indices].tolist(),'named_indices':{name:int(i)for name,i in zip(baseline,ids)},
        'chosen_before_confirmation_index':leaders['qualified']['index'],'chosen_allocation':leaders['qualified']['allocation'],
        'selection_results_sha256':sha(out/'selection-results.json'),'scope':'No candidate is added or selected using confirmation outcomes. All candidates already belong to the initially fixed full action universe.'}
    path=out/'confirmation-freeze.json'
    if path.exists():assert read(path)==frozen
    else:save(path,frozen)
    return frozen


def confirm_one(model,j,out,cfg,catalogs,native,frozen):
    path=out/'confirmation'/(model['name']+'.npz');n=cfg['confirmation_groups'];seed=cfg['confirmation_seed']+j
    if not path.exists():
        p=catalogs[model['catalog_folder']];rows=draw_population(p,model['weights'],3*n,seed).reshape(n,3,5);g=native.compress_groups(rows)
        q4=direct_matrix4(frozen['allocations'],g,4);sums,cross=paired_moments(q4*3,counts_from_groups(g,n))
        path.parent.mkdir(exist_ok=True)
        np.savez_compressed(path,triplets=rows,rank_sum4=sums,cross_sum16=cross,size=n,seed=seed)
    print('CONFIRM',j+1,model['name'],flush=True)
    return path


def confirm(models,catalogs,rules,out,cfg,native,frozen):
    values=[];crosses=[]
    with ThreadPoolExecutor(max_workers=cfg['workers'])as executor:
        jobs=[executor.submit(confirm_one,r,j,out,cfg,catalogs,native,frozen)for j,r in enumerate(models)]
        for job in jobs:
            with np.load(job.result())as d:values.append(d['rank_sum4']/(4*cfg['confirmation_groups']));crosses.append(d['cross_sum16'])
    v=np.array(values).T;cross=np.array(crosses);n=cfg['confirmation_groups'];c=len(frozen['indices']);f=len(models)
    variance=np.maximum((cross[:,np.arange(c),np.arange(c)]/16-n*v.T**2)/(n-1),0)
    comparisons=f*(c+c*(c-1)//2);alpha=cfg['confidence_alpha']/2
    error=empirical_bernstein_error(variance,n,comparisons,alpha,9)
    i=frozen['indices'].index(frozen['chosen_before_confirmation_index']);mask=rules['qualified']
    floor=float(v[i,mask].min());lower=float((v[i,mask]-error[mask,i]).min());upper=float((v[i,mask]+error[mask,i]).min())
    selection=read(out/'selection-results.json');empirical_global=selection['leaders']['qualified']['minimum']
    epsilon=9*np.sqrt(np.log(2*len(allocations())*f/alpha)/(2*cfg['selection_groups']))
    global_upper=empirical_global+epsilon;gap=max(0.,global_upper-lower)
    rows=[]
    for name,index in frozen['named_indices'].items():
        j=frozen['indices'].index(index)
        rows.append({'name':name,'allocation':frozen['allocations'][j],'minimum':float(v[j,mask].min()),'maximum':float(v[j,mask].max()),
            'robust_lower':float((v[j,mask]-error[mask,j]).min()),'robust_upper':float((v[j,mask]+error[mask,j]).min())})
    result={'working_allocation':frozen['chosen_allocation'],'chosen_index':frozen['chosen_before_confirmation_index'],
        'primary_field_count':int(mask.sum()),'all_evaluated_fields':f,'confirmation_candidates':c,'full_action_universe':len(allocations()),
        'confirmation_minimum':floor,'confirmation_maximum':float(v[i,mask].max()),'confirmation_robust_lower':lower,'confirmation_robust_upper':upper,
        'full_space_selection_cell_error_bound':float(epsilon),'full_space_optimum_upper':float(global_upper),'conditional_full_space_optimality_shortfall_bound':float(gap),
        'named_comparisons':rows,'joint_numerical_confidence':1-cfg['confidence_alpha'],
        'scope':'Global finite-action sampling bound plus independent fixed-candidate confirmation, conditional on the frozen population laws. Excludes population-model error and does not certify the hidden applicant optimum.'}
    np.savez_compressed(out/'confirmation-scores.npz',values=v,cross_sum16=cross,mean_error=error,names=[r['name']for r in models],indices=frozen['indices'])
    save(out/'confirmation-results.json',result)
    write_csv(out/'confirmation-candidates.csv',['allocation','minimum','maximum','robust_lower','robust_upper'],
        [[str(x),float(v[j,mask].min()),float(v[j,mask].max()),float((v[j,mask]-error[mask,j]).min()),float((v[j,mask]+error[mask,j]).min())]for j,x in enumerate(frozen['allocations'])])
    print('CONFIRMED RESULT',result,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--phase',choices=['all','discovery','selection','confirmation'],default='all')
    parser.add_argument('--output',type=Path,default=ROOT/'results/game2-fullspace-search');args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    cfg=read(ROOT/'config/game2-replication-search.json');population=ROOT/'results/game2-population-replication';validated=read(population/'validation.json')
    models=[r for r in validated['models']if r['qualified']or 'legacy87'in r['tags']or 'original17'in r['tags']];rules=masks_for(models)
    sources=[Path(__file__),ROOT/'game2_lab/fullscan.py',ROOT/'game2_lab/fullscan.cpp',ROOT/'config/game2-replication-search.json']
    frozen={'configuration':cfg,'models':models,'rules':{k:v.tolist()for k,v in rules.items()},'all_legal_actions':4598126,
        'source_input_sha256':{str(p.relative_to(ROOT)):sha(p)for p in sources+[population/'freeze.json',population/'model-freeze.json',population/'validation.json',ROOT/'data/parry-replay/lookup.cpp']},
        'scope':cfg['scope']}
    path=out/'freeze.json'
    if path.exists():assert read(path)==frozen,'Changed full-space study source or inputs'
    else:
        save(path,frozen)
        for p in sources:
            q=out/'source-snapshot'/p.relative_to(ROOT);q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
    family_names=read(ROOT/'data/parry-replay/results/field_fit_validated.json')['catalog'];catalogs={}
    for model in models:
        folder=model['catalog_folder']
        if folder not in catalogs:catalogs[folder]=get_catalog(ROOT/folder,family_names)
        assert model_key(catalogs[folder],model['weights'])==model['law_fingerprint']
    native=native_module(ROOT/'data/parry-replay');native._library();library()
    if args.phase in ['all','discovery']:complete_phase('discovery',models,catalogs,rules,out,cfg,native)
    if args.phase in ['all','selection']:
        minima,regrets,leaders=complete_phase('selection',models,catalogs,rules,out,cfg,native)
        freeze_confirmation(out,cfg,rules,minima,regrets,leaders)
    if args.phase in ['all','confirmation']:confirm(models,catalogs,rules,out,cfg,native,read(out/'confirmation-freeze.json'))
    for rel,digest in frozen['source_input_sha256'].items():assert sha(ROOT/rel)==digest,rel
    print('Full-space phase complete',args.phase,flush=True)


if __name__=='__main__':main()
