"""Refit catalog variants, test all splits, and stress every supplied family."""
import argparse
from datetime import datetime,timezone
import itertools
import json
from pathlib import Path
import platform
import shutil
import time
import numpy as np
from scipy.optimize import minimize
from game2_lab.cross_arena import (read,load_catalog,native_module,draw_population,sample_actions,
    uncompressed_groups,counts_from_groups,exact_sample_sums)
from game2_lab.catalog_stress import (CATALOG_VARIANTS,make_catalog,fingerprint,CubicCDF,
    choose_diverse,split_actions,paired_moments,cubic_minimum,empirical_bernstein_error)
from game2_lab.handoff import cdf_diagnostic
from run_population_study import ROOT,save,sha,write_csv


def store_catalog(path,catalog):
    path.mkdir(parents=True,exist_ok=True)
    if not(path/'rows.npz').exists():
        np.savez_compressed(path/'rows.npz',**{name:x for name,(x,w)in catalog.items()})
        np.savez_compressed(path/'weights.npz',**{name:w for name,(x,w)in catalog.items()})
    save(path/'definition.json',{'families':list(catalog),'fingerprints':{name:fingerprint(x,w)for name,(x,w)in catalog.items()}})


def get_catalog(path,names):
    with np.load(path/'rows.npz')as x,np.load(path/'weights.npz')as w:return {name:(x[name].copy(),w[name].copy())for name in names}


def prepare(catalog,path,native,cfg,index):
    if path.exists():return
    k=len(catalog);m=cfg['training_conditional_groups'];a=cfg['training_own_per_family']
    own,family=sample_actions(catalog,a,cfg['seed']+1000+index)
    ids=np.array(list(itertools.combinations_with_replacement(range(k),3)),dtype=int)
    rng=np.random.default_rng(cfg['seed']+2000+index);rows=np.empty((len(ids),m,3,5),dtype=np.uint8)
    names=list(catalog)
    for j,combo in enumerate(ids):
        for pos,kind in enumerate(combo):
            x,w=catalog[names[kind]];rows[j,:,pos]=x[rng.choice(len(x),m,p=w/w.sum())]
    pay=native.payoff_matrix(own,uncompressed_groups(rows.reshape(-1,3,5)))
    assert np.all(pay%3==0)
    sums=pay.reshape(len(own),len(ids),m).sum(axis=2,dtype=np.uint16)//3
    np.savez_compressed(path,own=own,families=family,ids=ids,rank_sum4=sums,conditional_triplets=rows,size=m)


def fit_catalog(kind,index,catalog,path,native,cfg,fields,public,thresholds):
    path.mkdir(parents=True,exist_ok=True);prepare(catalog,path/'coefficients.npz',native,cfg,index)
    with np.load(path/'coefficients.npz')as d:
        coefficients=d['rank_sum4'].astype(float)/(4*int(d['size']));ids=d['ids'];family=d['families']
    original=np.asarray(thresholds);dense=np.linspace(0,9,181);query=np.r_[original,dense]
    observed=np.array([(public<=t).mean()for t in query]);importance=np.r_[np.full(len(original),.5/len(original)),np.full(len(dense),.5/len(dense))]
    model=CubicCDF(coefficients,ids,family,query,observed,importance,.06)
    jobs=[('unrestricted',None,cfg['base_fit_starts']if index==0 else cfg['variant_fit_starts'],cfg['models_per_catalog'])]
    if index==0:jobs += [('omit_'+name,j,cfg['leave_one_out_fit_starts'],cfg['models_per_leave_one_out'])for j,name in enumerate(catalog)]
    all_selected=[]
    warm=[fields[j]['weights']for j in [0,12,15,2]]
    for job_number,(name,omitted,starts,keep)in enumerate(jobs):
        dest=path/(name+'.json')
        record=read(dest)if dest.exists()else {'catalog':kind,'constraint':name,'omitted_family':None if omitted is None else list(catalog)[omitted],'runs':[]}
        rng=np.random.default_rng(cfg['seed']+10000+index*1000+job_number)
        initials=[]
        for run in range(starts):
            initial=np.array(warm[run],float)if run<len(warm)else rng.dirichlet(np.full(len(catalog),.7))
            if omitted is not None:initial[omitted]=0
            initial/=initial.sum();initials.append(initial)
        for run in range(len(record['runs']),starts):
            model.bandwidth=[.06,.10,.16][run%3];model.last=None
            bounds=[(0,0)if j==omitted else(0,1)for j in range(len(catalog))]
            res=minimize(lambda w:model.stats(w)[0],initials[run],jac=lambda w:model.stats(w)[1],method='SLSQP',bounds=bounds,
                constraints={'type':'eq','fun':lambda w:w.sum()-1,'jac':lambda w:np.ones(len(catalog))},
                options={'maxiter':cfg['maximum_optimizer_iterations'],'ftol':cfg['optimizer_ftol']})
            weights=np.maximum(res.x,0)
            if omitted is not None:weights[omitted]=0
            weights/=weights.sum();loss,grad,u,cdf=model.stats(weights)
            record['runs'].append({'run':run,'bandwidth':model.bandwidth,'weights':weights.tolist(),'loss':loss,'iterations':int(res.nit),
                'solver_success':bool(res.success),'solver_message':str(res.message),'smoothed_max_cdf_gap':float(abs(cdf-observed).max()),
                'smoothed_mean_grid_gap':float(abs(cdf[len(original):]-observed[len(original):]).mean())})
            save(dest,record)
        record['selected']=choose_diverse(record['runs'],keep,cfg['model_l1_separation']);save(dest,record)
        for j,row in enumerate(record['selected']):all_selected.append({'name':kind+'__'+name+'__'+str(j),'catalog':kind,'constraint':name,'weights':row['weights'],'training_record':str(dest.relative_to(path.parent.parent)),'training_run':row['run']})
        print('FIT',kind,name,'selected',len(record['selected']),'best loss',round(min(r['loss']for r in record['runs']),6),flush=True)
    return all_selected


def validate_models(models,catalogs,out,cfg,native,public,thresholds):
    results=[]
    for j,row in enumerate(models):
        path=out/'validation'/(row['name']+'.npz');p=catalogs[row['catalog']]
        own,family=sample_actions(p,cfg['validation_own_per_family'],cfg['seed']+50000+CATALOG_VARIANTS.index(row['catalog']))
        n=cfg['validation_opponent_groups'];seed=cfg['seed']+60000+j
        if not path.exists():
            rows=draw_population(p,row['weights'],3*n,seed).reshape(n,3,5)
            u=native.evaluate(own,native.compress_groups(rows));path.parent.mkdir(exist_ok=True)
            np.savez_compressed(path,opponent_triplets=rows,values=u,own=own,families=family,size=n,seed=seed)
        with np.load(path)as d:u=d['values']
        mass=np.asarray(row['weights'])[family]/cfg['validation_own_per_family']
        metrics=cdf_diagnostic(u,mass,public)
        coarse=np.array([mass[u<=t].sum()for t in thresholds]);observed=np.array([(public<=t).mean()for t in thresholds]);gap=abs(coarse-observed)
        metrics['original_threshold_max_gap']=float(gap.max());metrics['original_threshold_mean_gap']=float(gap.mean())
        metrics['passes']=bool(metrics['maximum_cdf_gap']<=cfg['full_curve_maximum_tolerance']and metrics['mean_grid_cdf_gap']<=cfg['mean_grid_tolerance']and metrics['original_threshold_mean_gap']<=cfg['original_threshold_mean_tolerance']and metrics['mass_above_public_max_plus_point_two']<=cfg['upper_tail_mass_tolerance'])
        results.append({**row,'validation':metrics,'validation_seed':seed})
        print('VALIDATE',j+1,'/',len(models),row['name'],'gap',round(metrics['maximum_cdf_gap'],4),'pass',metrics['passes'],flush=True)
    save(out/'model-validation.json',{'models':results,'retained_count':sum(r['validation']['passes']for r in results),
        'scope':'Finite necessary score-CDF checks, not identification, posterior weights, or proof omitted families are absent.'})
    return [r for r in results if r['validation']['passes']]


def evaluate(models,catalogs,out,cfg,native,actions):
    splits=split_actions();values=[];split_values=[];moments=[]
    n=cfg['split_groups'];prefix=cfg['candidate_groups']
    for j,row in enumerate(models):
        path=out/'evaluation'/(row['name']+'.npz');seed=cfg['seed']+100000+j
        if not path.exists():
            rows=draw_population(catalogs[row['catalog']],row['weights'],3*n,seed).reshape(n,3,5)
            sums=exact_sample_sums(native,actions,native.compress_groups(rows[:prefix]),prefix)
            g=native.compress_groups(rows);q=native.payoff_matrix(splits,g);counts=counts_from_groups(g,n)
            split_sums,cross=paired_moments(q,counts)
            path.parent.mkdir(exist_ok=True)
            np.savez_compressed(path,triplets=rows,candidate_rank_sum4=sums,split_rank_sum4=split_sums,split_cross_sum16=cross,
                size=n,candidate_size=prefix,seed=seed)
        with np.load(path)as d:
            values.append(d['candidate_rank_sum4']/(4*prefix));split_values.append(d['split_rank_sum4']/(4*n));moments.append(d['split_cross_sum16'].copy())
        print('EVALUATE',j+1,'/',len(models),row['name'],flush=True)
    if not values:raise ValueError('No retained fields; do not fabricate a selection')
    values=np.array(values).T;split_values=np.array(split_values).T;moments=np.array(moments)
    names=[r['name']for r in models]
    np.savez_compressed(out/'candidate-scores.npz',values=values,allocations=actions,names=names)
    np.savez_compressed(out/'split-scores.npz',values=split_values,allocations=splits,names=names,cross_sum16=moments)
    selected=min(range(len(actions)),key=lambda i:(-values[i].min(),tuple(actions[i])))
    selected_split=int(split_values.min(axis=1).argmax())
    # Family-wise sampling bounds over all accepted fixed models, all split means and pairs.
    m=len(models)*101;mean_variance=(moments[:,np.arange(101),np.arange(101)]/16-n*split_values.T**2)/(n-1)
    mean_error=empirical_bernstein_error(mean_variance,n,m+len(models)*5050,.05,9)
    pair_error=[]
    for j in range(len(models)):
        mean=split_values[:,j];s2=(np.diag(moments[j])[:,None]+np.diag(moments[j])[None,:]-2*moments[j].astype(float))/16
        variance=np.maximum((s2-n*(mean[:,None]-mean[None,:])**2)/(n-1),0)
        pair_error.append(empirical_bernstein_error(variance,n,m+len(models)*5050,.05,18))
    pair_error=np.array(pair_error)
    np.savez_compressed(out/'split-confidence.npz',mean_error=mean_error,pair_error=pair_error)
    lower=(split_values.T-mean_error).min(axis=0);upper=(split_values.T+mean_error).min(axis=0)
    plausible=np.flatnonzero(upper>=lower.max()).tolist()
    # Model count is not used as a probability over catalog variants.
    result={'candidate_count':len(actions),'retained_field_count':len(models),'selected_allocation':actions[selected].tolist(),
        'selected_index':int(selected),'selected_minimum_mean':float(values[selected].min()),
        'selected_split':splits[selected_split].tolist(),'selected_split_C2':selected_split,'selected_split_minimum_mean':float(split_values[selected_split].min()),
        'sampling_plausible_split_C2_values':plausible,'sampling_confidence_scope':'Joint 95% bounds for fixed retained laws and all 101 split means and unordered pairs, using empirical Bernstein and a union bound. Excludes model error and selection beyond the frozen candidate pool.',
        'comparison_source':'https://www.cs.mcgill.ca/~colt2009/papers/012.pdf, Theorem 4',
        'rows':[{'C2':b,'C4':100-b,'minimum':float(split_values[b].min()),'maximum':float(split_values[b].max()),
            'worst_field':names[int(split_values[b].argmin())],'robust_lower':float(lower[b]),'robust_upper':float(upper[b]),
            'fields_with_higher_mean_than_36':int(np.sum(split_values[b]>split_values[36]))}for b in range(101)]}
    save(out/'selection.json',result)
    write_csv(out/'all-101-splits.csv',['C2','C4','minimum','maximum','worst_field','robust_lower','robust_upper','fields_with_higher_mean_than_36'],[[r[k]for k in ['C2','C4','minimum','maximum','worst_field','robust_lower','robust_upper','fields_with_higher_mean_than_36']]for r in result['rows']])
    return result


def stresses(catalogs,fields,out,cfg,native,actions):
    dest=out/'family-stress';dest.mkdir(exist_ok=True);unique={};mapping={}
    for kind,p in catalogs.items():
        mapping[kind]={}
        for name,(x,w)in p.items():
            key=fingerprint(x,w);mapping[kind][name]=key
            if key not in unique:unique[key]=(x,w)
    save(dest/'family-map.json',mapping);pure={}
    for j,(key,(x,w))in enumerate(unique.items()):
        path=dest/(key+'.npz');n=cfg['pure_family_groups'];seed=cfg['seed']+200000+j
        if not path.exists():
            rng=np.random.default_rng(seed);rows=x[rng.choice(len(x),(n,3),p=w/w.sum())]
            sums=exact_sample_sums(native,actions,native.compress_groups(rows),n)
            np.savez_compressed(path,triplets=rows,rank_sum4=sums,size=n,seed=seed)
        with np.load(path)as d:pure[key]=d['rank_sum4']/(4*n)
        if j%40==0:print('PURE FAMILY',j+1,'/',len(unique),flush=True)
    # Every split plus fixed outside-family comparison entries; full pool is evaluated in pure laws and calibrated laws.
    split_ids=[next(i for i,x in enumerate(actions.tolist())if x==y)for y in split_actions().tolist()]
    extra=[[15,18,49,12,6],[20,24,44,5,7],[0,0,0,0,100],[0,0,39,61,0],[0,3,35,57,5],[14,18,27,16,25]]
    shock_ids=split_ids+[next(i for i,x in enumerate(actions.tolist())if x==y)for y in extra if y not in split_actions().tolist()]
    shock_actions=actions[shock_ids];coeff=np.empty((len(fields),40,len(shock_ids),4));p=catalogs['original'];n=cfg['family_shock_conditional_groups']
    for j,field in enumerate(fields):
        base=draw_population(p,field['weights'],3*n,cfg['seed']+300000+j).reshape(n,3,5)
        base_value=exact_sample_sums(native,shock_actions,native.compress_groups(base),n)/(4*n)
        for f,(name,(x,w))in enumerate(p.items()):
            coeff[j,f,:,0]=base_value;coeff[j,f,:,3]=pure[mapping['original'][name]][shock_ids]
            for k in [1,2]:
                path=dest/(field['name']+'__'+name+'__k'+str(k)+'.npz');seed=cfg['seed']+400000+j*1000+f*10+k
                if not path.exists():
                    rng=np.random.default_rng(seed);rows=base.copy();rows[:,:k]=x[rng.choice(len(x),(n,k),p=w/w.sum())]
                    sums=exact_sample_sums(native,shock_actions,native.compress_groups(rows),n)
                    np.savez_compressed(path,triplets=rows,rank_sum4=sums,size=n,seed=seed)
                with np.load(path)as d:coeff[j,f,:,k]=d['rank_sum4']/(4*n)
        print('FAMILY SHOCKS',j+1,'/17',flush=True)
    np.savez_compressed(dest/'scores.npz',pure_values=np.column_stack(list(pure.values())),pure_keys=list(pure),allocations=actions,
        shock_coefficients=coeff,shock_action_indices=shock_ids)
    rows=[]
    for share in cfg['family_shock_shares']+[1.]:
        floor=cubic_minimum(coeff,share).min(axis=(0,1));b=int(floor[:101].argmax())
        rows.append({'maximum_replacement_share':share,'best_split_C2':b,'best_split_C4':100-b,'best_split_floor':float(floor[b]),'C2_36_floor':float(floor[36]),'C2_38_floor':float(floor[38]),'all_action_floors':floor.tolist()})
    summary={'unique_pure_family_laws':len(unique),'catalog_family_slots':len(catalogs)*40,'base_fields':17,'replacement_families':40,
        'conditional_K1_K2_banks':17*40*2,'shock_actions':len(shock_ids),'continuous_share_point_estimate_summary':rows,
        'scope':'Deliberate family stresses. Cubic minima cover every share in each interval for the estimated conditional coefficients. Sampling/model error remains; these are not calibrated applicant forecasts.'}
    save(dest/'results.json',summary)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=ROOT/'results/game2-catalog-stress')
    parser.add_argument('--phase',choices=['all','fits','evaluation','stresses'],default='all');args=parser.parse_args()
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True);cfg=read(ROOT/'config/game2-catalog-stress.json')
    vendor=ROOT/'data/parry-replay';fit,original,fields=load_catalog(vendor);native=native_module(vendor)
    public=np.array(read(vendor/'public-scores.json')['scores']);old_freeze=ROOT/'results/game2-cross-arena/freeze.json';pool=read(old_freeze)['pool'];actions=np.array([r['allocation']for r in pool],np.uint8)
    for x in split_actions().tolist():assert x in actions.tolist()
    sources=[Path(__file__),ROOT/'game2_lab/catalog_stress.py',ROOT/'config/game2-catalog-stress.json',ROOT/'tests/test_game2_catalog_stress.py']
    inputs=[old_freeze,vendor/'results/field_fit_validated.json',vendor/'results/fit_catalog.npz',vendor/'results/fit_catalog_weights.npz',vendor/'public-scores.json',ROOT/'results/game2-working-candidate-v2.json']
    hashes={str(p.relative_to(ROOT)):sha(p)for p in sources+inputs}
    definition={'configuration':cfg,'variants':CATALOG_VARIANTS,'pool':pool,'source_and_input_sha256':hashes,'versions':{'python':platform.python_version(),'numpy':np.__version__},'scope':cfg['scope']}
    path=out/'freeze.json'
    if path.exists():assert read(path)==definition,'Changed study inputs or sources; use a new output folder.'
    else:
        save(path,definition)
        for p in sources:
            q=out/'source-snapshot'/p.relative_to(ROOT);q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
    catalogs={}
    for j,kind in enumerate(CATALOG_VARIANTS):
        path=out/'catalogs'/kind
        if not(path/'rows.npz').exists():store_catalog(path,make_catalog(original,kind,cfg['variant_family_rows'],cfg['seed']+j*100))
        catalogs[kind]=get_catalog(path,list(original))
    if args.phase in ['all','fits']:
        selected=[]
        for j,kind in enumerate(CATALOG_VARIANTS):selected.extend(fit_catalog(kind,j,catalogs[kind],out/'training'/kind,native,cfg,fields,public,fit['thresholds']))
        selected += [{'name':'supplied_'+r['name'],'catalog':'original','constraint':'original_supplied','weights':r['weights']}for r in fields]
        save(out/'model-freeze.json',{'models':selected,'candidate_freeze_sha256':sha(out/'freeze.json'),'scope':'Models selected by training fit only, before fresh CDF validation or candidate evaluation.'})
    if args.phase in ['all','evaluation']:
        selected=read(out/'model-freeze.json')['models'];models=validate_models(selected,catalogs,out,cfg,native,public,fit['thresholds'])
        result=evaluate(models,catalogs,out,cfg,native,actions);print('SELECTION',result['selected_allocation'],result['selected_split'],flush=True)
    if args.phase in ['all','stresses']:print('STRESSES',stresses(catalogs,fields,out,cfg,native,actions)['unique_pure_family_laws'],flush=True)
    for rel,digest in hashes.items():assert sha(ROOT/rel)==digest,rel
    print('Phase complete',args.phase,flush=True)


if __name__=='__main__':main()
