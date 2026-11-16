"""Population-weight cubics, continuous family bounds, and failure diagnostics."""
from collections import Counter
from fractions import Fraction
from pathlib import Path
import argparse,json,time,platform,shutil
import numpy as np
from blotto_lab.core import Population
from game2_lab.core import fixed_sample_bound,focal_samples
from game2_lab.arena import random_actions,single_triplet_response,class_name
from game2_lab.human import payoff_arrays
from game2_lab.weights import (A,FAMILY_IDS,without_copies,pooled_population,conditional_triplets,family_tensor,
 copy_coefficients,cubic_value,continuous_minimum,robust_partition,representation_path)
from run_multiwave_study import ROOT
from run_population_study import save,sha,write_csv


def read(p):return json.loads(p.read_text())


def definitions(old,cfg):
    refs={}
    for group in ['official_rank','mixed_objectives']:
        for rule in ['replicator','payoff_replacement']:
            for snapshot in ['final','average']:
                paths=[p for p in old['paths'].values()if p['group']==group and p['rule']==rule]
                pooled=pooled_population([Population.from_dict(p[snapshot]['population'])for p in paths]);pop,mass=without_copies(pooled)
                key='__'.join([group,rule,snapshot]);refs[key]={'kind':'population','population':pop.to_dict(),'original_copy_mass':mass,'component_paths':[p['name']for p in paths],'snapshot':snapshot}
    for family in cfg['static_families']:refs['static__'+family]={'kind':'static','family':family,'variant_probabilities':[.5,.5],'conditioning':'Not exactly all-in five'}
    cells={f'f{i:02d}':{'kind':'family','ids':ids}for i,ids in enumerate(FAMILY_IDS)};indices={}
    for key,ref in refs.items():
        keys=[]
        for k in range(3):
            name=key+'__copies'+str(k);keys.append(name);cells[name]={'kind':'reference','reference':ref,'copies':k}
        indices[key]=keys+['f00']
    for j,(key,cell)in enumerate(cells.items()):cell['development_seed']=cfg['seed']+1000+j;cell['audit_seed']=cfg['seed']+10000+j
    qgrid=[[i,j,cfg['residual_grid_denominator']-i-j]for i in range(cfg['residual_grid_denominator']+1)for j in range(cfg['residual_grid_denominator']+1-i)]
    return {'references':refs,'cells':cells,'reference_indices':indices,'family_keys':[f'f{i:02d}'for i in range(20)],'qgrid_counts':qgrid,'qgrid_denominator':cfg['residual_grid_denominator']}


def candidate_field(old,cfg):
    actions=[tuple(v['allocation'])for v in old['candidates'].values()];anchors=list(dict.fromkeys(tuple(v['allocation'])for v in old['primary'].values()))
    for x in anchors:
        actions.append(single_triplet_response([x]*3)['allocation'])
        for donor in range(5):
            for recipient in range(5):
                if donor==recipient:continue
                for amount in cfg['local_transfer_amounts']:
                    if x[donor]>=amount:
                        y=list(x);y[donor]-=amount;y[recipient]+=amount;actions.append(tuple(y))
    actions+=random_actions(np.random.default_rng(cfg['seed']),cfg['new_random_candidates'])
    return list(dict.fromkeys(actions))


def criteria(values,d,cfg):
    keys=list(d['cells']);tensor=family_tensor(values[:,[keys.index(k)for k in d['family_keys']]])
    minima={str(cap):np.full(len(values),np.inf)for cap in cfg['copy_caps']}
    for counts in d['qgrid_counts']:
        coeff=copy_coefficients(tensor,np.array(counts)/d['qgrid_denominator'])
        for cap in cfg['copy_caps']:minima[str(cap)]=np.minimum(minima[str(cap)],continuous_minimum(coeff,cap)[0])
    reference=np.full(len(values),np.inf)
    for ids in d['reference_indices'].values():reference=np.minimum(reference,continuous_minimum(values[:,[keys.index(k)for k in ids]],.25)[0])
    rules={f'family_cap_{cap}':minima[str(cap)]for cap in cfg['copy_caps']}
    rules['references_cap_0.25']=reference;rules['combined_cap_0.25']=np.minimum(reference,minima['0.25'])
    return rules


def primary_failure_sums(actions,rows):
    maximum=rows.max(axis=1);ties=(rows==maximum[:,None,:]).sum(axis=1);records=[]
    for x in actions:
        greater=np.array(x)>maximum;equal=np.array(x)==maximum;credit=greater*12+equal*(12//(ties+1))
        samples=focal_samples(x,rows)
        records.append({'castle_win':greater.sum(axis=0).tolist(),'castle_tie':equal.sum(axis=0).tolist(),'castle_credit12':credit.sum(axis=0).tolist(),
            'placement_credit12':samples['finish_credits12'].sum(axis=0).tolist(),'outright_first':int(samples['outright_first'].sum()),'rank_tie':int(samples['rank_tie'].sum())})
    return records


def figures(out,r):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    selected=['human_100','previous_low_five','rank_tail_mean','rank_tail_minimum','combined_cap_0.25','family_cap_0.5']
    selected=[k for k in selected if k in r['primary']];seen=set();selected=[k for k in selected if r['primary'][k]['candidate_id']not in seen and not seen.add(r['primary'][k]['candidate_id'])]
    refs=list(r['definitions']['references']);chosen=[k for k in refs if k.startswith('official_rank')and k.endswith('final')]+['static__counter_cloud','static__middle_five']
    fig,axes=plt.subplots(2,2,figsize=(12,8));p=np.linspace(0,1,201)
    for ax,key in zip(axes.ravel(),chosen):
        for label in selected:
            cid=r['primary'][label]['candidate_id'];ax.plot(p,cubic_value(np.array(r['reference_coefficients'][cid][key]),p),label=str(tuple(r['candidates'][cid]['allocation'])),lw=1.4)
        ax.axhline(3.75,color='gray',ls=':',lw=1);ax.set_title(key.replace('official_rank__','').replace('static__','').replace('__',' / '));ax.set_ylim(0,9);ax.set_xlabel('Exact all-in-five opponent share');ax.set_ylabel('Expected ranking payout');ax.grid(alpha=.2)
    handles,labels=axes.ravel()[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=3,fontsize=8);fig.suptitle('Copy prevalence and residual population jointly change strategy values');fig.tight_layout(rect=[0,.09,1,.95]);fig.savefig(out/'copy-weight-curves.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,5));labels=[];a=[];b=[]
    for key,pth in r['representation'].items():
        labels.append(key.replace('__',' / '));a.append(pth['final_class_shares']['all_in_five']*100);b.append(pth['final_class_shares']['low_five']*100)
    pos=np.arange(len(labels));axes[0].barh(pos,a);axes[1].barh(pos,b,color='darkorange')
    for ax,title in zip(axes,['Exact all-in five','Fifth bid 0-3']):ax.set_yticks(pos,labels,fontsize=8);ax.set_xlabel('Final population share (%)');ax.set_title(title);ax.grid(axis='x',alpha=.2)
    fig.suptitle('Identical-label duplication changes flat replacement, not family-balanced replacement');fig.tight_layout();fig.savefig(out/'representation-control.png',dpi=170);plt.close(fig)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,default=ROOT/'config/game2-weights.json');parser.add_argument('--output',type=Path,default=ROOT/'results/game2-weights');parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    cfg=read(args.config);out=args.output.resolve();clock=time.perf_counter()
    sources=[Path(__file__),ROOT/'verify_game2_weights.py',ROOT/'game2_lab/weights.py',ROOT/'game2_lab/arena.py',ROOT/'game2_lab/human.py',ROOT/'game2_lab/core.py',ROOT/'blotto_lab/core.py',ROOT/'blotto_lab/merged.py',ROOT/'blotto_lab/multiwave.py',ROOT/'run_population_study.py',ROOT/'GAME2-WEIGHTS-PROTOCOL.md',ROOT/'tests/test_game2_weights.py']
    inputs=[ROOT/'results/game2-arena/results.json',ROOT/'results/game2-arena/seed-pool.json',ROOT/'results/game2-arena/counter-cloud.json',ROOT/'results/game2-arena-checkpoint-v1.json']
    signature={'configuration':cfg,'configuration_sha256':sha(args.config),'source_hashes':{str(p.relative_to(ROOT)):sha(p)for p in sources},'input_hashes':{str(p.relative_to(ROOT)):sha(p)for p in inputs},'environment':{'python':platform.python_version(),'numpy':np.__version__,'platform':platform.platform()}}
    if out.exists():
        if not args.resume:raise SystemExit('Output exists. Use --resume or another folder.')
        if read(out/'run-inputs.json')!=signature:raise SystemExit('Resume input/source mismatch')
        if (out/'manifest.json').exists():print('Completed. Run verification.');return
    else:
        if args.resume:raise SystemExit('No run to resume')
        out.mkdir(parents=True)
        for folder in ['development','audit','bounds','representation','source-snapshot']:(out/folder).mkdir()
        save(out/'run-inputs.json',signature)
        for p in sources:
            target=out/'source-snapshot'/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
    old=read(inputs[0]);cloud=read(inputs[2]);d=definitions(old,cfg);actions=candidate_field(old,cfg);candidates={f'w{i:04d}':{'allocation':x,'class':class_name(x)}for i,x in enumerate(actions)};cids=list(candidates);lookup={x:cid for x,cid in zip(actions,cids)}
    if not(out/'design.json').exists():save(out/'design.json',{'created_unix':time.time(),'definitions':d,'candidates':candidates})
    else:
        from blotto_lab.multiwave import simple_json
        assert read(out/'design.json')['definitions']==simple_json(d)
    development=[]
    for key,cell in d['cells'].items():
        filename=out/'development'/(key+'.npz')
        if filename.exists():
            with np.load(filename)as a:sums=a['rank_sum4']
        else:
            rows=conditional_triplets(cell,cfg['development_triplets'],cell['development_seed'],cloud);rank,_=payoff_arrays(actions,rows);sums=rank.sum(axis=1,dtype=np.int64);np.savez_compressed(filename,triplets=rows.astype(np.uint8),rank_sum4=sums)
        development.append(sums);print('Development conditional cell: '+key,flush=True)
    values=np.column_stack(development);objective=criteria(values/(4*cfg['development_triplets']),d,cfg)
    if not(out/'candidate-freeze.json').exists():
        primary={name:{'candidate_id':lookup[tuple(p['allocation'])],'allocation':p['allocation'],'origin':'previous frozen arena comparison'}for name,p in old['primary'].items()}
        for name,scores in objective.items():
            j=min(range(len(actions)),key=lambda i:(-round(float(scores[i]),11),actions[i]));primary[name]={'candidate_id':cids[j],'allocation':actions[j],'origin':'finite development search','development_objective':float(scores[j])}
        keep={p['candidate_id']for p in primary.values()};keep.update(lookup[tuple(old['candidates'][cid]['allocation'])]for cid in old['shortlist'])
        for name,scores in objective.items():
            if name.startswith('family_cap_'):
                keep.update(cids[j]for j in sorted(range(len(actions)),key=lambda i:(-round(float(scores[i]),11),actions[i]))[:cfg['shortlist_per_cap']])
        shortlist=[cid for cid in cids if cid in keep]
        np.savez_compressed(out/'development-scores.npz',rank_sum4=values)
        save(out/'candidate-freeze.json',{'created_unix':time.time(),'primary':primary,'shortlist':shortlist,'candidates':candidates,'design_sha256':sha(out/'design.json'),'development_scores_sha256':sha(out/'development-scores.npz'),'run_inputs_sha256':sha(out/'run-inputs.json')})
    freeze=read(out/'candidate-freeze.json');primary=freeze['primary'];shortlist=freeze['shortlist'];primary_ids=sorted(set(p['candidate_id']for p in primary.values()));freeze_hash=sha(out/'candidate-freeze.json')
    if not(out/'audit-start.json').exists():save(out/'audit-start.json',{'started_unix':time.time(),'freeze_sha256':freeze_hash})
    audit=[];failures={cid:{}for cid in primary_ids}
    for key,cell in d['cells'].items():
        filename=out/'audit'/(key+'.npz');failurefile=out/'audit'/(key+'-failures.json')
        if filename.exists()and failurefile.exists():
            with np.load(filename)as a:sums=a['rank_sum4'];castle=a['castle_sum12']
            direct=read(failurefile)
        else:
            rows=conditional_triplets(cell,cfg['audit_triplets'],cell['audit_seed'],cloud);rank,c=payoff_arrays([candidates[cid]['allocation']for cid in shortlist],rows)
            sums=rank.sum(axis=1,dtype=np.int64);castle=c.sum(axis=1,dtype=np.int64);np.savez_compressed(filename,triplets=rows.astype(np.uint8),rank4=rank,rank_sum4=sums,castle_sum12=castle)
            direct=primary_failure_sums([candidates[cid]['allocation']for cid in primary_ids],rows);save(failurefile,dict(zip(primary_ids,direct)));direct=dict(zip(primary_ids,direct))
        for cid in primary_ids:failures[cid][key]=direct[cid]
        audit.append(sums);print('Frozen precision cell: '+key,flush=True)
    means=np.column_stack(audit)/(4*cfg['audit_triplets']);keys=list(d['cells']);tensor=family_tensor(means[:,[keys.index(k)for k in d['family_keys']]])
    reference_coefficients={cid:{ref:means[j,[keys.index(k)for k in ids]].tolist()for ref,ids in d['reference_indices'].items()}for j,cid in enumerate(shortlist)}
    reference_worst={cid:{}for cid in shortlist}
    for cid in shortlist:
        for cap in cfg['copy_caps']:
            rows=[]
            for ref,coeff in reference_coefficients[cid].items():
                score,p=continuous_minimum(coeff,cap);rows.append((float(score),ref,float(p)))
            score,ref,p=min(rows);reference_worst[cid][str(cap)]={'score':score,'reference':ref,'copy_share':p}
    bounds={}
    for cid in primary_ids:
        bounds[cid]={};j=shortlist.index(cid)
        for cap in cfg['copy_caps']:
            filename=out/'bounds'/(cid+'__'+str(cap)+'.json')
            if not filename.exists():save(filename,robust_partition(tensor[j],cap,cfg['optimization_tolerance'],cfg['maximum_region_splits']))
            rec=read(filename);bounds[cid][str(cap)]={k:v for k,v in rec.items()if k!='nodes'};print('Continuous family bound: '+cid+' / '+str(cap),flush=True)
    seed_actions=read(inputs[1]);representation={};classes=['all_in_five','near_all_in_five','low_five','intermediate_five']
    for balanced in [False,True]:
        for duplicated in ['none','all_in_five','low_five','intermediate_five']:
            name=('balanced'if balanced else'flat')+'__'+duplicated;filename=out/'representation'/(name+'.npz');m=[cfg['representation_multiplier']if class_name(x)==duplicated else 1 for x in seed_actions]
            if filename.exists():
                with np.load(filename)as a:path={key:a[key]for key in a.files}
            else:path=representation_path(seed_actions,m,balanced,cfg,cfg['seed']+50000);np.savez_compressed(filename,**path)
            representation[name]={'balanced':balanced,'duplicated':duplicated,'multiplicity':m,'seed':cfg['seed']+50000,'file':str(filename.relative_to(out)),
                'final_class_shares':{k:float(sum(path['final'][i]for i,x in enumerate(seed_actions)if class_name(x)==k))for k in classes}}
            print('Representation control: '+name,flush=True)
    bound=fixed_sample_bound(cfg['audit_triplets'],len(shortlist)*len(d['cells']),cfg['sampling_alpha'])
    result={'configuration':cfg,'definitions':d,'candidates':candidates,'primary':primary,'shortlist':shortlist,'primary_ids':primary_ids,'reference_coefficients':reference_coefficients,
        'reference_worst':reference_worst,'family_bounds':bounds,'sampling_bound':bound,'failure_sums':failures,'representation':representation,'freeze_sha256':freeze_hash,
        'scope':'Known-model development weight and failure review; no final Game 2 selection or applicant forecast'}
    save(out/'results.json',result);figures(out,result)
    write_csv(out/'primary-robustness.csv',['rule','allocation','copy_cap','family_empirical_lower','family_empirical_upper','reference_empirical_minimum','reference','reference_copy_share'],
        [[name,str(tuple(p['allocation'])),cap,bounds[p['candidate_id']][str(cap)]['lower'],bounds[p['candidate_id']][str(cap)]['upper'],reference_worst[p['candidate_id']][str(cap)]['score'],reference_worst[p['candidate_id']][str(cap)]['reference'],reference_worst[p['candidate_id']][str(cap)]['copy_share']]for name,p in primary.items()for cap in cfg['copy_caps']])
    write_csv(out/'copy-curves.csv',['rule','allocation','reference','copy_share','payout'],[[name,str(tuple(v['allocation'])),ref,p/100,float(cubic_value(coeff,p/100))]for name,v in primary.items()for ref,coeff in reference_coefficients[v['candidate_id']].items()for p in range(101)])
    for source in sources:assert sha(source)==signature['source_hashes'][str(source.relative_to(ROOT))]
    save(out/'manifest.json',{'signature':signature,'freeze_sha256':freeze_hash,'completed_unix':time.time(),'elapsed_seconds':time.perf_counter()-clock,
        'artifacts':{str(p.relative_to(out)):sha(p)for p in sorted(out.rglob('*'))if p.is_file()and p.name!='manifest.json'}})
    print('Weight and failure review complete. Run independent verification.',flush=True)


if __name__=='__main__':main()
