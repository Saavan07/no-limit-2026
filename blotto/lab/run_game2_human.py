"""Human Game 2 hypotheses, exact family simplex, then frozen fresh triplets."""
from fractions import Fraction
from itertools import product,combinations_with_replacement
from pathlib import Path
import argparse,json,shutil,time
import numpy as np
from blotto_lab.core import Game,Population,iter_allocations
from game2_lab.core import group_scores12_4,reference_group,fixed_sample_bound
from game2_lab.human import (LITERALS,ANCHORS,anchor_support,triples_for_support,triple_weights,
    counts_for_mix,candidate_field,payoff_arrays,fresh_mixes,fresh_triplets)
from run_multiwave_study import ROOT
from run_population_study import save,sha,write_csv


def read(p):return json.loads(p.read_text())


def classification(x):
    if x[4]==100:return 'all_in'
    if x[4]>=80:return 'near_all_in'
    if x[4]<=3:return 'abandon_or_limp'
    return 'intermediate_five'


def clone_cases():
    a=LITERALS['human_100'];coarse=[tuple(25*v for v in x)for x in iter_allocations(Game(4,(1,2,3,4,5)))if tuple(25*v for v in x)!=a]
    result={}
    for k in [1,2,3]:
        enemy=list(combinations_with_replacement(coarse,3-k))
        groups=[[a]+[a]*k+list(xs)for xs in enemy];c,p=group_scores12_4(np.array(groups))
        cases={}
        for i,value in enumerate(p[:,0]):
            value=int(value)
            if value not in cases:cases[value]={'group':groups[i],'outcome':reference_group(groups[i])}
        result[k]={'coarse_groups_checked':len(groups),'outcomes':cases}
    result['unique_minimum']={'group':[a,(0,0,50,50,0),(0,100,0,0,0),(100,0,0,0,0)],'outcome':reference_group([a,(0,0,50,50,0),(0,100,0,0,0),(100,0,0,0,0)])}
    result['profitable_deviation']={'group':[(1,1,1,1,96),a,a,a],'outcome':reference_group([(1,1,1,1,96),a,a,a])}
    result['broad_counterexample']={'group':[(10,20,40,20,10),a,a,(2,5,35,58,0)],'outcome':reference_group([(10,20,40,20,10),a,a,(2,5,35,58,0)])}
    return result


def conditional_coefficients(ids,pay,support):
    specs={'abandon_only':[1,0,0],'broad_only':[0,1,0],'near_only':[0,0,1],
           'abandon_broad_equal':[Fraction(1,2),Fraction(1,2),0],
           'three_noncopy_equal':[Fraction(1,3)]*3}
    output={};copy_counts=(ids==0).sum(axis=1)
    for name,mix in specs.items():
        noncopy=[]
        for share,rows in zip(mix,list(ANCHORS.values())[1:]):noncopy.extend([Fraction(share)/len(rows)]*len(rows))
        # Give A and the complete non-copy reference equal mass; conditioning removes that convention.
        from math import lcm
        scale=lcm(*(q.denominator for q in noncopy));counts=[scale]+[int(q*scale)for q in noncopy]
        weights=triple_weights(ids,counts);a=[]
        for k in range(4):
            chosen=copy_counts==k;denominator=int(weights[chosen].sum());a.append(Fraction(int(pay[chosen].astype(np.int64)@weights[chosen]),4*denominator))
        coefficients=[a[0],3*(a[1]-a[0]),3*(a[0]-2*a[1]+a[2]),-a[0]+3*a[1]-3*a[2]+a[3]]
        assert sum(coefficients)==Fraction(15,4)
        curve=[{'copy_share':Fraction(i,100),'expected_payout':sum(v*Fraction(i,100)**j for j,v in enumerate(coefficients))}for i in range(101)]
        output[name]={'noncopy_family_weights':mix,'conditional_payouts_by_other_copy_count':a,'monomial_coefficients':coefficients,'curve':curve}
    return output


def plot(out,r):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8.5,4.8))
    for name,data in r['copy_curves'].items():ax.plot([float(v['copy_share'])for v in data['curve']],[float(v['expected_payout'])for v in data['curve']],label=name.replace('_',' '))
    ax.axhline(3.75,color='#555555',ls=':',label='Symmetric self-play reference 3.75')
    ax.set_xlabel('Assumed population fraction using exactly (0,0,0,0,100)');ax.set_ylabel('Expected rank payout of all-in five')
    ax.set_title('Exact clone-collision sensitivity\nEvery curve assumes a different declared non-copy population')
    ax.grid(alpha=.2);ax.legend(fontsize=8);fig.tight_layout();fig.savefig(out/'copy-sensitivity.png',dpi=170);plt.close(fig)
    names=list(r['primary']);matrix=[]
    for name in names:
        cid=r['primary'][name]['candidate_id'];matrix.append([float(r['fresh_summary'][cid][k])for k in ['mean','minimum','maximum']])
    fig,ax=plt.subplots(figsize=(9,6));im=ax.imshow(matrix,cmap='YlGnBu',vmin=0,vmax=9,aspect='auto')
    ax.set_yticks(range(len(names)),[n.replace('_',' ')for n in names],fontsize=8);ax.set_xticks(range(3),['Scenario average','Lowest scenario','Highest scenario'])
    for i,row in enumerate(matrix):
        for j,value in enumerate(row):ax.text(j,i,f'{value:.2f}',ha='center',va='center',color='white'if value>5 else 'black')
    ax.set_title('Frozen primary strategies on 36 new hypothesis scenarios\nScenario average is a coverage convention, not an applicant prediction')
    fig.colorbar(im,ax=ax,label='Expected ranking payout');fig.tight_layout();fig.savefig(out/'fresh-primary-comparison.png',dpi=170);plt.close(fig)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,default=ROOT/'config/game2-human.json');parser.add_argument('--output',type=Path,default=ROOT/'results/game2-human')
    args=parser.parse_args();out=args.output.resolve();cfg=read(args.config)
    if out.exists():raise SystemExit('Output exists. Choose a new folder.')
    out.mkdir(parents=True);(out/'evaluation').mkdir();clock=time.perf_counter()
    sources=[Path(__file__),ROOT/'verify_game2_human.py',ROOT/'GAME2-HUMAN-PROTOCOL.md',ROOT/'game2_lab/human.py',ROOT/'game2_lab/core.py',ROOT/'blotto_lab/core.py',ROOT/'tests/test_game2_human.py',ROOT/'run_population_study.py']
    inputs=[ROOT/'results/game1-selected-v1.json',ROOT/'results/game2-foundation-checkpoint-v1.json']
    signature={'configuration':cfg,'configuration_sha256':sha(args.config),'source_hashes':{str(p.relative_to(ROOT)):sha(p)for p in sources},'input_hashes':{str(p.relative_to(ROOT)):sha(p)for p in inputs}}
    save(out/'run-inputs.json',signature)
    for p in sources:
        target=out/'source-snapshot'/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
    field=candidate_field(cfg['seed']);actions=list(field);candidates={f'c{i:04d}':{'allocation':x,'roles':field[x],'class':classification(x)}for i,x in enumerate(actions)}
    cids=list(candidates);lookup={x:cid for cid,x in zip(cids,actions)};support=anchor_support();ids,triplets=triples_for_support(support)
    print(f'Exact anchor evaluation: {len(actions)} candidates, {len(ids)} unordered opponent triples',flush=True)
    p4,c12=payoff_arrays(actions,triplets)
    mixes=[];counts=[]
    for a,b,c in product(range(11),repeat=3):
        d=10-a-b-c
        if d>=0:
            mix=[Fraction(v,10)for v in [a,b,c,d]];mixes.append(mix);counts.append(counts_for_mix(mix,cfg['grid_mass']))
    weights=np.column_stack([triple_weights(ids,v)for v in counts]);scores=p4.astype(np.int64)@weights
    denominator=4*cfg['grid_mass']**3
    center_counts=counts_for_mix([Fraction(1,4)]*4,cfg['center_mass']);cw=triple_weights(ids,center_counts);center=p4.astype(np.int64)@cw
    center_denominator=4*cfg['center_mass']**3;robust=scores.min(axis=1)
    np.savez_compressed(out/'exact-arrays.npz',anchor_type_ids=ids,payout4=p4,castle12=c12,grid_weights=weights,grid_scores=scores,center_weights=cw,center_scores=center)
    save(out/'grid-definitions.json',{'family_order':list(ANCHORS),'anchors':ANCHORS,'support':support,'mixes':mixes,'counts':counts,'denominator':denominator,'center_counts':center_counts,'center_denominator':center_denominator})
    def choose(indices,value):return min(indices,key=lambda i:(-int(value[i]),actions[i]))
    def primary_fit(indices,value,label,origin):
        index=choose(indices,value);x=actions[index];return {'allocation':x,'candidate_id':cids[index],'origin':origin,'fit_objective':label}
    primary={label:{'allocation':x,'candidate_id':lookup[x],'origin':'literal human vector'}for label,x in LITERALS.items()}
    for label,x in [('game1_reference',(3,8,4,44,41)),('balanced_reference',(20,20,20,20,20))]:primary[label]={'allocation':x,'candidate_id':lookup[x],'origin':'fixed comparator'}
    primary['zero_five_center']=primary_fit([i for i,x in enumerate(actions)if x[4]==0],center,'equal-family center','analyst fit')
    primary['limp_five_center']=primary_fit([i for i,x in enumerate(actions)if 1<=x[4]<=3],center,'equal-family center','analyst fit')
    primary['broad_center']=primary_fit([i for i,x in enumerate(actions)if any(v in field[x]for v in ['broad_seeded','anchor_broad'])],center,'equal-family center','analyst fit')
    primary['overall_center']=primary_fit(range(len(actions)),center,'equal-family center','analyst fit')
    primary['worst_grid']=primary_fit(range(len(actions)),robust,'highest minimum over 286 declared mixes','analyst fit')
    kept={p['candidate_id']for p in primary.values()};kept.update(lookup[x]for x in support)
    for group in ['all_in','near_all_in','abandon_or_limp','intermediate_five']:
        indices=[i for i,x in enumerate(actions)if classification(x)==group]
        for value,limit in [(center,cfg['shortlist_center_per_class']),(robust,cfg['shortlist_robust_per_class'])]:
            kept.update(cids[i]for i in sorted(indices,key=lambda i:(-int(value[i]),actions[i]))[:limit])
    shortlist=[cid for cid in cids if cid in kept]
    definitions=[]
    for si,style in enumerate(cfg['styles']):
        for j,(name,mix)in enumerate(fresh_mixes()):definitions.append({'key':style+'__'+name,'style':style,'mix':mix,'size':cfg['evaluation_triplets'],'seed':cfg['seed']+10000+si*100+j})
    freeze={'created_unix':time.time(),'candidates':candidates,'primary':primary,'shortlist':shortlist,'definitions':definitions,
            'run_inputs_sha256':sha(out/'run-inputs.json'),'exact_arrays_sha256':sha(out/'exact-arrays.npz'),'grid_definitions_sha256':sha(out/'grid-definitions.json')}
    save(out/'frozen-candidates.json',freeze);freeze_sha=sha(out/'frozen-candidates.json');save(out/'evaluation-start.json',{'started_unix':time.time(),'freeze_sha256':freeze_sha})
    literal_index=actions.index(LITERALS['human_100']);curves=conditional_coefficients(ids,p4[literal_index],support)
    fresh={};fresh_scores={cid:{}for cid in shortlist};fresh_castle={cid:{}for cid in shortlist}
    for definition in definitions:
        mix=[Fraction(v)for v in definition['mix']];rows,types=fresh_triplets(mix,definition['style'],definition['size'],definition['seed'])
        pay,castle=payoff_arrays([candidates[cid]['allocation']for cid in shortlist],rows)
        key=definition['key'];file=out/'evaluation'/(key+'.npz');np.savez_compressed(file,triplets=rows.astype(np.uint8),family_ids=types.astype(np.uint8),payout4=pay,castle_sum12=castle.sum(axis=1,dtype=np.int64))
        fresh[key]={'definition':definition,'file':str(file.relative_to(out))}
        for i,cid in enumerate(shortlist):fresh_scores[cid][key]=Fraction(int(pay[i].sum(dtype=np.int64)),4*len(rows));fresh_castle[cid][key]=Fraction(int(castle[i].sum(dtype=np.int64)),12*len(rows))
        print('Fresh hypothesis challenge: '+key,flush=True)
    summary={cid:{'mean':sum(v.values())/len(v),'minimum':min(v.values()),'maximum':max(v.values()),'minimum_scenarios':[k for k,s in v.items()if s==min(v.values())]}for cid,v in fresh_scores.items()}
    exact_primary={name:{'center':Fraction(int(center[cids.index(p['candidate_id'])]),center_denominator),'worst_grid':Fraction(int(robust[cids.index(p['candidate_id'])]),denominator)}for name,p in primary.items()}
    result={'scope':'Human Game 2 hypotheses and first frozen population challenge, not final selection or applicant prediction',
       'configuration':cfg,'candidates':candidates,'primary':primary,'shortlist':shortlist,'anchor_families':ANCHORS,'grid_scenarios':len(mixes),
       'exact_primary':exact_primary,'clone_cases':clone_cases(),'copy_curves':curves,'fresh':fresh,'fresh_scores':fresh_scores,'fresh_castle_scores':fresh_castle,
       'fresh_summary':summary,'sampling_bound':fixed_sample_bound(cfg['evaluation_triplets'],len(shortlist)*len(definitions),cfg['sampling_alpha']),
       'freeze_sha256':freeze_sha,'next_stage':'adaptation and robustness arena using full rank payoffs','game1_recommendation':[3,8,4,44,41]}
    save(out/'results.json',result);plot(out,result)
    write_csv(out/'primary-summary.csv',['rule','allocation','anchor_center','anchor_worst_grid','fresh_scenario_mean','fresh_minimum','fresh_maximum'],
       [[name,str(p['allocation']),float(exact_primary[name]['center']),float(exact_primary[name]['worst_grid']),*[float(summary[p['candidate_id']][k])for k in ['mean','minimum','maximum']]]for name,p in primary.items()])
    write_csv(out/'copy-curves.csv',['noncopy_reference','all_in_population_fraction','expected_rank_payout'],
       [[name,float(p['copy_share']),float(p['expected_payout'])]for name,data in curves.items()for p in data['curve']])
    write_csv(out/'fresh-scores.csv',['candidate_id','allocation','scenario','expected_rank_payout','expected_castle_points'],
       [[cid,str(candidates[cid]['allocation']),key,float(value),float(fresh_castle[cid][key])]for cid,values in fresh_scores.items()for key,value in values.items()])
    for p in sources:assert sha(p)==signature['source_hashes'][str(p.relative_to(ROOT))]
    save(out/'manifest.json',{'signature':signature,'freeze_sha256':freeze_sha,'elapsed_seconds':time.perf_counter()-clock,'completed_unix':time.time(),
         'artifacts':{str(p.relative_to(out)):sha(p)for p in sorted(out.rglob('*'))if p.is_file()and p.name!='manifest.json'}})
    print('Game 2 human population study complete. Run independent verification.',flush=True)


if __name__=='__main__':main()
