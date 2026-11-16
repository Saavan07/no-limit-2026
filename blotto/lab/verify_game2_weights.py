"""Independent score checks and exact rational mixture-partition verification."""
from fractions import Fraction
from itertools import combinations_with_replacement,product
from math import lcm
from pathlib import Path
import argparse,json,time
import numpy as np
from blotto_lab.core import validate_allocation
from blotto_lab.multiwave import simple_json
from game2_lab.core import focal_samples,fixed_sample_bound
from game2_lab.human import payoff_arrays
from game2_lab.arena import class_name
from game2_lab.weights import conditional_triplets,FAMILY_IDS,family_tensor,copy_coefficients,cubic_value,continuous_minimum,representation_path
from run_game2_weights import definitions,candidate_field,criteria
from run_population_study import save,sha
from run_multiwave_study import ROOT


def read(p):return json.loads(p.read_text())


def rational_roots(cap):
    cap=Fraction(str(cap));eye=[[Fraction(int(i==j))for i in range(4)]for j in range(4)]
    if cap==0:return [eye[1:]]
    if cap==1:return [eye]
    base=eye[1:];top=[[cap*a+(1-cap)*b for a,b in zip(eye[0],row)]for row in base];v=base+top
    return [[v[i]for i in ids]for ids in [(0,1,2,5),(0,1,4,5),(0,3,4,5)]]


def integer_contract(tensor,v1,v2,v3):
    return sum(int(tensor[i,j,k])*a*b*c for i,a in enumerate(v1)if a for j,b in enumerate(v2)if b for k,c in enumerate(v3)if c)


def exact_value(tensor,w,denominator):
    scale=lcm(*(v.denominator for v in w));v=[int(x*scale)for x in w]
    return Fraction(integer_contract(tensor,v,v,v),denominator*scale**3)


def verify_partition(record,tensor,denominator):
    roots=rational_roots(record['copy_cap']);vertices={};children={};root_index=0;leaf_bounds=[];witness=None
    for index,node in enumerate(record['nodes']):
        assert node['id']==index
        if node['parent']is None:v=roots[root_index];root_index+=1
        else:
            parent=node['parent'];assert 0<=parent<index;source=vertices[parent];edge=record['nodes'][parent]['split'];assert edge is not None
            i,j=edge;assert 0<=i<len(source)and 0<=j<len(source)and i!=j
            ordinal=children.get(parent,0);assert ordinal<2;children[parent]=ordinal+1
            v=[row.copy()for row in source];mid=[(a+b)/2 for a,b in zip(source[i],source[j])];v[i if ordinal==0 else j]=mid
        vertices[index]=v;np.testing.assert_allclose(np.array([[float(x)for x in row]for row in v]),node['vertices'],atol=2e-13,rtol=0)
        points=v+[[sum(row[c]for row in v)/len(v)for c in range(4)]]
        if witness is None:
            for w in points:
                if max(abs(float(a)-b)for a,b in zip(w,record['witness']))<2e-13:witness=w;break
        if node['split']is None:
            scale=lcm(*(x.denominator for row in v for x in row));integer=[[int(x*scale)for x in row]for row in v]
            lower=min(integer_contract(tensor,integer[i],integer[j],integer[k])for i,j,k in combinations_with_replacement(range(len(v)),3))
            lower=Fraction(lower,denominator*scale**3);leaf_bounds.append(lower)
            assert node['lower']<=float(lower)+2e-12
    assert root_index==len(roots)and witness is not None
    for node in record['nodes']:assert children.get(node['id'],0)==(0 if node['split']is None else 2)
    lower=min(leaf_bounds);upper=exact_value(tensor,witness,denominator)
    assert record['lower']<=float(lower)+2e-12 and record['upper']>=float(upper)-2e-12
    assert abs(record['upper']-float(upper))<1e-8 and record['upper']-record['lower']>=float(upper-lower)-2e-12
    assert all(x>=0 for x in witness)and sum(witness)==1 and witness[0]<=Fraction(str(record['copy_cap']))
    return {'lower':lower,'upper':upper,'witness':witness,'gap':upper-lower,'leaves':len(leaf_bounds),'copy_cap':record['copy_cap']}


def independent_failures(x,rows):
    samples=focal_samples(x,rows);maximum=rows.max(axis=1);count=(rows==maximum[:,None,:]).sum(axis=1);gt=np.array(x)>maximum;eq=np.array(x)==maximum
    return {'castle_win':gt.sum(axis=0).tolist(),'castle_tie':eq.sum(axis=0).tolist(),'castle_credit12':(gt*12+eq*(12//(count+1))).sum(axis=0).tolist(),
      'placement_credit12':samples['finish_credits12'].sum(axis=0).tolist(),'outright_first':int(samples['outright_first'].sum()),'rank_tie':int(samples['rank_tie'].sum())},samples


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--study',type=Path,default=ROOT/'results/game2-weights');out=parser.parse_args().study.resolve();clock=time.perf_counter()
    r=read(out/'results.json');cfg=r['configuration'];m=read(out/'manifest.json');signature=read(out/'run-inputs.json');assert signature==m['signature']
    for key,digest in m['artifacts'].items():assert sha(out/key)==digest,key
    for key,digest in signature['source_hashes'].items():assert sha(ROOT/key)==sha(out/'source-snapshot'/key)==digest,key
    for key,digest in signature['input_hashes'].items():assert sha(ROOT/key)==digest,key
    freeze=read(out/'candidate-freeze.json');start=read(out/'audit-start.json');design=read(out/'design.json');freeze_hash=sha(out/'candidate-freeze.json')
    assert freeze_hash==r['freeze_sha256']==m['freeze_sha256']==start['freeze_sha256']and design['created_unix']<freeze['created_unix']<start['started_unix']<m['completed_unix']
    for k in ['primary','shortlist','candidates']:assert freeze[k]==r[k]
    for filename,key in [('design.json','design_sha256'),('development-scores.npz','development_scores_sha256'),('run-inputs.json','run_inputs_sha256')]:assert sha(out/filename)==freeze[key]
    old=read(ROOT/'results/game2-arena/results.json');cloud=read(ROOT/'results/game2-arena/counter-cloud.json');d=definitions(old,cfg);actions=candidate_field(old,cfg)
    assert simple_json(d)==r['definitions']==design['definitions']and actions==[tuple(v['allocation'])for v in r['candidates'].values()]
    for x in actions:validate_allocation(x)
    keys=list(d['cells']);development=[]
    for key,cell in d['cells'].items():
        expected=conditional_triplets(cell,cfg['development_triplets'],cell['development_seed'],cloud)
        with np.load(out/'development'/(key+'.npz'))as a:rows=a['triplets'];sums=a['rank_sum4']
        np.testing.assert_array_equal(rows,expected);calculated=payoff_arrays(actions,rows,candidate_batch=13,triplet_batch=1601)[0].sum(axis=1,dtype=np.int64);np.testing.assert_array_equal(sums,calculated);development.append(sums)
    values=np.column_stack(development)
    with np.load(out/'development-scores.npz')as a:np.testing.assert_array_equal(values,a['rank_sum4'])
    obj=criteria(values/(4*cfg['development_triplets']),d,cfg);cids=list(r['candidates'])
    for name,scores in obj.items():
        best=min(range(len(actions)),key=lambda i:(-round(float(scores[i]),11),actions[i]));assert r['primary'][name]['candidate_id']==cids[best]
    print('Complete finite development field, mixture coefficients, and selection verified.',flush=True)
    matrices=[];direct_cells=0
    for key,cell in d['cells'].items():
        expected=conditional_triplets(cell,cfg['audit_triplets'],cell['audit_seed'],cloud)
        with np.load(out/'audit'/(key+'.npz'))as a:rows=a['triplets'];rank=a['rank4'];sums=a['rank_sum4'];castles=a['castle_sum12']
        np.testing.assert_array_equal(rows,expected);assert np.all(rows.sum(axis=2)==100)
        if cell['kind']=='family':assert np.all((rows[:,:,4]==100).sum(axis=1)==list(cell['ids']).count(0))
        else:assert np.all((rows[:,:,4]==100).sum(axis=1)==cell['copies'])
        calculated,c=payoff_arrays([r['candidates'][cid]['allocation']for cid in r['shortlist']],rows,candidate_batch=11,triplet_batch=7001)
        np.testing.assert_array_equal(calculated,rank);np.testing.assert_array_equal(rank.sum(axis=1,dtype=np.int64),sums);np.testing.assert_array_equal(c.sum(axis=1,dtype=np.int64),castles)
        failurefile=read(out/'audit'/(key+'-failures.json'))
        for cid in r['primary_ids']:
            x=r['candidates'][cid]['allocation'];independent,samples=independent_failures(x,rows);j=r['shortlist'].index(cid)
            assert independent==r['failure_sums'][cid][key]==failurefile[cid]
            np.testing.assert_array_equal(samples['payout4'],rank[j]);assert samples['castle12'].sum()==castles[j]
            assert sum(independent['placement_credit12'])==12*len(rows)
            assert sum((j+1)*v for j,v in enumerate(independent['castle_credit12']))==int(castles[r['shortlist'].index(cid)])
            direct_cells+=len(rows)
        matrices.append(sums);print('Independent primary scoring: '+key,flush=True)
    audit=np.column_stack(matrices);means=audit/(4*cfg['audit_triplets']);family_indices=[keys.index(k)for k in d['family_keys']];tensors=family_tensor(audit[:,family_indices]);certificates={};leaf_count=0
    for j,cid in enumerate(r['shortlist']):
        for ref,ids in d['reference_indices'].items():
            coeff=means[j,[keys.index(k)for k in ids]];np.testing.assert_array_equal(coeff,r['reference_coefficients'][cid][ref])
        for cap in cfg['copy_caps']:
            choices=[]
            for ref,coeff in r['reference_coefficients'][cid].items():
                minimum,p=continuous_minimum(coeff,cap);choices.append((float(minimum),ref,float(p)))
                # A separate polynomial-root calculation audits all reported minima.
                a,b,c,e=coeff;power=[a,3*(b-a),3*(a-2*b+c),-a+3*b-3*c+e];points=[0.,cap]
                derivative=np.trim_zeros(np.array([3*power[3],2*power[2],power[1]]),'f')
                if len(derivative)>1:
                    points.extend(float(z.real)for z in np.roots(derivative)if abs(z.imag)<1e-8 and 0<=z.real<=cap)
                separate=min(float(cubic_value(coeff,v))for v in points);assert abs(separate-float(minimum))<1e-8
            score,ref,p=min(choices);assert r['reference_worst'][cid][str(cap)]=={'score':score,'reference':ref,'copy_share':p}
        if cid not in r['primary_ids']:continue
        certificates[cid]={}
        for cap in cfg['copy_caps']:
            record=read(out/'bounds'/(cid+'__'+str(cap)+'.json'));assert {k:v for k,v in record.items()if k!='nodes'}==r['family_bounds'][cid][str(cap)]
            certificate=verify_partition(record,tensors[j],4*cfg['audit_triplets']);certificates[cid][str(cap)]=certificate;leaf_count+=certificate['leaves']
        print('Exact rational continuous-family certificates: '+cid,flush=True)
    bound=fixed_sample_bound(cfg['audit_triplets'],len(r['shortlist'])*len(d['cells']),cfg['sampling_alpha']);assert bound==r['sampling_bound']
    seed_actions=read(ROOT/'results/game2-arena/seed-pool.json');balanced_traces=[];paths=0
    for name,spec in r['representation'].items():
        path=representation_path(seed_actions,spec['multiplicity'],spec['balanced'],cfg,spec['seed'])
        with np.load(out/spec['file'])as a:
            for key,v in path.items():np.testing.assert_array_equal(a[key],v)
        for group,share in spec['final_class_shares'].items():assert share==float(sum(path['final'][i]for i,x in enumerate(seed_actions)if class_name(x)==group))
        if spec['balanced']:balanced_traces.append(path['trace'])
        paths+=1
    for trace in balanced_traces[1:]:np.testing.assert_array_equal(trace,balanced_traces[0])
    for filename in ['game1-selected-v1.json','game2-foundation-checkpoint-v1.json','game2-human-checkpoint-v1.json','game2-arena-checkpoint-v1.json']:
        for key,digest in read(ROOT/'results'/filename)['source_artifact_sha256'].items():assert sha(ROOT/key)==digest,key
    save(out/'rational-certificates.json',certificates)
    acceptance={'passed':True,'verified_unix':time.time(),'elapsed_seconds':time.perf_counter()-clock,'verifier_sha256':sha(Path(__file__)),
      'finite_candidates':len(actions),'conditional_cells':len(d['cells']),'residual_compositions_screened':len(d['qgrid_counts']),'frozen_primary_rules':len(r['primary']),
      'unique_primary_allocations':len(r['primary_ids']),'frozen_precision_candidates':len(r['shortlist']),'development_triplets_regenerated':len(d['cells'])*cfg['development_triplets'],
      'precision_triplets_regenerated':len(d['cells'])*cfg['audit_triplets'],'primary_candidate_triplet_cells_foundation_checked':direct_cells,
      'continuous_family_certificates':len(r['primary_ids'])*len(cfg['copy_caps']),'partition_leaves_exactly_checked':leaf_count,'rational_certificates_sha256':sha(out/'rational-certificates.json'),
      'representation_paths_replayed':paths,'representation_steps_replayed':paths*cfg['representation_steps'],'balanced_duplicate_invariance':True,
      'earlier_checkpoints_unchanged':True,'freeze_sha256':freeze_hash,'simultaneous_uniform_mixture_sampling_bound':bound,
      'scope':'Known-family conditional mixture sensitivity and development failure review; no final selection or applicant forecast'}
    save(out/'acceptance.json',acceptance);print(json.dumps(acceptance,indent=2))


if __name__=='__main__':main()
