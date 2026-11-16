"""A frozen-candidate stress test of human-informed models with broader scouts."""
from fractions import Fraction
from pathlib import Path
import argparse
import json
import math
import shutil
import time
import numpy as np
from blotto_lab.core import best_response,mix_tables
from blotto_lab.historical import rows_to_table,sampled_metadata,validate_rows
from blotto_lab.human import marginal_counts
from blotto_lab.human_scouts import broad_scout_rows
from run_multiwave_study import ROOT,unpack_table
from run_population_study import save,sha,write_csv
from verify_population_study import rational


def verify(out):
    r=json.loads((out/'results.json').read_text());m=json.loads((out/'manifest.json').read_text())
    for p,h in m['artifacts'].items():assert sha(out/p)==h,p
    for p,h in m['signature']['source_hashes'].items():assert sha(ROOT/p)==sha(out/'source-snapshot'/p)==h,p
    for p,h in m['signature']['input_hashes'].items():assert sha(ROOT/p)==h,p
    assert sha(out/'candidate-freeze.json')==r['freeze_sha256']
    assert r['frozen_unix']<r['sampling_started_unix']
    tables={};cells=0
    for key,model in r['models'].items():
        with np.load(out/model['sample_file']) as a:rows=a['allocations']
        validate_rows(rows)
        generated,_=broad_scout_rows(tuple(i-1 for i in model['coalition']),model['law'],model['sample_size'],model['seed'],50 if model['heavy5'] else None)
        np.testing.assert_array_equal(rows,generated)
        assert sampled_metadata(rows,model['seed'])['draw_sha256']==model['draw_sha256']
        table=rows_to_table(rows,key);tables[key]=table
        assert table.numerators==tuple(tuple(row) for row in model['table']['numerators'])
        assert table.denominator==model['table']['denominator']
        for label,x in r['labels'].items():
            scores=np.zeros(len(rows),dtype=np.int16)
            for i in range(5):scores += (i+1)*(2*(x[i]>rows[:,i])+(x[i]==rows[:,i]))
            saved=model['scores'][label]
            assert rational(saved['points'])==Fraction(int(scores.sum()),2*len(rows))==table.expected_score(x)
            assert rational(saved['win_credit'])==Fraction(int(2*(scores>15).sum()+(scores==15).sum()),2*len(rows))
            cells+=1
    for name,reference in r['references'].items():
        keys=reference['model_keys'];table=mix_tables([tables[k] for k in keys],[1]*len(keys))
        for label,x in r['labels'].items():
            assert table.expected_score(x)==rational(reference['scores'][label]['points'])
            assert sum(rational(r['models'][k]['scores'][label]['win_credit']) for k in keys)/len(keys)==rational(reference['scores'][label]['win_credit'])
        br=best_response(table)
        assert list(br.allocation)==reference['evaluation_benchmark']['allocation']
        assert br.expected_score==rational(reference['evaluation_benchmark']['score'])
    save(out/'acceptance.json',{'passed':True,'verified_unix':time.time(),'verifier_sha256':sha(Path(__file__)),
        'sampled_populations_regenerated':len(r['models']),'saved_legal_draws':sum(m['sample_size'] for m in r['models'].values()),
        'point_win_cells_checked':cells,'frozen_candidates_unchanged':True,
        'scope':'IID accepted draws conditional on declared generators; no inference of applicant prevalence'})
    print(f'Broad-scout audit passed: {cells} point/win cells',flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=ROOT/'results/human-scout-stress');parser.add_argument('--verify',action='store_true')
    args=parser.parse_args();out=args.output.resolve()
    if args.verify:verify(out);return
    if out.exists():raise SystemExit('Output exists. Choose a new output folder.')
    out.mkdir(parents=True);(out/'samples').mkdir();start=time.perf_counter()
    sources=[Path(__file__),ROOT/'blotto_lab/human_scouts.py',ROOT/'blotto_lab/historical.py',ROOT/'blotto_lab/human.py',ROOT/'blotto_lab/core.py',ROOT/'run_population_study.py',ROOT/'run_multiwave_study.py',ROOT/'HUMAN-SCOUT-STRESS-PROTOCOL.md']
    paths=[ROOT/'results/human-challenge/results.json',ROOT/'results/human-challenge/post-definition-reference-optima.json']
    signature={'source_hashes':{str(p.relative_to(ROOT)):sha(p) for p in sources},'input_hashes':{str(p.relative_to(ROOT)):sha(p) for p in paths},'sample_size':100000,'seed':2026103111,'scout_range':[0,24]}
    for p in sources:
        target=out/'source-snapshot'/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
    old=json.loads(paths[0].read_text());fits=json.loads(paths[1].read_text())
    labels=old['labels'].copy();labels.update({'F_'+k:v['allocation'] for k,v in fits['reference_results'].items()})
    frozen={'created_unix':time.time(),'labels':labels,'source_inputs':signature['input_hashes']};save(out/'candidate-freeze.json',frozen)
    freeze_hash=sha(out/'candidate-freeze.json');sampling_start=time.time()
    models,tables,lookup={},{},{}
    seed=signature['seed']
    for coalition_1 in old['configuration']['coalitions']:
        coalition=tuple(i-1 for i in coalition_1)
        for law in ['uniform','value_multinomial']:
            for heavy in ([False,True] if 4 in coalition else [False]):
                key='broad_'+''.join(map(str,coalition_1))+'_'+law+('_heavy5' if heavy else '')
                draw_seed=seed+len(models)
                rows,meta=broad_scout_rows(coalition,law,signature['sample_size'],draw_seed,50 if heavy else None)
                sample_file='samples/'+key+'.npz';np.savez_compressed(out/sample_file,allocations=rows.astype(np.uint8))
                table=rows_to_table(rows,key);tables[key]=table;lookup[(coalition,law,heavy)]=key
                scores={}
                for label,x in labels.items():
                    points=np.zeros(len(rows),dtype=np.int16)
                    for i in range(5):points+=(i+1)*(2*(x[i]>rows[:,i])+(x[i]==rows[:,i]))
                    scores[label]={'points':table.expected_score(x),
                        'win_credit':Fraction(int(2*(points>15).sum()+(points==15).sum()),2*len(rows))}
                models[key]={'coalition':coalition_1,'law':law,'heavy5':heavy,'sample_file':sample_file,**meta,
                    **sampled_metadata(rows,draw_seed),'table':{'numerators':table.numerators,'denominator':table.denominator,'evidence':table.evidence},'scores':scores}
                print(f'Broader scout models: {len(models)}/16',flush=True)
    favorites={tuple(i-1 for i in c) for c in old['configuration']['favorite_coalitions']}
    coalitions=[tuple(i-1 for i in c) for c in old['configuration']['coalitions']]
    references={}
    for name,chosen,heavy in [('all_open',coalitions,False),('all_heavy5',coalitions,True),('favorites_open',[c for c in coalitions if c in favorites],False),('favorites_heavy5',[c for c in coalitions if c in favorites],True)]:
        keys=[lookup[(c,law,heavy and 4 in c)] for c in chosen for law in ['uniform','value_multinomial']]
        table=mix_tables([tables[k] for k in keys],[1]*len(keys));br=best_response(table)
        counts,den=marginal_counts(table,4)
        references[name]={'model_keys':keys,'scores':{label:{'points':table.expected_score(x),'win_credit':sum(models[k]['scores'][label]['win_credit'] for k in keys)/len(keys)} for label,x in labels.items()},
            'probability_fifth_26_to40':Fraction(sum(counts[26:41]),den),
            'evaluation_benchmark':{'allocation':br.allocation,'score':br.expected_score,'status':'Computed after frozen-candidate evaluation, not added to the comparison set'}}
    error=15*math.sqrt(math.log(10*len(models)/.05)/(2*signature['sample_size']))
    r={'models':models,'references':references,'labels':labels,'freeze_sha256':freeze_hash,
        'frozen_unix':frozen['created_unix'],'sampling_started_unix':sampling_start,'simultaneous_dkw_score_error':error,
        'scope':'Broader-scout follow-up after exact coalition refits; sampled generator stress, no applicant prevalence inference'}
    save(out/'results.json',r)
    write_csv(out/'scores.csv',['reference','label','allocation','points','win_credit'],
        [[name,label,str(tuple(labels[label])),float(s['points']),float(s['win_credit'])] for name,reference in references.items() for label,s in reference['scores'].items()])
    lines=['**Broader nonpriority-bid stress test**','',
        'This follow-up widens nonpriority bids from the exact study\'s 0,1,3 scout levels to independent '
        'uniform bids from 0 through 24. Human allocations, original machine baselines, range fits, '
        'and four exact-coalition refits were frozen before drawing these populations.','',
        '| Candidate | Open broad-scout points | Five-heavy broad-scout points |',
        '| --- | ---: | ---: |']
    display=old['human_labels']+['R_illustration','M_known_model_max_min','F_all_open','F_all_heavy5']
    for label in display:lines.append(f"| {label}, {tuple(labels[label])} | {float(references['all_open']['scores'][label]['points']):.4f} | {float(references['all_heavy5']['scores'][label]['points']):.4f} |")
    lines += ['',f"All 16 variants contain 100,000 saved legal accepted draws each. The simultaneous 95% "
        f"DKW point-score error bound is {error:.4f}, conditional on these fixed generators. This does "
        'not validate the distribution against actual applicants. Five-heavy sampling rejects complete '
        'joint proposals until the fifth bid is at least 50. The resulting conditioning can also '
        'change the distribution of outside bids.','',
        'Evaluation benchmarks are stored separately and were not inserted into the frozen candidate '
        'comparison. This is one explicit fresh generator stress after the exact-model refits, not '
        'an independent sample of people. Further model changes need further validation.','',
        'Reproduce with `python3 run_human_scout_stress.py --output results/human-scout-new`; verify '
        'with `python3 run_human_scout_stress.py --verify`. Sources, freezes, seeds, joint draws, '
        'marginal tables, and point/win results are saved.','']
    (out/'report.md').write_text('\n'.join(lines))
    assert sha(out/'candidate-freeze.json')==freeze_hash
    save(out/'manifest.json',{'signature':signature,'elapsed_seconds':time.perf_counter()-start,
        'completed_unix':time.time(),'artifacts':{str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='manifest.json'}})
    verify(out)


if __name__=='__main__':main()
