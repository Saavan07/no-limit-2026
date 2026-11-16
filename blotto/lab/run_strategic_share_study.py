"""Complete exact mixture-weight maps, with independent endpoint certificates."""
from fractions import Fraction
from pathlib import Path
import argparse
import json
import shutil
import time
import numpy as np
from blotto_lab.core import best_response, mix_tables
from blotto_lab.parametric import strategic_share_map
from run_multiwave_study import ROOT, unpack_table
from run_population_study import save, sha, write_csv
from verify_population_study import rational


def verify(out):
    result = json.loads((out/'results.json').read_text())
    manifest = json.loads((out/'manifest.json').read_text())
    for path,digest in manifest['artifacts'].items(): assert sha(out/path)==digest,path
    for path,digest in manifest['signature']['source_hashes'].items():
        assert sha(ROOT/path)==sha(out/'source-snapshot'/path)==digest,path
    for path,digest in manifest['signature']['input_hashes'].items(): assert sha(ROOT/path)==digest,path
    endpoint_checks=interior_checks=boundary_checks=grid_checks=0
    prior = json.loads((ROOT/'results/contamination-study/results.json').read_text())
    for name,data in result['maps'].items():
        a,b = [unpack_table(prior['reference_models'][key],key) for key in data['input_reference_keys']]
        previous = Fraction(0)
        for segment in data['segments']:
            lo,hi = rational(segment['lower_share']),rational(segment['upper_share'])
            assert lo==previous and lo<hi
            x = tuple(segment['allocation'])
            assert a.expected_score(x)==rational(segment['behavior_score'])
            assert b.expected_score(x)==rational(segment['strategic_score'])
            for s in [lo,hi]:
                table = mix_tables([a,b],[1-s,s])
                assert table.expected_score(x)==best_response(table).expected_score
                endpoint_checks += 1
            middle = (lo+hi)/2
            ordinary = best_response(mix_tables([a,b],[1-middle,middle]))
            assert ordinary.allocation==x
            assert ordinary.optimum_count==segment['open_interval_optimum_count']
            interior_checks += 1
            previous = hi
        assert previous==1
        for boundary in data['boundaries']:
            s = rational(boundary['share'])
            br = best_response(mix_tables([a,b],[1-s,s]))
            assert tuple(boundary['allocation'])==br.allocation
            assert rational(boundary['score'])==br.expected_score
            assert boundary['optimum_count']==br.optimum_count
            boundary_checks += 1
        for raw in prior['configuration']['solver_shares']:
            s = Fraction(raw)
            reference = unpack_table(prior['reference_models']['ref_'+name+'_s'+raw],'saved grid')
            br = best_response(reference)
            boundary = next((p for p in data['boundaries'] if rational(p['share'])==s),None)
            if boundary:
                assert tuple(boundary['allocation'])==br.allocation
                assert boundary['optimum_count']==br.optimum_count
            else:
                segment = next(p for p in data['segments'] if rational(p['lower_share'])<s<rational(p['upper_share']))
                assert tuple(segment['allocation'])==br.allocation
                assert segment['open_interval_optimum_count']==br.optimum_count
            grid_checks += 1
    acceptance = {'passed':True,'verified_unix':time.time(),'verifier_sha256':sha(Path(__file__)),
        'complete_maps':len(result['maps']),'endpoint_global_optimum_checks':endpoint_checks,
        'interior_exact_tie_count_checks':interior_checks,'boundary_exact_tie_count_checks':boundary_checks,
        'previous_grid_settings_reproduced':grid_checks,
        'certificate':'For each fixed vector the score difference from every legal alternative is linear in the mixture share. Global optimality at both endpoints proves the entire closed segment.',
        'scope':'Exact for declared saved references; no inference of real applicant weights'}
    save(out/'acceptance.json',acceptance)
    print(json.dumps(acceptance,indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=ROOT/'results/strategic-share-study')
    parser.add_argument('--verify',action='store_true')
    args=parser.parse_args(); out=args.output.resolve()
    if args.verify: verify(out); return
    if out.exists(): raise SystemExit('Output exists. Use --verify or a new output folder.')
    out.mkdir(parents=True); start=time.perf_counter()
    paths=[Path(__file__),ROOT/'blotto_lab/parametric.py',ROOT/'blotto_lab/core.py',
           ROOT/'run_population_study.py',ROOT/'run_multiwave_study.py',ROOT/'STRATEGIC-SHARE-PROTOCOL.md']
    input_path=ROOT/'results/contamination-study/results.json'
    signature={'source_hashes':{str(p.relative_to(ROOT)):sha(p) for p in paths},
               'input_hashes':{str(input_path.relative_to(ROOT)):sha(input_path)}}
    for p in paths:
        dest=out/'source-snapshot'/p.relative_to(ROOT); dest.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(p,dest)
    prior=json.loads(input_path.read_text()); maps={}
    for name in prior['configuration']['behavior_reference_weights']:
        keys=['ref_'+name+'_s0','ref_'+name+'_s1']
        a,b=[unpack_table(prior['reference_models'][key],key) for key in keys]
        mapped=strategic_share_map(a,b); mapped['input_reference_keys']=keys; maps[name]=mapped
        print(f"{name}: {len(mapped['segments'])} exact open intervals; {mapped['oracle_calls']} oracle calls",flush=True)
    result={'maps':maps,'scope':'Complete continuous strategic-share sensitivity; known reference hypotheses, no new holdout or human input'}
    save(out/'results.json',result)
    write_csv(out/'intervals.csv',['reference','lower_share','upper_share','allocation','behavior_score','strategic_score','interior_tie_count'],
        [[name,float(s['lower_share']),float(s['upper_share']),str(s['allocation']),float(s['behavior_score']),float(s['strategic_score']),s['open_interval_optimum_count']]
         for name,m in maps.items() for s in m['segments']])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    balanced=maps['balanced']
    for castle in range(5):
        xs,ys=[],[]
        for s in balanced['segments']:
            xs += [float(s['lower_share']),float(s['upper_share'])]; ys += [s['allocation'][castle]]*2
        axes[0].plot(xs,ys,label=f'Castle {castle+1}')
    axes[0].set_ylabel('Soldiers in an optimal fixed allocation')
    axes[0].set_title('Balanced behavior reference: every exact phase')
    axes[0].legend(fontsize=8)
    for name,m in maps.items():
        xs,ys=[],[]
        for s in m['segments']:
            for share in [s['lower_share'],s['upper_share']]:
                xs.append(float(share)); ys.append(float((1-share)*s['behavior_score']+share*s['strategic_score']))
        axes[1].plot(xs,ys,label=name.replace('_',' '))
    axes[1].set_ylabel('Optimal expected score for that declared mixture')
    axes[1].set_title('Different behavior priors imply different objectives')
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.set_xlabel('Assumed strategic-opponent share'); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(out/'complete-share-maps.png',dpi=180); plt.close(fig)
    lines=['**Game 1: complete strategic-share sensitivity maps**','',
        'The four declared behavior references now have exact best-response maps for every strategic '
        'share from zero to one. These replace a grid approximation with complete piecewise-linear '
        'envelopes. All input populations were already known hypotheses; no new applicant data or '
        'human strategy input has been used.','',
        '| Declared behavior reference | Open intervals with a fixed optimal response | Exact DP oracle calls |',
        '| --- | ---: | ---: |']
    for name,m in maps.items(): lines.append(f"| {name} | {len(m['segments'])} | {m['oracle_calls']} |")
    lines += ['', 'Every interval endpoint is an exact rational number in results.json. Its decimal '
        'representation is conditional mathematical precision, not precision about real applicant '
        'shares. intervals.csv gives a readable complete table. Boundary-only optima and exact '
        'tie counts are retained separately, including the endpoints zero and one.','',
        'At a fixed share, the optimizer searches all legal allocations through the exact marginal '
        'DP. The envelope discovery uses the intersection of known optimal lines to ask for a new '
        'global best response. Directional tie selection at endpoints preserves short phases. '
        'An independent ordinary-DP audit verifies the chosen allocation is globally optimal at '
        'both ends of each segment. Its difference from any other fixed allocation is linear in '
        'the share, so these endpoint checks certify the whole segment.','',
        'These maps optimize expected score for particular declared mixtures. They do not optimize '
        'worst-case score over all mixture weights or adversarial contamination. Those separate '
        'objectives are documented in the contamination study. No share or reference is identified '
        'as representative of Sequoia applicants, and no final submission is selected.','',
        'Run `python3 run_strategic_share_study.py --output results/share-map-new`; verify the delivered '
        'run with `python3 run_strategic_share_study.py --verify`. Source snapshots, input hashes, '
        'queries, exact interval and boundary results, plots, and acceptance checks are saved.','',
        '![Complete strategic-share maps](complete-share-maps.png)','']
    (out/'report.md').write_text('\n'.join(lines))
    for p in paths: assert sha(p)==signature['source_hashes'][str(p.relative_to(ROOT))]
    save(out/'manifest.json',{'signature':signature,'elapsed_seconds':time.perf_counter()-start,
        'completed_unix':time.time(),'artifacts':{str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='manifest.json'}})
    verify(out)


if __name__=='__main__': main()
