"""Independent integer-upper-bound audit of the frozen population selections."""
from fractions import Fraction
from itertools import islice
from pathlib import Path
import argparse
import hashlib
import json
import time

import numpy as np

from blotto_lab.core import DEFAULT_GAME, Population, ScoreTable, best_response, iter_allocations, security_of
from blotto_lab.historical import rows_to_table, sampled_metadata, validate_rows
from run_population_study import save, sha

ROOT = Path(__file__).resolve().parent


def rational(record):
    return Fraction(record['numerator'],record['denominator'])


def integer_upper_certificates(tables,objectives,expected,quantized_table,batch_size=77777):
    """All-action check using exact integer upper enclosures, without float scores.

    ceil(D*g_i) upper-bounds D*g_i. For regret, subtract floor(D*M).
    An action capable of tying or beating the claimed optimum must meet that
    optimum's ceiling threshold in the integer upper enclosure. Recheck those
    actions as Fractions. D only changes the number retained, not correctness.
    """
    scale = 10**9
    upper = np.array([[[ (n*scale+t.denominator-1)//t.denominator for n in row]
                       for row in t.numerators] for t in tables],dtype=np.int64)
    optima = [best_response(t).expected_score for t in tables]
    lower_optima = np.array([v.numerator*scale//v.denominator for v in optima],dtype=np.int64)
    assert DEFAULT_GAME.total_value*scale+DEFAULT_GAME.castles < np.iinfo(np.int64).max
    qrows = np.array(quantized_table.numerators,dtype=np.int64)
    assert sum(max(row) for row in quantized_table.numerators)<np.iinfo(np.int64).max
    retained = {name:[] for name in objectives}
    thresholds = {name:-((-expected[name].numerator*scale)//expected[name].denominator) for name in objectives}
    qbest,qwinner,qties = None,None,0
    actions = iter_allocations()
    done = 0
    while True:
        batch = list(islice(actions,batch_size))
        if not batch:
            break
        array = np.asarray(batch,dtype=np.uint8)
        values = np.zeros((len(tables),len(batch)),dtype=np.int64)
        qvalues = np.zeros(len(batch),dtype=np.int64)
        for i in range(5):
            values += upper[:,i,array[:,i]]
            qvalues += qrows[i,array[:,i]]
        for name,obj in objectives.items():
            scores = values[obj['model_indices']]
            if obj['mode']=='minimax_regret':
                scores = scores-lower_optima[obj['model_indices'],None]
            mask = scores.min(axis=0)>=thresholds[name]
            retained[name].extend(tuple(map(int,x)) for x in array[mask])
        maximum = int(qvalues.max())
        if qbest is None or maximum>qbest:
            qbest,qties = maximum,0
            qwinner = tuple(map(int,array[np.flatnonzero(qvalues==maximum)[0]]))
        if maximum==qbest:
            qties += int((qvalues==maximum).sum())
        done += len(batch)
    certificates = {}
    for name,obj in objectives.items():
        exact_values = []
        for x in retained[name]:
            scores = [tables[i].expected_score(x) for i in obj['model_indices']]
            if obj['mode']=='minimax_regret':
                scores = [v-optima[i] for v,i in zip(scores,obj['model_indices'])]
            exact_values.append((min(scores),x))
        maximum = max(v for v,x in exact_values)
        winners = [x for v,x in exact_values if v==maximum]
        certificates[name] = {'exact_value':maximum,'winner':min(winners),'tied_optima':len(winners),
            'integer_upper_contenders':len(retained[name]),'integer_scale':scale,'actions_scanned':done}
    qcertificate = {'allocation':qwinner,'score':Fraction(qbest,quantized_table.denominator),
                    'optimum_count':qties,'actions_scanned':done}
    return certificates,qcertificate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--study',type=Path,default=ROOT/'results/population-study')
    parser.add_argument('--equilibrium',type=Path,default=ROOT/'results/equilibrium-benchmark')
    args = parser.parse_args()
    start = time.perf_counter()
    study,equilibrium = args.study,args.equilibrium
    result = json.loads((study/'results.json').read_text())
    freeze = json.loads((study/'frozen-candidates.json').read_text())
    manifest = json.loads((study/'manifest.json').read_text())
    assert sha(study/'frozen-candidates.json')==result['freeze_sha256']==manifest['freeze_sha256']
    assert manifest['frozen_unix']<manifest['evaluation_started_unix']
    for relative,digest in manifest['artifacts'].items():
        assert sha(study/relative)==digest,relative
    for relative,digest in manifest['implementation'].items():
        assert sha(ROOT/relative)==digest,relative
    assert result['candidates']==freeze['candidates']
    assert result['selected']==freeze['selected']
    assert sha(study/'development-models.json')==freeze['development_models_sha256']
    tables = {}
    sample_checks = 0
    for key,model in result['models'].items():
        t = model['table']
        tables[key] = ScoreTable(DEFAULT_GAME,tuple(tuple(row) for row in t['numerators']),t['denominator'],key)
        optimum = best_response(tables[key])
        assert list(optimum.allocation)==model['best_response']['allocation']
        assert optimum.expected_score==rational(model['best_response']['score'])
        assert optimum.optimum_count==model['best_response']['optimum_count']
        if 'sample_file' in model:
            with np.load(study/model['sample_file']) as saved:
                rows = saved['allocations']
            validate_rows(rows)
            assert sampled_metadata(rows,model['seed'])['draw_sha256']==model['draw_sha256']
            rebuilt = rows_to_table(rows,key)
            assert rebuilt.numerators==tables[key].numerators
            assert rebuilt.denominator==tables[key].denominator
            sample_checks += 1
    for cid,item in result['candidates'].items():
        for key,table in tables.items():
            score = table.expected_score(item['allocation'])
            assert score==rational(result['scores'][cid][key])
            assert best_response(table).expected_score-score==rational(result['regrets'][cid][key])
        pure = security_of(Population.from_people([tuple(item['allocation'])]))
        assert pure['security']==rational(result['candidate_security'][cid]['security'])
    # Exact response beliefs can be checked against every saved cloud support.
    for suffix in ('development','validation'):
        cloud = json.loads((study/f'response-cloud-{suffix}.json').read_text())
        pop = Population.from_dict(cloud['population'])
        observed = Population.from_people([tuple(b['response']) for b in cloud['beliefs']])
        assert pop.entries==observed.entries
    q = Population.from_dict(json.loads((equilibrium/'mixture.json').read_text()))
    qinfo = security_of(q)
    eqcert = json.loads((equilibrium/'certificate.json').read_text())
    assert qinfo['exploitability']==rational(eqcert['exploitability'])
    assert qinfo['exploitability']<=rational(eqcert['tolerance'])
    eqmanifest = json.loads((equilibrium/'manifest.json').read_text())
    for relative,digest in eqmanifest['source_hashes'].items():
        assert sha(ROOT/relative)==digest,relative
    for relative,digest in eqmanifest['artifacts'].items():
        assert sha(equilibrium/relative)==digest,relative
    dev_keys = result['phases']['development']
    expected = {name:rational(obj['maximized_value']) for name,obj in result['selected'].items()}
    certificates,independent_q = integer_upper_certificates([tables[k] for k in dev_keys],
        result['selected'],expected,q.table())
    for name,certificate in certificates.items():
        assert certificate['exact_value']==expected[name]
        assert list(certificate['winner'])==result['selected'][name]['allocation']
        assert certificate['tied_optima']==result['selected'][name]['optimum_count']
    assert independent_q['allocation']==qinfo['best_response'].allocation
    assert independent_q['score']==qinfo['best_response'].expected_score
    assert independent_q['optimum_count']==qinfo['best_response'].optimum_count
    acceptance = {'passed':True,'verified_unix':time.time(),'elapsed_seconds':time.perf_counter()-start,
        'verification_source_sha256':sha(Path(__file__)),'freeze_sha256':sha(study/'frozen-candidates.json'),
        'sampled_legal_joint_pools_rebuilt':sample_checks,'frozen_candidates':len(result['candidates']),
        'scenario_tables':len(tables),'score_regret_cells_checked':len(tables)*len(result['candidates']),
        'independent_integer_upper_fullspace_certificates':certificates,
        'independent_integer_equilibrium_best_response':independent_q,
        'scope':'global certificates for specified saved models; no claim about applicant distribution'}
    save(study/'acceptance.json',acceptance)
    print(json.dumps({'passed':True,'seconds':acceptance['elapsed_seconds'],'samples':sample_checks,
                      'candidates':len(result['candidates']),'models':len(tables)},indent=2))


if __name__=='__main__':
    main()
