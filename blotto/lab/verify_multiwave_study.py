"""Independent full-space and accounting checks of the multiwave study."""
from collections import Counter
from fractions import Fraction
from pathlib import Path
import argparse
import json
import time
import numpy as np
from blotto_lab.core import Population, best_response, mix_tables, security_of, validate_allocation
from blotto_lab.historical import rows_to_table, sampled_metadata, validate_rows
from blotto_lab.multiwave import hypothesis_rows, mutate
from run_multiwave_study import GROUP_NAMES, ROOT, unpack_table
from run_population_study import save, sha
from verify_population_study import integer_upper_certificates, rational


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--study',type=Path,default=ROOT/'results/multiwave-study')
    out = parser.parse_args().study.resolve()
    clock = time.perf_counter()
    result,freeze,manifest,initial = [read(out/f) for f in ['results.json','frozen-candidates.json','manifest.json','run-inputs.json']]
    cfg,models = result['configuration'],result['models']
    for relative,digest in manifest['artifacts'].items():
        assert sha(out/relative)==digest,relative
    for relative,digest in initial['implementation'].items():
        assert sha(ROOT/relative)==digest,relative
        assert sha(out/'source-snapshot'/relative)==digest,relative
    for relative,digest in initial['prior_inputs'].items():
        assert sha(ROOT/relative)==digest,relative
    assert manifest['signature']==initial and cfg==initial['configuration']
    assert freeze['candidates']==result['candidates'] and freeze['selected']==result['selected']
    assert freeze['created_unix']==manifest['frozen_unix']<manifest['evaluation_started_unix']
    marker = read(out/'evaluation-start.json')
    assert marker['started_unix']==manifest['evaluation_started_unix']
    assert marker['freeze_sha256']==sha(out/'frozen-candidates.json')==result['freeze_sha256']
    for file,field in [('development-models.json','development_models_sha256'),('seed-population.json','seed_population_sha256'),('run-inputs.json','inputs_sha256')]:
        assert sha(out/file)==freeze[field]
    tables = {k:unpack_table(m,k) for k,m in models.items()}
    optima = {k:best_response(t) for k,t in tables.items()}
    previous = read(ROOT/'results/population-study/results.json')
    samples = regenerated = adaptive_tables = 0
    for key,model in models.items():
        br = optima[key]
        assert list(br.allocation)==model['best_response']['allocation']
        assert br.expected_score==rational(model['best_response']['score'])
        assert br.optimum_count==model['best_response']['optimum_count']
        if model['kind']=='legacy_model':
            old = previous['models'][model['original_key']]
            assert tables[key].numerators==tuple(tuple(row) for row in old['table']['numerators'])
            assert tables[key].denominator==old['table']['denominator']
            if 'sample_file' not in old:
                continue
            metadata,path = old,ROOT/'results/population-study'/old['sample_file']
        elif model['kind']=='direct_generator':
            metadata,path = model,out/model['sample_file']
        elif model['kind'] in ('adaptive_generation','adaptive_archive'):
            record = read(out/model['population_file'])
            population = Population.from_dict(record[model['population_field']])
            assert population.table().numerators==tables[key].numerators
            assert population.table().denominator==tables[key].denominator
            adaptive_tables += 1
            continue
        else:
            continue
        with np.load(path) as data:
            rows = data['allocations']
        validate_rows(rows)
        assert sampled_metadata(rows,metadata['seed'])['draw_sha256']==metadata['draw_sha256']
        rebuilt = rows_to_table(rows,key)
        assert rebuilt.numerators==tables[key].numerators and rebuilt.denominator==tables[key].denominator
        if model['kind']=='direct_generator':
            np.testing.assert_array_equal(rows,hypothesis_rows(model['family'],model['sample_size'],model['seed']))
            regenerated += 1
        samples += 1
    seed_population = Population.from_dict(read(out/'seed-population.json'))
    group_keys = result['group_keys']
    groups = [mix_tables([tables[k] for k in group_keys[g]],[1]*len(group_keys[g])) for g in GROUP_NAMES]
    behavior_tables = [tables[k] for k in result['behavior_keys']]
    fresh = ['validation__'+f for f in cfg['development_families']+cfg['validation_novel_families']]
    validation_groups = groups[:-1]+[mix_tables([tables[k] for k in fresh],[1]*len(fresh))]
    validation_specialists = behavior_tables+[tables[k] for k in fresh]
    audit_groups = [tables['final_audit__'+f] for f in cfg['audit_novel_families']]
    response_checks = optimum_checks = mutations = waves = 0
    first_generations = {}
    for path in result['arena_paths']:
        counts,latest = Counter(dict(seed_population.entries)),seed_population
        phase,replicate,policy = path['phase'],path['replicate'],path['policy']
        base_seed = cfg['seed']+({'development':10000,'validation':3000000,'final_audit':5000000}[phase])+replicate*1000000
        for wave in range(1,cfg['waves']+1):
            record = read(out/'arenas'/path['label']/f'wave-{wave:02}.json')
            if phase=='validation':
                active_groups,specialists = validation_groups,validation_specialists
            elif phase=='final_audit' and wave>cfg['waves']//2:
                active_groups,specialists = audit_groups,audit_groups
            else:
                active_groups,specialists = groups,behavior_tables
            memory = latest if policy=='latest_only' else Population.from_counts(counts)
            assert record['memory_mass_before']==memory.total_mass
            assert record['environment_start_wave']==(cfg['waves']//2+1 if phase=='final_audit' and wave>cfg['waves']//2 else 1)
            memory_table = memory.table()
            rng = np.random.default_rng(base_seed+wave*10000)
            entries = []
            audit_jobs = {0,1,len(specialists)-1,len(specialists),64,95,127,cfg['responses_per_wave']-1}
            for j,episode in enumerate(record['episodes']):
                tie_seed,mutation_seed = int(rng.integers(0,2**62)),int(rng.integers(0,2**62))
                add_mutation = bool(rng.random()<cfg['mutation_probability'])
                assert episode['tie_seed']==tie_seed and episode['mutation_seed']==mutation_seed
                assert episode['mutation_added']==add_mutation
                if j<len(specialists):
                    target = specialists[j]
                    assert episode['belief']=={'kind':'specialist','index':j}
                else:
                    concentration = float(rng.choice(cfg['belief_concentrations']))
                    weights = rng.multinomial(1000,rng.dirichlet(np.full(len(active_groups),concentration))).tolist()
                    strategic = int(rng.choice(cfg['strategic_share_percentages']))
                    assert episode['belief']=={'kind':'group_and_memory','group_integer_weights':weights,
                        'strategic_share_percentage':strategic,'concentration':concentration}
                    target = mix_tables(list(active_groups)+[memory_table],[w*(100-strategic) for w in weights]+[strategic*1000])
                x = validate_allocation(episode['response']['allocation'])
                value = target.expected_score(x)
                assert value==rational(episode['response']['expected_score'])
                if j in audit_jobs:
                    independent = best_response(target)
                    assert value==independent.expected_score
                    assert independent.optimum_count==episode['response']['optimum_count']
                    optimum_checks += 1
                entries.append(x)
                if add_mutation:
                    child = mutate(x,mutation_seed,cfg['mutation_steps'],cfg['mutation_max_transfer'])
                    assert list(child)==episode['mutation']
                    assert target.expected_score(child)==rational(episode['mutation_target_score'])<=value
                    entries.append(child)
                    mutations += 1
                response_checks += 1
            assert len(record['episodes'])==cfg['responses_per_wave']
            latest = Population.from_people(entries)
            assert latest.entries==Population.from_dict(record['generation']).entries
            counts.update(entries)
            archive = Population.from_counts(counts)
            assert archive.entries==Population.from_dict(record['archive']).entries
            for field,population in [('generation_security',latest),('archive_security',archive)]:
                br = best_response(population.table())
                assert br.expected_score==rational(record[field]['best_response']['expected_score'])
                assert Fraction(15)-br.expected_score==rational(record[field]['security'])
                assert br.expected_score-Fraction(15,2)==rational(record[field]['exploitability'])
            if wave==1:
                key = phase,replicate
                if key in first_generations:
                    assert first_generations[key]==latest.entries
                else:
                    first_generations[key]=latest.entries
            waves += 1
        checkpoint = read(out/'arenas'/path['label']/'checkpoint.json')
        assert checkpoint['wave']==cfg['waves']
        assert Population.from_dict(checkpoint['archive']).entries==archive.entries
        assert Population.from_dict(checkpoint['latest']).entries==latest.entries
        print(f"Verified episodes and mass: {path['label']}",flush=True)
    partitions = {p:[k for k,m in models.items() if m['partition']==p] for p in {m['partition'] for m in models.values()}}
    for cid,item in result['candidates'].items():
        for key,table in tables.items():
            value = table.expected_score(item['allocation'])
            assert value==rational(result['scores'][cid][key])
            assert optima[key].expected_score-value==rational(result['regrets'][cid][key])
        for partition,keys in partitions.items():
            assert min(rational(result['scores'][cid][k]) for k in keys)==rational(result['summary'][cid][partition]['minimum'])
            assert max(rational(result['regrets'][cid][k]) for k in keys)==rational(result['summary'][cid][partition]['worst_regret'])
        pure = security_of(Population.from_people([tuple(item['allocation'])]))
        assert pure['security']==rational(result['security'][cid]['security'])
        assert list(pure['best_response'].allocation)==result['security'][cid]['best_response']['allocation']
    q = Population.from_dict(read(ROOT/'results/equilibrium-benchmark/mixture.json'))
    assert tables['certified_equilibrium_control'].numerators==q.table().numerators
    expected = {k:rational(x['maximized_value']) for k,x in result['selected'].items()}
    print('Independent integer-bound scan of all 4,598,126 allocations',flush=True)
    certificates,qcertificate = integer_upper_certificates([tables[k] for k in result['development_model_order']],
        result['selected'],expected,q.table(),batch_size=55555)
    for name,certificate in certificates.items():
        assert certificate['exact_value']==expected[name]
        assert list(certificate['winner'])==result['selected'][name]['allocation']
        assert certificate['tied_optima']==result['selected'][name]['optimum_count']
    assert qcertificate['score']==best_response(q.table()).expected_score
    assert sha(out/'frozen-candidates.json')==manifest['freeze_sha256']
    acceptance = {'passed':True,'verified_unix':time.time(),'elapsed_seconds':time.perf_counter()-clock,
        'verifier_sha256':sha(Path(__file__)),'frozen_candidates':len(result['candidates']),
        'scenario_tables':len(models),'score_regret_cells':len(models)*len(result['candidates']),
        'legal_samples_rebuilt':samples,'new_samples_regenerated':regenerated,'adaptive_tables_rebuilt':adaptive_tables,
        'arena_waves_verified':waves,'response_scores_beliefs_and_legality_verified':response_checks,
        'independent_core_dp_optimal_response_checks':optimum_checks,'mutations_verified':mutations,
        'complete_memory_accounting':True,'coupled_first_generations_match':True,
        'source_and_artifact_hashes_verified':True,'freeze_sha256':manifest['freeze_sha256'],
        'independent_integer_fullspace_certificates':certificates,'equilibrium_best_response':qcertificate,
        'scope':'Exact conditional-model and accounting verification; no applicant-population inference'}
    save(out/'acceptance.json',acceptance)
    print(json.dumps({k:v for k,v in acceptance.items() if not isinstance(v,dict)},indent=2))


if __name__=='__main__':
    main()
