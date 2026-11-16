"""Independent outcome-pattern and integer-bound certificates over every action."""
from fractions import Fraction
from itertools import islice
from pathlib import Path
import argparse
import json
import time
import numpy as np
from blotto_lab.core import DEFAULT_GAME, Population, best_response, iter_allocations, mix_tables, score
from blotto_lab.contamination import FrontierObjective, pattern_security_twice
from run_multiwave_study import ROOT, unpack_table
from run_population_study import save, sha
from verify_population_study import rational


def read(path): return json.loads(path.read_text())


def check_selection(row,frontier,counts,first):
    epsilon = rational(row['epsilon'])
    if epsilon==1:
        maximum_class = max(counts)
        maximum,allocation,ties = Fraction(maximum_class,2),first[maximum_class],counts[maximum_class]
        classes = [maximum_class]
    else:
        values = {c:(1-epsilon)*rational(p['nominal_value'])+epsilon*Fraction(c,2) for c,p in frontier.items()}
        maximum = max(values.values())
        classes = sorted(c for c,v in values.items() if v==maximum)
        allocation = min(tuple(frontier[c]['allocation']) for c in classes)
        ties = sum(frontier[c]['optimum_count'] for c in classes)
    assert rational(row['value'])==maximum
    assert tuple(row['allocation'])==allocation
    assert row['optimum_count']==ties and row['optimal_security_classes']==classes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--study',type=Path,default=ROOT/'results/contamination-study')
    out = parser.parse_args().study.resolve()
    clock = time.perf_counter()
    result,manifest = read(out/'results.json'),read(out/'manifest.json')
    signature = manifest['signature']
    for relative,digest in manifest['artifacts'].items(): assert sha(out/relative)==digest,relative
    for relative,digest in signature['source_hashes'].items():
        assert sha(ROOT/relative)==sha(out/'source-snapshot'/relative)==digest,relative
    for relative,digest in signature['input_hashes'].items(): assert sha(ROOT/relative)==digest,relative
    assert signature==read(out/'run-inputs.json')
    known_order = result['known_model_order']
    tables = [unpack_table(result['known_models'][k],k) for k in known_order]+[None]*len(result['reference_models'])
    objectives = [FrontierObjective(o['name'],tuple(o['indices']),o['mode']) for o in result['objectives']]
    for obj in objectives:
        if obj.name in result['reference_models']:
            tables[obj.indices[0]] = unpack_table(result['reference_models'][obj.name],obj.name)
    assert all(t is not None for t in tables)
    group_tables = {g:mix_tables([tables[known_order.index(k)] for k in keys],[1]*len(keys)) for g,keys in result['groups'].items()}
    behavior_groups = ['analytic','historical','anchors','support_rounding','new_reasoning']
    strategic_groups = ['legacy_strategic','adaptive_generations','adaptive_archives']
    strategic = mix_tables([group_tables[g] for g in strategic_groups],[1]*3)
    for obj in objectives:
        if obj.name in result['reference_models']:
            ref = result['reference_models'][obj.name]
            behavior = mix_tables([group_tables[g] for g in behavior_groups],ref['behavior_group_weights'])
            share = rational(ref['strategic_share'])
            reconstructed = mix_tables([behavior,strategic],[1-share,share])
            assert reconstructed.numerators==tables[obj.indices[0]].numerators
            assert reconstructed.denominator==tables[obj.indices[0]].denominator
    counts = {int(c):n for c,n in result['class_counts'].items()}
    first = {int(c):tuple(x) for c,x in result['class_first'].items()}
    frontiers = {name:{int(c):p for c,p in f.items()} for name,f in result['frontiers'].items()}
    with np.load(out/'security-classes.npz') as data: saved_security = data['security_twice']
    assert len(saved_security)==DEFAULT_GAME.actions==sum(counts.values())
    optima = [best_response(t).expected_score for t in tables]
    scale = 10**9
    upper = np.array([[[(n*scale+t.denominator-1)//t.denominator for n in row] for row in t.numerators] for t in tables],dtype=np.int64)
    lower_optima = np.array([v.numerator*scale//v.denominator for v in optima],dtype=np.int64)
    assert 15*scale+5<np.iinfo(np.int64).max
    thresholds = []
    for obj in objectives:
        lookup = np.full(31,np.iinfo(np.int64).max,dtype=np.int64)
        for c,p in frontiers[obj.name].items():
            value = rational(p['nominal_value'])
            lookup[c] = -((-value.numerator*scale)//value.denominator)
        thresholds.append(lookup)
    tied_counts = {o.name:{c:0 for c in counts} for o in objectives}
    winners = {o.name:{c:None for c in counts} for o in objectives}
    contenders = {o.name:{c:0 for c in counts} for o in objectives}
    rng = np.random.default_rng(result['configuration']['verification_seed'])
    sample_target = result['configuration']['independent_dp_samples']
    audit_indices = set()
    for c in counts:
        indices = np.flatnonzero(saved_security==c)
        chosen = rng.choice(indices,size=min(len(indices),sample_target//len(counts)),replace=False)
        audit_indices.update(map(int,chosen))
    while len(audit_indices)<sample_target:
        audit_indices.add(int(rng.integers(0,DEFAULT_GAME.actions)))
    iterator,done,dp_checks = iter_allocations(),0,0
    independent_counts = {c:0 for c in counts}
    independent_first = {}
    batch_size = result['configuration']['verification_batch_size']
    while True:
        batch = list(islice(iterator,batch_size))
        if not batch: break
        actions = np.asarray(batch,dtype=np.uint8)
        security = pattern_security_twice(actions)
        np.testing.assert_array_equal(security,saved_security[done:done+len(batch)])
        for c in np.unique(security):
            c = int(c)
            positions = np.flatnonzero(security==c)
            independent_counts[c] += len(positions)
            independent_first.setdefault(c,tuple(map(int,actions[positions[0]])))
        for index in sorted(i for i in audit_indices if done<=i<done+len(batch)):
            x = tuple(map(int,actions[index-done]))
            independent = 15-best_response(Population.from_people([x]).table()).expected_score
            assert independent==Fraction(int(security[index-done]),2)
            dp_checks += 1
        scores = np.zeros((len(tables),len(batch)),dtype=np.int64)
        for i in range(5): scores += upper[:,i,actions[:,i]]
        for j,obj in enumerate(objectives):
            selected = scores[list(obj.indices)]
            if obj.mode=='regret': selected = selected-lower_optima[list(obj.indices),None]
            upper_values = selected.min(axis=0)
            indices = np.flatnonzero(upper_values>=thresholds[j][security])
            for index in indices:
                c = int(security[index]); x = tuple(map(int,actions[index]))
                values = [tables[i].expected_score(x) for i in obj.indices]
                if obj.mode=='regret': values = [v-optima[i] for v,i in zip(values,obj.indices)]
                value = min(values); expected = rational(frontiers[obj.name][c]['nominal_value'])
                assert value<=expected,(obj.name,c,x,value,expected)
                contenders[obj.name][c] += 1
                if value==expected:
                    tied_counts[obj.name][c] += 1
                    old = winners[obj.name][c]
                    winners[obj.name][c] = x if old is None else min(x,old)
        done += len(batch)
        if done%1000000<batch_size or done==DEFAULT_GAME.actions:
            print(f'Independent security and objective scan: {done:,}/{DEFAULT_GAME.actions:,}',flush=True)
    assert independent_counts==counts and independent_first==first
    assert dp_checks==sample_target
    for obj in objectives:
        for c,p in frontiers[obj.name].items():
            assert winners[obj.name][c]==tuple(p['allocation']),(obj.name,c)
            assert tied_counts[obj.name][c]==p['optimum_count'],(obj.name,c)
    cells = 0
    for cid,item in result['candidates'].items():
        x = tuple(item['allocation'])
        info = result['security'][cid]
        guarantee = rational(info['guarantee'])
        counter = tuple(info['counter'])
        assert score(x,counter)==guarantee
        assert 15-best_response(Population.from_people([x]).table()).expected_score==guarantee
        for i,k in enumerate(known_order):
            value = tables[i].expected_score(x)
            assert value==rational(result['scores'][cid][k])>=guarantee
            assert optima[i]-value==rational(result['regrets'][cid][k])
            cells += 1
    for name,rows in result['epsilon_grid'].items():
        for row in rows: check_selection(row,frontiers[name],counts,first)
    for name,envelope in result['envelopes'].items():
        previous = Fraction(0)
        for segment in envelope['segments']:
            lo,hi = rational(segment['lower_epsilon']),rational(segment['upper_epsilon'])
            assert lo==previous and lo<hi
            c = segment['security_class']
            for e in [lo,hi]:
                chosen = (1-e)*rational(frontiers[name][c]['nominal_value'])+e*Fraction(c,2)
                for k,p in frontiers[name].items():
                    assert chosen>=(1-e)*rational(p['nominal_value'])+e*Fraction(k,2)
            previous = hi
        assert previous==1
        for row in envelope['boundary_ties']: check_selection(row,frontiers[name],counts,first)
    for row in result['security_floors']:
        allowed = [p for c,p in frontiers[row['objective']].items() if Fraction(c,2)>=rational(row['minimum_security'])]
        maximum = max(rational(p['nominal_value']) for p in allowed)
        tied = [p for p in allowed if rational(p['nominal_value'])==maximum]
        assert rational(row['value'])==maximum
        assert tuple(row['allocation'])==min(tuple(p['allocation']) for p in tied)
        assert row['optimum_count']==sum(p['optimum_count'] for p in tied)
    acceptance = {'passed':True,'verified_unix':time.time(),'elapsed_seconds':time.perf_counter()-clock,
        'verifier_sha256':sha(Path(__file__)),'all_actions_independently_classified':done,
        'outcome_pattern_method':'Enumerate independent opponent loss/tie/win states, minimum legal costs, highest feasible score',
        'independent_dp_security_checks':dp_checks,'conditional_objectives':len(objectives),
        'security_classes':len(counts),'conditional_global_optima_verified':len(objectives)*len(counts),
        'all_exact_tie_counts_verified':True,'counter_and_score_regret_cells_checked':cells,
        'continuous_envelopes_verified':len(result['envelopes']),'security_floor_results_checked':len(result['security_floors']),
        'class_counts':counts,'class_first':first,'integer_upper_scale':scale,
        'retained_integer_upper_contenders':contenders,
        'scope':'Exact mathematical certificates for stated saved models; no new holdout or applicant inference'}
    save(out/'acceptance.json',acceptance)
    print(json.dumps({k:v for k,v in acceptance.items() if not isinstance(v,dict)},indent=2))


if __name__=='__main__': main()
