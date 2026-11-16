"""Symmetric double oracle, with an exact full-action certificate after every LP."""
from fractions import Fraction
from pathlib import Path
import argparse
import json
import time

import numpy as np
import scipy
from scipy.optimize import linprog

from blotto_lab.core import DEFAULT_GAME, Population, score_twice, security_of
from run_population_study import encode, save, sha, write_csv

ROOT = Path(__file__).resolve().parent


def centered_matrix(actions,game=DEFAULT_GAME):
    rows = np.asarray(actions,dtype=np.int64)
    result = np.full((len(rows),len(rows)),-game.total_value,dtype=np.int64)
    for i,v in enumerate(game.values):
        result += 2*v*(rows[:,i,None]>rows[None,:,i])+v*(rows[:,i,None]==rows[None,:,i])
    if not np.array_equal(result,-result.T):
        raise AssertionError('Centered constant-sum matrix must be antisymmetric')
    return result


def integer_masses(probabilities,total=10**15):
    """Quantize numerical LP probabilities; exact verification follows this step."""
    raw = np.maximum(np.asarray(probabilities,dtype=np.float64),0)
    if not np.isfinite(raw).all() or raw.sum()<=0:
        raise ValueError('Invalid probability vector')
    scaled = raw/raw.sum()*total
    masses = [int(np.floor(x)) for x in scaled]
    difference = total-sum(masses)
    if not 0<=difference<=len(masses):
        raise ValueError('Unexpected floating normalization error')
    order = sorted(range(len(masses)),key=lambda j:(-(scaled[j]-masses[j]),j))
    for j in order[:difference]:
        masses[j] += 1
    assert sum(masses)==total and min(masses)>=0
    return masses


def solve_restricted(actions,game=DEFAULT_GAME):
    centered = centered_matrix(actions,game).astype(np.float64)/2
    n = len(actions)
    # max opponent payoff above V/2 <= t. Restricted symmetric value is zero.
    constraints = np.column_stack((centered,-np.ones(n)))
    objective = np.zeros(n+1)
    objective[-1] = 1
    equality = np.zeros((1,n+1))
    equality[0,:n] = 1
    result = linprog(objective,A_ub=constraints,b_ub=np.zeros(n),A_eq=equality,
        b_eq=np.ones(1),bounds=[(0,None)]*n+[(None,None)],method='highs',
        options={'primal_feasibility_tolerance':1e-10,'dual_feasibility_tolerance':1e-10,
                 'ipm_optimality_tolerance':1e-12})
    if not result.success:
        raise RuntimeError(f'LP failed: {result.message}')
    masses = integer_masses(result.x[:n])
    population = Population.from_counts([(x,m) for x,m in zip(actions,masses) if m],game,
                                        'quantized restricted equilibrium')
    diagnostics = {'solver_objective':float(result.fun),'solver_status':int(result.status),
        'solver_iterations':int(result.nit),'restricted_max_violation':float(np.max(centered@result.x[:n])),
        'solver':'scipy.optimize.linprog(method=highs)','scipy':scipy.__version__,
        'probability_quantization_total':10**15,'lp_probabilities':result.x[:n].tolist()}
    return population,diagnostics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=ROOT/'results/equilibrium-benchmark')
    parser.add_argument('--tolerance',default='0.00000001')
    parser.add_argument('--max-iterations',type=int,default=1000)
    parser.add_argument('--resume',action='store_true')
    args = parser.parse_args()
    out = args.output.resolve()
    tolerance = Fraction(args.tolerance)
    if tolerance<=0:
        raise SystemExit('Tolerance must be positive')
    paths = [Path(__file__),ROOT/'blotto_lab/core.py',ROOT/'run_population_study.py',
             ROOT/'results/first-pass/arena-mixture.json',ROOT/'results/population-study/frozen-candidates.json']
    source_hashes = {str(p.relative_to(ROOT)):sha(p) for p in paths}
    if out.exists() and not args.resume:
        raise SystemExit('Output exists. Use --resume or choose a new folder.')
    out.mkdir(parents=True,exist_ok=True)
    checkpoint = out/'checkpoint.json'
    if args.resume:
        state = json.loads(checkpoint.read_text())
        if state['source_hashes']!=source_hashes:
            raise SystemExit('Checkpoint inputs or implementation changed.')
        actions = [tuple(x) for x in state['actions']]
        trace = state['trace']
    else:
        archive = Population.from_dict(json.loads((ROOT/'results/first-pass/arena-mixture.json').read_text()))
        actions = [x for x,m in archive.entries]
        trace = []
    start = time.perf_counter()
    for iteration in range(len(trace)+1,args.max_iterations+1):
        population,diagnostics = solve_restricted(actions)
        exact = security_of(population)
        response = exact['best_response'].allocation
        record = {'iteration':iteration,'restricted_actions':len(actions),'support_size':len(population.entries),
            'security':exact['security'],'exploitability':exact['exploitability'],
            'best_response':response,'lp_objective':diagnostics['solver_objective'],
            'iteration_elapsed_seconds':time.perf_counter()-start}
        trace.append(encode(record))
        print(f"Iteration {iteration}: {len(actions)} restricted actions, exact gap {float(exact['exploitability']):.12g}",flush=True)
        save(out/'mixture.json',population.to_dict())
        save(out/'certificate.json',{**exact,'solver_diagnostics':diagnostics,
             'tolerance':tolerance,'achieved':exact['exploitability']<=tolerance})
        if exact['exploitability']>tolerance:
            if response in actions:
                raise RuntimeError('Numerical restricted solution fails certificate and oracle is already present')
            actions.append(response)
        state = {'source_hashes':source_hashes,'actions':actions,'trace':trace}
        temporary = out/'checkpoint.tmp'
        save(temporary,state)
        temporary.replace(checkpoint)
        if exact['exploitability']<=tolerance:
            break
    else:
        raise SystemExit('Iteration cap reached. Saved state can be resumed.')
    frozen_path = ROOT/'results/population-study/frozen-candidates.json'
    frozen = json.loads(frozen_path.read_text())
    scores = {cid:population.table().expected_score(item['allocation']) for cid,item in frozen['candidates'].items()}
    save(out/'frozen-candidate-scores.json',{'freeze_sha256':sha(frozen_path),'scores':scores,
         'scope':'post-freeze adversarial audit only; no candidates added or changed'})
    write_csv(out/'progress.csv',['iteration','restricted_actions','support_size','security','exploitability'],
        [[r['iteration'],r['restricted_actions'],r['support_size'],r['security']['decimal'],r['exploitability']['decimal']] for r in trace])
    selected_rows = []
    for name,item in frozen['selected'].items():
        cid = item['candidate_id']
        selected_rows.append(f"| {name} | {tuple(item['allocation'])} | {float(scores[cid]):.8f} |")
    report = ['**Game 1 equilibrium benchmark: exact certificate**','',
        f"The double-oracle solver reached a full-action exploitability of {float(exact['exploitability']):.12g} points. "
        f"Its saved integer-weight mixture guarantees 7.5 minus {float(exact['exploitability']):.12g} expected points "
        'against every legal opponent. The symmetric game value is exactly 7.5.','',
        f"The final legal support has {len(population.entries)} allocations, after {len(trace)} restricted solves. "
        'The initial restricted set used all 774 distinct allocations in the earlier response archive. '
        'Each step reweights retained actions by a symmetric zero-sum LP, computes an exact full-space '
        'best response, and adds that response if the security gap exceeds the declared tolerance. '
        'Actions are retained, including those receiving zero current weight.','',
        'The LP uses floating arithmetic with primal and dual feasibility tolerances 1e-10. '
        'It proposes a mixture; it does not supply the final guarantee. Probabilities are converted '
        'to legal integer masses totaling 10^15. The saved rational mixture is then verified by the '
        'exact dynamic-programming oracle across the full action space. The certificate already '
        'includes LP error and quantization error. It establishes the displayed tolerance, not an '
        'exact rational equilibrium with zero exploitability.','',
        '**Independent audit of previously frozen selections**','',
        '| Selection | Fixed vector | Score against certified mixture |',
        '| --- | --- | ---: |',*selected_rows,'',
        'These are post-freeze adversarial diagnostics. They did not change the population-study '
        'shortlist or its robust winners. Scoring near 7.5 against this mixture is not an applicant '
        'forecast. A fixed vector does not inherit the mixture\'s security guarantee. Every fixed '
        'vector remains counterable.','',
        '`mixture.json` preserves exact masses; `certificate.json` stores the rational certificate '
        'and numerical solver diagnostics; `checkpoint.json` retains every restricted action and '
        'the progress trace. Resume with `python3 run_equilibrium_benchmark.py --resume`.','',
        'LP interface and tolerances: [SciPy HiGHS documentation](https://docs.scipy.org/doc/scipy/reference/optimize.linprog-highs.html). '
        'The guarantee comes from the project\'s exact best-response check.','']
    (out/'report.md').write_text('\n'.join(report))
    save(out/'manifest.json',{'source_hashes':source_hashes,'scipy':scipy.__version__,
        'numpy':np.__version__,'exact_tolerance':tolerance,'completed_unix':time.time(),
        'elapsed_seconds_this_invocation':time.perf_counter()-start,
        'artifacts':{str(p.relative_to(out)):sha(p) for p in sorted(out.iterdir()) if p.is_file() and p.name!='manifest.json'}})
    print('Certified benchmark saved.',flush=True)


if __name__=='__main__':
    main()
