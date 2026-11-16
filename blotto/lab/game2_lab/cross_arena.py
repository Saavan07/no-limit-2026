"""Frozen Parry inputs, independently loaded; original native scoring bridge."""
from pathlib import Path
import importlib.util
import json
import numpy as np
from blotto_lab.core import validate_allocation


def read(path):
    return json.loads(Path(path).read_text())


def load_catalog(folder):
    folder = Path(folder)
    fit = read(folder / 'results/field_fit_validated.json')
    catalog = {}
    with np.load(folder / 'results/fit_catalog.npz') as rows, np.load(folder / 'results/fit_catalog_weights.npz') as mass:
        if set(rows.files) != set(fit['catalog']) or set(mass.files) != set(fit['catalog']):
            raise ValueError('Catalog family mismatch')
        for name in fit['catalog']:
            x, w = rows[name].copy(), mass[name].copy()
            if x.ndim != 2 or x.shape[1] != 5 or w.shape != (len(x),):
                raise ValueError('Invalid catalog shape')
            if np.any(x < 0) or np.any(x != np.floor(x)) or np.any(x.sum(axis=1) != 100):
                raise ValueError('Illegal catalog allocations')
            if not np.all(np.isfinite(w)) or np.any(w < 0) or w.sum() <= 0:
                raise ValueError('Invalid catalog masses')
            catalog[name] = (x, w)
    fields = []
    for r in fit['models']:
        if not r['retained']:
            continue
        w = np.asarray(r['weights'], dtype=float)
        if len(w) != len(catalog) or np.any(w < 0) or not np.isfinite(w).all() or abs(w.sum()-1) > 1e-10:
            raise ValueError('Invalid field weights')
        fields.append({'name': 'fit_'+str(r['run']), 'weights': r['weights'],
                       'strict': r['validation_cdf_max_error'] <= .05 and r['validation_cdf_mean_error'] <= .02 and r['validation_population_mass_above_observed_max_plus_point2'] <= .003,
                       'original_model': r})
    return fit, catalog, fields


def draw_population(catalog, weights, n, seed):
    """Preserve the original family-first RNG ordering exactly."""
    weights = np.asarray(weights, dtype=float)
    rng = np.random.default_rng(seed)
    families = rng.choice(len(catalog), n, p=weights/weights.sum())
    rows = np.empty((n, 5), dtype=np.uint8)
    for j, (x, w) in enumerate(catalog.values()):
        ids = np.flatnonzero(families == j)
        rows[ids] = x[rng.choice(len(x), len(ids), p=w/w.sum())]
    return rows


def sample_actions(catalog, n, seed):
    rng = np.random.default_rng(seed)
    rows, families = [], []
    for j, (x, w) in enumerate(catalog.values()):
        rows.extend(x[rng.choice(len(x), n, p=w/w.sum())])
        families.extend([j]*n)
    return np.asarray(rows, dtype=np.uint8), np.asarray(families)


def native_module(folder):
    spec = importlib.util.spec_from_file_location('parry_frozen_arena', Path(folder)/'arena.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def uncompressed_groups(rows):
    """Retain group order, allowing exact cell-by-cell engine checks."""
    maxima = rows.max(axis=1)
    masks = ((rows == maxima[:, None, :])*(1 << np.arange(3))[None, :, None]).sum(axis=1)
    codes = ((masks-1)*(7**np.arange(5))).sum(axis=1)
    return (np.ascontiguousarray(maxima, dtype=np.uint8), np.ascontiguousarray(codes, dtype=np.int32), np.ones(len(rows))/len(rows))


def counts_from_groups(groups, n):
    counts = np.rint(groups[2]*n).astype(np.uint64)
    if int(counts.sum()) != n or np.max(abs(groups[2]-counts/n)) > 1e-10:
        raise ValueError('Compression masses do not recover integer counts')
    return counts


def exact_sample_sums(native, actions, groups, n):
    pay = native.payoff_matrix(actions, groups)
    if np.any(pay % 3):
        raise ValueError('Rank payout is not an integer number of quarters')
    counts = counts_from_groups(groups, n)
    return np.einsum('ij,j->i', pay, counts, dtype=np.uint64)//3


def summary(values, fields, actions):
    """Maximin and regret are separate, declared sensitivity rules."""
    values = np.asarray(values)
    strict = np.asarray([r['strict'] for r in fields])
    result = {}
    for name, mask in [('all_17', np.ones(len(fields), dtype=bool)), ('strict_9', strict)]:
        v = values[:, mask]
        floor = v.min(axis=1)
        regret = (v.max(axis=0)-v).max(axis=1)
        maximin = min(range(len(actions)), key=lambda j: (-floor[j], tuple(actions[j])))
        minimax_regret = min(range(len(actions)), key=lambda j: (regret[j], tuple(actions[j])))
        result[name] = {'maximin_index': maximin, 'minimax_regret_index': minimax_regret,
                        'minimum': floor.tolist(), 'maximum_regret': regret.tolist()}
    return result


def fixed_pool(our_freeze, final_audit, memory):
    pool, seen = [], {}
    def add(x, source):
        x = validate_allocation(tuple(int(v) for v in x))
        if x not in seen:
            seen[x] = len(pool)
            pool.append({'allocation': list(x), 'sources': []})
        pool[seen[x]]['sources'].append(source)
    for key, row in our_freeze['pool'].items():
        add(row['allocation'], 'our_handoff_'+key)
    for i, row in enumerate(final_audit['candidates']):
        add(row['allocation'], 'parry_final_'+str(i))
    for label in ['pool', 'memory_actions']:
        for i, x in enumerate(memory[label]):
            add(x, 'parry_checkpoint_'+label+'_'+str(i))
    add(memory['global_best_response'], 'parry_checkpoint_best_response')
    return pool
