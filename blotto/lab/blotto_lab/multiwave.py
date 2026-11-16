"""Unfitted legal reasoning hypotheses and retained-history response experiments."""
from collections import Counter
from fractions import Fraction
from itertools import combinations
from pathlib import Path
from random import Random
import hashlib
import json

import numpy as np

from .core import DEFAULT_GAME, BestResponse, Population, ScoreTable, mix_tables, security_of, validate_allocation
from .historical import round_rows_to_grid, validate_rows

VALUES = np.arange(1,6)


def hypothesis_rows(family,size,seed):
    if type(size) is not int or size<=0:
        raise ValueError('Sample size must be a positive integer')
    rng = np.random.default_rng(seed)
    rows = np.zeros((size,5),dtype=np.int64)
    if family=='subset_value':
        subsets = [s for k in (2,3,4) for s in combinations(range(5),k) if sum(i+1 for i in s)>=8]
        weights = np.array([sum(i+1 for i in s)**2 for s in subsets],dtype=float)
        choices = rng.choice(len(subsets),size=size,p=weights/weights.sum())
        for j,subset in enumerate(subsets):
            selected = np.flatnonzero(choices==j)
            p = VALUES[list(subset)].astype(float)
            draws = rng.multinomial(100-len(subset),rng.dirichlet(12*p/p.sum(),size=len(selected)))+1
            rows[selected[:,None],list(subset)] = draws
    elif family=='cheap_scouts':
        rows[:,:2] = rng.integers(0,5,size=(size,2))
        rows[:,2:] = rng.multinomial(100-rows[:,:2].sum(axis=1),rng.dirichlet([4.5,6,7.5],size=size))
    elif family=='fortress_five':
        rows[:,:4] = rng.integers(0,9,size=(size,4))
        rows[:,4] = 100-rows[:,:4].sum(axis=1)
    elif family=='skip_one':
        weights = 1/VALUES
        omitted = rng.choice(5,size=size,p=weights/weights.sum())
        for j in range(5):
            selected = np.flatnonzero(omitted==j)
            fields = [k for k in range(5) if k!=j]
            p = VALUES[fields].astype(float)
            rows[selected[:,None],fields] = rng.multinomial(96,rng.dirichlet(30*p/p.sum(),size=len(selected)))+1
    elif family=='round_tens':
        rows = rng.multinomial(10,rng.dirichlet(8*VALUES/15,size=size))*10
    elif family=='anchor_snipers':
        anchors = [(20,20,20,20,20),(7,13,20,27,33),(0,12,23,30,35),(0,0,26,34,40)]
        templates = [(anchor,s) for anchor in anchors for k in (3,4)
                     for s in combinations(range(5),k)
                     if sum(i+1 for i in s)>=8 and sum(anchor[i]+1 for i in s)<=100]
        choices = rng.integers(0,len(templates),size=size)
        for j,(anchor,fields) in enumerate(templates):
            selected = np.flatnonzero(choices==j)
            bids = np.array([anchor[k]+1 for k in fields])
            p = VALUES[list(fields)].astype(float)
            rows[selected[:,None],list(fields)] = bids+rng.multinomial(100-int(bids.sum()),p/p.sum(),size=len(selected))
    elif family=='heterogeneous_power':
        beta = rng.uniform(-.25,2,size=size)
        p = VALUES[None,:]**beta[:,None]
        p /= p.sum(axis=1,keepdims=True)
        concentration = np.exp(rng.uniform(np.log(1),np.log(50),size=size))
        gamma = rng.gamma(concentration[:,None]*p)
        rows = rng.multinomial(100,gamma/gamma.sum(axis=1,keepdims=True))
    elif family=='monotone_ladder':
        rows = np.sort(rng.multinomial(100,rng.dirichlet(np.ones(5),size=size)),axis=1)
        for _ in range(2):
            indices = np.arange(size)
            donor = rng.integers(0,5,size=size)
            receiver = rng.integers(0,4,size=size)
            receiver += receiver>=donor
            amount = rng.integers(0,np.minimum(2,rows[indices,donor])+1)
            rows[indices,donor] -= amount
            rows[indices,receiver] += amount
    elif family=='inverse_value':
        p = VALUES.astype(float)**-1.3
        rows = rng.multinomial(100,rng.dirichlet(9*p/p.sum(),size=size))
    elif family=='pair_commitment':
        pairs = list(combinations(range(5),2))
        choices = rng.integers(0,len(pairs),size=size)
        for j,pair in enumerate(pairs):
            selected = np.flatnonzero(choices==j)
            other = [k for k in range(5) if k not in pair]
            rows[selected[:,None],other] = rng.integers(0,3,size=(len(selected),3))
            remaining = 100-rows[selected].sum(axis=1)
            rows[selected[:,None],list(pair)] = rng.multinomial(remaining,rng.dirichlet([.35,.35],size=len(selected)))
    elif family=='grid_three':
        rows = rng.multinomial(32,rng.dirichlet(np.ones(5),size=size))*3
        fields = np.argsort(rng.random((size,5)),axis=1)[:,:4]
        rows[np.arange(size)[:,None],fields] += 1
    elif family=='rotating_templates':
        templates = [(0,0,26,34,40),(0,21,26,26,27),(4,4,4,8,80)]
        choices = rng.integers(0,15,size=size)
        pool = np.array([np.roll(x,k) for x in templates for k in range(5)])
        rows = pool[choices].copy()
        for _ in range(2):
            indices = np.arange(size)
            donor = rng.integers(0,5,size=size)
            receiver = rng.integers(0,4,size=size)
            receiver += receiver>=donor
            amount = rng.integers(0,np.minimum(3,rows[indices,donor])+1)
            rows[indices,donor] -= amount
            rows[indices,receiver] += amount
    elif family=='budget_blocks':
        lower = rng.choice([20,80],size=size)+rng.integers(-8,9,size=size)
        rows[:,:3] = rng.multinomial(lower,rng.dirichlet(20*np.array([1,2,3])/6,size=size))
        rows[:,3:] = rng.multinomial(100-lower,rng.dirichlet(10*np.array([4,5])/9,size=size))
    elif family=='even_thresholds':
        rows = rng.multinomial(100,rng.dirichlet(VALUES.astype(float),size=size))
        rows = round_rows_to_grid(rows,rng,unit=2)
        selected = np.flatnonzero(rng.random(size)<.5)
        donor = np.argmax(rows[selected],axis=1)
        receiver = rng.integers(0,4,size=len(selected))
        receiver += receiver>=donor
        rows[selected,donor] -= 1
        rows[selected,receiver] += 1
    else:
        raise ValueError('Unknown reasoning hypothesis')
    validate_rows(rows)
    return rows


def uniform_best_response(table,seed):
    """An exact global response, uniformly sampled from all tied maximizers."""
    game = table.game
    n,b = game.castles,game.budget
    dp = [[None]*(b+1) for _ in range(n+1)]
    ways = [[0]*(b+1) for _ in range(n+1)]
    dp[n][0],ways[n][0] = 0,1
    evaluations = 0
    for i in range(n-1,-1,-1):
        row = table.numerators[i]
        for left in range(b+1):
            maximum,count = None,0
            for k in range(left+1):
                evaluations += 1
                future = dp[i+1][left-k]
                if future is None:
                    continue
                value = row[k]+future
                if maximum is None or value>maximum:
                    maximum,count = value,ways[i+1][left-k]
                elif value==maximum:
                    count += ways[i+1][left-k]
            dp[i][left],ways[i][left] = maximum,count
    rng = Random(seed)
    allocation,left = [],b
    for i in range(n):
        draw = rng.randrange(ways[i][left])
        for k in range(left+1):
            future = dp[i+1][left-k]
            if future is not None and table.numerators[i][k]+future==dp[i][left]:
                count = ways[i+1][left-k]
                if draw<count:
                    allocation.append(k)
                    left -= k
                    break
                draw -= count
    return BestResponse(tuple(allocation),Fraction(dp[0][b],table.denominator),ways[0][b],
                        evaluations,'exact DP; uniform among tied global optima')


def mutate(action,seed,steps=2,max_transfer=8,game=DEFAULT_GAME):
    action = list(validate_allocation(action,game))
    if steps<0 or max_transfer<=0:
        raise ValueError('Invalid mutation settings')
    if game.castles<2 or game.budget==0:
        return tuple(action)
    rng = Random(seed)
    for _ in range(steps):
        donor = rng.choice([i for i,k in enumerate(action) if k])
        receiver = rng.randrange(game.castles-1)
        receiver += receiver>=donor
        amount = rng.randint(1,min(max_transfer,action[donor]))
        action[donor] -= amount
        action[receiver] += amount
    return validate_allocation(action,game)


def table_digest(table):
    payload = [table.game.budget,table.game.values,table.denominator,table.numerators]
    return hashlib.sha256(json.dumps(payload,separators=(',',':')).encode()).hexdigest()


def fraction_json(x):
    return {'numerator':x.numerator,'denominator':x.denominator,'decimal':float(x)}


def simple_json(x):
    from dataclasses import asdict,is_dataclass
    if isinstance(x,Fraction):
        return fraction_json(x)
    if is_dataclass(x):
        return simple_json(asdict(x))
    if isinstance(x,dict):
        return {str(k):simple_json(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):
        return [simple_json(v) for v in x]
    return x


def atomic_json(path,obj):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(simple_json(obj),sort_keys=True,indent=2)+'\n')
    temporary.replace(path)


def run_path(folder,policy,seed_population,environments,cfg,seed,callback=None,max_wave=None):
    """Checkpointed complete waves, with identical exogenous seeds across policies.

    environments contains (first_wave, group_tables, specialist_tables). A regime
    change updates targets, not the seed support or candidate shortlist.
    """
    if policy not in ('latest_only','retain_history'):
        raise ValueError('Unknown memory policy')
    folder = Path(folder)
    folder.mkdir(parents=True,exist_ok=True)
    game = seed_population.game
    module_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    core_hash = hashlib.sha256((Path(__file__).parent/'core.py').read_bytes()).hexdigest()
    signature = {'policy':policy,'seed':seed,'seed_population':seed_population.to_dict(),
        'settings':{k:cfg[k] for k in ['responses_per_wave','mutation_probability','mutation_steps',
            'mutation_max_transfer','strategic_share_percentages','belief_concentrations']},
        'environments':[(start,[table_digest(t) for t in groups],[table_digest(t) for t in specialists])
                        for start,groups,specialists in environments],
        'module_sha256':module_hash,'core_sha256':core_hash}
    digest = hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest()
    checkpoint = folder/'checkpoint.json'
    if checkpoint.exists():
        state = json.loads(checkpoint.read_text())
        if state['input_digest']!=digest:
            raise ValueError('Arena checkpoint input or source mismatch')
        archive = Population.from_dict(state['archive'])
        latest = Population.from_dict(state['latest'])
        completed = state['wave']
        rebuilt = Counter(dict(seed_population.entries))
        for previous_wave in range(1,completed+1):
            previous = json.loads((folder/f'wave-{previous_wave:02}.json').read_text())
            if previous['input_digest']!=digest or previous['wave']!=previous_wave:
                raise ValueError('Arena history mismatch')
            generation = Population.from_dict(previous['generation'])
            rebuilt.update(dict(generation.entries))
            if Population.from_counts(rebuilt,game).entries!=Population.from_dict(previous['archive']).entries:
                raise ValueError('Arena archive mass mismatch')
        if completed and (latest.entries!=generation.entries or archive.entries!=Population.from_counts(rebuilt,game).entries):
            raise ValueError('Arena checkpoint does not match saved history')
    else:
        archive = latest = seed_population
        completed = 0
    target = cfg['waves'] if max_wave is None else max_wave
    if target<completed or target>cfg['waves']:
        raise ValueError('Invalid wave target')
    for wave in range(completed+1,target+1):
        environment = max((env for env in environments if env[0]<=wave),key=lambda e:e[0])
        _,groups,specialists = environment
        memory = latest if policy=='latest_only' else archive
        memory_table = memory.table()
        rng = np.random.default_rng(seed+wave*10000)
        entries,episodes = [],[]
        for j in range(cfg['responses_per_wave']):
            tie_seed = int(rng.integers(0,2**62))
            mutation_seed = int(rng.integers(0,2**62))
            add_mutation = bool(rng.random()<cfg['mutation_probability'])
            if j<len(specialists):
                target_table = specialists[j]
                belief = {'kind':'specialist','index':j}
            else:
                concentration = float(rng.choice(cfg['belief_concentrations']))
                weights = rng.multinomial(1000,rng.dirichlet(np.full(len(groups),concentration))).tolist()
                strategic = int(rng.choice(cfg['strategic_share_percentages']))
                mixed_weights = [w*(100-strategic) for w in weights]+[strategic*1000]
                target_table = mix_tables(list(groups)+[memory_table],mixed_weights,name='arena belief')
                belief = {'kind':'group_and_memory','group_integer_weights':weights,
                          'strategic_share_percentage':strategic,'concentration':concentration}
            response = uniform_best_response(target_table,tie_seed)
            entries.append(response.allocation)
            episode = {'belief':belief,'response':response,'tie_seed':tie_seed,
                       'mutation_seed':mutation_seed,'mutation_added':add_mutation}
            if add_mutation:
                child = mutate(response.allocation,mutation_seed,cfg['mutation_steps'],
                               cfg['mutation_max_transfer'],game)
                entries.append(child)
                episode.update({'mutation':child,'mutation_target_score':target_table.expected_score(child)})
            episodes.append(episode)
        latest = Population.from_people(entries,game,name=f'{policy} wave {wave}')
        counts = Counter(dict(archive.entries))
        counts.update(entries)
        archive = Population.from_counts(counts,game,name=f'{policy} full shadow history {wave}')
        gen_security,archive_security = security_of(latest),security_of(archive)
        record = {'wave':wave,'policy':policy,'environment_start_wave':environment[0],
            'memory_mass_before':memory.total_mass,'generation':latest.to_dict(),
            'archive':archive.to_dict(),'generation_security':gen_security,
            'archive_security':archive_security,'episodes':episodes,'input_digest':digest}
        atomic_json(folder/f'wave-{wave:02}.json',record)
        state = {'wave':wave,'input_digest':digest,'archive':archive.to_dict(),'latest':latest.to_dict()}
        atomic_json(checkpoint,state)
        if callback:
            callback(wave,latest,archive,gen_security,archive_security)
    return [json.loads((folder/f'wave-{wave:02}.json').read_text()) for wave in range(1,target+1)]
