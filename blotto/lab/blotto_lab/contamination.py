"""Exact pure counters and global nominal-performance/security frontiers."""
from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations, islice, product
import math
import numpy as np
from .core import DEFAULT_GAME, best_response, iter_allocations, validate_allocation
from .robust import float_tables


@dataclass(frozen=True)
class FrontierObjective:
    name: str
    indices: tuple[int,...]
    mode: str = 'score'


def loss_sets(game):
    """Sets costing more than the all-tie score cannot improve our minimum."""
    return [s for k in range(1,game.castles) for s in combinations(range(game.castles),k)
            if 2*sum(game.values[i] for i in s)<=game.total_value]


def security_twice(actions,game=DEFAULT_GAME):
    original = np.asarray(actions)
    if not np.issubdtype(original.dtype,np.integer):
        raise ValueError('Security calculation requires integer allocations')
    rows = np.asarray(actions,dtype=np.int64)
    if rows.ndim!=2 or rows.shape[1]!=game.castles or np.any(rows<0) or not np.all(rows.sum(axis=1)==game.budget):
        raise ValueError('Invalid legal action batch')
    security = np.full(len(rows),game.total_value,dtype=np.int64)
    for lost in loss_sets(game):
        remaining = [i for i in range(game.castles) if i not in lost]
        values = sorted(game.values[i] for i in remaining)
        # Tied fields are the cheapest remaining fields after the opponent wins
        # the most valuable fields affordable from the released troop budget.
        tied_score = np.array([sum(values[:max(len(values)-k,0)]) for k in range(game.castles+1)])
        released = rows[:,lost].sum(axis=1)
        valid = (rows[:,lost]>0).all(axis=1)
        candidate = 2*sum(game.values[i] for i in lost)+tied_score[np.minimum(released,game.castles)]
        security[valid] = np.minimum(security[valid],candidate[valid])
    return security


def exact_counter(action,game=DEFAULT_GAME):
    x = validate_allocation(action,game)
    best,chosen = game.total_value,None
    for lost in loss_sets(game):
        if any(x[i]==0 for i in lost):
            continue
        remaining = sorted((i for i in range(game.castles) if i not in lost),key=lambda i:(-game.values[i],i))
        winners = remaining[:min(sum(x[i] for i in lost),len(remaining))]
        tied = [i for i in remaining if i not in winners]
        value = 2*sum(game.values[i] for i in lost)+sum(game.values[i] for i in tied)
        if value<best:
            best,chosen = value,(lost,winners)
    if chosen is None:
        return x,Fraction(best,2)
    lost,winners = chosen
    y = [0 if i in lost else x[i] for i in range(game.castles)]
    for i in winners:
        y[i] += 1
    y[winners[0]] += game.budget-sum(y)
    return validate_allocation(y,game),Fraction(best,2)


def pattern_security_twice(actions,game=DEFAULT_GAME):
    """Independent loss/tie/win outcome enumeration, with first-feasible screening."""
    rows = np.asarray(actions,dtype=np.int64)
    result = np.full(len(rows),game.total_value,dtype=np.int64)
    unresolved = np.arange(len(rows))
    patterns = []
    for pattern in product((0,1,2),repeat=game.castles):
        # Opponent state: lose=0, tie=1, win=2. Patterns with no loss and some
        # wins exceed the common budget. All-tie initializes the fallback.
        lost = [i for i,k in enumerate(pattern) if k==0]
        if not lost:
            continue
        opponent_score = sum(game.values[i]*k for i,k in enumerate(pattern))
        if opponent_score<=game.total_value:
            continue
        active = [i for i,k in enumerate(pattern) if k]
        wins = pattern.count(2)
        patterns.append((opponent_score,lost,active,wins))
    for opponent_score,lost,active,wins in sorted(patterns,key=lambda p:-p[0]):
        if not len(unresolved):
            break
        selected = rows[unresolved]
        valid = (selected[:,lost]>0).all(axis=1)
        cost = selected[:,active].sum(axis=1)+wins
        feasible = valid & (cost<=game.budget)
        result[unresolved[feasible]] = 2*game.total_value-opponent_score
        unresolved = unresolved[~feasible]
    return result


def objective_value(x,tables,obj,optima):
    values = [tables[i].expected_score(x) for i in obj.indices]
    if obj.mode=='regret':
        values = [v-optima[i] for v,i in zip(values,obj.indices)]
    return min(values)


def search_frontiers(tables,objectives,batch_size=50000,callback=None):
    tables = tuple(tables)
    game = tables[0].game
    if game.total_value>255 or type(batch_size) is not int or batch_size<=0:
        raise ValueError('Unsupported security encoding or batch size')
    floating,errors = float_tables(tables)
    optima = [best_response(t).expected_score for t in tables]
    optimum_floats = np.array([float(x) for x in optima])
    enclosures = []
    for obj in objectives:
        if not obj.indices or obj.mode not in ('score','regret') or any(i<0 or i>=len(tables) for i in obj.indices):
            raise ValueError('Invalid objective')
        bound = max(errors[i] for i in obj.indices)
        if obj.mode=='regret':
            bound = max(errors[i]+abs(Fraction.from_float(float(optima[i]))-optima[i]) for i in obj.indices)
            bound += Fraction.from_float(math.ulp(float(2*game.total_value)))
        enclosures.append(math.nextafter(float(bound),math.inf))
    best,kept = [{} for _ in objectives],[{} for _ in objectives]
    counts,first = {},{}
    security_chunks = []
    iterator,done = iter_allocations(game),0
    while True:
        batch = list(islice(iterator,batch_size))
        if not batch:
            break
        actions = np.asarray(batch,dtype=np.uint8)
        security = security_twice(actions,game)
        security_chunks.append(security.astype(np.uint8))
        classes = np.unique(security)
        positions = {int(c):np.flatnonzero(security==c) for c in classes}
        for c,indices in positions.items():
            counts[c] = counts.get(c,0)+len(indices)
            first.setdefault(c,tuple(map(int,actions[indices[0]])))
        scores = np.zeros((len(tables),len(batch)),dtype=np.float64)
        for i in range(game.castles):
            scores += floating[:,i,actions[:,i]]
        for j,obj in enumerate(objectives):
            selected = scores[list(obj.indices)]
            if obj.mode=='regret':
                selected = selected-optimum_floats[list(obj.indices),None]
            values = selected.min(axis=0)
            for c,indices in positions.items():
                current = values[indices]
                maximum = max(best[j].get(c,-math.inf),float(current.max()))
                best[j][c] = maximum
                threshold = math.nextafter(maximum-2*enclosures[j],-math.inf)
                survivors = []
                for old_actions,old_values in kept[j].get(c,[]):
                    mask = old_values>=threshold
                    if mask.any():
                        survivors.append((old_actions[mask],old_values[mask]))
                mask = current>=threshold
                if mask.any():
                    survivors.append((actions[indices[mask]].copy(),current[mask].copy()))
                kept[j][c] = survivors
        done += len(batch)
        if callback:
            callback(done,game.actions)
    frontiers = {}
    for j,obj in enumerate(objectives):
        entries = {}
        for c in sorted(counts):
            maximum,winner,ties,contenders = None,None,0,0
            for actions,_ in kept[j][c]:
                for row in actions:
                    x = tuple(map(int,row))
                    value = objective_value(x,tables,obj,optima)
                    contenders += 1
                    if maximum is None or value>maximum:
                        maximum,winner,ties = value,x,1
                    elif value==maximum:
                        ties += 1
                        winner = min(winner,x)
            entries[c] = {'security':Fraction(c,2),'allocation':winner,'nominal_value':maximum,
                'optimum_count':ties,'exact_contenders':contenders,'float_error_bound':enclosures[j]}
        frontiers[obj.name] = entries
    return {'frontiers':frontiers,'class_counts':counts,'class_first':first,'evaluated':done,
            'security_twice':np.concatenate(security_chunks)}


def select_epsilon(frontier,epsilon,class_counts,class_first):
    e = Fraction(epsilon)
    if not 0<=e<=1:
        raise ValueError('Epsilon must be between zero and one')
    if e==1:
        c = max(class_counts)
        return {'allocation':class_first[c],'value':Fraction(c,2),'security':Fraction(c,2),
                'optimal_security_classes':[c],'optimum_count':class_counts[c]}
    values = {c:(1-e)*p['nominal_value']+e*p['security'] for c,p in frontier.items()}
    maximum = max(values.values())
    winners = [c for c,v in values.items() if v==maximum]
    c = min(winners,key=lambda c:frontier[c]['allocation'])
    return {'allocation':frontier[c]['allocation'],'value':maximum,'security':frontier[c]['security'],
            'optimal_security_classes':winners,'optimum_count':sum(frontier[k]['optimum_count'] for k in winners)}


def continuous_envelope(frontier,class_counts,class_first):
    points = {Fraction(0),Fraction(1)}
    for a,b in combinations(frontier.values(),2):
        numerator = b['nominal_value']-a['nominal_value']
        denominator = (a['security']-a['nominal_value'])-(b['security']-b['nominal_value'])
        if denominator:
            crossing = numerator/denominator
            if 0<crossing<1:
                points.add(crossing)
    points = sorted(points)
    segments = []
    for lo,hi in zip(points,points[1:]):
        winner = select_epsilon(frontier,(lo+hi)/2,class_counts,class_first)
        c = winner['optimal_security_classes'][0]
        if segments and segments[-1]['security_class']==c:
            segments[-1]['upper_epsilon'] = hi
        else:
            segments.append({'lower_epsilon':lo,'upper_epsilon':hi,'security_class':c,
                             'allocation':winner['allocation'],'security':winner['security'],
                             'nominal_value':frontier[c]['nominal_value']})
    boundaries = sorted({s['lower_epsilon'] for s in segments}|{Fraction(1)})
    return {'segments':segments,'boundary_ties':[{'epsilon':e,**select_epsilon(frontier,e,class_counts,class_first)} for e in boundaries]}
