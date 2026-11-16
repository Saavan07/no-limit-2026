"""Full-action robust selection with bounded float screening and exact finalists.

Every rational table entry's conversion error is measured exactly. Each score uses
at most n additions of nonnegative values bounded by V. n*ulp(2V) bounds their
rounding error. min/max do not magnify the largest absolute error. Retaining all
actions within twice the resulting uniform bound of the best floating objective
therefore retains every exact maximizer. Retained actions are ranked as Fractions.
"""

from dataclasses import dataclass
from fractions import Fraction
from itertools import islice
import math

import numpy as np

from .core import ScoreTable, best_response, iter_allocations


@dataclass(frozen=True)
class Objective:
    name: str
    model_indices: tuple[int,...]
    mode: str = "max_min"


def float_tables(tables):
    game = tables[0].game
    if game.total_value>=2**40 or game.budget>255:
        raise ValueError("This bounded-screen implementation requires small values and bids <=255")
    if any(t.game != game for t in tables):
        raise ValueError("Tables use different games")
    array = np.empty((len(tables),game.castles,game.budget+1),dtype=np.float64)
    errors = []
    arithmetic = game.castles*Fraction.from_float(math.ulp(float(2*game.total_value)))
    for s,table in enumerate(tables):
        error = Fraction(0)
        for i,row in enumerate(table.numerators):
            entry_errors = []
            for k,numerator in enumerate(row):
                value = Fraction(numerator,table.denominator)
                if not 0<=value<=game.values[i]:
                    raise ValueError("Screening bound requires legal nonnegative local scores")
                floating = float(value)
                array[s,i,k] = floating
                entry_errors.append(abs(Fraction.from_float(floating)-value))
            error += max(entry_errors)
        errors.append(error+arithmetic)
    return array,errors


def exact_objective(action,tables,indices,mode,optima):
    scores = [tables[i].expected_score(action) for i in indices]
    if mode=='max_min':
        return min(scores)
    if mode=='minimax_regret':
        return min(score-optima[i] for score,i in zip(scores,indices))
    raise ValueError("Unknown objective")


def search(tables,objectives,batch_size=100000,callback=None):
    tables = tuple(tables)
    if not tables or not objectives or type(batch_size) is not int or batch_size<=0:
        raise ValueError("Need tables, objectives, and a positive batch size")
    game = tables[0].game
    floats,errors = float_tables(tables)
    optima = [best_response(t).expected_score for t in tables]
    optimum_floats = np.array([float(x) for x in optima])
    objective_errors = []
    for obj in objectives:
        if not obj.model_indices or any(i<0 or i>=len(tables) for i in obj.model_indices):
            raise ValueError("Invalid model subset")
        if obj.mode not in ('max_min','minimax_regret'):
            raise ValueError("Unknown objective")
        bound = max(errors[i] for i in obj.model_indices)
        if obj.mode=='minimax_regret':
            bound = max(errors[i]+abs(Fraction.from_float(float(optima[i]))-optima[i])
                        for i in obj.model_indices)
            bound += Fraction.from_float(math.ulp(float(2*game.total_value)))
        upward = math.nextafter(float(bound),math.inf)
        assert Fraction.from_float(upward)>=bound
        objective_errors.append(upward)
    best = [-math.inf]*len(objectives)
    kept = [[] for _ in objectives]
    iterator = iter_allocations(game)
    evaluated = 0
    while True:
        batch = list(islice(iterator,batch_size))
        if not batch:
            break
        actions = np.asarray(batch,dtype=np.uint8)
        scores = np.zeros((len(tables),len(actions)),dtype=np.float64)
        for i in range(game.castles):
            scores += floats[:,i,actions[:,i]]
        for j,obj in enumerate(objectives):
            selected = scores[list(obj.model_indices)]
            if obj.mode=='minimax_regret':
                selected = selected-optimum_floats[list(obj.model_indices),None]
            approximate = selected.min(axis=0)
            best[j] = max(best[j],float(approximate.max()))
            threshold = math.nextafter(best[j]-2*objective_errors[j],-math.inf)
            retained = []
            for old_actions,old_scores in kept[j]:
                mask = old_scores>=threshold
                if np.any(mask):
                    retained.append((old_actions[mask],old_scores[mask]))
            mask = approximate>=threshold
            if np.any(mask):
                retained.append((actions[mask].copy(),approximate[mask].copy()))
            kept[j] = retained
        evaluated += len(actions)
        if callback:
            callback(evaluated,game.actions)
    results = {}
    for j,obj in enumerate(objectives):
        exact_best,winner,ties,contenders = None,None,0,0
        for candidates,_ in kept[j]:
            for row in candidates:
                action = tuple(int(k) for k in row)
                value = exact_objective(action,tables,obj.model_indices,obj.mode,optima)
                contenders += 1
                if exact_best is None or value>exact_best:
                    exact_best,winner,ties = value,action,1
                elif value==exact_best:
                    ties += 1
                    winner = min(winner,action)
        results[obj.name] = {"allocation":winner,"maximized_value":exact_best,
            "optimum_count":ties,"exact_contenders":contenders,"evaluated":evaluated,
            "uniform_float_error_bound":objective_errors[j],
            "mode":obj.mode,"model_indices":obj.model_indices}
    return results
