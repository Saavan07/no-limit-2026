"""Feature-based historical hypotheses, not transformed historical submissions.

The original game rewards match wins on 10/13 fields. This module estimates
features and explicitly changes field count under alternative transfer assumptions.
It does not infer the distribution of Sequoia applicants.
"""

from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
import csv
import hashlib
import math

import numpy as np

from .core import DEFAULT_GAME, ScoreTable, table_from_histograms


SOURCE_NAMES = {
    1: "castle-solutions.csv", 2: "castle-solutions-2.csv",
    3: "castle-solutions-3.csv", 4: "castle-solutions-4.csv",
    5: "castle-solutions-5.csv",
}
EXPECTED_BLOBS = {
    1: "7902cbb4475f920e1ed099c88deee67194faa5a4",
    2: "080b4d0e9e1f97b8c3cd4377896642bd922b7d18",
    3: "60df9408219cd26de2e3ee4a70d6897a32c6484c",
    4: "3a6850cf4a95dd8979dc08065a92d94bd56ffb4b",
    5: "30beae934d490444233b47c175d8e8a9a2b8f243",
}


def load_round(folder, number):
    path = Path(folder)/SOURCE_NAMES[number]
    raw = path.read_bytes()
    blob = hashlib.sha1(b"blob "+str(len(raw)).encode()+b"\0"+raw).hexdigest()
    if blob != EXPECTED_BLOBS[number]:
        raise ValueError(f"Historical file differs from pinned snapshot: {path.name}")
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = next(reader)
        columns = [i for i,name in enumerate(header) if name.startswith("Castle ")]
        values = [[int(row[i]) for i in columns] for row in reader if row]
    array = np.asarray(values, dtype=np.int64)
    validate_rows(array, budget=100, castles=13 if number==5 else 10)
    return array, {"round": number, "file":path.name,"git_blob_sha1":blob,
                   "sha256":hashlib.sha256(raw).hexdigest(),"records":len(array),
                   "castles":array.shape[1]}


def validate_rows(array, budget=100, castles=5):
    if array.ndim != 2 or array.shape[1] != castles or len(array)==0:
        raise ValueError("Invalid allocation array shape")
    if not np.issubdtype(array.dtype,np.integer) or np.any(array<0):
        raise ValueError("Allocation arrays must contain nonnegative integers")
    if not np.all(array.sum(axis=1)==budget):
        raise ValueError("Allocation array violates exact budget")


def fit_features(array):
    budget = int(array[0].sum())
    validate_rows(array,budget,array.shape[1])
    shares = array.astype(np.float64)/budget
    mean = shares.mean(axis=0)
    if np.any(mean<=0):
        raise ValueError("Mean profile has empty fields; this fit is not identified")
    n = array.shape[1]
    rank = (np.arange(n)+0.5)/n
    centered = np.log(rank)-np.log(rank).mean()
    beta = float(np.dot(centered,np.log(mean)-np.log(mean).mean())/np.dot(centered,centered))
    concentration = float(np.square(shares).sum(axis=1).mean())
    squared_mean = float(np.dot(mean,mean))
    ratio = (concentration-squared_mean)/(1-squared_mean)
    if ratio <= 1/budget:
        alpha, projection = 10000.0, "underdispersed relative to multinomial; concentration capped"
    else:
        raw_alpha = (1-ratio)/(ratio-1/budget)
        alpha = float(np.clip(raw_alpha,0.02,10000))
        projection = "none" if alpha==raw_alpha else "concentration parameter capped"
    return {"source_castles":n,"source_records":len(array),"budget":budget,
            "mean_shares":mean.tolist(),"power_exponent":beta,
            "mean_squared_share_sum":concentration,"dirichlet_total_alpha":alpha,
            "moment_fit_projection":projection,
            "whole_vector_multiple_of_five_rate":float(np.all(array%5==0,axis=1).mean()),
            "any_zero_rate":float(np.any(array==0,axis=1).mean())}


def transfer_parameters(features, profile="power", concentration="fixed_total"):
    n = features['source_castles']
    target_rank = (np.arange(5)+0.5)/5
    if profile == "power":
        logs = features['power_exponent']*np.log(target_rank)
        probabilities = np.exp(logs-logs.max())
    elif profile == "rank_curve":
        source_rank = (np.arange(n)+0.5)/n
        probabilities = np.interp(target_rank,source_rank,np.asarray(features['mean_shares']))
    else:
        raise ValueError("Unknown attention-profile transfer")
    probabilities = probabilities/probabilities.sum()
    if concentration == "fixed_total":
        alpha = features['dirichlet_total_alpha']
    elif concentration == "fixed_per_field":
        alpha = features['dirichlet_total_alpha']*5/n
    else:
        raise ValueError("Unknown concentration transfer")
    return {"profile":profile,"concentration_transfer":concentration,
            "probabilities":probabilities.tolist(),"total_alpha":float(alpha),
            "rounding_probability_proxy":features['whole_vector_multiple_of_five_rate'],
            "interpretation":"Uncertain feature transfer, not a fit to Sequoia submissions"}


def round_rows_to_grid(rows, rng, unit=5):
    budget = int(rows[0].sum())
    validate_rows(rows,budget,rows.shape[1])
    if budget%unit:
        raise ValueError("Grid must divide budget")
    lower = rows//unit
    remainder = rows%unit
    missing = budget//unit-lower.sum(axis=1)
    # Small continuous keys only break equal integer remainders; no fixed-label bias.
    order = np.argsort(-(remainder+rng.random(rows.shape)*0.01),axis=1)
    for j in range(rows.shape[1]):
        selected = np.flatnonzero(missing>j)
        lower[selected,order[selected,j]] += 1
    result = lower*unit
    validate_rows(result,budget,rows.shape[1])
    return result


def sample_transferred(parameters, size, seed):
    rng = np.random.Generator(np.random.PCG64(seed))
    alpha = parameters['total_alpha']*np.asarray(parameters['probabilities'])
    shares = rng.dirichlet(alpha,size=size)
    rows = rng.multinomial(100,shares)
    mask = rng.random(size)<parameters['rounding_probability_proxy']
    if np.any(mask):
        rows[mask] = round_rows_to_grid(rows[mask],rng)
    validate_rows(rows)
    return rows


def sample_synthetic(family, size, seed):
    rng = np.random.Generator(np.random.PCG64(seed))
    if family in ('equal_jitter','value_jitter'):
        anchor = (20,20,20,20,20) if family=='equal_jitter' else (7,13,20,27,33)
        rows = np.tile(anchor,(size,1)).astype(np.int64)
        indices = np.arange(size)
        for _ in range(4):
            donor = rng.integers(0,5,size=size)
            receiver = rng.integers(0,4,size=size)
            receiver += receiver>=donor
            amount = rng.integers(0,np.minimum(6,rows[indices,donor])+1)
            rows[indices,donor] -= amount
            rows[indices,receiver] += amount
    elif family == 'sparse':
        rows = np.zeros((size,5),dtype=np.int64)
        supports = rng.integers(2,5,size=size)
        for k in (2,3,4):
            selected = np.flatnonzero(supports==k)
            fields = np.argsort(rng.random((len(selected),5)),axis=1)[:,:k]
            values = rng.multinomial(100-k,rng.dirichlet(np.ones(k),size=len(selected)))+1
            rows[selected[:,None],fields] = values
    elif family == 'middle_focus':
        rows = np.zeros((size,5),dtype=np.int64)
        rows[:,0] = rng.integers(0,4,size=size)
        rows[:,4] = rng.integers(0,4,size=size)
        rows[:,1:4] = rng.multinomial(100-rows[:,0]-rows[:,4],rng.dirichlet(np.ones(3),size=size))
    elif family == 'top_focus':
        rows = np.zeros((size,5),dtype=np.int64)
        rows[:,:3] = rng.integers(0,4,size=(size,3))
        rows[:,3:] = rng.multinomial(100-rows[:,:3].sum(axis=1),rng.dirichlet([.6,.6],size=size))
    elif family == 'one_up_grid':
        rows = rng.multinomial(20,rng.dirichlet(np.ones(5),size=size))*5
        rows[:,4] += rows[:,0]
        rows[:,0] = 0
        rows[:,1:5] += 1
        rows[:,4] -= 4
        # If the last field cannot fund all four scouts, transfer from its richest rival.
        negative = np.flatnonzero(rows[:,4]<0)
        for index in negative:
            deficit = -int(rows[index,4])
            donor = int(np.argmax(rows[index,:4]))
            rows[index,donor] -= deficit
            rows[index,4] += deficit
    else:
        raise ValueError("Unknown synthetic family")
    validate_rows(rows)
    return rows


def rows_to_table(rows, name):
    validate_rows(rows)
    hist = [np.bincount(rows[:,i],minlength=101).tolist() for i in range(5)]
    return table_from_histograms(hist,len(rows),DEFAULT_GAME,name,
        "exact conditional on saved sampled marginal counts; generator is a hypothesis")


def sampled_metadata(rows, seed):
    return {"sample_size":len(rows),"seed":seed,"bit_generator":"PCG64",
            "draw_sha256":hashlib.sha256(rows.astype('<i2').tobytes()).hexdigest()}
