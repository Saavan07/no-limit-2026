"""Transparent response experiments. Arena frequencies are not applicant estimates."""

from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
import json
import hashlib

from .core import (
    DEFAULT_GAME, Game, Population, best_response, table_from_histograms,
    validate_allocation,
)


def fraction_record(value):
    return {"numerator": value.numerator, "denominator": value.denominator,
            "decimal": float(value)}


def implementation_fingerprint():
    folder = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for name in ("core.py", "arena.py"):
        digest.update(name.encode("ascii"))
        digest.update((folder/name).read_bytes())
    return digest.hexdigest()


@dataclass
class ResponseArchive:
    game: Game
    counts: Counter
    steps: int = 0
    last_added: tuple | None = None

    @classmethod
    def start(cls, action=(20,20,20,20,20), game=DEFAULT_GAME):
        action = validate_allocation(action, game)
        return cls(game, Counter({action: 1}), 0, action)

    def population(self):
        return Population.from_counts(self.counts, self.game, "empirical response history")

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {"format_version": 2, "method": "symmetric fictitious play",
                "implementation_sha256": implementation_fingerprint(),
                "tie_break": "lexicographically smallest exact best response",
                "steps": self.steps, "last_added": list(self.last_added),
                "population": self.population().to_dict()}
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(data, indent=2) + "\n")
        temporary.replace(path)

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text())
        if data.get("format_version") != 2 or data.get("method") != "symmetric fictitious play":
            raise ValueError("Unsupported checkpoint")
        if data.get("implementation_sha256") != implementation_fingerprint():
            raise ValueError("Checkpoint implementation differs from the current core or arena")
        if data.get("tie_break") != "lexicographically smallest exact best response":
            raise ValueError("Unsupported tie policy")
        population = Population.from_dict(data["population"])
        steps = data["steps"]
        if type(steps) is not int or steps < 0 or population.total_mass != steps+1:
            raise ValueError("Checkpoint steps and masses disagree")
        last = validate_allocation(data["last_added"], population.game)
        if last not in dict(population.entries):
            raise ValueError("Last action missing from archive")
        return cls(population.game, Counter(dict(population.entries)), steps, last)


def advance_history(state, target_steps, record_every=100, checkpoint=None,
                    checkpoint_every=250, callback=None):
    """Best respond to the complete empirical history; keep every observed action.

    Computation is exact, including the full-space security certificate. A fixed
    iteration count is not itself a convergence certificate. Resume depends only
    on integer counts and the deterministic tie rule, not an unrecorded RNG.
    """
    if type(target_steps) is not int or target_steps < state.steps:
        raise ValueError("Target cannot precede checkpoint")
    if record_every <= 0 or checkpoint_every <= 0:
        raise ValueError("Recording intervals must be positive")
    hist = state.population().histograms()
    total = state.steps+1
    trace = []
    value = Fraction(state.game.total_value, 2)
    while True:
        table = table_from_histograms(hist, total, state.game, "response history")
        response = best_response(table)
        if state.steps == target_steps or state.steps % record_every == 0 or state.steps < 10:
            gap = response.expected_score-value
            record = {"step": state.steps, "history_mass": total,
                      "archive_size": len(state.counts), "last_added": list(state.last_added),
                      "best_response": list(response.allocation),
                      "exploitability": fraction_record(gap),
                      "security": fraction_record(value-gap)}
            trace.append(record)
            if callback:
                callback(record)
        if state.steps == target_steps:
            break
        action = response.allocation
        state.counts[action] += 1
        state.last_added = action
        state.steps += 1
        total += 1
        for i, k in enumerate(action):
            hist[i][k] += 1
        if checkpoint and state.steps % checkpoint_every == 0:
            state.save(checkpoint)
    if checkpoint:
        state.save(checkpoint)
    return trace


def latest_only(initial=(20,20,20,20,20), max_steps=100):
    """A forgetting baseline: respond only to the current pure champion."""
    current = validate_allocation(initial)
    seen, trace = {}, []
    for step in range(max_steps):
        if current in seen:
            return {"trace": trace, "cycle_start": seen[current],
                    "cycle_length": step-seen[current], "repeated_action": list(current)}
        seen[current] = step
        response = best_response(Population.from_people([current]).table())
        trace.append({"step": step, "allocation": list(current),
                      "counter": list(response.allocation),
                      "counter_score": fraction_record(response.expected_score),
                      "current_security": fraction_record(Fraction(15)-response.expected_score)})
        current = response.allocation
    return {"trace": trace, "cycle_start": None, "cycle_length": None,
            "repeated_action": None}
