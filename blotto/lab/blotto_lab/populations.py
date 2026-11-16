"""Explicit initial controls and synthetic hypotheses, with no fitted applicant shares."""

from dataclasses import dataclass
from random import Random

from .core import DEFAULT_GAME, Game, Population, ScoreTable, multinomial, uniform_composition


@dataclass(frozen=True)
class Scenario:
    key: str
    title: str
    table: ScoreTable
    description: str
    status: str
    population: Population | None = None

    def metadata(self):
        result = {"key": self.key, "title": self.title, "description": self.description,
                  "status": self.status, "table_evidence": self.table.evidence}
        if self.population:
            result.update(total_mass=self.population.total_mass,
                          distinct_vectors=len(self.population.entries))
        return result


def random_composition(rng: Random, budget: int, castles: int):
    """Uniform stars-and-bars composition, not independent soldier assignment."""
    if castles == 1:
        return (budget,)
    cuts = [-1] + sorted(rng.sample(range(budget + castles - 1), castles - 1))
    cuts.append(budget + castles - 1)
    return tuple(cuts[i + 1] - cuts[i] - 1 for i in range(castles))


def jittered_population(anchor, size, seed, name, transfers=4, max_transfer=6):
    rng = Random(seed)
    actions = []
    for _ in range(size):
        action = list(anchor)
        for _ in range(transfers):
            donor, receiver = rng.sample(range(5), 2)
            amount = rng.randint(0, min(max_transfer, action[donor]))
            action[donor] -= amount
            action[receiver] += amount
        actions.append(tuple(action))
    return Population.from_people(actions, name=name)


def sparse_population(size, seed):
    rng = Random(seed)
    actions = []
    for _ in range(size):
        support_size = rng.choice((2, 3, 4))
        indices = sorted(rng.sample(range(5), support_size))
        positive = random_composition(rng, 100 - support_size, support_size)
        action = [0] * 5
        for i, value in zip(indices, positive):
            action[i] = value + 1
        actions.append(tuple(action))
    return Population.from_people(actions, name="sparse_sample")


def initial_scenarios(seed=20261029, population_size=2000):
    """Finite hypotheses are evaluated exactly conditional on their saved sample.

    Seeded sample frequencies are not measured applicant frequencies. No final
    holdout is created or consumed by this exploratory diagnostic.
    """
    if type(population_size) is not int or population_size <= 0:
        raise ValueError("Population size must be positive")
    result = [
        Scenario("uniform_legal", "Uniform legal vectors", uniform_composition(),
                 "Every legal integer vector has equal probability.", "exact control"),
        Scenario("soldier_random", "Independent soldier placement", multinomial(),
                 "Each soldier independently chooses one of five castles uniformly.", "exact control"),
        Scenario("round_grid", "Multiples-of-five grid", uniform_composition(unit=5),
                 "Every legal vector whose bids are divisible by five has equal probability.",
                 "exact control; hypothetical rounding model"),
    ]
    descriptions = [
        ("equal_jitter", "Near equal allocation", (20,20,20,20,20), seed+101,
         "Start at (20,20,20,20,20); four random transfers, each at most six troops."),
        ("value_jitter", "Near value-proportional allocation", (7,13,20,27,33), seed+202,
         "Start at (7,13,20,27,33); four random transfers, each at most six troops."),
    ]
    for key, title, anchor, derived_seed, description in descriptions:
        population = jittered_population(anchor, population_size, derived_seed, key)
        result.append(Scenario(key, title, population.table(), description,
                               "unfitted synthetic hypothesis; exact for saved finite sample", population))
    population = sparse_population(population_size, seed+303)
    result.append(Scenario("sparse_sample", "Random sparse allocations", population.table(),
        "Uniformly choose support size 2, 3, or 4 and a subset; sample a uniform positive composition there.",
        "unfitted synthetic hypothesis; exact for saved finite sample", population))
    return result
