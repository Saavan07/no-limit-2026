"""Exact Game 1 utilities. Scores are Fractions; local tables use common integers.

Only tables derived from legal distributions should be interpreted as populations.
Shape validation on a general ScoreTable does not certify joint marginal feasibility.
"""

from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from functools import reduce
from math import comb, gcd, lcm
from typing import Iterable, Sequence

Action = tuple[int, ...]


@dataclass(frozen=True)
class Game:
    budget: int = 100
    values: tuple[int, ...] = (1, 2, 3, 4, 5)

    def __post_init__(self):
        if type(self.budget) is not int or self.budget < 0:
            raise ValueError("Budget must be a nonnegative integer")
        if not isinstance(self.values, tuple) or not self.values:
            raise ValueError("Values must be a nonempty tuple")
        if any(type(v) is not int or v <= 0 for v in self.values):
            raise ValueError("Castle values must be positive integers")

    @property
    def castles(self):
        return len(self.values)

    @property
    def total_value(self):
        return sum(self.values)

    @property
    def actions(self):
        return comb(self.budget + self.castles - 1, self.castles - 1)


DEFAULT_GAME = Game()


def validate_allocation(x: Sequence[int], game: Game = DEFAULT_GAME) -> Action:
    try:
        action = tuple(x)
    except TypeError as exc:
        raise ValueError("Allocation must be an integer sequence") from exc
    if len(action) != game.castles:
        raise ValueError(f"Expected {game.castles} castle bids")
    if any(type(v) is not int or v < 0 for v in action):
        raise ValueError("Bids must be nonnegative integers, excluding booleans")
    if sum(action) != game.budget:
        raise ValueError(f"Bids must sum to {game.budget}")
    return action


def score_twice(x: Sequence[int], y: Sequence[int], game=DEFAULT_GAME) -> int:
    """Deliberately direct reference scorer, in half-point units."""
    x, y = validate_allocation(x, game), validate_allocation(y, game)
    result = 0
    for a, b, value in zip(x, y, game.values):
        if a > b:
            result += 2 * value
        elif a == b:
            result += value
    return result


def score(x, y, game=DEFAULT_GAME) -> Fraction:
    return Fraction(score_twice(x, y, game), 2)


def iter_allocations(game=DEFAULT_GAME):
    """Labeled compositions, in lexicographic order, without storing the action set."""
    budget = game.budget
    if game.castles == 5:
        for a in range(budget + 1):
            for b in range(budget - a + 1):
                for c in range(budget - a - b + 1):
                    for d in range(budget - a - b - c + 1):
                        yield a, b, c, d, budget - a - b - c - d
    else:
        def visit(left, n, prefix):
            if n == 1:
                yield prefix + (left,)
            else:
                for k in range(left + 1):
                    yield from visit(left - k, n - 1, prefix + (k,))
        yield from visit(budget, game.castles, ())


@dataclass(frozen=True)
class ScoreTable:
    game: Game
    numerators: tuple[tuple[int, ...], ...]
    denominator: int
    name: str = "unnamed"
    evidence: str = "specified model"

    def __post_init__(self):
        if type(self.denominator) is not int or self.denominator <= 0:
            raise ValueError("Table denominator must be a positive integer")
        if len(self.numerators) != self.game.castles:
            raise ValueError("Wrong number of score rows")
        for row in self.numerators:
            if len(row) != self.game.budget + 1 or any(type(x) is not int for x in row):
                raise ValueError("Score rows must contain an integer for every legal bid")

    def numerator(self, x):
        x = validate_allocation(x, self.game)
        return sum(row[k] for row, k in zip(self.numerators, x))

    def expected_score(self, x) -> Fraction:
        return Fraction(self.numerator(x), self.denominator)


def table_from_histograms(histograms, total_mass, game=DEFAULT_GAME,
                          name="empirical", evidence="exact for supplied weighted support"):
    """Compress known legal joint draws. Marginal validity alone is not joint validity."""
    if type(total_mass) is not int or total_mass <= 0:
        raise ValueError("Total mass must be positive")
    if len(histograms) != game.castles:
        raise ValueError("Wrong number of histograms")
    rows = []
    for value, hist in zip(game.values, histograms):
        if len(hist) != game.budget + 1 or any(type(c) is not int or c < 0 for c in hist):
            raise ValueError("Invalid histogram")
        if sum(hist) != total_mass:
            raise ValueError("Every histogram must sum to total mass")
        before, row = 0, []
        for count in hist:
            row.append(value * (2 * before + count))
            before += count
        rows.append(tuple(row))
    return ScoreTable(game, tuple(rows), 2 * total_mass, name, evidence)


@dataclass(frozen=True)
class Population:
    """A weighted legal support. Integer masses are retained, never deduplicated away.

    from_people preserves actual multiplicities. Abstract masses and empirical arena
    frequencies do not estimate applicant prevalence unless independently justified.
    """
    game: Game
    entries: tuple[tuple[Action, int], ...]
    name: str = "population"

    def __post_init__(self):
        seen = set()
        if not self.entries:
            raise ValueError("Population must not be empty")
        for action, mass in self.entries:
            validate_allocation(action, self.game)
            if action in seen or type(mass) is not int or mass <= 0:
                raise ValueError("Entries must be unique with positive integer masses")
            seen.add(action)

    @classmethod
    def from_people(cls, actions: Iterable[Action], game=DEFAULT_GAME, name="population"):
        counts = Counter(validate_allocation(x, game) for x in actions)
        return cls.from_counts(counts, game, name)

    @classmethod
    def from_counts(cls, counts, game=DEFAULT_GAME, name="population"):
        merged = Counter()
        pairs = counts.items() if hasattr(counts, "items") else counts
        for action, mass in pairs:
            action = validate_allocation(action, game)
            if type(mass) is not int or mass <= 0:
                raise ValueError("Masses must be positive integers")
            merged[action] += mass
        return cls(game, tuple(sorted(merged.items())), name)

    @property
    def total_mass(self):
        return sum(mass for _, mass in self.entries)

    def histograms(self):
        hist = [[0] * (self.game.budget + 1) for _ in self.game.values]
        for action, mass in self.entries:
            for i, k in enumerate(action):
                hist[i][k] += mass
        return hist

    def table(self):
        return table_from_histograms(self.histograms(), self.total_mass,
                                     self.game, self.name)

    def reference_mean(self, x):
        numerator = sum(m * score_twice(x, y, self.game) for y, m in self.entries)
        return Fraction(numerator, 2 * self.total_mass)

    def excluding_one(self, action):
        action = validate_allocation(action, self.game)
        counts = dict(self.entries)
        if action not in counts or self.total_mass <= 1:
            raise ValueError("Cannot exclude that person from this population")
        counts[action] -= 1
        if counts[action] == 0:
            del counts[action]
        return Population.from_counts(counts, self.game, self.name + ": leave-one-out")

    def to_dict(self):
        return {"name": self.name, "budget": self.game.budget,
                "values": list(self.game.values),
                "entries": [[list(x), m] for x, m in self.entries]}

    @classmethod
    def from_dict(cls, data):
        game = Game(data["budget"], tuple(data["values"]))
        return cls.from_counts(((tuple(x), m) for x, m in data["entries"]),
                               game, data["name"])


def uniform_composition(game=DEFAULT_GAME, unit=1):
    if type(unit) is not int or unit <= 0 or game.budget % unit:
        raise ValueError("Grid unit must be positive and divide the budget")
    b, n = game.budget // unit, game.castles
    total = comb(b + n - 1, n - 1)
    counts = [0] * (game.budget + 1)
    for k in range(b + 1):
        counts[k * unit] = int(k == b) if n == 1 else comb(b - k + n - 2, n - 2)
    return table_from_histograms([counts] * n, total, game,
        "uniform_legal" if unit == 1 else f"uniform_grid_{unit}",
        "exact mathematical control, not a human prior")


def multinomial(game=DEFAULT_GAME):
    """Each soldier independently chooses a castle uniformly; exact binomial marginals."""
    n, b = game.castles, game.budget
    counts = [comb(b, k) * (n - 1) ** (b - k) for k in range(b + 1)]
    return table_from_histograms([counts] * n, n ** b, game, "soldier_random",
                                "exact mathematical control, not a human prior")


def mix_tables(tables, weights, name="mixture"):
    tables, weights = tuple(tables), tuple(weights)
    if not tables or len(tables) != len(weights):
        raise ValueError("Need a weight for every table")
    if any(type(w) is not int and not isinstance(w, Fraction) for w in weights):
        raise ValueError("Use integers or Fractions for exact weights")
    weights = tuple(Fraction(w) for w in weights)
    if min(weights) < 0 or sum(weights) <= 0:
        raise ValueError("Weights must be nonnegative with a positive sum")
    if any(t.game != tables[0].game for t in tables):
        raise ValueError("Cannot mix different games")
    weights = tuple(w / sum(weights) for w in weights)
    terms = [(t, w) for t, w in zip(tables, weights) if w]
    denominator = lcm(*(t.denominator * w.denominator for t, w in terms))
    factors = [(t, w.numerator * (denominator // (t.denominator * w.denominator)))
               for t, w in terms]
    game = tables[0].game
    rows = tuple(tuple(sum(t.numerators[i][k] * f for t, f in factors)
                       for k in range(game.budget + 1)) for i in range(game.castles))
    common = reduce(gcd, (x for row in rows for x in row), denominator)
    return ScoreTable(game, tuple(tuple(x // common for x in row) for row in rows),
                      denominator // common, name, "exact specified mixture")


@dataclass(frozen=True)
class BestResponse:
    allocation: Action
    expected_score: Fraction
    optimum_count: int
    evaluations: int
    method: str


def best_response(table: ScoreTable) -> BestResponse:
    """Suffix DP gives the lexicographically smallest optimum and exact tie count."""
    n, b = table.game.castles, table.game.budget
    dp = [[None] * (b + 1) for _ in range(n + 1)]
    ways = [[0] * (b + 1) for _ in range(n + 1)]
    dp[n][0], ways[n][0] = 0, 1
    evaluations = 0
    for i in range(n - 1, -1, -1):
        row = table.numerators[i]
        for spent in range(b + 1):
            current = None
            for k in range(spent + 1):
                evaluations += 1
                previous = dp[i + 1][spent - k]
                if previous is None:
                    continue
                candidate = row[k] + previous
                if current is None or candidate > current:
                    current, ways[i][spent] = candidate, ways[i + 1][spent - k]
                elif candidate == current:
                    ways[i][spent] += ways[i + 1][spent - k]
            dp[i][spent] = current
    remaining, action = b, []
    for i in range(n):
        for k in range(remaining + 1):
            previous = dp[i + 1][remaining - k]
            if previous is not None and table.numerators[i][k] + previous == dp[i][remaining]:
                action.append(k)
                remaining -= k
                break
    action = validate_allocation(action, table.game)
    return BestResponse(action, Fraction(dp[0][b], table.denominator), ways[0][b],
                        evaluations, "exact dynamic program; lexicographic ties")


def exhaustive_best_response(table: ScoreTable) -> BestResponse:
    """Independent full scan, sharing no dynamic-programming logic."""
    best, winner, ties, count = None, None, 0, 0
    rows = table.numerators
    if table.game.castles == 5:
        # Inline score lookups to keep full-space verification practical in Python.
        for x in iter_allocations(table.game):
            a,b,c,d,e = x
            value = rows[0][a]+rows[1][b]+rows[2][c]+rows[3][d]+rows[4][e]
            count += 1
            if best is None or value > best:
                best, winner, ties = value, x, 1
            elif value == best:
                ties += 1
    else:
        for x in iter_allocations(table.game):
            value = sum(row[k] for row, k in zip(rows, x))
            count += 1
            if best is None or value > best:
                best, winner, ties = value, x, 1
            elif value == best:
                ties += 1
    return BestResponse(winner, Fraction(best, table.denominator), ties, count,
                        "complete enumeration; lexicographic ties")


def constructive_counter(x):
    """A legal >=10-point counter for the exact application instance."""
    x = validate_allocation(x)
    if x[4] >= 4:
        selected = (0,1,2,3)
    elif x[1] + x[2] >= 3:
        selected = (0,3,4)
    else:
        selected = (1,2,4)
    counter = [0] * 5
    for i in selected:
        counter[i] = x[i] + 1
    counter[selected[0]] += 100 - sum(counter)
    return validate_allocation(counter)


def security_of(population: Population):
    """Exact guarantee for this mixture, independent of how it was generated."""
    response = best_response(population.table())
    value = Fraction(population.game.total_value, 2)
    advantage = response.expected_score - value
    return {"best_response": response, "exploitability": advantage,
            "security": value - advantage}
