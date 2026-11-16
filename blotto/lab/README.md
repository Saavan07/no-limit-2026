# Blotto Lab

Exact scoring and reproducible population experiments for two allocation games. See [the research overview](../README.md) for the full sequence and final recommendations.

## Run

Python 3.10 or newer. The foundation has no external dependencies:

```sh
python3 -m unittest discover -s tests -p test_foundation.py -v
```

For the public computational tests:

```sh
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install -e '.[population]'
python3 run_public_tests.py
```

The public runner excludes only the five tests requiring the omitted Parry catalog and native bridge. It includes all other original tests, including full allocation indexing and independent four-player scoring checks. The omitted native full-scan scoring checks require the external C++ payoff table as well as `clang++`. The runner reports the excluded test names rather than silently claiming the original private suite passed.

## Structure

- `blotto_lab/`: exact two-player scoring, populations, best responses, human hypotheses, and adapting arenas.
- `game2_lab/`: four-player placement scoring, joint population sampling, catalog stress, and complete-action search.
- `run_*.py`, `verify_*.py`: historical experiment drivers and independent audit routines.
- `config/`, `*-PROTOCOL.md`: versioned experiment definitions, seeds, and selection rules.
- `results/`: selected decision records and aggregate evidence supporting the public research reports.
- `data/riddler/`: FiveThirtyEight data and original license. The strategies inform hypotheses rather than directly estimate No Limit entrant frequencies.

## Large studies and omitted inputs

The historical full-space study searched 4,598,126 actions against model-defined sampled fields. The central reproduction sequence is in `GAME2-REPLICATION-SEARCH-PROTOCOL.md`. The original run recorded Python 3.13.5, NumPy 2.4.4, SciPy 1.17.1, and Matplotlib 3.10.8; sampled draws can vary with library versions.

This public release omits `data/parry-replay`, `data/parry-handoff`, full frozen study banks, and large discovery/selection arrays. Several runners depend on those inputs and will not complete from this checkout alone. Do not interpret missing-input errors as a newly reproduced research result. No replacement dataset is silently substituted. The original research reported 122 tests; the fresh publication verification is recorded separately in [verification](../../docs/verification.md).

The source of the original external package is not redistributed. Our integration code and research findings preserve its role in challenging the initial assumptions. All original research runners remain inspectable, but the lightweight commands above are the supported public entry points.
