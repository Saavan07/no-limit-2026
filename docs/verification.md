# Verification and reproduction

Publication checks were run on November 15, 2026. The bot and archived submission ZIPs are historical originals. Documentation, the public test selector, and the archive verifier were added after the event.

## Checks completed

| Check | Result | Scope |
| --- | --- | --- |
| Public Blotto tests | 117 passed | Exact foundations, populations, four-player scores, selection rules, and allocation indexing. Five original checks require omitted external inputs and are explicitly listed by the runner. |
| Organizer poker package check | Passed | Python 3.12.13, regional SDK, final `agent.py`; source SHA-256 matched the frozen Round 6 package. |
| Submission integrity | Passed | All six ZIPs match their recorded round hashes; Round 6 ZIP contains exactly the published final source. |
| Results reconciliation | Passed | Eight practices, six scored rounds, weights 1/2/3/5/8/13, weighted total 487.3205. |
| Button-steal repricing | Reproduced | Round 4: -0.0882 per table; Round 5: -0.1137. A 15 percentage-point fold-rate improvement changes these to +0.0240 and +0.0030. |

The public Blotto suite ran on macOS with Python 3.13.5, NumPy 2.5.5, SciPy 1.18.1, and Matplotlib 3.11.2. These are the publication-test versions, not the historical population-generation environment. See the lab README for the recorded research environment and sampling caveat.

The five excluded tests comprise two external catalog/native bridge checks and three full-scan scoring checks that include the external C++ payoff table. The historical study reported 122 tests. This release does not claim a fresh 122-test pass or an independent rerun of the entire population search.

## Poker runtime

Obtain the organizer's **Pittsburgh double-board pineapple starter kit** through the [regional competition page](https://pokerbot.sequoiacap.com/tournament/competitions/2a001a79-3ac7-45eb-b662-e5fa3039a95a). Access may require a participant account. This repository does not bundle the kit or promise that the historical download remains available.

From the downloaded kit, use Python 3.12 and its original setup commands:

```sh
python3.12 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
pip install -e ./sdk/agent-protocol -e ./sdk/engine -e ./sdk/agent-sdk
```

With that environment active, set `REPO` to your checkout and check the published agent:

```sh
REPO=/path/to/no-limit-2026
poker-agent check "$REPO/poker/bot/agent.py"
poker-agent play "$REPO/poker/bot" examples/check_or_fold examples/min_raiser --tables 20 --no-history
```

The play command uses the kit's example opponents and is a functional smoke test. It does not measure competitive strength. The original pre-submission [180-table smoke test](../poker/evidence/final-smoke-test.json) had zero failed decisions against those simple opponents. That historical result is distinct from the fresh publication package check.

The competition action budget was 1 ms with a 100 ms time bank per table, with 64 KB memory permitted and less than 16 KB recommended. Final official play recorded 2,741,180 decisions, zero missed decisions, and zero restarts. These are observed results for that version and environment, not a universal latency guarantee.

## Frozen artifact identity

- Final source: `26718653482e04c43cd4b20534aed71213d45dde888e374971915e854c307f6f`
- Final ZIP: `f84deecf56fe364addf4d284fd008372100def58a7a44a15eea0bbdf1885039d`
- Round 6 submission: `ae62fa87-0e55-41d8-ab14-676fd2caa15c`

Run `python3 scripts/verify_archive.py` from any directory to check the six packages and score ledger. [Submission manifest](../poker/evidence/submission-manifest.json)

## Evidence limits

The public aggregate CSVs support the descriptive results and sensitivity calculations. Raw tournament hands remain private. Consequently the original replay, fitted-model training, opponent comparisons, and conditional payoff estimates cannot all be independently regenerated from this checkout.

The final pressure replay measures hand-end ICM against folding, not eventual table points. Round-to-round comparisons changed opponents and deals. The intended Practice 9 A/B trial never ran. The Blotto search covers the whole action space conditional on specified sampled models, not every possible population. These limits are part of the findings.
