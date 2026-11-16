# Blotto research

This work prepared for No Limit's allocation-game application. It began with exact scoring and expanded into population modeling, adversarial testing, historical feature transfers, adapting strategy populations, and independent replication.

| Game | Objective | Final research choice |
| --- | --- | --- |
| Game 1 | Expected castle points against a modeled opponent; 100 troops, five castles worth 1 through 5, ties split value. | (3,8,4,44,41) |
| Game 2 | Expected 9/4/2/0 placement payout in four-player groups, including shared castles and tied ranks. | (0,2,35,63,0) |

The chosen allocations depend on the opponent assumptions and selection objective. They are not universal optimal strategies. No official Blotto score or application receipt is established by this archive.

## Research progression

1. **Foundations.** Exact rational reference scoring, allocation validation, weighted populations, best-response dynamic programming, and small-game exhaustive comparisons.
2. **Population hypotheses.** Handcrafted families, historical Riddler feature transfers, adapting arenas, multiwave training, contamination tests, and strategic-share sensitivity. Historical entrants were not assumed to represent the current applicant field.
3. **Game 1 selection.** Frozen candidate comparison, failure analysis, and fresh challenges led to (3,8,4,44,41). The [selection record](lab/results/game1-selected-v1.json) preserves its assumptions and counterexamples.
4. **Game 2 foundations and arenas.** Joint four-player evaluation, exact tie handling, correlated castle allocations, candidate populations, and placement-aware responses exposed why raw castle score was insufficient.
5. **External challenge.** An independently supplied Parry research package disagreed with the earlier balanced choice. We replayed and audited the comparison, checked population assumptions, and tested omitted and replacement families. See [report 26](research/26-game2-cross-arena-replay.md).
6. **Model stress and replication.** Catalog changes shifted the preferred castle pair. [Reports 27](research/27-game2-population-family-and-split-challenge.md) and [28](research/28-game2-overfitting-and-lock-readiness-audit.md) document those failures and the requirements for a stronger confirmation.
7. **Complete-action search.** Independent catalog rebuilding, refitting, qualification screens, and a full-action search produced the final conditional choice in [report 29](research/29-game2-independent-replication-and-full-allocation-search.md).

## Final Game 2 evidence

Every one of **4,598,126** legal allocations was searched against constructed sampled fields. Ninety-nine population hypotheses passed both diagnostic screens. The candidate was fixed before fresh confirmation on a 22-entry panel. Its minimum confirmed expected payout was about **5.000** across the selection fields. That floor is conditional on those models, not a guarantee against arbitrary players.

The research retained nearby alternatives, counterexamples, failed hypotheses, and model-removal sensitivity. Exhaustive action coverage does not validate the opponent distribution. The [versioned record](lab/results/game2-working-candidate-v4.json) includes numerical uncertainty and scope.

## Reproduction boundaries

The [lab](lab/README.md) contains our scoring and research source, configurations, protocols, tests, and selected aggregate evidence. Public Riddler data is included with its original license and attribution. Original private messages, the externally supplied Parry source package, raw applicant identity lists, and large intermediate simulation banks are excluded.

The core and public unit tests can run without that private package. Some historical cross-arena tests and full-study runners require the separately supplied Parry files or generated banks. They remain available as source for transparency, with their missing inputs documented, and are not advertised as a one-command reproduction of the entire study. Historical reports retain their checkpoint wording; report 29 supersedes earlier recommendations.
