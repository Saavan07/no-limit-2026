# Experiments and opponent research

## How we chose changes

The useful loop was: identify an observed weakness, isolate a policy change, test against the saved baseline, check legality and speed, then decide whether the evidence justified a submission. Practice ranks were signals to investigate, not causal experiments. Opponents could change their bots between rounds.

We split work between bot development and opponent research. Named-team profiles helped offline comparisons and representative test opponents. At runtime the bot only sees anonymous seats and observable actions. Hidden cards are not always known, and showdown hands are a selected subset of play.

## Final preflop pressure

The pressure candidate expanded three specific situations: medium-depth reraises, short-stack big-blind reshoves with six to eight players remaining, and selected closing calls after our open against a short blind shove. It preserved existing discard logic and basic payout protections.

Post-event replay used the first 2,500 Round 6 histories. At the first qualifying changed decision in each table, 1,017 tables differed from the baseline. All those baseline decisions were folds; the actual bot took the new action with no automatic decision. The mean realized hand-end ICM gain relative to folding was **+0.833 points per affected hand**, with standard error about **0.087**. This is a one-hand placement-equity proxy, not measured eventual table-score improvement. The ordered sample, changed continuation path, and other teams' versions limit causal interpretation. [Aggregate evidence](../poker/evidence/final-pressure.csv)

A separately scoped reshove study found 247 folds in 338 observed shoves (73.1%). Restricting to the first such change per table gave 213 folds in 292 shoves (72.9%) and about +0.136 hand-end ICM points per sampled table, standard error 0.027. This 292-table analysis has a different first-change definition from the 227 reshoves in the three-policy comparison. The samples must not be combined as independent evidence.

## Broader button pressure: rejected

A proposed wider button opening range targeted hands with estimated preflop strength around 0.35 to 0.398 at stacks of at least 18 big blinds. Comparable observed opens faced only about 43% to 44% complete folds and about 31% to 35% reraises. Our sensitivity model estimated roughly -0.09 to -0.11 placement-equity points per table under those response frequencies. Even a favorable 15 percentage-point increase in folds produced at most about +0.024.

The change was not a credible last-minute breakthrough, so we rejected it. [The executable sensitivity calculation](../poker/analysis/button_steal.py) uses public aggregate counts and fixed conditional payoff estimates. It reproduces the repricing, not the original private-hand simulations used to estimate those payoffs.

## What the leaders still did better

Our Round 6 paid-finish rate was 45.75%, compared with 49.45% for fifth in that round. Our win rate was 16.515%, compared with 21.55%. Relative to Round 5, our paid-finish rate improved, but wins did not. The final gap cannot be explained solely by failing to cash.

In the 2,500-table extract, our 800 heads-up appearances began with 49.0% of the chips on average. The five highest-scoring teams in Round 6 began their 264 observed heads-up appearances with 53.2%. That suggests chip advantage on arrival deserves separate analysis from heads-up decisions. It does not prove that an individual aggression rule caused their success.

## Next experiments

1. Compare heads-up decisions at matched starting stacks, position, board texture, and betting history.
2. Evaluate pressure against a mixture of tighter and more call-heavy opponents, including opponents adapted to our own policy.
3. Run an actual table-stable randomized comparison with assignment recorded before play. The canceled Practice 9 provides no A/B evidence.
4. Separate payout-protection losses, missed value, and arrival-stack disadvantage. Optimize eventual table points, not just a convenient hand-end proxy.

No private raw histories or opponent-identifying datasets are published here. The aggregate evidence is sufficient to inspect the reported claims, but not to independently reconstruct every replay.
