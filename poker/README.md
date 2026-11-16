# 74suited: tournament performance and strategy evolution

**Sequoia Capital, OpenAI, and Citadel Securities | No Limit 2026 | Pittsburgh**

Selected to compete in a tournament with **8,000+ applicants** across the event, I competed solo as **Saav07** on November 12, 2026. I built the initial double-board pineapple agent during the **60-minute build window**, then used practice evidence and opponent analysis to refine it between scored rounds. [Event context and sources](../docs/provenance.md#event-context-and-sources)

The bot competed in **120,000 simulated tables across six scored rounds**. Its final version completed **2,741,180 decisions with zero missed decisions and zero restarts**, placing **25th of 112 active teams** in Round 6. The overall Pittsburgh finish was **27th**, with **487.3205 weighted points**.

This case study shows how the strategy evolved, which changes earned a place in the final submission, and where the evidence remains limited. The regional variant required one discard after turn betting, with the retained pair used on both boards.

## Six scored rounds

Each round used 20,000 tables. The organizer multiplied our official average placement points by the round weight before adding them to the standings. The table gives our rank among teams that played that round. [Official competition page](https://pokerbot.sequoiacap.com/tournament/competitions/2a001a79-3ac7-45eb-b662-e5fa3039a95a)

| Round | Weight | Rank | Points per table | Weighted points | Missed decisions | Main strategy in this version |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 1x | 37/109 | 14.7245 | 14.7245 | 0 | Equity estimates on both boards and price-sensitive river calls. |
| 2 | 2x | 28/111 | 15.3135 | 30.6270 | 0 | Learned discard scorer, fitted flop and turn call values, and table-level response counts. |
| 3 | 3x | 38/112 | 14.6870 | 44.0610 | 0 | Payout-aware short calls, opening-hand estimates and targeted blind steals. |
| 4 | 5x | 28/112 | 15.2855 | 76.4275 | 16 | Big-blind reshoves, selective calls after opening and postflop response tracking. |
| 5 | 8x | 31/112 | 14.8180 | 118.5440 | 0 | More short-stack open shoves and revised four-handed small-blind opens. |
| 6 | 13x | 25/112 | 15.6105 | 202.9365 | 0 | Short-stack reshoves at fuller tables, medium-depth reraises and selected calls against blind shoves. |

## Where the final points came from

| Entry | Paid finish | Top two | Win |
| --- | ---: | ---: | ---: |
| Round 5, our baseline | 42.63% | 29.62% | 16.64% |
| Round 6, our final bot | 45.75% | 31.57% | 16.52% |
| Fifth in Round 6 | 49.45% | 36.96% | 21.55% |

The payouts were 50/30/20. Every paid finish therefore contributed at least 20 points, reaching the top two added 10, and winning added a further 20. These rates describe the scores. Opponents and deals changed between rounds, so the difference between Rounds 5 and 6 cannot isolate the effect of the final changes.

## Did the pressure changes show up in real hands?

We replayed both saved bots on the first 2,500 scored Round 6 histories. We stopped at the first qualifying preflop decision per table on which the two versions disagreed. That gave 1,017 tables. On every one, the old policy chose a fold, the recorded bot took the new action, and no decision was automatic.

| First change | Tables | Mean hand-end placement equity above folding |
| --- | ---: | ---: |
| Medium-depth reraise | 651 | +0.47 |
| Short-stack reshove with six to eight players remaining | 227 | +1.21 |
| Call after our open against a short blind shove | 139 | +1.89 |

Across the 1,017 changed hands the gain was about +0.83 ICM points per hand. Here ICM estimates the value of a chip stack in table-placement points. The comparison prices the immediate fold exactly from the public stacks and uses the actual outcome of the hand played. The sample is an ordered extract. It tracks three specific decisions, ends after one hand and does not measure the change in eventual table finishes. [Aggregate counts and sampling uncertainty](evidence/final-pressure.csv)

In the same extract, our 800 heads-up appearances began with 49.0% of the chips on average. Opponents from the five highest-scoring Round 6 teams began their 264 observed heads-up appearances with 53.2%. On the full official result we won 52.3% of our top-two finishes. A follow-up study should compare heads-up decisions at matched starting stacks before attributing the remaining gap to a particular betting mistake.

## What changed during practice

The first bot averaged 7.252 points, placing 64th of 82 active teams. By Practice 8 we averaged 15.656, placing 26th of 112. Eight practices ran. Practice 9 never occurred, so the prepared table-level A/B comparison yielded no observations. The score ledger contains each practice and scored round, with submission IDs, team counts, placements and official score uncertainty. It uses the organizer's reported points. In Round 2, the published placement counts imply a mean 0.0005 higher than the published mean.

## Reproduce and inspect

- [Read the strategy and experiments](strategy.md). The final bot runs locally at decision time. See how it makes a single discard for both boards and why some preflop pressure qualified for the final.
- [Inspect the exact final agent](bot/agent.py) or [download its original submission ZIP](74suited.zip). The Round 6 entry was `ae62fa87-0e55-41d8-ab14-676fd2caa15c`. ZIP SHA-256 is `f84deecf56fe364addf4d284fd008372100def58a7a44a15eea0bbdf1885039d`.
- [Open the round-by-round score ledger](evidence/rounds.csv) and [the 180-table validation result](evidence/final-smoke-test.json). The ZIP passed the organizer's package check and local smoke test before submission.
- [Run the button-steal experiment](analysis/button_steal.py) with the [aggregated field observations](evidence/button-steal.csv). It uses only standard Python.
- [Review verification and reproduction boundaries](../docs/verification.md). The original submitted code and post-event analysis are labeled separately.
