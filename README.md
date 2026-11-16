# 74suited | No Limit 2026

**A poker bot built for No Limit, hosted by Sequoia Capital, OpenAI, and Citadel Securities.**

I was selected for the Pittsburgh regional from a tournament-wide pool of **8,000+ applicants**.[^selection] Competing solo, I had **60 minutes to build the initial bot** after the organizers revealed double-board pineapple, a poker variant we had to adapt to on the spot. I then refined the strategy between rounds using practice results and opponent hand histories.

My bot, **74suited**, combined equity estimates across both boards, models for discarding and calling, and adjustments based on opponents' observed play. It competed in **120,000 simulated tables across six scored rounds**. In the final round, it handled **2,741,180 decisions with zero missed decisions and zero restarts**.

## What I built and tested

- **A fast poker agent:** combined lookup tables and fitted models with compact table memory to work within a 1 ms action budget and a 100 ms time bank per table.
- **A strategy for incomplete information:** evaluated both boards together, accounted for tournament payouts, and adapted to observable behavior at anonymous seats.
- **An evidence-driven development loop:** analyzed losses, compared candidate changes with a saved baseline, rejected weak hypotheses, and validated each submission before its deadline.
- **A Blotto research toolkit:** built exact scoring, population models, and a search over **4,598,126 legal allocations** under specified opponent assumptions. The public research suite has **117 passing tests**.
- **A reproducible competition record:** preserved all six scored submissions, package hashes, experiment evidence, and the full results ledger.

I used **OpenAI Codex** throughout the project for research, coding, parallel analysis, and testing. I directed the strategy priorities and experiment choices. During play, the bot ran locally using Python code and fitted models.

## Competition results

| Measure | Result |
| --- | --- |
| Pittsburgh overall finish | **27th**, with 487.3205 weighted points |
| Final-round finish | **25th of 112 active teams** |
| Final-round evaluation | **20,000 tables**, averaging 15.6105 points per table |
| Final-round reliability | **2,741,180 decisions; zero misses or restarts** |
| Full scored evaluation | **120,000 tables across six rounds** |

[Official event](https://pokerbot.sequoiacap.com/) | [Full tournament results](poker/README.md) | [Verification and limitations](docs/verification.md)

## Start here

| Area | What is included |
| --- | --- |
| [Blotto research](blotto/README.md) | Exact scoring, dynamic-programming responses, population models, adversarial challenges, full-action search, and versioned findings. |
| [Preparation and development timeline](docs/timeline.md) | How I prepared before the variant reveal and organized build, practice, and final submission decisions. |
| [Poker strategy](poker/strategy.md) | Two-board equity, a shared discard decision, payout-aware play, anonymous opponent profiles, and preflop pressure. |
| [Exact final agent](poker/bot/agent.py) | The frozen Round 6 source, including its fitted models and lookup tables. |
| [Experiments and opponent analysis](docs/experiments.md) | Evidence behind accepted and rejected changes, including the limits of our hand replays. |
| [Run and verify](docs/verification.md) | Local checks, dependency boundaries, package hashes, and what can be reproduced from this public repository. |
| [Provenance](docs/provenance.md) | My role, Codex assistance, external data, organizer tooling, and publication scope. |

## How the bot works

Each player receives three private cards and plays against two community boards. In this regional's released rules, the discard happens **after turn betting and before the rivers**. The same two retained cards serve both boards, with half of the pot allocated to each board. Eight-player table finishes pay 50/30/20.

The bot combines preflop hand-strength tables, sampled board equity, a fitted joint discard scorer, postflop call estimates, and payout-aware short-stack decisions. It tracks observable actions within a table and uses those counts conservatively. Seats are anonymous and memory resets at each table, so offline named-team research does not become a live team-identity lookup.

The final update expanded selected preflop pressure: big-blind reshoves at fuller tables, medium-stack reraises, and some calls against short blind shoves. We kept the tested discard logic and basic payout protections. A broader button-steal idea failed its sensitivity check and was rejected.

## Quick checks

These checks need only standard Python:

```sh
python3 scripts/verify_archive.py
python3 poker/analysis/button_steal.py
cd blotto/lab
python3 -m unittest discover -s tests -p test_foundation.py -v
```

For the larger public Blotto test suite, install the optional dependencies and follow [the lab instructions](blotto/lab/README.md). To run the poker agent, obtain the organizer's regional starter kit and install its SDK as described in [verification](docs/verification.md). The SDK and private competition hand histories are not bundled.

## Engineering lessons

A reliable bot is only the starting point. The largest remaining gap was playing strength, especially translating strong table finishes into wins. The final bot reached paid positions more often than our Round 5 baseline, but its win rate was nearly unchanged. A successful next iteration would test matched-stack heads-up play and stronger opponent mixtures rather than treating a rank change as proof of a single improvement.

The Blotto work reached a similar lesson from a different direction: exhaustive search removes uncertainty about which action wins under a specified model, but it does not remove uncertainty about whether that model describes real opponents.

[^selection]: The 8,000+ applicant figure was announced by the organizers at the Pittsburgh competition, as reported by Saavan. It describes the tournament-wide application pool, not the Pittsburgh field or a finishing percentile. See [event context and sources](docs/provenance.md#event-context-and-sources).
