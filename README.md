# No Limit 2026: 74suited

I competed solo as **Saav07** in Sequoia's Pittsburgh No Limit regional on November 12, 2026. This repository brings together my Blotto research, competition preparation, and **74suited**, the double-board pineapple poker bot I built and refined with OpenAI Codex during the event. The name is a tribute to my favorite poker hand, seven-four suited.

The project connects exact game scoring, population modeling, fast decision policies, and experiments under a live submission deadline. It includes the final submitted agent, all six scored submission packages, research code, and a results ledger. Codex helped with research, implementation, parallel analysis, and testing; the deployed bot used local Python logic and fitted models, with no live language-model calls.

## Results

- **27th overall**, with **487.3205 weighted points**. We did not reach our top-five target.
- **25th of 112 active teams in Round 6**, averaging **15.6105 points across 20,000 tables**.
- **2,741,180 decisions in Round 6**, with **zero missed decisions and zero restarts**.
- Blotto research searched **all 4,598,126 legal allocations** against sampled, model-defined opponent fields. This was a conditional strategy study, not a solution against every possible opponent.

[Official competition](https://pokerbot.sequoiacap.com/tournament/competitions/2a001a79-3ac7-45eb-b662-e5fa3039a95a) | [Full tournament results](poker/README.md) | [Verification and limitations](docs/verification.md)

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

These two checks need only standard Python:

```sh
python3 scripts/verify_archive.py
python3 poker/analysis/button_steal.py
cd blotto/lab
python3 -m unittest discover -s tests -p test_foundation.py -v
```

For the larger public Blotto test suite, install the optional dependencies and follow [the lab instructions](blotto/lab/README.md). To run the poker agent, obtain the organizer's regional starter kit and install its SDK as described in [verification](docs/verification.md). The SDK and private competition hand histories are not bundled.

## What I learned

A reliable bot is only the starting point. The largest remaining gap was playing strength, especially translating strong table finishes into wins. The final bot reached paid positions more often than our Round 5 baseline, but its win rate was nearly unchanged. A successful next iteration would test matched-stack heads-up play and stronger opponent mixtures rather than treating a rank change as proof of a single improvement.

The Blotto work reached a similar lesson from a different direction: exhaustive search removes uncertainty about which action wins under a specified model, but it does not remove uncertainty about whether that model describes real opponents.
