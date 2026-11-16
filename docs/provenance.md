# Provenance and publication scope

## Event context and sources

No Limit was hosted by **Sequoia Capital, OpenAI, and Citadel Securities**. The [official event website](https://pokerbot.sequoiacap.com/) identifies the hosts, the 60-minute build format, and the November 12, 2026 Pittsburgh regional.

The **8,000+ applicants** figure was announced by the organizers at the Pittsburgh competition, as reported by Saavan. It refers to the tournament-wide application pool. It is not a claim that 8,000 people competed in Pittsburgh, that Saavan placed 27th nationally, or that his acceptance rate can be calculated from regional team counts. We have not independently recovered a recording or written organizer source for that announcement.

The results ledger establishes 20,000 tables in each of six scored regional rounds, totaling 120,000 tables. The 60-minute claim describes the initial build phase; subsequent versions incorporated changes between rounds.

## Authorship and assistance

Saavan Kiran competed solo as Saav07 and directed the project: preparation, strategy priorities, risk tolerance, and final submission goals. OpenAI Codex assisted with research, implementation, tests, opponent analysis, and iteration. The published code should be understood as AI-assisted competition work, not as an unaided implementation claim.

The submitted poker agent makes decisions with local Python code, lookup tables, sampled equity, fitted models, and table memory. It does not call a language model during play. The exact Round 6 source is preserved unchanged; publication scripts and documentation were added after the event.

## External material

- **Sequoia No Limit:** event rules, tournament system, starter SDK, and recorded results. [Event rules](https://pokerbot.sequoiacap.com/rules.html), [regional competition](https://pokerbot.sequoiacap.com/tournament/competitions/2a001a79-3ac7-45eb-b662-e5fa3039a95a). The organizer SDK is a runtime dependency and is not vendored.
- **FiveThirtyEight Riddler:** historical allocation submissions. The copied dataset keeps its [original README](../blotto/lab/data/riddler/README.md) and [license](../blotto/lab/data/riddler/LICENSE). [Original dataset](https://github.com/fivethirtyeight/data/tree/master/riddler-castles).
- **Parry research package:** an externally supplied Blotto study used for adversarial comparison and model challenges. Its original implementation and private handoff are not redistributed. Our integration code and findings identify that dependency explicitly.
- **Poker runtime libraries:** the organizer environment supplies its SDK and allowed packages, including `eval7`. Those packages are not included in the submitted-source archive.

No blanket open-source license is asserted for the whole repository. External materials retain their own notices. Publication of this archive does not relicense third-party code or data.

## What is public

The exact final agent, all six scored ZIPs, aggregate round results, selected experiment evidence, our Blotto research source/tests/protocols, and the post-event narrative are included. Absolute workstation paths in research documents were converted to repository links. The bot source and submission packages were not edited.

School email, invitations, API credits, tokens, SSH files, private hand histories, raw third-party handoffs, and upload helpers remain outside the repository. Large intermediate banks and caches are omitted. This is a curated project record, not a dump of the private workspace.
