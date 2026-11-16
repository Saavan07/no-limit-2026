# How 74suited plays double-board pineapple

Eight players start a table. Each player receives three private cards. Two flops, two turns, and two rivers share one betting round per street. The crucial timing detail in Pittsburgh was that players kept all three cards through turn betting. They discarded one card face down before the rivers. Each board awarded half of every contested pot. Table placement paid 50, 30 and 20 points.

## One decision inside the bot

The tournament engine calls `Agent.act(state, memory)`. In the final Python file, `Agent.choose` routes the decision to preflop, discard or postflop logic. All gameplay computation runs locally. Codex helped write and analyze the bot during development. A hand requires no model API, internet connection or access to another team's identity.

- **Preflop.** A three-card lookup gives a starting estimate. The bot adjusts for position, effective stacks, number of opponents and prior action. Facing an all-in, it considers how calling changes expected placement points. This matters most near the payouts, when losing chips and winning the same number of chips can have different tournament value.
- **Flop and turn.** A bounded sample of legal runouts estimates each board's share. The bot considers the weaker board, pot size, price to continue and action history. An apparently strong hand on one board can still be a poor call for the whole pot.
- **Discard.** A fitted scorer compares the available pairs with both turns visible. The chosen discard is irreversible and applies to both boards. An evaluator that maximizes one board independently can choose incompatible pairs.
- **River.** Both final boards are known. The bot estimates contested pot share and uses price and prior betting to decide whether to pay. Archived calls let us measure the cost of paying off aggression at different prices. Showdowns are a selected sample.
- **Table memory.** The engine returns a small byte buffer to the bot on later decisions at that table. Seat-level opening and response counts can adjust thresholds as opportunities accumulate. A new table starts with empty memory. Named-team comparisons were an offline research aid.

## Why the final build applied more pressure

The largest observed fold rate was specific to a betting sequence. A single button or small-blind opener had raised, everyone else was out, and the big blind could shove. In Rounds 3 and 4 the opener folded to 80.9% of 1,139 and 79.0% of 1,155 comparable field shoves. Applying the new six- through eight-handed rule to 2,500 historical tables per round identified 262 and 300 first changes. The previous policy had folded in 561 of those 562 hands.

We priced those opportunities using responses and contested showdowns from the other round. The short-stack extension projected +0.124 and +0.146 placement points per original table. Reducing the assumed fold rate by eight percentage points gave +0.081 and +0.098. Those estimates depend on which hidden hands opponents would call with and how their bots changed. They describe the first changed action on each table. Later actions can interact with other changes.

The final policy also reraised some late openers at medium depth with a smaller three-bet, folding the bluff to a qualifying four-bet. It called selected short blind shoves after our own open when tournament payout estimates supported it. The Round 5 discard and postflop code carried forward. The whole-table split built for a possible Practice 9 never played a round. Every Round 6 table used the same policy.

## Two decisions worth reproducing

**Shove or small open at 13 to 17 big blinds.** In 5,500 sampled Round 5 tables, 1,071 first qualifying medium-strength opens shoved. Among 777 close matches to earlier small opens, the observed hand-end tournament equity gain was +0.308 points per switched decision, with a standard error of 0.156. Broader matching gave +0.190 with a standard error of 0.149. The comparison changed across source rounds. It supports the decision while leaving uncertainty about its true size.

**Steal more from the button.** A 2.2-big-blind opening extension for weak button hands would have changed 2,221 folds in a 2,000-table Round 4 sample and 2,313 in a separate Round 5 sample. Comparable opens got both blinds to fold just 43% to 44% of the time. At least one blind reraised 31% to 35%. Matching the number of remaining players, opener depth and shortest defending stack gave modeled changes of -0.089 and -0.114 points per table. Even an assumed 15-point increase in fold frequency at the expense of calls gave only +0.024 and +0.003. We declined the change before the final freeze.

## What can be reproduced

The release source embeds its lookup tables and fixed model parameters. With Python 3.12 and the Pittsburgh starter kit installed as described in its README, run the following from the starter-kit directory. Replace the example path with this folder's absolute path.

```sh
poker-agent check /path/to/74suited-retrospective/74suited.zip
poker-agent play /path/to/74suited-retrospective/bot examples/check_or_fold examples/min_raiser --variant double_board_turn_discard --seed 893227 --tables 180 --no-history
```

The second command is a functional smoke test against examples. The historical validation isolated one table at a time on the build box, used seed 893227 and returned zero failed decisions across 180 tables. Its example opponents and three-seat format make the score unsuitable for predicting an eight-player regional result. The official Round 6 decision and restart counts are the strongest reliability evidence.

To rerun the button-steal calculation without the SDK, run `python3 analysis/button_steal.py` from this directory. The inputs are counts of actual field responses and candidate opportunities. The response-specific payoffs are simulation estimates. The script reproduces the pooled estimate and the favorable fold-rate scenario. The CSV also records the separate stack- and player-count-matched result. Its standard-error column covers sampling in the pooled field responses and opportunities. Differences between simulated and event opponents create additional uncertainty.

## What the final field did

A separate replay isolated the six- through eight-handed big-blind reshove in 2,500 Round 6 tables. Opponents folded on 247 of 338 shoves (73.1%), below the 79% to 81% seen in Rounds 3 and 4. On the first qualifying reshove per affected table, opponents folded 213 of 292 times. The archived Round 5 policy folded all 292 positions. Counting the other sampled tables as zero, those first reshoves gained an estimated **0.136 hand-end ICM points per sampled table over folding**, with a standard error of 0.027. The pre-final projection was +0.124 to +0.146.

This reshove-only count includes tables on which a different pressure decision had already occurred. The retrospective's comparison across all three changes stops at the first change of any kind and attributes 227 tables to the reshove. The two analyses answer different questions and their estimates overlap.
