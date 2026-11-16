# Preparation and development timeline

## Blotto preparation, October to early November

The application research began with two allocation games. Game 1 assigns 100 troops across five castles worth 1 through 5, with tied castles splitting their value. Game 2 uses four-player groups and converts castle totals into placement payouts of 9/4/2/0, with tied ranks sharing the relevant payouts. Maximizing raw castle points and maximizing expected placement payout are different objectives.

We built exact reference scorers first, then marginal and dynamic-programming tools, hypothetical human populations, historical feature transfers, adapting strategy populations, failure analysis, and frozen confirmation studies. Historical Riddler data helped construct hypotheses; it was not a sample of actual No Limit applicants. External research challenged the initial Game 2 model and led to independent population rebuilding and full-action search.

The final research recommendations were Game 1 **(3,8,4,44,41)** and Game 2 **(0,2,35,63,0)**. These are versioned research decisions. This repository does not establish an official application score or receipt. See [Blotto](../blotto/README.md).

## Before the regional

I reviewed the invitation, official rules, setup instructions, and research on poker bots. Preparation focused on the environment and a decision process for an unknown variant. We registered the tournament API and SSH access, checked the assigned remote build box, and tested connectivity again on venue Wi-Fi. The tournament API authorized team and submission operations. The OpenAI API credits were development resources, separate from the poker action interface.

We planned a fast legal baseline, targeted improvements, and a validated fallback. A full poker equilibrium solver or a large coevolving bot population was not realistic inside the build window. The proposed approach was to identify variant-specific decisions, build an inexpensive policy, and use practice histories to choose the next change. We used GPT-6 Astra at High reasoning as the main development setting and focused specialists for independent analysis when useful.

This is a historical account of our setup, not permission guidance for a future event. Follow that event's current rules for what may be prepared in advance.

## Reveal and initial build

The announced variant was double-board pineapple. The released rules resolved an important initial ambiguity: discard occurs after turn betting, before the rivers. We built around one retained pair for both boards, split-pot equity, legal action sizes, and table-placement payouts.

The first 20 minutes were build-only. Practice opportunities followed at approximately ten-minute intervals, with additional practice between scored rounds. The first practice exposed a weak initial policy: 64th of 82 active teams and 7.252 points per table. Development then concentrated on actual hand decisions rather than placing too much weight on a single rank.

## Practice and scored rounds

Across eight practices and six scored rounds, changes included more price-sensitive river calls, a fitted discard scorer, postflop call models, anonymous response counts, short-stack payout protections, and selected preflop pressure. The [round ledger](../poker/evidence/rounds.csv) records the submitted versions and official outcomes.

Opponent research became a separate workstream. The development lead retained integration, validation, and uploads. Reports compared leaders under similar positions and stack depths, emphasized short-stack and heads-up play, and distinguished observed actions from inferred ranges. Heavy calculations were coordinated on remote resources after the Mac became hot; workers reused shared extracts instead of repeatedly downloading missing data.

Six scored rounds used Fibonacci weights of **1, 2, 3, 5, 8, and 13**. Late-round decisions therefore had a large effect on the total. Mystery prizes did not replace the main objective of reaching the top five.

## Final decision

We prepared a table-stable 50/50 experiment for a possible Practice 9. That practice did not occur, so the split generated no comparison data and was not the final strategy. We selected the pressure candidate using existing validation and replay evidence, preserved the baseline, and rejected an unsupported broader button-steal change. The final package is archived unchanged in [poker/74suited.zip](../poker/74suited.zip).

## After the event

We reconciled all rounds, preserved the six scored packages, replayed final pressure decisions, and compared finish distributions. Publication adds documentation and independent archive checks. It does not retrofit the final bot and present the result as the code used in competition.
