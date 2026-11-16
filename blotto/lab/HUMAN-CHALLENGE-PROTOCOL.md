**Game 1 human-strategy challenge, version 1**

The user supplied five literal vectors, a proposed allocation range, five priority
coalitions, a castle-5 bid-gap hypothesis, and an evolutionary-frequency conjecture.
Keep the nine primary machine comparison vectors and 193-vector archive fixed from
`results/game1-machine-checkpoint-v1.json`. Verify its pinned artifact hashes. Do not
reweight or replace those baseline vectors after reading the human explanation.

Literal vectors, in castle order: (1,1,20,8,70), (5,15,25,35,20), (0,10,30,30,30),
(3,3,2,41,51), (10,10,0,60,20). Interpret the stated ranges literally as
[2,3],[5,7],[22,25],[36,42],[25,30], intersected with an exact total of 100.
There are 114 such vectors. Also declare an exploratory relaxation of every endpoint
by two soldiers, clipped to legal bounds. This relaxation is an analyst construction,
not a strategy explicitly proposed by the humans.

First score the literal inputs and the entire literal/relaxed ranges on the 121
previously known population models and the declared balanced behavior and 50%
strategic references. Select range representatives by four stated criteria:
max-min score, minimax regret, expected balanced-reference score, and expected
half-strategic-reference score. Save and hash these selections before constructing
new priority-coalition populations or learning simulations. Range selection is
exploratory fitting to known scenarios, not an independent success claim.

Construct exact legal populations for all five priority coalitions: {3,5}, {4,5},
{1,2,5}, {1,3,4}, {2,3,4}. These are all the inclusion-minimal castle sets worth at
least eight. Give each nonpriority field zero, one, or three scouts. On priority
fields require one soldier each and allocate the remaining budget using either
uniform positive compositions or exact value-weighted multinomial masses.
For coalitions containing castle 5, add a separate condition requiring its bid to be
at least 50. This produces 48 declared variants. Do not silently impose that
restriction on all coalition players. Preserve exact joint supports and multiplicity.

Compare the five literal vectors, primary machine baselines, range representatives,
and larger exploratory archive on every new variant. Report exact expected points
and win probabilities separately. Uniform family weights are declared comparison
conventions, never inferred prevalence. Reference mixtures balance priority sets,
allocation laws, and scout levels; a five-heavy reference substitutes the conditional
variants only where castle 5 is a priority. Two-favorite references weight {3,5} and
{1,3,4} equally, without claiming those are actually the most common.

Test the castle-5 gap directly through marginal bid masses. For the illustrative
range vector (2,6,24,40,28), compare points conditional on losing castle 5, including
losses by at least 20 soldiers. Conditional relationships need legal joint supports;
Game 1 expected points alone require only marginals. Also calculate exact local
transfer gains instead of assuming fixed upper bounds on scouts.

For learning, freeze a finite legal candidate pool: the old machine archive, five
literal vectors, all literal-range vectors, known-scenario range representatives,
and value-proportional coalition anchors. Compare points fitness with binary
head-to-head-win fitness, each under replicator updating and the literal rule that
replaces frequencies proportionally to current payoffs. Use two declared starting
conditions, mutation rates zero and 1%, and 1,000 discrete updates. This gives 16
paths. The replacement rule is a separate behavioral hypothesis, not the replicator
equation or a Nash-equilibrium condition. Record final and time-averaged populations.

Quantize final/average mixtures to integer masses and compute full-space official
point exploitability with the exact DP. Report restricted win exploitability only;
no full-space win-probability oracle has been established. No trajectory is an
estimate of applicant frequencies, and no convergence is assumed. The candidate
pool includes human hypotheses, so these are adaptive stress constructions, not
independent human-population holdouts. Keep approximate dynamics and exact score
certificates distinct. No global claim is made for restricted win-rate best responses.

Verify exact scoring and win indicators, coalition counts and weights, marginal
recovery, source and freeze hashes, and deterministic learning recovery. Publish all
primary comparisons; label the larger archive and range fitting as exploratory.
The four-player ideas are reserved for Game 2. This study chooses no final submission.
