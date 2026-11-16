**Game 1 weighting and adversarial-contamination study, version 1**

This is a controlled mathematical sensitivity study. All 122 scenarios and 74
candidates from the multiwave study are now known evidence. There is no new holdout
claim. Do not alter previous runs. The human explanation remains reserved and unused.

Exclude the certified equilibrium control from the nominal population set. Retain
all 121 other scenarios. Partition them into analytic controls, historical transfers,
anchor perturbations, support/rounding patterns, new reasoning families, legacy
strategic constructions, adaptive generations, and adaptive archives. For every
nominal set, compute max-min score and minimax regret over all legal allocations.

Nominal sets are: all known models; all without the old empirical history archive;
all behavior models; the previous 71-model development set; that previous set without
the old history archive; and each leave-one-group-out variant of the full known set.
These are ablations of stated objectives, not estimates of which groups matter in
actual applications. Compare every selected vector on the common full 121-model set.

Create four declared behavior references: equal weight among five behavior groups,
historical-emphasis group weights (1,4,1,1,1), anchor-emphasis weights (1,1,4,1,1),
and removal of analytic controls (0,1,1,1,1). Within-group weights are equal. Strategic
reference weights the three strategic groups equally, then each model equally inside
its group. Strategic shares are 0,25,50,75,100 percent. These priors are conventions,
not inferred applicant frequencies. Preserve the exact mixture definitions.

For each objective, search every allocation conditional on each attainable exact
pure-vector security level. Certify the optimum and tie count for every such class.
For raw-score objectives and declared-reference expected scores, derive the entire
continuous contamination frontier from these conditional optima. For epsilon in
[0,1], the objective is (1-epsilon)*nominal_score(x)+epsilon*W(x), where W is the
minimum score against any legal pure opponent. For max-min populations, nominal_score
is the smallest score across the chosen scenario set. The bound also covers arbitrary
mixtures of those scenarios and an arbitrary contaminating distribution. The model
allows a worst-case opponent selected against our fixed submission; epsilon is not
an estimate of how many applicants will know or target us.

Do not call an epsilon-weighted version of nominal regret the minimax regret of a
contaminated game. Nominal regret changes its benchmark after contamination. Instead,
report genuine nominal minimax regret subject to pure-security floors as a separate
tradeoff. At epsilon=1, all maximum-security vectors tie, irrespective of nominal
performance. Count all of them and use the lexicographically smallest legal vector.

Compute pure security by the following exact structural method. If the opponent
loses fields L, their minimum losing bids are zero and those fields must have x_i>0.
Setting all remaining fields to ties costs B-sum_L x_i. Each tie converted to a win
costs one more soldier, so the opponent converts the highest-value remaining fields
first, up to min(sum_L x_i, n-|L|) fields. Spare troops can be placed in a field already
won. Include the all-tie outcome. Minimize our resulting score over all nonempty
proper L. This is an exact counter calculation, not a heuristic allocation search.

Verify it against the existing exact DP on complete tiny games and a large fixed
sample of the real game. Independently recompute every real-game security class by
enumerating loss/tie/win patterns, and independently verify every conditional optimum
using integer upper bounds over the full action space. Retain witnesses, counts,
exact transition fractions, configurations, source snapshots, and input hashes.

No new population resembles applicants merely because it is optimized exactly.
Scores remain exact conditional on existing saved model tables. Sampling guarantees
from earlier studies do not cover model validity or inferred applicant prevalence.
This study selects no final application vector and does not start Game 2.
