**Game 2 human hypotheses and population sensitivity, version 1**

Preserve the user's three literal vectors: (2,2,3,4,89), (1,1,1,1,96), and
(0,0,0,0,100). Preserve the reasoning as hypotheses: a unique all-in five can
secure repeated high placements; clone collisions reduce its value; abandoning
five can exploit wasted fifth-field spending; broadly spread allocations might
be weak. Analyst-generated allocations are labeled separately. No applicant
frequencies, equilibrium shares, or final submission are inferred.

Check exact scoring first. A unique all-in five has at least five castle points,
so its rank payout is at least four. With clone opponents, zero-bid castles still
split their values. Enumerate outcome patterns and retain feasible witnesses for
one, two, or three other clones. Prove the all-in strategy's pure worst-case payout
is one, using the unique-copy bound and tied identical players plus a legal witness.
Show that pure all-in-five self-play is not an equilibrium by a legal unilateral
deviation earning nine. Do not infer absence or presence in a mixed equilibrium.

The known development population has four explicitly declared families: exact
all-in five (one anchor), abandon/limp-five (five anchors), broader allocation
(five anchors), near-all-in five (the two literal 89/96 anchors). Within-family
anchor weights are equal. Evaluate the complete family simplex on a 10% grid
(286 settings), plus the equal-family center. Integer masses encode probabilities,
not 100 actual applicants. Retain full joint vectors and exact iid triple weights.

Generate a candidate field independently of these evaluation outcomes: literal
vectors, every anchor, the fixed Game 1 recommendation, a scout/3+4 grid with fifth
bids 0,1,3, and 384 seeded broad/coarse legal vectors. Exact conditional scores on
the finite anchor models support comparison within this finite field only. No
global Game 2 best response or certificate over all 4,598,126 vectors is claimed.

Predeclare primary comparisons: the three human literals; the Game 1 reference;
the balanced reference; best exact-zero-five, best small-positive-five, best broad,
overall center-score leader, and worst-grid-score leader. Fit only on the known
anchor models. Retain top center and worst-grid alternatives, anchors, and primaries
in a shortlist capped by the stated selection procedure. Freeze the field, all
primary choices, shortlist, sources, grid weights, and fresh generator definitions
before constructing fresh evaluation triplets.

For all-in five, derive exact conditional coefficients a_k for k other all-in
players, under five declared non-copy references. Expected iid payout is the cubic
sum_k binomial(3,k)*p^k*(1-p)^(3-k)*a_k. Save coefficients and curves. The probability
of no other exact copy is (1-p)^3; a small all-in population does not make zero-copy
groups rare. Distinguish exact copies from near-all-in five players.

After the freeze, test twelve declared family mixes under three new legal generator
styles, using 65,536 shared iid opponent triples per scenario. Fresh variants change
scouts, zero/positive fifth bids, concentration, rounding, and near-all-in bids.
Report all frozen primary outcomes and conditional sample uncertainty. Scenario
counts and a pooled average are comparison conventions, not applicant forecasts.
The same triplets are shared across candidates. Keep sampled error and model error
separate. This stage does not contain an evolving Game 2 arena; that comes after
these structural hypotheses and the first population challenge.

A faster scorer may reuse opponent maxima and tied-winner masks, but must reproduce
the foundation integer scorer exactly. Verify all candidate/anchor triples, all
probability masses and self-play calibration, clone witnesses, polynomial values,
fresh sample regeneration, every sample score with the optimized kernel, direct
foundation scoring for every primary on all fresh triples, and freeze integrity.
Preserve the selected Game 1 vector. New fitting after this challenge would need a
subsequent fresh challenge before final selection.
