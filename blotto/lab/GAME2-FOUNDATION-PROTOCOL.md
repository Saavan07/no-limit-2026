**Game 2 foundation, version 1**

Preserve the selected Game 1 vector and all earlier studies. This stage establishes
Game 2 scoring and evaluation, not a population forecast or final strategy. The
user's reserved four-player ideas have not yet been supplied or incorporated.

Official rule: four players allocate 100 integer troops across five castles valued
1,2,3,4,5. Split each castle among maximum bidders, sum castle points per player,
then rank those totals. Pay 9,4,2,0, averaging occupied prizes for a ranking tie.
The exact regrouping count and implementation are not public. Treat three distinct
opponent people in a finite population as one explicit model, and independent
opponent draws from a probability distribution as another. Do not silently replace
one by the other or interpret artificial integer masses as applicant counts.

Implement a direct Fraction reference scorer and an independent vectorized integer
scorer. Castle scores use units of 1/12, sufficient for ties among 1,2,3,4 players.
Rank payouts use units of 1/4 for the stated prize vector: the three-way occupied
prize sums are divisible by three, and other ties divide by two or four. This
payout scale is specific to these prizes, not arbitrary rank contests.

Verify all castle splits and all ranking tie patterns, conservation of total castle
points and of 15 total rank payout, identical-strategy four-way ties, player
permutation invariance, invalid allocations, and all 50,625 four-player profiles
for a reduced-budget five-castle game. Reproduce scores with both implementations.

Implement exact compressed opponent-triplet enumeration. Independent sampling
uses multinomial weights on unordered vector triples; finite sampling uses
products of combinations of person multiplicities. Verify finite compression
against direct combinations of distinct people, including duplicate strategies and
exclusion of one focal applicant. Finite people may submit identical allocations.

Implement reproducible shared triplet sampling of full legal joint vectors. Finite
triplets must have distinct person IDs; iid triplets may repeat a support vector.
Use the same triplets to compare candidates. Conditional sample means use exact
integer aggregates. Report fixed-sample Hoeffding bounds separately from model
uncertainty; do not reuse a fixed-sample bound for arbitrary adaptive stopping.

Construct exact counterexamples showing (a) five castle points can win outright
or place second depending on opponents, (b) identical one-castle marginals can yield
different expected rank payouts, and (c) ordering candidates by expected castle
points can disagree with ordering by expected rank payout. Mark discovered examples
as illustrative constructions, not fresh holdout or optimal-strategy evidence.

Run small declared controls and baseline allocations to exercise the foundation.
No Game 1 marginal best-response DP or two-player minimax guarantee is transferred
to the rank-payoff objective. No global Game 2 best-response oracle, equilibrium
certificate, applicant distribution, or winning final vector is claimed.

Save rules, research scope, source/input hashes, exact fixtures and evaluations,
sample definitions and IDs, benchmark timing, verification, and a report. New broad
deep research is not needed before the exact scorer exists. If the next stage finds
a specific gap, research multi-player rank-payoff search or sample-allocation methods
with explicit applicability to this discrete four-player problem.
