**Population study protocol, version 1**

This protocol and its configuration are saved before the population study is run.
The original five Riddler files have already been inspected for basic data quality.
The split below withholds transferred scenarios from candidate selection. It is not
a claim that nobody has seen the source data, and it is not external validation on
Sequoia applicants. Source rounds can share people and are not independent cohorts.

Development uses three analytic controls, fresh equal/value-jitter and sparse
samples, and rounds 1/2 transferred through two attention profiles. All historical
transfers preserve inferred total Dirichlet concentration in development. Historical
vectors are never truncated or merged into five fields. Whole-vector divisibility
by five is used as a rounding proxy, not as an identified rounding probability.

An adapting-opponent model contains exact best responses to 512 seeded integer
belief-weight vectors over the development behavior models. The previous 5,001-mass
full-history mixture supplies a second strategic opponent. Their frequencies are
computational constructions, not forecasts of applicant behavior.

Four objectives are optimized over every legal allocation: worst score and worst
regret across behavior models, and those same objectives including the two strategic
models. There is no inferred overall applicant prior. A sensitivity reference gives
equal weight to four model groups (analytic controls, anchor perturbations, sparse,
historical), then equal weight inside each group. This is a declared comparison
convention. It is not a fitted population estimate. Solver shares vary separately.

The shortlist includes the four robust winners, each development best response,
the declared-reference sensitivity winners, and the equal/value anchors. The full
shortlist and selection inputs are hashed and saved before evaluation models are
constructed. Score ties use the lexicographically smallest vector. The exact number
of tied robust optima is retained, but no tied alternatives are chosen after seeing
evaluation results.

Validation uses fresh samples of the development synthetic families, rounds 3/4
transferred by both profiles, middle-focused and one-above-grid families, and a fresh
response cloud with a different seed. Final audit uses round 5's 13-field data under
both profiles and two concentration transfers, plus a top-two-field family. Evaluate
the entire frozen shortlist, compute each scenario's exact best-response benchmark,
and report every result. Do not replace selected candidates after these challenges.

Finite-sample scores are exact conditional on saved marginal counts. DKW bounds are
simultaneous over all candidates and all sampled scenarios, conditional on the
specified fitted generators. They do not cover feature-fitting error, historical
transfer error, unknown applicant mixture weights, or arbitrary future strategies.
Response-cloud distributions are exact finite constructions and do not use an IID
sampling claim about their relationship to applicants.

Pure-vector contamination stress replaces an epsilon fraction of each modeled
population with a legal best response to that vector. This evaluates frozen
candidates, not a global optimization of a contamination objective. A single fixed
allocation has at most 5 worst-case points in this game. An arena mixture's security
does not transfer to a selected fixed vector.

If evaluation exposes a weakness, report it. Any subsequent redesign starts a new
study with new challenges. This study does not choose or submit the application
answer, and it does not implement the four-player game.
