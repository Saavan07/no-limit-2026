**Game 1 failure attribution and weight sensitivity, version 1**

This is a diagnostic stage using known outcomes from the completed merged study.
All prior development, validation, and audit results are now known evidence.
No new holdout claim, applicant-frequency estimate, or final submission is made.
The merged checkpoint, its source files, and all previous studies stay unchanged.

Focus on the eight predeclared merged primary allocations and the previous
machine max-min allocation. For each, retain every tie for lowest score and
highest regret over all non-control scenarios, plus every highest-regret tie
inside the previous audit set. Deduplicate the resulting critical scenarios.
This outcome-driven rule is explicit. Counts of selected cases have no prevalence
meaning. Keep their original phase and evidence kind.

For every selected candidate/scenario pair, reconstruct exact castle marginals
from the saved legal score table. Attribute score and regret to each castle,
report the global best response, local win/tie/loss probabilities, and the best
single-troop transfer. Castle contributions may have either sign. Local transfers
are diagnostics, not certificates of a global optimum. Preserve marginal atoms
and source metadata; do not infer joint strategy correlations from marginals.

Use the declared midpoint prior from the merged study as a common baseline.
For every critical scenario Q, compute the complete exact expected-score response
map for P(alpha)=(1-alpha)*P0+alpha*Q, alpha in [0,1], both unrestricted and subject
to a pure-security floor of two. The security floor is equivalent, for this exact
100-troop, values-1-through-5 game, to x1<=3. Prove this and verify it across every
legal vector independently using outcome patterns. Do not generalize it to other
budgets or castle values.

Alpha is the mass replaced with Q; Q may already overlap with P0. It is not an
estimate of total prevalence of that opponent type. Report the first optimal-line
change and the first alpha at which the baseline-optimal vector loses at least
0.10, 0.25, or 0.50 points relative to the optimal response in the same action
class. Those opportunity costs are conditional sensitivity measures, not evidence
that a particular alpha is realistic.

Also compute the exact upper envelope among the nine frozen focus candidates.
Report their pairwise tradeoffs against the same mixture rather than silently
substituting newly optimized allocations for them. New vectors on global maps are
exploratory diagnostics and will need a subsequent fresh challenge before final
selection. Do not replace a frozen primary using these known outcomes.

Discover global response intervals with exact rational oracles and directional
tie selection. Independently certify every segment endpoint by a separately
implemented constrained dynamic program, and verify interior/boundary allocations
and exact tie counts. Linearity then certifies entire intervals. Compare against
complete enumeration in small games, including ties and cap restrictions.
Verify all saved attribution cells, marginal reconstruction, single-troop moves,
source hashes, and selected-extreme accounting. Keep the Game 2 phase reserved.
