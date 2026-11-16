> Historical research checkpoint. Later recommendations supersede earlier choices. The final choice is in report 29. Statements about application submission describe the checkpoint date only.

**Game 2: checking the feedback-loop concern before locking**

The specific winner-frequency feedback loop did not occur in the latest study. That does not establish that model overfitting or misspecification has been ruled out. **Keep (0,0,36,64,0) as the working candidate; do not treat Game 2 as ready to lock yet.**

The latest stage did not run a reproducing evolutionary arena. It froze the 760 candidates, fitted hypothetical population weights to the published incumbent-score distribution, screened those fits on independent numerical samples, then evaluated the frozen entries. The fitting loss and retention screen did not reward a model for defeating either candidate. All 101 exact C3/C4 splits were already in the frozen pool before that comparison. The C3/C4 candidate was not introduced after seeing the result of the 87-field test.

There was candidate-informed model design: two catalog variants deliberately emphasized castle-4 threshold bids and a C2/C4-targeting family. Those were deliberate stress assumptions. Removing both variants leaves 81 accepted fields and **does not change the C3/C4 recommendation**. Thus those targeted additions are not the cause of the switch.

**The population choice still matters substantially**

| Existing population set | Minimum for (0,36,0,64,0) | Minimum for (0,0,36,64,0) |
|---|---:|---:|
| Original 17 supplied fitted fields | 5.376 | 5.128 |
| All 87 accepted expanded fields | 4.884 | 5.130 |
| Expanded set excluding equal_active_targets__unrestricted__2 | 5.367 | 5.130 |

The original 17 are the unchanged source models on the earlier replay's fresh draws. The latter two rows use the expanded study's million-group draws. Differences between draws should not be confused with model changes. Within the 87-field comparison, the best C2/C4 split is 37/63, with minimum 4.891. Removing the influential field restores 36/64 as that family's best split.

Only **one** of the 87 individual model removals reverses the preferred castle pair: removing **equal_active_targets__unrestricted__2**. Removing the full equal-centered catalog also reverses the decision. Keeping only the unchanged catalog and its omissions leaves 59 fields and again favors C2/C4. This is an important qualification to the previous headline.

A minimum-across-populations rule is designed to respond to one sufficiently adverse plausible field. Dependence on that field is not, by itself, proof of overfitting. But the public score distribution does not tell us how likely that field is. The confidence bounds in the previous report measure simulation error within the supplied hypotheses; they do not validate the hypotheses as actual applicant populations.

**Was the influential population filled with the previous winner?**

No. Its exact (0,36,0,64,0) share is **0.0096%**. For a broader C2/C4-focused class, defined as at least 16 troops on each target and at most 15 total on the other castles, the share is **7.53%**. The same class has share **8.97%** in the original supplied fit_13 population, which instead favors C2/C4. A large new prevalence of this class does not explain that particular change.

Those are explicit numerical class definitions, not observed human archetype labels. The [all-field class-share table](../lab/results/game2-overfit-audit/actual-C2-C4-class-shares.csv) also includes outside-troop cutoffs of zero, eight and twenty-four. Changes in allocation shape, other families and their joint bids still matter even if the broad C2/C4 percentage is lower.

The new entry also is not an unconditional counter that dominates the old one in a four-person game. With one C3/C4 entrant and three copies of the C2/C4 entry, the C3/C4 entrant earns nine. With one C2/C4 entrant and three copies of the C3/C4 entry, the C2/C4 entrant earns nine. The exact clone controls are saved in the audit result. Directly defeating one allocation is insufficient evidence of population performance.

**How it performed before the expanded models**

(0,0,36,64,0) already earned **5.128 to 6.861** on the original seventeen supplied population hypotheses. It was a strong entry there, but C2/C4 earned more in **13 of 17** fields; C3/C4 earned more in the other 4. It did not win the original seventeen-field maximin comparison.

On our older adapting arenas, the new C3/C4 entry averaged **3.757**, compared with **3.787** for the previous C2/C4 entry and **4.180** for our historical balanced choice. Those constructed arena populations were not themselves good public-score matches. They remain stress evidence, and the new candidate was not their overall winner either.

**Lock readiness**

The result supports a provisional C3/C4 choice under the expanded worst-population criterion. It does not justify saying that all forms of overfitting are ruled out or that final selection is complete. Before locking, independently rebuild and refit the influential equal-centered assumptions, and expand allocation discovery beyond the 760 frozen entries. In particular, small scout bids around the C3/C4 leaders have not received a complete legal-action search under the broadened model set.

This audit used existing saved results and exact rational controls. It did not evolve the arena, add copies of a successful allocation, fit new populations, draw a new holdout, or alter earlier freezes and recommendations. The [population-subset comparisons](../lab/results/game2-overfit-audit/population-subset-comparisons.csv) and [all 87 single-model removals](../lab/results/game2-overfit-audit/all-87-single-model-removals.csv) make the sensitivity reviewable.
