"""Reprice the candidate's marginal button opens using the published field counts."""
import csv
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "evidence" / "button-steal.csv"

# Estimated change in hand-end tournament equity versus folding, conditional on
# each response. These came from the original paired follow-on simulations.
FOLD_GAIN = 0.419
CALL_GAIN = -0.254
RERAISE_GAIN = -0.646


def estimate(row, extra_folds):
    donors = int(row["comparable_opens"])
    opportunities = int(row["changed_folds"]) / int(row["tables"])
    fold = int(row["both_fold"]) / donors + extra_folds
    reraise = int(row["any_reraise"]) / donors
    call = 1 - fold - reraise
    if min(fold, reraise, call) < 0:
        raise ValueError("The assumed response rates are incompatible.")
    return opportunities * (fold * FOLD_GAIN + call * CALL_GAIN + reraise * RERAISE_GAIN)


if __name__ == "__main__":
    print("Round   Field response   If folds rise 15pp")
    with DATA.open(newline="") as handle:
        for row in csv.DictReader(handle):
            print(f"{row['round']:>5} {estimate(row, 0):>16.4f} {estimate(row, 0.15):>20.4f}")
