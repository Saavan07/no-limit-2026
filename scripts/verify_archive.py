"""Verify the frozen packages and recompute the official weighted total."""
from pathlib import Path
from decimal import Decimal
import csv
import hashlib
import json
import zipfile

ROOT = Path(__file__).resolve().parents[1]
manifest = json.loads((ROOT / "poker/evidence/submission-manifest.json").read_text())
for item in manifest:
    path = ROOT / item["path"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"], path
    with zipfile.ZipFile(path) as archive:
        assert archive.testzip() is None, path
        assert archive.namelist() == ["agent.py"], path
with zipfile.ZipFile(ROOT / "poker/submissions/round-6.zip") as archive:
    assert archive.read("agent.py") == (ROOT / "poker/bot/agent.py").read_bytes()
assert (ROOT / "poker/74suited.zip").read_bytes() == (ROOT / "poker/submissions/round-6.zip").read_bytes()
rows = list(csv.DictReader((ROOT / "poker/evidence/rounds.csv").open()))
scored = [r for r in rows if Decimal(r["multiplier"]) > 0]
assert len(scored) == 6 and len(rows) == 14
assert [Decimal(r["multiplier"]) for r in scored] == list(map(Decimal, [1, 2, 3, 5, 8, 13]))
total = sum(Decimal(r["mean_table_points"]) * Decimal(r["multiplier"]) for r in scored)
assert total == Decimal("487.3205"), total
for row, item in zip(scored, manifest):
    assert row["local_scored_zip_sha256"] == item["sha256"]
print("Six scored packages verified; final source matches Round 6.")
print("Eight practices and six scored rounds; weighted total:", total)
