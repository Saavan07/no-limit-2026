"""Run the original tests that do not require the omitted external package."""
import unittest
from pathlib import Path

OMITTED = {
    "test_game2_fullscan.FullScanTests.test_all_actions_on_reduced_games_against_independent_foundation",
    "test_game2_fullscan.FullScanTests.test_exhaustive_profiles_in_small_games",
    "test_game2_fullscan.FullScanTests.test_full_budget_sparse_and_zero_tie_groups",
    "test_game2_cross_arena.CrossArenaTests.test_frozen_catalog_and_field_order",
    "test_game2_cross_arena.CrossArenaTests.test_native_and_foundation_group_cells_and_compression",
}

def flatten(suite):
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from flatten(item)
        else:
            yield item

if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    tests = list(flatten(unittest.defaultTestLoader.discover(str(root / "tests"))))
    for name in sorted(OMITTED):
        print("Not included, requires external Parry package:", name, flush=True)
    suite = unittest.TestSuite(t for t in tests if t.id() not in OMITTED)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)
