"""Exact tools for the additive head-to-head application game."""

from .core import (
    DEFAULT_GAME, Game, Population, ScoreTable, best_response,
    exhaustive_best_response, iter_allocations, mix_tables, score,
    score_twice, uniform_composition, multinomial, validate_allocation,
)

__version__ = "0.1.0"

__all__ = [
    "DEFAULT_GAME", "Game", "Population", "ScoreTable", "best_response",
    "exhaustive_best_response", "iter_allocations", "mix_tables", "score",
    "score_twice", "uniform_composition", "multinomial", "validate_allocation",
]
