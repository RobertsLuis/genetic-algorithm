from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class GenerationRecord:
    """Best individual of a single generation."""

    generation_index: int
    x_value: int
    objective_value: float


@dataclass
class RunResult:
    """Full result of one GA execution."""

    population_size: int
    seed: int | None
    best_genome: str
    best_x: int
    best_objective_value: float
    history: List[GenerationRecord]


@dataclass
class PopulationStatistics:
    """Aggregated statistics for many runs of a same population size."""

    population_size: int
    best_x_overall: int
    best_objective_overall: float
    mean_best_objective: float
    worst_best_objective: float
