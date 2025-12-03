from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence

from ga_types.ga_types import PopulationStatistics, RunResult


def ensure_directories() -> tuple[Path, Path]:
    """Ensure logs/ and outputs/ directories exist and return their paths."""
    logs_directory = Path("logs")
    outputs_directory = Path("outputs")

    logs_directory.mkdir(exist_ok=True)
    outputs_directory.mkdir(exist_ok=True)

    return logs_directory, outputs_directory


def get_timestamp_string() -> str:
    """Return timestamp string in YYYYMMDD_HHMMSS format."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def determine_next_run_number(logs_directory: Path) -> int:
    """Inspect log files and return the next available run number."""
    highest_run_number = 0

    for path in logs_directory.glob("log*_*.txt"):
        stem_parts = path.stem.split("_")
        if len(stem_parts) < 2 or stem_parts[0] != "log":
            continue

        candidate = None
        if len(stem_parts) >= 3 and stem_parts[1] == "run":
            candidate = stem_parts[2]
        elif len(stem_parts) >= 2:
            candidate = stem_parts[1]

        if candidate is None:
            continue

        try:
            run_number = int(candidate)
        except ValueError:
            continue

        highest_run_number = max(highest_run_number, run_number)

    return highest_run_number + 1


def build_main_log_path(logs_directory: Path, run_number: int, timestamp: str) -> Path:
    """Build path for main log file of a run."""
    return logs_directory / f"log_run_{run_number:03d}_{timestamp}.txt"


def build_result_plot_path(
    outputs_directory: Path, run_number: int, timestamp: str
) -> Path:
    """Build path for convergence plot file of a run."""
    return outputs_directory / f"result_run_{run_number:03d}_{timestamp}.png"


def format_population_statistics_table(
    statistics_by_population: Dict[int, PopulationStatistics],
) -> str:
    """Format a text table comparing statistics for each population size."""
    header = (
        "Population | Best x | Best f(x) | Mean of best f(x) | Worst of best f(x)\n"
        "---------- | ------ | --------- | ----------------- | ------------------"
    )
    rows: List[str] = [header]

    for population_size in sorted(statistics_by_population.keys()):
        stats = statistics_by_population[population_size]
        row = (
            f"{stats.population_size:^10} | "
            f"{stats.best_x_overall:^6} | "
            f"{stats.best_objective_overall:^9.4f} | "
            f"{stats.mean_best_objective:^17.4f} | "
            f"{stats.worst_best_objective:^18.4f}"
        )
        rows.append(row)

    return "\n".join(rows)


def write_main_run_log(
    run_number: int,
    timestamp: str,
    log_file_path: Path,
    population_sizes: Sequence[int],
    number_of_generations: int,
    mutation_rate: float,
    runs_per_population: int,
    statistics_by_population: Dict[int, PopulationStatistics],
    best_run_by_population: Dict[int, RunResult],
) -> None:
    """Write a concise summary log for the whole main.py execution."""
    # NOTE: High-level experiment summary for later inspection
    summary_table = format_population_statistics_table(statistics_by_population)

    with log_file_path.open("w", encoding="utf-8") as log_file:
        log_file.write(f"Genetic Algorithm – main run #{run_number:03d}\n")
        log_file.write(f"Timestamp: {timestamp}\n\n")

        log_file.write("Configuration\n")
        log_file.write("-------------------------\n")
        log_file.write(
            f"Population sizes: {', '.join(str(size) for size in population_sizes)}\n"
        )
        log_file.write(f"Generations per run: {number_of_generations}\n")
        log_file.write(f"Mutation rate: {mutation_rate}\n")
        log_file.write(f"Runs per population: {runs_per_population}\n\n")

        log_file.write("Summary table (best-of-100 per population)\n")
        log_file.write("-------------------------------------------\n")
        log_file.write(summary_table)
        log_file.write("\n\n")

        log_file.write("Best runs by population\n")
        log_file.write("-----------------------\n")
        for population_size in sorted(best_run_by_population.keys()):
            best_run = best_run_by_population[population_size]
            log_file.write(
                f"Population {population_size}: "
                f"x = {best_run.best_x}, "
                f"f(x) = {best_run.best_objective_value:.4f}, "
                f"seed = {best_run.seed}\n"
            )
