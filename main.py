from __future__ import annotations

import statistics
from typing import Dict, List

from algorithm import run_genetic_algorithm_for_population
from ga_types.ga_types import PopulationStatistics, RunResult
from io_utils import (
    build_main_log_path,
    build_result_plot_path,
    determine_next_run_number,
    ensure_directories,
    format_population_statistics_table,
    get_timestamp_string,
    write_main_run_log,
)
from plot_service import save_convergence_plot

# NOTE: Global configuration for the GA experiments
NUMBER_OF_GENERATIONS = 10
MUTATION_RATE = 0.05
NUMBER_OF_RUNS_PER_POPULATION = 100
POPULATION_SIZES = (4, 8, 12)


def main() -> None:
    """Run all experiments and orchestrate logging and plotting."""
    print("Starting genetic algorithm experiments...\n")

    logs_directory, outputs_directory = ensure_directories()
    run_number = determine_next_run_number(logs_directory)
    timestamp = get_timestamp_string()

    global_run_number = 1

    statistics_by_population: Dict[int, PopulationStatistics] = {}
    best_values_by_population: Dict[int, List[float]] = {}
    best_run_by_population: Dict[int, RunResult] = {}

    run_result_for_plot: RunResult | None = None
    run_number_for_plot: int | None = None

    # NOTE: Outer loop – different population sizes
    for population_size in POPULATION_SIZES:
        print(
            f"Running {NUMBER_OF_RUNS_PER_POPULATION} runs for population size {population_size}..."
        )
        best_values_by_population[population_size] = []

        # Inner loop – multiple runs per population
        for run_index in range(1, NUMBER_OF_RUNS_PER_POPULATION + 1):
            run_result = run_genetic_algorithm_for_population(
                population_size=population_size,
                number_of_generations=NUMBER_OF_GENERATIONS,
                mutation_rate=MUTATION_RATE,
                seed=None,
            )

            best_values_by_population[population_size].append(
                run_result.best_objective_value
            )

            existing_best_run = best_run_by_population.get(population_size)
            if (
                existing_best_run is None
                or run_result.best_objective_value
                < existing_best_run.best_objective_value
            ):
                best_run_by_population[population_size] = run_result

            if population_size == 8 and run_result_for_plot is None:
                run_result_for_plot = run_result
                run_number_for_plot = global_run_number

            print(
                f"  Run {run_index:3d}/{NUMBER_OF_RUNS_PER_POPULATION} "
                f"(global #{global_run_number:03d}): "
                f"best x = {run_result.best_x}, "
                f"f(x) = {run_result.best_objective_value:.4f}"
            )

            global_run_number += 1

        best_values = best_values_by_population[population_size]
        mean_best_value = statistics.fmean(best_values)
        worst_best_value = max(best_values)
        best_run = best_run_by_population[population_size]

        statistics_by_population[population_size] = PopulationStatistics(
            population_size=population_size,
            best_x_overall=best_run.best_x,
            best_objective_overall=best_run.best_objective_value,
            mean_best_objective=mean_best_value,
            worst_best_objective=worst_best_value,
        )

        print(
            f"Finished population size {population_size}: "
            f"best f(x) = {best_run.best_objective_value:.4f} at x = {best_run.best_x}, "
            f"mean best f(x) = {mean_best_value:.4f}, "
            f"worst best f(x) = {worst_best_value:.4f}.\n"
        )

    print("Global summary:\n")
    summary_table = format_population_statistics_table(statistics_by_population)
    print(summary_table)

    log_file_path = build_main_log_path(logs_directory, run_number, timestamp)
    write_main_run_log(
        run_number=run_number,
        timestamp=timestamp,
        log_file_path=log_file_path,
        population_sizes=POPULATION_SIZES,
        number_of_generations=NUMBER_OF_GENERATIONS,
        mutation_rate=MUTATION_RATE,
        runs_per_population=NUMBER_OF_RUNS_PER_POPULATION,
        statistics_by_population=statistics_by_population,
        best_run_by_population=best_run_by_population,
    )
    print(f"\nExecution log written to: {log_file_path}")

    if run_result_for_plot is not None and run_number_for_plot is not None:
        plot_path = build_result_plot_path(outputs_directory, run_number, timestamp)
        save_convergence_plot(run_result_for_plot, plot_path)
        print(f"Convergence plot for population size 8 written to: {plot_path}")


if __name__ == "__main__":
    main()
