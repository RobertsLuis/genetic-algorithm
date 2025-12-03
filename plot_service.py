from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from ga_types.ga_types import RunResult


def save_convergence_plot(run_result: RunResult, file_path: Path) -> Path:
    """Save a PNG plot with the best f(x) per generation for one run."""
    # NOTE: Simple convergence curve
    generation_indices = [record.generation_index for record in run_result.history]
    objective_values = [record.objective_value for record in run_result.history]

    plt.figure(figsize=(8, 4))
    plt.plot(generation_indices, objective_values, marker="o", linewidth=2)
    plt.title("Best f(x) per generation")
    plt.xlabel("Generation")
    plt.ylabel("Best f(x)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(file_path)
    plt.close()

    return file_path
