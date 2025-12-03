from __future__ import annotations

import random
from typing import List, Sequence, Tuple

from ga_types.ga_types import GenerationRecord, RunResult

# NOTE: Basic encoding configuration for x in [-15, 15]
GENOME_LENGTH = 5          # 1 bit sign, 4 bits magnitude
MAGNITUDE_BITS = GENOME_LENGTH - 1
MAX_MAGNITUDE = (1 << MAGNITUDE_BITS) - 1

MIN_X = -MAX_MAGNITUDE
MAX_X = MAX_MAGNITUDE


def evaluate_function(x_value: int) -> float:
    """Compute f(x) = x^2 - 4x + 4."""
    return x_value * x_value - 4 * x_value + 4


def encode_integer(x_value: int) -> str:
    """Encode x in [-15, 15] as 5-bit genome (sign + magnitude)."""
    if x_value < MIN_X or x_value > MAX_X:
        raise ValueError(f"x must be in [{MIN_X}, {MAX_X}], got {x_value}.")

    sign_bit = "1" if x_value >= 0 else "0"
    magnitude = abs(x_value)
    magnitude_bits = format(magnitude, f"0{MAGNITUDE_BITS}b")
    return sign_bit + magnitude_bits


def decode_genome(genome: str) -> int:
    """Decode 5-bit genome into integer x."""
    if len(genome) != GENOME_LENGTH or any(bit not in {"0", "1"} for bit in genome):
        raise ValueError(f"Invalid genome: {genome!r}")

    sign = 1 if genome[0] == "1" else -1
    magnitude = int(genome[1:], 2)
    x_value = sign * magnitude

    if x_value < MIN_X or x_value > MAX_X:
        raise ValueError(f"Decoded value {x_value} is outside the allowed range.")

    return x_value


class GeneticAlgorithm:
    """Genetic algorithm specialized for minimizing f(x) on [-15, 15]."""

    def __init__(
        self,
        population_size: int,
        number_of_generations: int,
        mutation_rate: float,
        seed: int | None = None,
    ) -> None:
        """Configure GA with population, generations, mutation rate and seed."""
        self.population_size = population_size
        self.number_of_generations = number_of_generations
        self.mutation_rate = mutation_rate
        self.seed = seed
        self.random_generator = random.Random(seed)

    def run(self) -> RunResult:
        """Run one GA execution and return the best individual and history."""
        # NOTE: Standard GA loop – init, evolve, track best
        population = self._create_initial_population()

        best_genome, best_x, best_value = self._get_best_individual(population)
        history: List[GenerationRecord] = [
            GenerationRecord(
                generation_index=0,
                x_value=best_x,
                objective_value=best_value,
            )
        ]

        for generation_index in range(1, self.number_of_generations + 1):
            population = self._create_next_generation(population)

            generation_best_genome, generation_best_x, generation_best_value = (
                self._get_best_individual(population)
            )

            history.append(
                GenerationRecord(
                    generation_index=generation_index,
                    x_value=generation_best_x,
                    objective_value=generation_best_value,
                )
            )

            if generation_best_value < best_value:
                best_genome = generation_best_genome
                best_x = generation_best_x
                best_value = generation_best_value

        return RunResult(
            population_size=self.population_size,
            seed=self.seed,
            best_genome=best_genome,
            best_x=best_x,
            best_objective_value=best_value,
            history=history,
        )

    def _create_initial_population(self) -> List[str]:
        """Create first generation with random x values in the valid range."""
        population: List[str] = []
        for _ in range(self.population_size):
            random_x = self.random_generator.randint(MIN_X, MAX_X)
            population.append(encode_integer(random_x))
        return population

    def _get_best_individual(self, population: Sequence[str]) -> Tuple[str, int, float]:
        """Return genome, x and f(x) of the best individual in a population."""
        best_genome: str | None = None
        best_x: int | None = None
        best_value: float | None = None

        for genome in population:
            x_value = decode_genome(genome)
            objective_value = evaluate_function(x_value)

            if best_value is None or objective_value < best_value:
                best_value = objective_value
                best_genome = genome
                best_x = x_value

        assert best_genome is not None and best_x is not None and best_value is not None
        return best_genome, best_x, best_value

    def _create_next_generation(self, current_population: Sequence[str]) -> List[str]:
        """Build next generation with selection, crossover and mutation."""
        next_population: List[str] = []

        while len(next_population) < self.population_size:
            parent_a = self._select_parent(current_population)
            parent_b = self._select_parent(current_population)

            child_a, child_b = self._crossover(parent_a, parent_b)
            mutated_child_a = self._mutate(child_a)
            mutated_child_b = self._mutate(child_b)

            next_population.append(mutated_child_a)
            if len(next_population) < self.population_size:
                next_population.append(mutated_child_b)

        return next_population

    def _select_parent(self, population: Sequence[str]) -> str:
        """Select parent using tournament of size 2 (best of two)."""
        competitor_a, competitor_b = self.random_generator.sample(population, k=2)

        x_a = decode_genome(competitor_a)
        x_b = decode_genome(competitor_b)

        value_a = evaluate_function(x_a)
        value_b = evaluate_function(x_b)

        return competitor_a if value_a <= value_b else competitor_b

    def _crossover(self, parent_a: str, parent_b: str) -> Tuple[str, str]:
        """Apply one-point crossover to two parents."""
        crossover_point = self.random_generator.randint(1, GENOME_LENGTH - 1)
        child_a = parent_a[:crossover_point] + parent_b[crossover_point:]
        child_b = parent_b[:crossover_point] + parent_a[crossover_point:]
        return child_a, child_b

    def _mutate(self, genome: str) -> str:
        """Flip bits in genome with probability given by mutation_rate."""
        genome_bits = list(genome)

        for index in range(GENOME_LENGTH):
            if self.random_generator.random() < self.mutation_rate:
                genome_bits[index] = "1" if genome_bits[index] == "0" else "0"

        return "".join(genome_bits)


def run_genetic_algorithm_for_population(
    population_size: int,
    number_of_generations: int,
    mutation_rate: float,
    seed: int | None = None,
) -> RunResult:
    """Helper to run one GA execution with the given configuration."""
    algorithm = GeneticAlgorithm(
        population_size=population_size,
        number_of_generations=number_of_generations,
        mutation_rate=mutation_rate,
        seed=seed,
    )
    return algorithm.run()
