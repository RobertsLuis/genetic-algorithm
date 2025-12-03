# Genetic Algorithm – Quadratic Function Minimization

This project implements a genetic algorithm to minimize the function:

\[
f(x) = x^2 - 4x + 4
\]

with **x encoded as a 5-bit binary vector** (1 bit for sign, 4 bits for magnitude), and **x ∈ [-15, 15]**.

The algorithm runs multiple experiments with different population sizes (4, 8, and 12 individuals) and logs statistics about the best solutions found.


Output examples:

<img width="898" height="449" alt="image" src="https://github.com/user-attachments/assets/f5028f7e-95f5-4710-b8d0-c8d09f5e72af" />
<img width="644" height="453" alt="image" src="https://github.com/user-attachments/assets/a453541f-3571-46a5-950e-29e791046809" />


---

## 1. Project goals

- Encode integers in the range **[-15, 15]** using **5 bits**  
  - First bit: sign (1 = positive/zero, 0 = negative)  
  - Remaining 4 bits: magnitude (0–15 in binary)
- Run a **genetic algorithm** with:
  - One-point crossover  
  - Bit-flip mutation  
  - Up to **10 generations** per run
- Repeat the process **100 times** for each population size (4, 8, 12)
- Compare populations using a summary table
- Generate a **convergence plot** showing the best `f(x)` per generation for one run with population size 8

---

## 2. Project structure

```text
.
├── algorithm.py              # GeneticAlgorithm implementation and helpers
├── main.py                   # Entry point – runs all experiments
├── io_utils.py               # Paths, logging, file naming, summary table
├── plot_service.py           # Convergence plot (matplotlib)
├── ga_types/
│   └── ga_types.py           # Dataclasses for results and statistics
├── logs/                     # (generated) text logs for each main run
├── outputs/                  # (generated) PNG plots for each main run
├── requirements.txt
└── README.md
```

## 3. Installation

(Recommended) Create and activate a virtual environment:
```
python -m venv venv
source venv/bin/activate      # Linux/macOS
# or
venv\Scripts\activate         # Windows
```

Install dependencies:
```
pip install -r requirements.txt
```

## 4. How to run
```
python main.py
```
