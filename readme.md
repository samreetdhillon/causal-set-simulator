# Causal Set Simulator

A Python simulator for studying causal sets in 2D or 3D Minkowski spacetime. Generate random sprinklings, compute observables, perform statistical analyses, and visualize results with Hasse diagrams.

```
Read the full project report [here](https://drive.google.com/file/d/10XACWFf3s4tN7q19EYO7oer3a2Uxh0Ax/view?usp=sharing)
```

---

## Features

- **Sprinkling**: Uniformly distribute points in a causal diamond (2D or 3D Minkowski spacetime)
- **Causal Relations**: Compute causal matrix using lightcone conditions
- **Observables**: Calculate ordering fraction, Myrheim–Meyer dimension, longest chain, and largest antichain
- **Analysis Modes**: Single (one causal set), Monte Carlo (statistical averages), Scaling (N-dependence), and Percolation (random partial orders)
- **Visualization**: Interactive Hasse diagrams (2D/3D) and scaling plots
- **Parallel Computing**: Efficient multi-process scaling studies

---

## Project Structure

```
cst-demo/
├── main.py                      # Entry point - runs interactive interface
├── requirements.txt             # Python dependencies
├── CITATION.cff                 # Citation metadata
├── src/                         # Core modules
│   ├── __init__.py             # Package info (version: 1.0.0)
│   ├── core_operations.py      # Sprinkling, causal relations, percolation
│   ├── observables.py          # Dimension estimation, chains, antichains
│   ├── monte_carlo.py          # Statistical analysis and scaling studies
│   ├── visualization.py        # Hasse diagrams and result plots
│   └── interface.py            # Interactive command-line interface
├── notebooks/                   # (Empty - for Jupyter notebooks)
└── outputs/                     # (Empty - for generated results)
```

### Module Breakdown

- **core_operations.py**: Core causal set functions
  - `sprinkle(N, dim, T, rng)`: Generate N uniformly random points in a causal diamond
  - `causal_matrix(points, dim)`: Compute binary causal reachability matrix using lightcone condition
  - `transitive_percolation(N, p, T, rng)`: Generate causal sets via random partial order model
- **observables.py**: Computable quantities on causal sets
  - `ordering_fraction(R)`: Fraction of causally related pairs
  - `estimate_dimension(r)`: Invert Myrheim–Meyer relation to estimate spacetime dimension
  - `longest_chain_length(R)`: Maximum length chain via dynamic programming
  - `largest_antichain(R)`: Maximum antichain width (Dilworth's theorem)
- **monte_carlo.py**: Statistical and scaling analysis
  - `run_single_trial(N, dim)`: Execute one causal set and compute observables
  - `scaling_study(N_list, dim, trials)`: Parallel Monte Carlo for multiple N values
  - `monte_carlo_dimension(N, dim, trials)`: Average dimension estimate
  - `monte_carlo_longest_chain(N, dim, trials)`: Average chain length
- **visualization.py**: Plotting utilities
  - `hasse_edges_from_R(R)`: Extract covering relations for transitive reduction
  - `plot_causet(points, R, T, dim, title, show, save_path, draw_hasse)`: Draw Hasse diagram (2D and 3D)
- **interface.py**: Interactive user interface
  - `run_single_mode(dim, N)`: Single causal set generation and analysis
  - `run_mc_mode(dim, N, trials)`: Monte Carlo statistical study
  - `run_scaling_mode(dim, N_list, trials)`: Scaling analysis with plots
  - `run_interactive_interface()`: Main prompt-based menu

---

## Requirements

- Python 3.13+
- **numpy**: Numerical computations and random number generation
- **matplotlib**: Visualization (2D and 3D plotting)
- **networkx**: Graph algorithms (Hasse diagram drawing)
- **scipy**: Special functions and numerical optimization (for Myrheim–Meyer inversion)
- **tqdm**: Progress bars for long-running studies

Install dependencies:

```sh
pip install -r requirements.txt
```

## Usage

### Interactive Mode

Run the interactive interface:

```sh
python main.py
```

You will be prompted to:

1. Select spacetime dimension (2 or 3)
2. Choose an analysis mode: `single`, `mc`, `scaling`, or `percolation`
3. Specify parameters (number of points, trials, etc.)

The program computes observables and displays visualizations.

### Command-Line API

You can also import modules directly:

```python
from src.core_operations import sprinkle, causal_matrix
from src.observables import ordering_fraction, estimate_dimension, longest_chain_length, largest_antichain
from src.monte_carlo import scaling_study

# Generate a causal set
points = sprinkle(N=100, dim=2)
R = causal_matrix(points, dim=2)

# Compute observables
r = ordering_fraction(R)
d = estimate_dimension(r)
L = longest_chain_length(R)
A = largest_antichain(R)

print(f"Ordering fraction: {r:.3f}")
print(f"Estimated dimension: {d:.2f}")
print(f"Longest chain: {L}")
print(f"Largest antichain: {A}")
```

## Analysis Modes

### `single`

Generates one causal set, computes all observables, and displays a Hasse diagram.

- **Output**: Ordering fraction, estimated dimension, longest chain, largest antichain, plus visualization
- **Use case**: Explore individual causal sets

### `mc` (Monte Carlo)

Runs multiple causal set simulations and computes mean and standard deviation of observables.

- **Output**: Statistical summary (mean ± std) for all observables
- **Use case**: Measure fluctuations and convergence of estimators

### `scaling`

Studies how observables scale with the number of points. Performs Monte Carlo analysis across multiple N values.

- **Output**: Tables of scaling data, 4-panel plot showing N-dependence of r, d, L, and AC
- **Use case**: Study continuum limit behavior and critical scaling

### `percolation`

Generates a causal set using a random partial order (transitive percolation) rather than geometric sprinkling.

- **Parameters**: Probability p of direct causal links
- **Output**: Observables and Hasse diagram for the percolated causal set
- **Use case**: Explore non-geometric causal structures

## Example Output

```
Welcome to the Causal Set Simulator!
Enter spacetime dimension (2 or 3): 2
Enter number of sprinkled points (e.g., 20): 100
Choose mode: single / mc / scaling / percolation
> single

Ordering fraction: 0.476
Estimated dimension (Myrheim–Meyer): 2.05
Longest chain length: 14
Largest antichain size: 8
[Hasse diagram displayed]
```

---

## Theory

This simulator is based on **Causal Set Theory** (CST), a discrete approach to quantum gravity. Key concepts:

- **Sprinkling**: In CST, spacetime is approximated as a countable, locally finite poset arising from Poisson processes in continuous spacetime
- **Ordering Fraction** ($r$): The density of causal relations; converges to a dimension-dependent theoretical value
- **Myrheim–Meyer Dimension**: Inverts the relation $$r(d) = \frac{\Gamma(d+1)\Gamma(d/2)}{4\Gamma(3d/2)}$$ to estimate $d$
- **Chains and Antichains**: Longest chain estimates the "height" of spacetime; largest antichain its "width"

The simulator allows quantitative exploration of these structures.

---

## Citation

If you use this software in research, please cite:

```bibtex
@software{dhillon2025causalset,
  author       = {Dhillon, Samreet},
  title        = {Causal Set Simulator},
  year         = 2025,
  url          = {https://github.com/samreetdhillon/cst-demo},
  version      = {1.0.0}
}
```

Or using the provided `CITATION.cff` file.

---

## References

- Surya, S. (2019). "The Causal Set Approach to Quantum Gravity." _Living Reviews in Relativity_, 22(1), 5.
- Bombelli, L., Lee, J., Meyer, D., & Sorkin, R. (1987). "Space-time as a causal set." _Physical Review Letters_, 59(5), 521.
- Dowker, F. (2005). "Causal sets and the deep structure of spacetime." _arXiv preprint gr-qc/0508109_.

---

## License

This project is provided as-is for research and educational purposes.

---
