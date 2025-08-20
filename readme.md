# Causal Set Simulator

This project simulates and analyzes random causal sets ("sprinklings") in 2D or 3D Minkowski spacetime. It provides tools to generate causal sets, compute key observables, and visualize results.

---

## Features

- **Sprinkling**: Randomly generate points in a causal diamond in 2D or 3D spacetime.
- **Causal Matrix**: Construct the causal relation matrix for the sprinkled points.
- **Observables**: Compute ordering fraction, Myrheim–Meyer dimension estimate, longest chain, and largest antichain.
- **Curvature Proxy**: Estimate a simple curvature measure for the causal set.
- **Monte Carlo Studies**: Estimate observables over many random sprinklings.
- **Scaling Studies**: Analyze how observables scale with the number of points.
- **Visualization**: Plot Hasse diagrams and scaling results.

---

## Files

- [`causet_mc.py`](causet_mc.py): Core library for generating and analyzing causal sets.
- [`demo_plot.py`](demo_plot.py): Interactive script for running simulations and plotting results.
- Example output files: `sprinkling_stats.csv`, `csg_stats.csv`, `longest_chain_vs_N.png`.

---

## Requirements

- Python 3.7+
- `numpy`
- `matplotlib`
- `networkx`

Install dependencies with:

```sh
pip install numpy matplotlib networkx
```

---

## Usage

Run the interactive demo:

```sh
python demo_plot.py
```

You will be prompted to select:

- Spacetime dimension (2 or 3)
- Number of sprinkled points
- Mode: `single` (one causal set), `mc` (Monte Carlo), or `scaling` (study vs N)

### Modes

- **single**: Generates and analyzes one causal set, prints observables, and shows a Hasse diagram.
- **mc**: Runs multiple sprinklings, computes mean and std of observables.
- **scaling**: Runs Monte Carlo studies for a list of N values, prints and plots scaling of observables.

### New/Updated Features

- **Curvature Proxy**: Both single and Monte Carlo modes now print a "curvature proxy" (mean ± std), which gives a rough idea of the set's curvature (should be ≈0 for flat spacetime).
- **Scaling Study**: Now prints and plots the mean and standard deviation for all observables (ordering fraction, dimension, longest chain, largest antichain) as N increases.
- **Improved Output**: Results are printed in a clear, tabular format for each N in scaling mode.
- **Visualization**: Scaling mode now shows four plots: ordering fraction, dimension, longest chain, and largest antichain vs N.

---

## Example

```
Welcome to the Causal Set Simulator!
Enter spacetime dimension (2 or 3): 2
Enter number of sprinkled points (e.g., 20): 100
Choose mode: single / mc / scaling: single
Ordering fraction: 0.52
Estimated dimension (Myrheim–Meyer): 2.08
Longest chain length: 14
Largest antichain size: 7
Curvature proxy (mean±std): 0.003 ± 0.012  (≈0 flat)
```

---

## References

- Myrheim, J. (1978). Statistical geometry.
- Meyer, D. A. (1988). The dimension of causal sets.

---

\*Created for causal set theory
