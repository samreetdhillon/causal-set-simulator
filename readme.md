# Causal Set Simulator

This project simulates and analyzes random causal sets ("sprinklings") in 2D or 3D Minkowski spacetime. It provides tools to generate causal sets, compute key observables, and visualize results. Several models and analysis modes are included, such as percolation, action proxies, and classical sequential growth.

---

## Features

- **Sprinkling**: Randomly generate points in a causal diamond in 2D or 3D spacetime.
- **Causal Matrix**: Construct the causal relation matrix for the sprinkled points.
- **Observables**: Compute ordering fraction, Myrheim–Meyer dimension estimate, longest chain, and largest antichain.
- **Curvature Proxy**: Estimate a simple curvature measure for the causal set.
- **Benincasa–Dowker Action Proxy**: Compute an action-like observable for causal sets.
- **Transitive Percolation Model**: Generate random causal sets using percolation.
- **Classical Sequential Growth (CSG) Model**: Generate causal sets using the Rideout–Sorkin sequential growth process.
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

- Python 3.13+
- `numpy`
- `matplotlib`
- `networkx`

Install dependencies with:

```sh
pip install numpy matplotlib networkx
```

## Usage

Run the interactive demo:

```sh
python demo_plot.py
```

You will be prompted to select:

- Spacetime dimension (2 or 3)
- Mode: Choose from `single`, `mc`, `scaling`, `percolation`, `action`, or `csg`.
- Number of sprinkled points (as required for the mode)

For each observable, the program prints the computed value. Users can refer to the documentation or literature for the physical interpretation of these results in the context of Causal Set Theory.

### Modes Explained

- **single**: Generates and analyzes one causal set, prints observables (ordering fraction, dimension, longest chain, largest antichain, curvature proxy), and shows a Hasse diagram.
- **mc**: Runs multiple sprinklings, computes mean and std of all observables, including the curvature proxy.
- **scaling**: Runs Monte Carlo studies for a list of N values, prints and plots scaling of all observables.
- **percolation**: Generates a causal set using the transitive percolation model, prints observables, and shows a Hasse diagram.
- **action**: Computes the Benincasa–Dowker action proxy for either sprinkled or percolated sets, over multiple trials.
- **csg**: Generates a causal set using the Rideout–Sorkin CSG model (with geometric or custom weights), prints observables, and shows a Hasse diagram.

### New/Updated Features

- **Curvature Proxy**: All relevant modes print a "curvature proxy" (mean ± std), which gives a rough idea of the set's curvature (should be ≈0 for flat spacetime).
- **Benincasa–Dowker Action Proxy**: The `action` mode computes an action-like observable for causal sets, useful for comparing different models.
- **Classical Sequential Growth (CSG) Model**: The `csg` mode demonstrates the Rideout–Sorkin sequential growth process for small N.
- **Scaling Study**: Prints and plots the mean and standard deviation for all observables (ordering fraction, dimension, longest chain, largest antichain) as N increases.
- **mc**: Runs multiple sprinklings, computes mean and std of observables.
- **scaling**: Runs Monte Carlo studies for a list of N values, prints and plots scaling of observables.

---

## Example

```
Welcome to the Causal Set Simulator!
Enter spacetime dimension (2 or 3): 2
Enter number of sprinkled points (e.g., 20): 100
Choose mode: single / mc / scaling / percolation / action / csg: single
Ordering fraction: 0.52
Estimated dimension (Myrheim–Meyer): 2.08
Longest chain length: 14
Largest antichain size: 7
Curvature proxy (mean±std): 0.003 ± 0.012  (≈0 flat)
```

---

## References

- Surya, S. (2019). The Causal Set Approach to Quantum Gravity.

---

\*Created for causal set theory
