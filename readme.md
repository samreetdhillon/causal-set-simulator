# Causal Set Simulator

This project simulates and analyzes random causal sets ("sprinklings") in 2D or 3D Minkowski spacetime. It provides tools to generate causal sets, compute key observables, and visualize results.

---

## Features

- **Sprinkling**: Randomly generate points in a causal diamond in 2D or 3D spacetime.
- **Causal Matrix**: Construct the causal relation matrix for the sprinkled points.
- **Observables**: Compute ordering fraction, Myrheim–Meyer dimension estimate, longest chain, and largest antichain.
- **modes**: Study causal sets is four different modes: single, monte carlo, scaling, and transitive percolation.
- **Visualization**: Plot Hasse diagrams and scaling results.

---

## Project Structure

```
cst-demo/
├── main.py                      # Entry point - runs the simulator
├── requirements.txt             # Project dependencies
├── src/                         # Source code modules
│   ├── __init__.py             # Package initialization
│   ├── core_operations.py      # Sprinkling, causal matrix, percolation
│   ├── observables.py          # Ordering fraction, dimension estimation, chains
│   ├── monte_carlo.py          # MC simulations and scaling studies
│   ├── visualization.py        # Plotting and Hasse diagrams
│   └── interface.py            # User interaction logic
├── causalset/                   # Legacy code (deprecated)
└── outputs/                     # Output directory for results
```

### Module Breakdown

- **core_operations.py**: Core causal set operations including sprinkling, causal matrix computation, transitive percolation, and interval calculations
- **observables.py**: Observable quantities and estimators (ordering fraction, Myrheim-Meyer dimension, longest chain, largest antichain)
- **monte_carlo.py**: Statistical analysis including Monte Carlo dimension estimation and scaling studies
- **visualization.py**: Plotting tools for visualizing causal sets with Hasse diagrams
- **interface.py**: User interface for interactive simulation with mode-specific functions

---

## Requirements

- Python 3.13+
- `numpy`
- `matplotlib`
- `networkx`

Install dependencies with:

```sh
pip install -r requirements.txt
```

## Usage

Run the interactive demo:

```sh
python main.py
```

You will be prompted to select:

- Spacetime dimension (2 or 3)
- Mode: Choose from `single`, `mc`, `scaling`, or `percolation`.
- Number of sprinkled points (as required for the mode)

For each observable, the program prints the computed value. Users can refer to the documentation or literature for the physical interpretation of these results in the context of Causal Set Theory.

### Modes Explained

- **single**: Generates and analyzes one causal set, prints observables (ordering fraction, dimension, longest chain, largest antichain), and shows a Hasse diagram.
- **mc**: Based on Monte Carlo technoque, this mode estimates observables (including their mean and std) over many random sprinklings.
- **scaling**: Analyzes how observables scale with the number of points. Runs Monte Carlo studies for a list of N values, prints and plots scaling of all observables.
- **percolation**: Generates a causal set using the transitive percolation model, prints observables, and shows a Hasse diagram.

---

## Example

```
Welcome to the Causal Set Simulator!
Enter spacetime dimension (2 or 3): 2
Enter number of sprinkled points (e.g., 20): 100
Choose mode: single / mc / scaling / percolation
Ordering fraction: 0.52
Estimated dimension (Myrheim–Meyer): 2.08
Longest chain length: 14
Largest antichain size: 7
```

---

## References

- Surya, S. (2019). The Causal Set Approach to Quantum Gravity.

---
