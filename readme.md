# Causal Set Simulator

This project simulates and analyzes random causal sets ("sprinklings") in 2D or 3D Minkowski spacetime. It provides tools to generate causal sets, compute key observables, and visualize results.

## Features

- **Sprinkling**: Randomly generate points in a causal diamond in 2D or 3D spacetime.
- **Causal Matrix**: Construct the causal relation matrix for the sprinkled points.
- **Observables**: Compute ordering fraction, Myrheim–Meyer dimension estimate, longest chain, and largest antichain.
- **Monte Carlo Studies**: Estimate observables over many random sprinklings.
- **Scaling Studies**: Analyze how observables scale with the number of points.
- **Visualization**: Plot Hasse diagrams and scaling results.

## Files

- [`causet_mc.py`](causet_mc.py): Core library for generating and analyzing causal sets.
- [`demo_plot.py`](demo_plot.py): Interactive script for running simulations and plotting results.
- [`sprinkling_stats.csv`](sprinkling_stats.csv), [`csg_stats.csv`](csg_stats.csv): Example output data from scaling studies.
- `longest_chain_vs_N.png`: Example plot image.

## Requirements

- Python 3.7+
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
- Number of sprinkled points
- Mode: `single` (one causal set), `mc` (Monte Carlo), or `scaling` (study vs N)

### Modes

- **single**: Generates and analyzes one causal set, prints observables, and shows a Hasse diagram.
- **mc**: Runs multiple sprinklings, computes mean and std of observables.
- **scaling**: Runs Monte Carlo studies for a list of N values, prints and plots scaling of observables.

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
```

## References

- Myrheim, J. (1978). Statistical geometry.
- Meyer, D. A. (1988). The dimension of causal sets.

---

*Created for causal set theory exploration and visualization.*