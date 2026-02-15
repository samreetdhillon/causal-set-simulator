# Causal Set Simulator

A high-performance Python simulator for manifold dimension recovery and scaling analysis in discrete Causal Set Theory.

## Problem

Causal Set Theory models spacetime as a discrete partially ordered set, but extracting geometric information such as dimension and structure from a finite causal set is non-trivial. Researchers need reliable tools to generate causal sets, compute observables, and study scaling behavior across system sizes.

## Solution

This project provides a compact simulator that can sprinkle points into Minkowski causal diamonds, compute causal relations, evaluate key observables (ordering fraction, Myrheim-Meyer dimension, chains, and antichains), and visualize the resulting Hasse diagrams.

## Methodology

- **Sprinkling**: Uniformly generate points in a 2D or 3D causal diamond.
- **Causal relations**: Build reachability matrices using lightcone conditions.
- **Observables**: Compute ordering fraction, dimension estimates, longest chains, and largest antichains.
- **Analysis**: Run single-shot, Monte Carlo, scaling, and percolation studies.
- **Visualization**: Plot 2D/3D Hasse diagrams and scaling trends.

Notebooks in `notebooks/` provide a guided walkthrough of these steps with small, fast examples.

## Results

- **(1+1)D Sprinkling:** $d \approx 2.00$
- **(2+1)D Sprinkling (Truncated):** $d \approx 2.82$
- **(2+1)D Percolation:** $d \approx 3.04$

A sample 3D single-run Hasse diagram:

![3D single run](outputs/3d-single.png)

## Relevance

The simulator enables quick experimentation with causal set observables and scaling behavior, supporting research and teaching in discrete spacetime models. Read the full project report [here](https://drive.google.com/drive/folders/1MXPDxqLKaaHKJ13FCGTtGhj1Y3F-C2i3?usp=sharing).

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
├── notebooks/                   # Jupyter notebooks
│   ├── 00_project_overview.ipynb
│   ├── 01_core_operations.ipynb
│   ├── 02_observables.ipynb
│   ├── 03_monte_carlo_and_scaling.ipynb
│   └── 04_percolation_and_visualization.ipynb
└── outputs/                     # (Empty - for generated results)
```

## Requirements

- Python 3.13+
- **numpy**: Numerical computations and random number generation
- **matplotlib**: Visualization (2D and 3D plotting)
- **networkx**: Graph algorithms (Hasse diagram drawing)
- **scipy**: Special functions and numerical optimization (for Myrheim-Meyer inversion)
- **tqdm**: Progress bars for long-running studies

Install dependencies:

```sh
pip install -r requirements.txt
```

## References

- Luca Bombelli, Joohan Lee, David Meyer, and Rafael D. Sorkin. Space-time as a causal set.
  Physical Review Letters, 59:521–524, 1987. doi: 10.1103/PhysRevLett.59.521.
- Jan Myrheim. Statistical geometry. Technical Report TH-2538, CERN, 1978.
- David A. Meyer. The Dimension of Causal Sets. PhD thesis, Massachusetts Institute of
  Technology, 1988. URL https://dspace.mit.edu/handle/1721.1/14356.
- Graham Brightwell and Ruth Gregory. Structure of random discrete spacetime. Physical
  Review Letters, 66:260–263, 1991. doi: 10.1103/PhysRevLett.66.260.
  A Simple Causal Set Simulator 9
- Robert P. Dilworth. A decomposition theorem for partially ordered sets. Annals of Mathe-
  matics, 51(1):161–166, 1950. doi: 10.2307/1969503.
- Sumati Surya. The causal set approach to quantum gravity. Living Reviews in Relativity, 22
  (1):5, 2019. doi: 10.1007/s41114-019-0023-1.
- Dionigi M. T. Benincasa and Fay Dowker. Scalar field action on a causal set. Physical Review
  Letters, 104:181301, 2010. doi: 10.1103/PhysRevLett.104.181301.

- Dowker, F. (2005). "Causal sets and the deep structure of spacetime." _arXiv preprint gr-qc/0508109_.
