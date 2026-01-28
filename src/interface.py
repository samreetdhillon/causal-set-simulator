import matplotlib.pyplot as plt
import numpy as np
from src.core_operations import sprinkle, causal_matrix, transitive_percolation
from src.observables import ordering_fraction, estimate_dimension, longest_chain_length, largest_antichain
from src.monte_carlo import monte_carlo_dimension, monte_carlo_longest_chain, scaling_study
from src.visualization import plot_causet


def run_single_mode(dim, N):
    """Run single causal set generation and analysis"""
    points = sprinkle(N, dim=dim)
    R = causal_matrix(points, dim=dim)
    f = ordering_fraction(R)
    d_est = estimate_dimension(f)
    L = longest_chain_length(R)
    AC = largest_antichain(R)

    print(f"Ordering fraction: {f:.3f}")
    if d_est is not None:
        print(f"Estimated dimension (Myrheim–Meyer): {d_est:.2f}")
    print(f"Longest chain length: {L}")
    print(f"Largest antichain size: {AC}")
    plot_causet(points, R, T=1.0, dim=dim, title="Single Causal Set")

def run_mc_mode(dim, N, trials):
    """Run Monte Carlo analysis using the parallelized scaling_study engine"""
    print(f"\nStarting Monte Carlo study: N={N}, dim={dim}, trials={trials}...")
    
    # Use the existing scaling_study to do the heavy lifting in parallel
    results = scaling_study([N], dim=dim, trials=trials)

    # Extract values (since [N] was a list of one, results are at index 0)
    m_d = results['dimension_mean'][0]
    s_d = results['dimension_std'][0]
    
    m_L = results['longest_chain_mean'][0]
    s_L = results['longest_chain_std'][0]
    
    m_AC = results['largest_antichain_mean'][0]
    s_AC = results['largest_antichain_std'][0]
    
    m_r = results['ordering_fraction_mean'][0]
    s_r = results['ordering_fraction_std'][0]

    # Print Report
    print("-" * 30)
    print(f"RESULTS FOR N={N} ({trials} trials)")
    print("-" * 30)
    print(f"Ordering fraction:   {m_r:.4f} ± {s_r:.4f}")
    
    if m_d is not None:
        print(f"Estimated Dimension: {m_d:.2f} ± {s_d:.2f}")
    else:
        print("Estimated Dimension: Could not invert Myrheim-Meyer.")
        
    print(f"Longest Chain (L):  {m_L:.2f} ± {s_L:.2f}")
    print(f"Largest Antichain:  {m_AC:.2f} ± {s_AC:.2f}")
    print("-" * 30)

def run_scaling_mode(dim, N_list, trials):
    """Run scaling study and plot results"""
    results = scaling_study(N_list, dim=dim, trials=trials)

    print("\nScaling Study Results:")
    for i, N in enumerate(results['N']):
        print(f"N={N}: r={results['ordering_fraction_mean'][i]:.3f} ± {results['ordering_fraction_std'][i]:.3f}, "
            f"d={results['dimension_mean'][i]:.2f} ± {results['dimension_std'][i]:.2f}, "
            f"L={results['longest_chain_mean'][i]:.1f} ± {results['longest_chain_std'][i]:.1f}, "
            f"AC={results['largest_antichain_mean'][i]:.1f} ± {results['largest_antichain_std'][i]:.1f}")

    # --- Plot scaling study ---
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    axs = axs.flatten()

    axs[0].errorbar(results['N'], results['ordering_fraction_mean'], yerr=results['ordering_fraction_std'],
                    fmt='o-', capsize=5, label='Ordering fraction r')
    axs[0].set_xlabel('N')
    axs[0].set_ylabel('r')
    axs[0].set_title('Ordering fraction vs N')
    axs[0].grid(True)

    axs[1].errorbar(results['N'], results['dimension_mean'], yerr=results['dimension_std'],
                    fmt='o-', capsize=5, label='Dimension d', color='green')
    axs[1].set_xlabel('N')
    axs[1].set_ylabel('Estimated dimension d')
    axs[1].set_title('Myrheim–Meyer dimension vs N')
    axs[1].grid(True)

    axs[2].errorbar(results['N'], results['longest_chain_mean'], yerr=results['longest_chain_std'],
                    fmt='o-', capsize=5, label='Longest chain L', color='orange')
    
    axs[2].set_xscale('log')
    axs[2].set_yscale('log')
    axs[2].set_xlabel('N')
    axs[2].set_ylabel('Longest chain L')
    axs[2].set_title('Longest chain vs N')
    axs[2].grid(True)
    axs[3].errorbar(results['N'], results['largest_antichain_mean'], yerr=results['largest_antichain_std'],
                    fmt='o-', capsize=5, label='Largest antichain AC', color='red')
    axs[3].set_xlabel('N')
    axs[3].set_ylabel('Largest antichain AC')
    axs[3].set_title('Largest antichain vs N')
    axs[3].grid(True)

    plt.tight_layout()
    plt.show()


def run_percolation_mode(dim, N, p):
    """Run transitive percolation analysis"""
    points, R = transitive_percolation(N, p=p, T=1.0)
    f = ordering_fraction(R)
    d_est = estimate_dimension(f)
    L = longest_chain_length(R)
    AC = largest_antichain(R)
    print(f"Percolation model with p={p:.3f}")
    print(f"Ordering fraction: {f:.3f}")
    if d_est is not None:
        print(f"Estimated dimension (Myrheim–Meyer): {d_est:.2f}")
    print(f"Longest chain length: {L}")
    print(f"Largest antichain size: {AC}")
    plot_causet(points, R, dim=dim, title=f"Transitive Percolation (p={p:.3f})")


def run_interactive_interface():
    """Main interactive interface for the causal set simulator"""
    print("Welcome to the Causal Set Simulator!")

    dim = int(input("Enter spacetime dimension (2 or 3): "))
    mode = input("Choose mode: single / mc / scaling / percolation: ").strip().lower()

    # Only ask for N if needed for the selected mode (but not for scaling, which asks for a list later)
    if mode in ["single", "mc", "percolation"]:
        N = int(input("Enter number of sprinkled points (e.g., 20): "))

    if mode == "single":
        run_single_mode(dim, N)

    elif mode == "mc":
        trials = int(input("Enter number of trials: "))
        run_mc_mode(dim, N, trials)

    elif mode == "scaling":
        N_input = input("Enter list of N values (comma-separated, e.g., 20,50,100): ")
        N_list = [int(n.strip()) for n in N_input.split(",")]
        trials = int(input("Enter number of Monte Carlo trials per N: "))
        run_scaling_mode(dim, N_list, trials)

    elif mode == "percolation":
        # generate a transitive percolation causal set and show stats
        p = float(input("Enter percolation probability p (e.g., 0.05): "))
        run_percolation_mode(dim, N, p)

    else:
        print("Invalid mode. Choose 'single', 'mc', 'scaling', 'percolation', 'action' or 'csg'.")
