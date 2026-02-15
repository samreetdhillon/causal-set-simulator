import matplotlib.pyplot as plt
import numpy as np
from src.core_operations import sprinkle, causal_matrix, transitive_percolation
from src.observables import ordering_fraction, estimate_dimension, longest_chain_length, largest_antichain
from src.monte_carlo import monte_carlo_dimension, monte_carlo_longest_chain, scaling_study
from src.visualization import plot_causet

def run_single_mode(dim, N, padding=0.2):
    """Run single causal set generation with Bulk Truncation"""
    # 1. Sprinkle N points into the global diamond (T=1.0)
    points = sprinkle(N, dim=dim)
    
    # 2. Define the 'Bulk' sub-interval tips
    # We look at an interval between -0.5 + padding and 0.5 - padding
    t_min, t_max = -0.5 + padding, 0.5 - padding
    
    # 3. Filter points: only keep those inside the inner diamond
    # A point is in the diamond if |spatial_dist| < (t_max - t) and (t - t_min)
    t = points[:, 0]
    if dim == 2:
        x = points[:, 1]
        inner_mask = (t > t_min) & (t < t_max) & (np.abs(x) < (0.5 - padding - np.abs(t)))
    else: # dim == 3
        x, y = points[:, 1], points[:, 2]
        r = np.sqrt(x**2 + y**2)
        inner_mask = (t > t_min) & (t < t_max) & (r < (0.5 - padding - np.abs(t)))
    
    inner_points = points[inner_mask]
    N_inner = len(inner_points)
    
    if N_inner < 2:
        print("Error: Padding too high or N too low. No points in the bulk!")
        return

    # 4. Generate Causal Matrix for the BULK points only
    R_inner = causal_matrix(inner_points, dim=dim)
    
    # 5. Calculate Observables
    f = ordering_fraction(R_inner)
    d_est = estimate_dimension(f)
    L = longest_chain_length(R_inner)
    AC = largest_antichain(R_inner)

    print(f"--- Bulk Truncation Analysis (Inner N: {N_inner}) ---")
    print(f"Ordering fraction: {f:.3f}")
    if d_est is not None:
        print(f"Estimated dimension (MM): {d_est:.2f}")
    print(f"Longest chain length: {L}")
    print(f"Largest antichain size: {AC}")
    
    # Just plot the inner causet to see the improved structure
    plot_causet(inner_points, R_inner, dim=dim, title=f"Bulk Causal Set (dim={dim})")


def run_mc_mode(dim, N, trials, padding=0.0):
    """Run Monte Carlo analysis using the parallelized scaling_study engine"""
    print(f"\nStarting Monte Carlo study: N={N}, dim={dim}, trials={trials}...")
    
    # Use scaling_study to do the heavy lifting in parallel
    padding=padding
    results = scaling_study([N], dim=dim, trials=trials, padding=padding)

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

def run_scaling_mode(dim, N_list, trials, padding=0.0):
    """Run scaling study and plot results"""
    padding=padding
    results = scaling_study(N_list, dim=dim, trials=trials, padding=padding)

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
    
    # Dimension selection
    dim = int(input("Enter spacetime dimension (2 or 3): "))
    
    # Mode selection via number keys
    print("\nChoose mode:")
    print("1. Single Run")
    print("2. Monte Carlo (MC)")
    print("3. Scaling Study")
    print("4. Transitive Percolation")
    mode_choice = input("Select an option (1-4): ").strip()

    mode_map = {'1': 'single', '2': 'mc', '3': 'scaling', '4': 'percolation'}
    mode = mode_map.get(mode_choice)

    if not mode:
        print("Invalid selection. Please choose 1-4.")
        return

    # Check for Bulk Truncation if in 3D and not Percolation
    padding = 0.0
    if dim == 3 and mode in ["single", "mc", "scaling"]:
        use_trunc = input("Enable Bulk Truncation for better 3D accuracy? (y/n): ").lower()
        if use_trunc == 'y':
            padding = float(input("Enter padding (0.1 to 0.2 recommended): "))

    # Get N for modes that require a single N value
    if mode in ["single", "mc", "percolation"]:
        N = int(input(f"Enter number of sprinkled points (e.g., 20): "))

    # Execute Modes
    if mode == "single":
        # Pass padding to run_single_mode
        run_single_mode(dim, N, padding=padding)

    elif mode == "mc":
        trials = int(input("Enter number of trials: "))
        run_mc_mode(dim, N, trials, padding=padding)

    elif mode == "scaling":
        N_input = input("Enter list of N values (comma-separated, e.g., 50,100,200): ")
        N_list = [int(n.strip()) for n in N_input.split(",")]
        trials = int(input("Enter number of Monte Carlo trials per N: "))
        run_scaling_mode(dim, N_list, trials, padding=padding)

    elif mode == "percolation":
        p = float(input("Enter percolation probability p (e.g., 0.05): "))
        run_percolation_mode(dim, N, p)

    else:
        print("Invalid selection.")