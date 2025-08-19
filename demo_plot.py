from causet_mc import (
    sprinkle,
    causal_matrix,
    ordering_fraction,
    estimate_dimension,
    plot_causet,
    monte_carlo_dimension,
    longest_chain_length,
    monte_carlo_longest_chain,
    largest_antichain,
    scaling_study
)

import matplotlib.pyplot as plt

if __name__ == "__main__":
    print("Welcome to the Causal Set Simulator!")

    dim = int(input("Enter spacetime dimension (2 or 3): "))
    N = int(input("Enter number of sprinkled points (e.g., 20): "))
    mode = input("Choose mode: single / mc / scaling: ").strip().lower()

    if mode == "single":
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
        plot_causet(points, R, dim=dim, title="Single Causal Set")
        

    elif mode == "mc":
        trials = int(input("Enter number of trials: "))
        mean_d, std_d = monte_carlo_dimension(N, dim, trials=trials)
        mean_L, std_L = monte_carlo_longest_chain(N, dim=dim, trials=trials)
        if mean_d is not None:
            print(f"Monte Carlo estimated dimension: {mean_d:.2f} ± {std_d:.2f} (from {trials} trials)")
        else:
            print("Could not compute dimension estimate.")
        print(f"Monte Carlo longest chain: {mean_L:.2f} ± {std_L:.2f}")
        AC_list = []
        for _ in range(trials):
            pts = sprinkle(N, dim=dim)
            R = causal_matrix(pts, dim=dim)
            AC_list.append(largest_antichain(R))
        mean_AC = np.mean(AC_list)
        std_AC = np.std(AC_list)
        print(f"Largest antichain (MC): {mean_AC:.2f} ± {std_AC:.2f}")

    elif mode == "scaling":
        N_input = input("Enter list of N values (comma-separated, e.g., 20,50,100): ")
        N_list = [int(n.strip()) for n in N_input.split(",")]
        trials = int(input("Enter number of Monte Carlo trials per N: "))

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
    
    else:
        print("Invalid mode. Choose 'single', 'mc', or 'scaling'.")

    
