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
    scaling_study,
    curvature_proxy,
    transitive_percolation,
    interval_cardinalities,
    midpoint_scaling_dimension,
    benincasa_dowker_action_proxy,
    csg_sequential_growth
)

import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    print("Welcome to the Causal Set Simulator!")

    dim = int(input("Enter spacetime dimension (2 or 3): "))
    N = int(input("Enter number of sprinkled points (e.g., 20): "))
    mode = input("Choose mode: single / mc / scaling / percolation / action / csg: ").strip().lower()

    if mode == "single":
        points = sprinkle(N, dim=dim)
        R = causal_matrix(points, dim=dim)
        f = ordering_fraction(R)
        d_est = estimate_dimension(f)
        L = longest_chain_length(R)
        AC = largest_antichain(R)
        mean_dev, std_dev = curvature_proxy(R, sample_pairs=20, alpha=1.0)

        print(f"Ordering fraction: {f:.3f}")
        if d_est is not None:
            print(f"Estimated dimension (Myrheim–Meyer): {d_est:.2f}")
        print(f"Longest chain length: {L}")
        print(f"Largest antichain size: {AC}")
        plot_causet(points, R, dim=dim, title="Single Causal Set")
        print(f"Curvature proxy (mean±std): {mean_dev:.3f} ± {std_dev:.3f}  (≈0 flat)")

        

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
        devs=[]
        for _ in range(trials):
            pts = sprinkle(N, dim=dim)
            R = causal_matrix(pts, dim=dim)
            m, _ = curvature_proxy(R, sample_pairs=20, alpha=1.0)
            devs.append(m)
        print(f"Curvature proxy (MC): {np.mean(devs):.3f} ± {np.std(devs):.3f}  (≈0 flat)")
    
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

    elif mode == "percolation":
        # generate a transitive percolation causal set and show stats
        p = float(input("Enter percolation probability p (e.g., 0.05): "))
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
        plot_causet(points, R, dim=2, title=f"Transitive Percolation (p={p:.3f})")

    elif mode == "action":
        # compare action-proxy for sprinkled vs percolated sets
        choice = input("Compare which ensembles? (sprinkle / percolation): ").strip().lower()
        trials = int(input("Number of trials to average (e.g., 50): "))
        sample_pairs = int(input("Sample pairs per run for action proxy (e.g., 200): "))

        actions = []
        for _ in range(trials):
            if choice == "sprinkle":
                pts = sprinkle(N, dim=2)
                R = causal_matrix(pts, dim=2)
            else:  # percolation
                p = float(input("Enter percolation p (use same p for all runs): ")) if trials==1 else float(input("Enter percolation p (e.g., 0.05): "))
                pts, R = transitive_percolation(N, p=p)
            total_action, mean_dev, std_dev = benincasa_dowker_action_proxy(R, points=pts, sample_pairs=sample_pairs, alpha=1.0)
            actions.append(total_action)

        print(f"Action-proxy ({choice}) over {trials} trials: mean={np.mean(actions):.3f} ± {np.std(actions):.3f}")
    
    elif mode == "csg":
        # Classical Sequential Growth (CSG) demo
        print("CSG mode: exact Rideout–Sorkin sequential growth (enumerates order ideals).")
        print("Warning: exact CSG uses exponential enumeration; keep N small (<=18 recommended).")
        use_geom = input("Use geometric coupling t_k = a^k? (y/n): ").strip().lower() == 'y'
        if use_geom:
            a = float(input("Enter geometric parameter a (e.g., 0.5): "))
            points, R = csg_sequential_growth(N, geometric_a=a)
            print(f"CSG run (geometric a={a}) completed for N={N}")
        else:
            # ask for comma-separated t_k weights
            t_input = input("Enter t_k weights comma-separated (or leave blank for uniform): ").strip()
            if t_input == "":
                t_weights = None
            else:
                t_weights = [float(x.strip()) for x in t_input.split(",")]
            points, R = csg_sequential_growth(N, t_weights=t_weights)
            print(f"CSG run completed for N={N} with custom t_k (len={len(t_weights) if t_weights else 'default'})")

        f = ordering_fraction(R)
        d_est = estimate_dimension(f)
        L = longest_chain_length(R)
        AC = largest_antichain(R)
        mean_dev, std_dev = curvature_proxy(R, sample_pairs=20, alpha=1.0)

        print(f"Ordering fraction: {f:.3f}")
        if d_est is not None:
            print(f"Estimated dimension (Myrheim–Meyer): {d_est:.2f}")
        print(f"Longest chain length: {L}")
        print(f"Largest antichain size: {AC}")
        print(f"Curvature proxy (mean±std): {mean_dev:.3f} ± {std_dev:.3f}  (≈0 flat)")

        plot_causet(points, R, dim=2, title=f"CSG causal set (N={N})")

    else:
        print("Invalid mode. Choose 'single', 'mc', or 'scaling'.")

    
