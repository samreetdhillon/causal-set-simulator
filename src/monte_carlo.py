import numpy as np
from src.core_operations import sprinkle, causal_matrix
from src.observables import ordering_fraction, estimate_dimension, longest_chain_length, largest_antichain

# ------------------ Scaling/Monte Carlo Studies ------------------

def scaling_study(N_list, dim=2, trials=50):

    # Compute mean and std of observables for different N values.

    results = {
        'N': [],
        'ordering_fraction_mean': [],
        'ordering_fraction_std': [],
        'dimension_mean': [],
        'dimension_std': [],
        'longest_chain_mean': [],
        'longest_chain_std': [],
        'largest_antichain_mean': [],
        'largest_antichain_std': [],
    }

    for N in N_list:
        r_list = []
        d_list = []
        L_list = []
        AC_list = []

        for _ in range(trials):
            pts = sprinkle(N, dim=dim)
            R = causal_matrix(pts, dim=dim)
            r = ordering_fraction(R)
            r_list.append(r)
            d_list.append(estimate_dimension(r))
            L_list.append(longest_chain_length(R))
            AC_list.append(largest_antichain(R))

        results['N'].append(N)
        results['ordering_fraction_mean'].append(np.mean(r_list))
        results['ordering_fraction_std'].append(np.std(r_list))
        results['dimension_mean'].append(np.mean(d_list))
        results['dimension_std'].append(np.std(d_list))
        results['longest_chain_mean'].append(np.mean(L_list))
        results['longest_chain_std'].append(np.std(L_list))
        results['largest_antichain_mean'].append(np.mean(AC_list))
        results['largest_antichain_std'].append(np.std(AC_list))

    return results

def monte_carlo_dimension(N, dim=2, trials=50):
    """
    Perform multiple sprinklings and average dimension estimate.
    Returns mean and std of estimated dimension.
    """
    estimates = []
    for _ in range(trials):
        pts = sprinkle(N, dim=dim)
        R = causal_matrix(pts, dim=dim)
        f = ordering_fraction(R)
        d_est = estimate_dimension(f)
        if d_est is not None:
            estimates.append(d_est)

    if len(estimates) == 0:
        return None, None

    mean_d = np.mean(estimates)
    std_d = np.std(estimates)
    return mean_d, std_d

def monte_carlo_longest_chain(N, dim=2, trials=50):
    """
    Compute longest chain over multiple sprinklings.
    Returns mean and std of longest chain lengths.
    """
    L_list = []
    for _ in range(trials):
        pts = sprinkle(N, dim=dim)
        R = causal_matrix(pts, dim=dim)
        L_list.append(longest_chain_length(R))
    mean_L = np.mean(L_list)
    std_L = np.std(L_list)
    return mean_L, std_L
