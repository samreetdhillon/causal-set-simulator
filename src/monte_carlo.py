import numpy as np
from src.core_operations import sprinkle, causal_matrix
from src.observables import ordering_fraction, estimate_dimension, longest_chain_length, largest_antichain
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

def run_single_trial(N, dim):
    """Worker function for a single simulation run."""
    pts = sprinkle(N, dim=dim)
    R = causal_matrix(pts, dim=dim)
    
    r = ordering_fraction(R)
    d_est = estimate_dimension(r)
    L = longest_chain_length(R)
    AC = largest_antichain(R)
    
    return r, d_est, L, AC

def scaling_study(N_list, dim=2, trials=50):
    results = {
        'N': [],
        'ordering_fraction_mean': [], 'ordering_fraction_std': [],
        'dimension_mean': [], 'dimension_std': [],
        'longest_chain_mean': [], 'longest_chain_std': [],
        'largest_antichain_mean': [], 'largest_antichain_std': [],
    }

    num_procs = max(1, cpu_count() - 1) if dim == 2 else 1 # Forces 3D to stay in 1 process
    # We use a Pool to handle the parallel execution
    with Pool(processes=num_procs) as pool:
        for N in tqdm(N_list, desc="Scaling Study"):
            
            # This prepares 'trials' number of tasks for this specific N
            # pool.starmap runs run_single_trial(N, dim) multiple times in parallel
            trial_data = pool.starmap(run_single_trial, [(N, dim)] * trials)
            
            # trial_data is now a list of tuples: [(r, d, L, AC), (r, d, L, AC), ...]
            # We "unzip" this into separate lists for statistics
            r_list, d_list, L_list, AC_list = zip(*trial_data)
            
            # Filter out None values from d_list if any
            d_list = [d for d in d_list if d is not None]

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
