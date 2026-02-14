import numpy as np
from src.core_operations import sprinkle, causal_matrix
from src.observables import ordering_fraction, estimate_dimension, longest_chain_length, largest_antichain
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

def run_single_trial(N, dim, padding=0.0):
    """Worker function for a single simulation run with optional truncation."""
    # 1. Sprinkle the total number of points
    pts = sprinkle(N, dim=dim)
    
    # 2. Apply Bulk Truncation if padding is provided
    if padding > 0:
        t = pts[:, 0]
        t_min, t_max = -0.5 + padding, 0.5 - padding
        
        if dim == 2:
            # Spatial distance |x| < (width at time t)
            mask = (t > t_min) & (t < t_max) & (np.abs(pts[:, 1]) < (0.5 - padding - np.abs(t)))
        else: # dim == 3
            # Radial distance sqrt(x^2 + y^2) < (width at time t)
            r_sq = pts[:, 1]**2 + pts[:, 2]**2
            mask = (t > t_min) & (t < t_max) & (np.sqrt(r_sq) < (0.5 - padding - np.abs(t)))
        
        pts = pts[mask]

    # 3. Calculate metrics ONLY if we have enough points left
    if len(pts) > 1:
        # Use the faster vectorized matrix function
        R = causal_matrix(pts, dim)
        
        r = ordering_fraction(R)
        d_est = estimate_dimension(r)
        L = longest_chain_length(R)
        AC = largest_antichain(R)
        
        return r, d_est, L, AC
    
    # Return failure defaults if truncation emptied the set
    return 0, None, 0, 0

def scaling_study(N_list, dim=2, trials=50, padding=0.0):
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
            trial_tasks = [(N, dim, padding)] * trials
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
