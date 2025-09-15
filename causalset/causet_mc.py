import numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt
import networkx as nx

# ------------------ Sprinkling ------------------

def sprinkle(N, dim=2, T=1.0, rng=None):
    """
    Sprinkle N points uniformly in a d-dimensional Minkowski causal diamond.
    dim = 2 or 3
    """
    rng = np.random.default_rng() if rng is None else rng
    points = []

    if dim == 2:
        while len(points) < N:
            t = rng.uniform(-T/2, T/2)
            x = rng.uniform(-T/2, T/2)
            if abs(x) <= (T/2 - abs(t)):
                points.append((t, x))
    elif dim == 3:
        while len(points) < N:
            t = rng.uniform(-T/2, T/2)
            rmax = T/2 - abs(t)
            x = rng.uniform(-rmax, rmax)
            y = rng.uniform(-rmax, rmax)
            if x**2 + y**2 <= rmax**2:
                points.append((t, x, y))
    else:
        raise ValueError("dim must be 2 or 3")
    
    return np.array(points)

# ------------------ Causal relations ------------------

def causal_matrix(points, dim=None):
    """
    Construct causal adjacency (reachability) matrix.
    R[i,j] = 1 if i precedes j in causal set (inside lightcone)
    """
    N = len(points)
    R = np.zeros((N, N), dtype=int)
    for i in range(N):
        for j in range(N):
            if points[i][0] < points[j][0]:  # time ordering
                dt = points[j][0] - points[i][0]
                dx2 = np.sum((np.array(points[j][1:]) - np.array(points[i][1:]))**2)
                if dx2 <= dt**2:  # within lightcone
                    R[i, j] = 1
    return R

# ------------------ Growth models & helpers ------------------

def _transitive_closure_bool(R):
    """
    Compute transitive closure of boolean adjacency R (NxN) using repeated squaring/DP.
    R can be int/bool; result is int matrix (0/1).
    O(N^3) worst-case but fine for moderate N.
    """
    reach = (R != 0).astype(bool)
    if reach.size == 0:
        return reach.astype(int)
    while True:
        # (reach @ reach) produces integer counts; >0 turns into boolean reachability in 2 steps
        new = reach | ((reach @ reach) > 0)
        if np.array_equal(new, reach):
            break
        reach = new
    return reach.astype(int)


def transitive_percolation(N, p=0.1, T=1.0, rng=None):
    """
    Generate a causal set via transitive percolation (random partial order).
    - N: number of elements
    - p: probability for each future-directed pair to have a direct link
    - T: time embedding span (for plotting). We produce simple spacetime coords (t,x)
         with times sorted so that i<j implies t_i <= t_j (monotone).
    Returns: points (N,2) array: (t, x) for plotting/embedding, and reachability matrix R (N,N)
    NOTE: The model builds an upper-triangular random adjacency matrix then closes transitively.
    """
    rng = np.random.default_rng() if rng is None else rng
    # create random times and sort them (so time-ordering is present)
    t = rng.uniform(-T/2, T/2, size=N)
    order = np.argsort(t)
    t_sorted = t[order]
    # create spatial coordinate for plotting
    x = rng.uniform(-T/2, T/2, size=N)
    x_sorted = x[order]
    points = np.column_stack((t_sorted, x_sorted))

    # build upper-triangular random adjacency (i precedes j only if i < j in time-order)
    R = np.zeros((N, N), dtype=int)
    # for i < j: add direct link with probability p
    iu = np.triu_indices(N, k=1)
    rand = rng.random(size=len(iu[0]))
    R[iu] = (rand < p).astype(int)

    # transitive closure
    R = _transitive_closure_bool(R)

    return points, R

# ------------------ Intervals and Local Structures ------------------

def _interval_elements(R, p, q):
    """Return indices in the Alexandrov interval I(p,q) = { r | p ≺ r ≺ q }."""
    # R[p, r] True means p ≺ r ; R[r, q] True means r ≺ q
    between = np.where(R[p, :] & R[:, q])[0]
    return between[(between != p) & (between != q)]


def interval_cardinalities(R):
    """
    Return list of sizes of Alexandrov intervals I(p,q) for all comparable pairs p≺q.
    Uses _interval_elements helper from your file.
    """
    comps = np.argwhere(R)
    sizes = []
    for p, q in comps:
        I = _interval_elements(R, p, q)
        sizes.append(len(I))
    return sizes

# ------------------ Observables/Estimators ------------------

def ordering_fraction(R):
    """Fraction of related pairs r = #relations / total pairs"""
    N = R.shape[0]
    total_pairs = N*(N-1)
    if total_pairs == 0:
        return 0.0
    related_pairs = np.sum(R)
    return related_pairs / total_pairs

def estimate_dimension(r):
    """
    Estimate spacetime dimension from ordering fraction using
    the Myrheim–Meyer relation: r(d) = 1 - 1/2^(d-1)
    Inversion: d = 1 + log(1/(1-r))/log(2)
    """
    if r <= 0 or r >= 1:
        return None
    return 1 + np.log(1/(1-r)) / np.log(2)

def longest_chain_length(R):
    """Longest chain length using dynamic programming on DAG R"""
    N = R.shape[0]
    L = np.ones(N, dtype=int)
    for j in range(N):
        preds = np.where(R[:, j])[0]
        if preds.size:
            L[j] = 1 + np.max(L[preds])
    return int(np.max(L)) if N > 0 else 0

def _longest_chain_length_from_to(S, s, t):
    """
    Longest chain length in DAG S from node index s to node index t.
    S is boolean reachability on the subposet.
    """
    n = S.shape[0]
    # topological order: we can just use numeric since S is upper-triangular if original was time-ordered,
    # but to be safe, do a DP using reachability.
    L = np.zeros(n, dtype=int)
    order = range(n)
    for j in order:
        preds = np.where(S[:, j])[0]
        if preds.size:
            L[j] = max(L[p] for p in preds) + 1
        else:
            L[j] = 1
    # We want paths that start at s and end at t. If no path, return 0.
    if not S[s, t] and s != t:
        return 0
    # Recompute DP constrained to nodes reachable from s and that reach t
    reach_from_s = S[s, :]
    reach_to_t = S[:, t]
    mask = reach_from_s | (np.arange(n) == s)
    mask &= reach_to_t | (np.arange(n) == t)
    idx = np.where(mask)[0]
    if idx.size == 0:
        return 0
    M = S[np.ix_(idx, idx)]
    # map local indices
    s_loc = int(np.where(idx == s)[0][0])
    t_loc = int(np.where(idx == t)[0][0])
    # DP again on M
    L2 = np.zeros(len(idx), dtype=int)
    for j in range(len(idx)):
        preds = np.where(M[:, j])[0]
        if preds.size:
            L2[j] = max(L2[p] for p in preds) + 1
        else:
            L2[j] = 1
    return int(L2[t_loc])

def largest_antichain(R):
    """
    Estimate the size of the largest antichain in a causal set.
    Greedy approximation: pick nodes not related to any already chosen.
    """
    N = R.shape[0]
    remaining = set(range(N))
    antichain = []

    while remaining:
        node = remaining.pop()
        antichain.append(node)
        # Remove all nodes related to 'node' (both ways)
        related = set(np.where(R[node, :] | R[:, node])[0])
        remaining -= related

    return len(antichain)

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


# ------------------ Hasse diagram plotting ------------------

def plot_causet(points, R, dim=2, title="Causal Set"):
    """
    Plot Hasse diagram (2D: space-time, 3D: just scatter)
    """
    N = len(points)
    G = nx.DiGraph()
    G.add_nodes_from(range(N))
    edges = np.argwhere(R)
    for i,j in edges:
        G.add_edge(i,j)

    plt.figure(figsize=(6,6))
    if dim == 2:
        pos = {i: (points[i,1], points[i,0]) for i in range(N)}
        nx.draw(G, pos, with_labels=False, node_size=50, node_color="skyblue", arrows=True)
        plt.xlabel("space (x)")
        plt.ylabel("time (t)")
    else:  # 3D just scatter
        plt.scatter(points[:,1], points[:,2], c='skyblue', s=50)
        plt.xlabel("x")
        plt.ylabel("y")
    plt.title(title)
    plt.show()