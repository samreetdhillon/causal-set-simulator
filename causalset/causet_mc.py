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


def midpoint_scaling_dimension(points, R, sample_pairs=50):
    """
    Midpoint-scaling estimator for dimension (rough heuristic).
    For sampled comparable pairs (p,q):
      - pick r in I(p,q) whose time coordinate is closest to midpoint of t_p and t_q
      - compute V_full = |I(p,q)|, V_half = |I(p,r)|
      - estimate d ≈ log(V_full / V_half) / log(2)
    Returns (mean_d, std_d). If estimator cannot be computed, returns (nan, nan).
    NOTE: This is a pragmatic implementation of midpoint-scaling for sprinkled causal sets.
    """
    n = R.shape[0]
    comps = np.argwhere(R)
    if comps.size == 0:
        return float('nan'), float('nan')
    rng = np.random.default_rng()
    idx = rng.choice(len(comps), size=min(sample_pairs, len(comps)), replace=False)
    pairs = comps[idx]

    ests = []
    for p, q in pairs:
        I = _interval_elements(R, p, q)
        if len(I) < 2:  # need at least one interior element to pick a midpoint
            continue
        # choose node r in I whose time coordinate is nearest to midpoint of p and q
        # if points array not provided or time missing, skip
        try:
            t_p = points[p][0]
            t_q = points[q][0]
            midpoint_time = 0.5 * (t_p + t_q)
            # find r in I with t closest to midpoint_time
            times = np.array([points[i][0] for i in I])
            idx_r = np.argmin(np.abs(times - midpoint_time))
            r = I[idx_r]
        except Exception:
            # fallback: pick a random interior element
            r = int(np.random.choice(I))

        V_full = len(I)
        I_pr = _interval_elements(R, p, r)
        V_half = len(I_pr)
        # require both volumes to be >0 to estimate
        if V_full > 0 and V_half > 0 and V_half < V_full:
            d_est = np.log(V_full / V_half) / np.log(2)
            # numeric sanity check
            if not np.isfinite(d_est):
                continue
            ests.append(d_est)

    if len(ests) == 0:
        return float('nan'), float('nan')
    return float(np.mean(ests)), float(np.std(ests))


def benincasa_dowker_action_proxy(R, points=None, sample_pairs=200, alpha=1.0):
    """
    A BD-inspired action proxy (not a full derivation) that sums normalized deviations
    of interval volumes from a 2D-flat expectation. This gives a scalar 'action-like'
    number that can be used to compare sprinkled vs percolated sets.

    For sampled comparable pairs (p,q):
      - compute I = |I(p,q)|
      - compute L = longest chain length between p and q
      - expected_flat ~ 0.5 * alpha * L^2  (alpha calibration)
      - local contribution: (I - expected_flat)/max(1,I)
    Returns: total_action = sum_local_contrib, and (mean, std) of the sampled deviations.
    """
    n = R.shape[0]
    comps = np.argwhere(R)
    if comps.size == 0:
        return 0.0, float('nan'), float('nan')
    rng = np.random.default_rng()
    idx = rng.choice(len(comps), size=min(sample_pairs, len(comps)), replace=False)
    pairs = comps[idx]

    devs = []
    for p, q in pairs:
        I = _interval_elements(R, p, q)
        # build induced subposet and get longest chain length L
        subset = np.concatenate(([p], I, [q]))
        S = R[np.ix_(subset, subset)]
        L = _longest_chain_length_from_to(S, 0, len(subset)-1)
        I_size = len(I)
        expected = 0.5 * alpha * (L**2)
        dev = (I_size - expected) / max(1, I_size)
        devs.append(dev)

    devs = np.asarray(devs, dtype=float)
    total_action_proxy = float(np.sum(devs))
    return total_action_proxy, float(np.mean(devs)), float(np.std(devs))

# ------------------ Classical Sequential Growth (CSG) ------------------

def _is_order_ideal(R, subset):
    """
    Check whether `subset` (iterable of node indices) is an order ideal in the poset R.
    An order ideal I has the property: if v in I and u ≺ v then u in I.
    R is reachability (transitive closure) boolean/int matrix.
    """
    if len(subset) == 0:
        return True
    subset = set(subset)
    for v in list(subset):
        preds = set(np.where(R[:, v])[0])
        # if any predecessor of v is not in subset, it's not an ideal
        if not preds.issubset(subset):
            return False
    return True


def enumerate_order_ideals(R):
    """
    Return a list of order ideals of the poset given by reachability R.
    WARNING: naive enumeration is O(2^n) in the number of elements. Use for small n (<=18-20).
    Each ideal is returned as a numpy array of node indices (sorted).
    """
    n = R.shape[0]
    ideals = []
    if n == 0:
        return ideals
    # iterate over bitmasks 0..2^n - 1
    # For moderate n this is expensive; acceptable for small demonstration runs.
    for mask in range(1 << n):
        # quick skip: build subset
        subset = [i for i in range(n) if (mask >> i) & 1]
        if _is_order_ideal(R, subset):
            ideals.append(np.array(subset, dtype=int))
    return ideals


def csg_sequential_growth(N, t_weights=None, geometric_a=None, rng=None, T=1.0):
    """
    Exact Classical Sequential Growth (CSG) generator building N elements.
    - N: number of elements to generate
    - t_weights: iterable/list giving nonnegative t_k weights for k=0..(N-1). If provided,
                 during each birth when there are m existing elements, weight for an ideal
                 I of size k is t_weights[k]. If t_weights shorter than needed, missing
                 entries are treated as 0.
    - geometric_a: convenience parameter; if provided, uses t_k = a**k (geometric family).
    - rng: numpy RNG
    - T: embedding time span for plotting (time increases with births)
    Returns: points (N,2) array (t,x) and reachability matrix R (N,N)
    
    Notes:
    - This implements the exact Rideout–Sorkin rule of choosing an order ideal I
      at each birth with probability proportional to t_{|I|}.
    - Complexity: enumerating all order ideals at each step is exponential; keep N small (<=20).
    """
    rng = np.random.default_rng() if rng is None else rng

    # Build t_weights array if geometric_a given
    if geometric_a is not None:
        # generate t_k = a^k for k=0..N-1
        a = float(geometric_a)
        t_weights = [a**k for k in range(N)]
    elif t_weights is None:
        # default to simple percolation-like choice: t_k = 1 for all k (uniform by size)
        t_weights = [1.0] * N
    else:
        t_weights = list(t_weights)

    # prepare arrays
    points = np.zeros((N, 2), dtype=float)  # (t, x)
    R = np.zeros((N, N), dtype=int)

    # We will grow elements one by one: element indices 0..N-1 in birth order.
    for new in range(N):
        if new == 0:
            # first element: no ancestors
            points[0, 0] = -T/2 + rng.uniform() * T  # place somewhere
            points[0, 1] = rng.uniform(-T/2, T/2)
            # R row/col remain zeros
            continue

        # compute current poset reachability among existing nodes 0..new-1
        R_sub = R[:new, :new].copy()
        # enumerate order ideals of current poset
        ideals = enumerate_order_ideals(R_sub)
        if len(ideals) == 0:
            # fallback: empty ideal only
            ideals = [np.array([], dtype=int)]

        # compute weights for each ideal: weight = t_{|I|}
        weights = np.array([ t_weights[min(len(I), len(t_weights)-1)] if len(t_weights) > 0 else 0.0
                            for I in ideals ], dtype=float)
        # if all weights zero (unlikely), fall back to uniform
        if np.all(weights == 0):
            weights = np.ones_like(weights)

        probs = weights / np.sum(weights)
        choice_idx = rng.choice(len(ideals), p=probs)
        chosen_I = ideals[choice_idx]

        # add links from every element in chosen_I to new element (direct link)
        for v in chosen_I:
            R[int(v), new] = 1

        # ensure transitive closure among existing nodes plus new
        R[:new+1, :new+1] = _transitive_closure_bool(R[:new+1, :new+1])

        # assign embedding coordinates: time strictly increasing with birth index (small jitter)
        # simple linear spacing in time for birth order (use jitter for better visualization)
        points[new, 0] = -T/2 + (new / float(N)) * T + rng.uniform(-0.001, 0.001)
        points[new, 1] = rng.uniform(-T/2, T/2)

    return points, R


# ------------------ Observables ------------------

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

def scaling_study(N_list, dim=2, trials=50):
    """
    Compute mean and std of observables for different N values.

    Returns a dictionary:
    {
        'N': [...],
        'ordering_fraction_mean': [...],
        'ordering_fraction_std': [...],
        'dimension_mean': [...],
        'dimension_std': [...],
        'longest_chain_mean': [...],
        'longest_chain_std': [...],
        'largest_antichain_mean': [...],
        'largest_antichain_std': [...],
    }
    """
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

def _interval_elements(R, p, q):
    """Return indices in the Alexandrov interval I(p,q) = { r | p ≺ r ≺ q }."""
    # R[p, r] True means p ≺ r ; R[r, q] True means r ≺ q
    between = np.where(R[p, :] & R[:, q])[0]
    return between[(between != p) & (between != q)]

def curvature_proxy(R, sample_pairs=50, alpha=1.0):
    """
    Flatness-deviation proxy in 2D spirit:
    For random comparable pairs (p,q), compare |I(p,q)| to alpha * L(p,q)^2 / 2.
    Returns mean ± std of normalized deviation over sampled pairs.
    
    - R : reachability (transitive closure)
    - sample_pairs : how many (p,q) with p≺q to sample
    - alpha : calibration factor (~1). Tune so that flat 2D runs average near 0.
    
    Output: (mean_dev, std_dev), where each deviation is
        dev(p,q) = ( |I| - (alpha/2)*L^2 ) / max(1, |I|)
    so ~0 means "flat-like", positive means "more volume than flat", negative "less".
    """
    n = R.shape[0]
    comps = np.argwhere(R)  # all p≺q
    if comps.size == 0:
        return np.nan, np.nan
    rng = np.random.default_rng()
    idx = rng.choice(len(comps), size=min(sample_pairs, len(comps)), replace=False)
    pairs = comps[idx]

    devs = []
    # dynamic-programming longest-chain on induced subposet between p and q
    for p, q in pairs:
        I = _interval_elements(R, p, q)
        # Build induced order among {p} ∪ I ∪ {q}
        subset = np.concatenate(([p], I, [q]))
        sub_index = {v:i for i,v in enumerate(subset)}
        S = R[np.ix_(subset, subset)]
        # Longest-chain length from p to q within S
        L = _longest_chain_length_from_to(S, 0, len(subset)-1)
        # Interval size
        I_size = len(I)
        # Flat 2D expectation ~ (alpha/2) * L^2 (alpha is a calibration constant)
        expected = 0.5 * alpha * (L**2)
        dev = (I_size - expected) / max(1, I_size)
        devs.append(dev)

    devs = np.asarray(devs, dtype=float)
    return float(np.mean(devs)), float(np.std(devs))

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

# ------------------ Monte Carlo dimension estimation ------------------

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
