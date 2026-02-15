import numpy as np
import networkx as nx
from scipy.special import gamma
from scipy.optimize import brentq

# ------------------ Observables/Estimators ------------------

def ordering_fraction(R):
    """Fraction of related pairs r = #relations / total pairs"""
    N = R.shape[0]
    total_pairs = N*(N-1)
    if total_pairs == 0:
        return 0.0
    related_pairs = np.sum(R)
    return related_pairs / total_pairs

def myrheim_meyer_func(d, r_observed):
    """
    The theoretical relation for a d-dimensional causal diamond:
    f(d) = Gamma(d+1) * Gamma(d/2) / (4 * Gamma(3d/2))
    """
    if d <= 0:
        return 1.0  # Avoid division by zero/negative
    
    theoretical_r = (gamma(d + 1) * gamma(d / 2)) / (4 * gamma(1.5 * d))
    return theoretical_r - r_observed

def estimate_dimension(r):
    """
    Accurately estimate spacetime dimension d by finding the root of 
    the Myrheim–Meyer relation.
    """
    if r <= 0 or r >= 0.5: # r=0.5 is the limit as d -> 0
        return None
    
    try:
        # We search for d in the range [0.5, 10.0]
        d_est = brentq(myrheim_meyer_func, 0.5, 10.0, args=(r,))
        return d_est
    except ValueError:
        # If no root is found in the interval
        return None

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
    Computes the exact width of the poset (largest antichain)
    using Dilworth's Theorem via Maximum Bipartite Matching.
    """
    N = R.shape[0]
    # Create a bipartite graph
    G = nx.Graph()

    u_nodes = [f"u{i}" for i in range(N)]
    v_nodes = [f"v{i}" for i in range(N)]
    G.add_nodes_from(u_nodes)
    G.add_nodes_from(v_nodes)

    for i in range(N):
        for j in range(N):
            if R[i, j] == 1:
                # Bipartite mapping: node i in set U connects to node j in set V
                G.add_edge(f"u{i}", f"v{j}")
    
    # Matching size
    matching = nx.bipartite.hopcroft_karp_matching(G, top_nodes=u_nodes)
    # Matching size is len(matching) // 2 because it returns dict with both directions
    return N - (len(matching) // 2)
