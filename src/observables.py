import numpy as np

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
