import numpy as np

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
            max_r = T/2 - abs(t) # The radius of the disk at time t
            
            if max_r > 0:
                # Use sqrt(uniform) to ensure uniform density across the disk area
                r = max_r * np.sqrt(rng.uniform(0, 1))
                theta = rng.uniform(0, 2 * np.pi)
                
                x = r * np.cos(theta)
                y = r * np.sin(theta)
                
                points.append([t, x, y])

    else:
        raise ValueError("dim must be 2 or 3")
    pts = np.array(points)
    return pts[np.argsort(pts[:, 0])]  # sort by time coordinate

# ------------------ Causal relations ------------------
def causal_matrix(points, dim):
    N = len(points)
    # Ensure points are sorted by time (they should be from sprinkle, but this is safe)
    # If points aren't sorted, the DP longest chain might fail.
    
    t = points[:, 0]
    # We want R[i, j] = 1 IF point i is in the past of point j
    # This means t[j] - t[i] > 0
    t_row = t[:, np.newaxis] # Past point candidate
    t_col = t[np.newaxis, :] # Future point candidate
    
    dt = t_col - t_row # Positive if col is in the future of row
    
    if dim == 2:
        x = points[:, 1]
        dx = x[np.newaxis, :] - x[:, np.newaxis]
        mask = (dt > 0) & (dt**2 - dx**2 >= 0)
    elif dim == 3:
        x = points[:, 1]
        y = points[:, 2]
        dx = x[np.newaxis, :] - x[:, np.newaxis]
        dy = y[np.newaxis, :] - y[:, np.newaxis]
        mask = (dt > 0) & (dt**2 - dx**2 - dy**2 >= 0)
        
    R = np.zeros((N, N), dtype=int)
    R[mask] = 1
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
# Generate a causal set via transitive percolation (random partial order). 
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
