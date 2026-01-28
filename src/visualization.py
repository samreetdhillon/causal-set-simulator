import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# ------------------ Hasse diagram plotting ------------------
# plotting helper: compute Hasse (cover) edges from reachability matrix R
def hasse_edges_from_R(R):
    """
    Return a list of covering edges (i,j) such that R[i,j]==1 and there is NO k (k != i,j) with R[i,k] and R[k,j].
    This gives the transitive-reduction / Hasse edges for drawing.
    """
    N = R.shape[0]
    edges = []
    for i in range(N):
        for j in range(N):
            if R[i, j]:
                # check if there exists intermediate k
                intermediate = False
                for k in range(N):
                    if k == i or k == j:
                        continue
                    if R[i, k] and R[k, j]:
                        intermediate = True
                        break
                if not intermediate:
                    edges.append((i, j))
    return edges


def plot_causet(points, R, T=1.0, dim=2, title="Causal Set", show=True, save_path=None, draw_hasse=True):
    # Plot Hasse diagram embedding.
    N = len(points)
    if N == 0:
        print("Empty causal set: nothing to plot.")
        return None

    # edges to draw
    if draw_hasse:
        edges = hasse_edges_from_R(R)
    else:
        edges = [tuple(e) for e in np.argwhere(R)]

    if dim == 2 or points.shape[1] == 2:
        # 1. Create the figure and axis first
        fig, ax = plt.subplots(figsize=(6, 6))
    
        # 2. Draw the Lightcone/Diamond boundaries
        t_vals = np.linspace(-T/2, T/2, 100)
        # Boundary lines: |x| = T/2 - |t|
        ax.plot(T/2 - np.abs(t_vals), t_vals, 'r--', alpha=0.3, label="Causal Boundary")
        ax.plot(-(T/2 - np.abs(t_vals)), t_vals, 'r--', alpha=0.3)
        
        # 3. Setup the Graph
        pos = {i: (points[i, 1], points[i, 0]) for i in range(N)}
        G = nx.DiGraph()
        G.add_nodes_from(range(N))
        G.add_edges_from(edges)

        # 4. Draw the actual Causet elements
        nx.draw_networkx_nodes(G, pos, node_size=40, node_color="skyblue", ax=ax, edgecolors='black', linewidths=0.5)
        nx.draw_networkx_edges(
            G, pos, edgelist=edges, arrowstyle='-|>', arrowsize=10,
            connectionstyle='arc3,rad=0.0', ax=ax, alpha=0.4
        )
        
        # 5. Formatting
        ax.set_xlabel('Space (x)')
        ax.set_ylabel('Time (t)')
        ax.set_title(title)
        ax.grid(True, linestyle=':', alpha=0.5)

    else:
        # 3D plotting: use (x,y,t) with time as z (upward flow)
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # 1. Draw the Diamond Wireframe (Circular Cross-sections)
        # We create circles at different time slices 'tz'
        n_slices = 10
        theta = np.linspace(0, 2 * np.pi, 50)
        for tz in np.linspace(-T/2, T/2, n_slices):
            radius = T/2 - np.abs(tz)
            if radius > 0:
                cx = radius * np.cos(theta)
                cy = radius * np.sin(theta)
                cz = np.full_like(theta, tz)
                ax.plot(cx, cy, cz, 'r--', alpha=0.15) # Light red dashed rings

        # 2. Draw the vertical "tips" axis
        ax.plot([0, 0], [0, 0], [-T/2, T/2], 'r--', alpha=0.2)

        # 3. Scatter the points
        xs = points[:, 1]
        ys = points[:, 2]
        zs = points[:, 0] # Time
        ax.scatter(xs, ys, zs, s=40, c=zs, cmap='viridis', edgecolors='k', alpha=0.8)

        # 4. Draw Hasse edges as 3D lines
        for i, j in edges:
            ax.plot([points[i, 1], points[j, 1]], 
                    [points[i, 2], points[j, 2]], 
                    [points[i, 0], points[j, 0]], 
                    color='black', linewidth=0.5, alpha=0.3)

        ax.set_xlabel('Space (x)')
        ax.set_ylabel('Space (y)')
        ax.set_zlabel('Time (t)')
        ax.set_title(title)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, ax