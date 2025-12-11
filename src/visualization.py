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


def plot_causet(points, R, dim=2, title="Causal Set", show=True, save_path=None, draw_hasse=True):
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
        # 2D embedding: (x, t) so time on vertical axis
        fig, ax = plt.subplots(figsize=(6, 6))
        pos = {i: (points[i, 1], points[i, 0]) for i in range(N)}  # x horizontal, t vertical
        G = nx.DiGraph()
        G.add_nodes_from(range(N))
        G.add_edges_from(edges)

        # nodes
        nx.draw_networkx_nodes(G, pos, node_size=60, node_color="skyblue", ax=ax)
        # edges with arrowheads
        nx.draw_networkx_edges(
            G, pos, edgelist=edges, arrowstyle='-|>', arrowsize=12,
            connectionstyle='arc3,rad=0.0', ax=ax
        )
        ax.set_xlabel('space (x)')
        ax.set_ylabel('time (t)')
        ax.set_title(title)
        ax.invert_yaxis()  # optional: make earlier times at top if you prefer

    else:
        # 3D plotting: use (x,y,t) but put time as z for clarity (or choose order you like)
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        xs = points[:, 1]
        ys = points[:, 2]
        zs = points[:, 0]
        ax.scatter(xs, ys, zs, s=40)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('t')
        ax.set_title(title)

        # draw Hasse edges as 3D lines (no arrows)
        for i, j in edges:
            x_line = [points[i, 1], points[j, 1]]
            y_line = [points[i, 2], points[j, 2]]
            z_line = [points[i, 0], points[j, 0]]
            ax.plot(x_line, y_line, z_line, linewidth=0.8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, ax