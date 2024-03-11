import networkx as nx


def assign_track_id(nx_sol):
    """Assign unique integer track ID to each node.

    Nodes that have more than one incoming edge, or more than
    two children trigger a new track ID.

    Args:
        nx_sol (nx.DiGraph): directed solution graph
    """
    roots = [node for node in nx_sol.nodes if nx_sol.in_degree(node) == 0]
    nx.set_node_attributes(nx_sol, -1, "track-id")
    track_id = 1
    for root in roots:
        if nx_sol.out_degree(root) == 0:
            nx_sol.nodes[root]["track-id"] = track_id

        for edge_key in nx.edge_dfs(nx_sol, root):
            source, dest = edge_key[0], edge_key[1]
            source_out = nx_sol.out_degree(source)
            # true root
            if nx_sol.in_degree(source) == 0 and nx_sol.nodes[source]["track-id"] == -1:
                nx_sol.nodes[source]["track-id"] = track_id
            # source is splitting or destination has multiple parents
            if source_out > 1:
                track_id += 1
            elif nx_sol.in_degree(dest) > 1:
                if nx_sol.nodes[dest]["track-id"] != -1:
                    continue
                else:
                    track_id += 1
            nx_sol.nodes[dest]["track-id"] = track_id

        track_id += 1
    return track_id


def assign_intertrack_edges(nx_g: "nx.DiGraph"):
    """Currently assigns is_intertrack_edge=True for all edges
    that has more than one incoming edge and/or more than one
    outgoing ede.

    Args:
        g (nx.DiGraph): directed tracking graph
    """
    nx.set_edge_attributes(nx_g, 0, name="is_intertrack_edge")
    for e in nx_g.edges:
        src, dest = e
        # source has two children
        if len(nx_g.out_edges(src)) > 1:
            nx_g.edges[e]["is_intertrack_edge"] = 1
        # destination has two parents
        if len(nx_g.in_edges(dest)) > 1:
            nx_g.edges[e]["is_intertrack_edge"] = 1
