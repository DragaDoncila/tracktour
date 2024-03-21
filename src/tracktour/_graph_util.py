import copy

import networkx as nx
import numpy as np
import pandas as pd


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
                # we've already allocated for this node
                if nx_sol.nodes[dest]["track-id"] != -1:
                    continue
                else:
                    track_id += 1
            nx_sol.nodes[dest]["track-id"] = track_id

        track_id += 1
    return track_id


def remove_merges(nx_g: "nx.DiGraph", location_keys: list = ["y", "x"]):
    """Remove merge nodes from the graph. Returns a copy.

    For each node with more than one incoming edge, keep the
    edge with source closest to the merge node.

    Args:
        nx_g (nx.DiGraph): directed tracking graph
    """
    n_incoming = nx_g.in_degree()
    merge_nodes = [node for node, degree in n_incoming if degree > 1]
    furthest_parent_edges = []
    for node in merge_nodes:
        merge_coords = np.asarray([nx_g.nodes[node][key] for key in location_keys])
        parent_coords = {
            parent: np.asarray([nx_g.nodes[parent][key] for key in location_keys])
            for parent in nx_g.predecessors(node)
        }
        closest = min(
            parent_coords, key=lambda x: np.linalg.norm(merge_coords - parent_coords[x])
        )
        parent_coords.pop(closest)
        furthest_parent_edges.extend(
            [(parent, node) for parent in parent_coords.keys()]
        )
    mergeless = copy.deepcopy(nx_g)
    mergeless.remove_edges_from(furthest_parent_edges)
    return mergeless


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


def get_ctc_tracks(tracked_sol: "nx.DiGraph", frame_key: str = "t"):
    """Given nx solution graph, return CTC formatted track df.

    Each row in the df defines a track using ID, start frame, end frame, parent

    Parameters
    ----------
    tracked_sol : nx.DiGraph
        networkx solution graph with track-id assigned
    """
    node_df = pd.DataFrame.from_dict(tracked_sol.nodes, orient="index")
    t_only = node_df[["track-id", frame_key]]
    grouped = t_only.groupby("track-id")
    start_frame_idx = grouped.idxmin()
    end_frames_idx = grouped.idxmax()

    tids = []
    start_frames = []
    end_frames = []
    parent_tids = []
    for tid in grouped.groups:
        tids.append(tid)
        start_node = start_frame_idx.loc[tid].iloc[0]
        start_frame = tracked_sol.nodes[start_node][frame_key]
        predecessors = list(tracked_sol.predecessors(start_node))
        if len(predecessors) == 0:
            parent_tid = 0
        else:
            parent_tid = tracked_sol.nodes[predecessors[0]]["track-id"]
        end_frame = tracked_sol.nodes[end_frames_idx.loc[tid].iloc[0]][frame_key]
        start_frames.append(start_frame)
        end_frames.append(end_frame)
        parent_tids.append(parent_tid)
    track_df = pd.DataFrame(
        {
            "track-id": tids,
            "start": start_frames,
            "end": end_frames,
            "parent": parent_tids,
        }
    )
    return track_df
