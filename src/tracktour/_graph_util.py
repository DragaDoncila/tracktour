import copy

import networkx as nx
import numpy as np
import pandas as pd


def assign_track_id(nx_sol):
    """Assign unique integer track ID to each node.

    Nodes that have more than one incoming edge, more than
    two children, or an outgoing skip-edge trigger a new track ID.

    Args:
        nx_sol (nx.DiGraph): directed solution graph
    """
    div_merge_skip = set()
    for node in nx_sol.nodes:
        # node is a merge
        if nx_sol.in_degree(node) > 1:
            merge_edges = set(nx_sol.in_edges(node))
            div_merge_skip.update(merge_edges)
        # node is a division
        if nx_sol.out_degree(node) > 1:
            div_edges = set(nx_sol.out_edges(node))
            div_merge_skip.update(div_edges)
        # node is source of a skip edge
        if nx_sol.out_degree(node) == 1:
            source_node = nx_sol.nodes[node]
            dest_node_id = list(nx_sol.successors(node))[0]
            dest_node = nx_sol.nodes[dest_node_id]
            if source_node["t"] + 1 != dest_node["t"]:
                div_merge_skip.add((node, dest_node_id))
            # node is a specific parent link that has been assigned
            # in other processing e.g. by user, or reading from ground truth
            elif nx_sol.edges[node, dest_node_id].get("manual_parent_link", False):
                div_merge_skip.add((node, dest_node_id))
    non_div_merge = set(nx_sol.edges) - div_merge_skip
    subg = nx_sol.edge_subgraph(non_div_merge)
    ccs = nx.connected_components(subg.to_undirected())
    tid_dict = {}
    max_id = 0
    for i, cc in enumerate(ccs, start=1):
        tid_dict.update({node: i for node in cc})
        max_id = i
    for node in nx_sol.nodes:
        if node not in tid_dict:
            tid_dict[node] = max_id + 1
            max_id += 1
    nx.set_node_attributes(nx_sol, tid_dict, "track-id")
    return max_id


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
    outgoing edge.

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
