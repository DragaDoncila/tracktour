import os

import igraph
import networkx as nx
import pandas as pd
from traccuracy import TrackingGraph

from ._io_util import get_im_centers, load_tiff_frames


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


def get_traccuracy_graph_nx(sol_nx: "nx.DiGraph", seg_ims: "np.ndarray"):
    nx_g = get_migration_subgraph(sol_nx)
    assign_intertrack_edges(nx_g)
    track_graph = TrackingGraph(nx_g, label_key="pixel_value", segmentation=seg_ims)
    return track_graph


def load_gt_info(gt_path, coords_path=None, return_ims=False):
    if coords_path is not None:
        ims = load_tiff_frames(gt_path)
        coords = pd.read_csv(coords_path)
    else:
        ims, coords, min_t, max_t, corners = get_im_centers(gt_path)
    # srcs = []
    # dests = []
    # is_parent = []
    # for label_val in range(coords['label'].min(), coords['label'].max()):
    #     gt_points = coords[coords.label == label_val].sort_values(by='t')
    #     track_edges = [(gt_points.index.values[i], gt_points.index.values[i+1]) for i in range(0, len(gt_points)-1)]
    #     if len(track_edges):
    #         sources, targets = zip(*track_edges)
    #         srcs.extend(sources)
    #         dests.extend(targets)
    #         is_parent.extend([0 for _ in range(len(sources))])

    # man_track = pd.read_csv(os.path.join(gt_path, 'man_track.txt'), sep=' ', header=None)
    # man_track.columns = ['current', 'start_t', 'end_t', 'parent']
    # child_tracks = man_track[man_track.parent != 0]
    # for index, row in child_tracks.iterrows():
    #     parent_id = row['parent']
    #     parent_end_t = man_track[man_track.current == parent_id]['end_t'].values[0]
    #     parent_coords = coords[(coords.label == parent_id) & (coords.t == parent_end_t)]
    #     child_coords = coords[(coords.label == row['current']) & (coords.t == row['start_t'])]
    #     srcs.append(parent_coords.index.values[0])
    #     dests.append(child_coords.index.values[0])
    #     is_parent.append(1)

    # edges = pd.DataFrame({
    #     'sources': srcs,
    #     'dests': dests,
    #     'is_parent': is_parent
    # })
    if not return_ims:
        return coords
    return ims, coords
