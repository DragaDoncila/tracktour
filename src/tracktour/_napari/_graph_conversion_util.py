from collections import defaultdict

import networkx as nx
import numpy as np
import pandas as pd
from napari.layers import Labels, Points, Tracks

from tracktour._graph_util import assign_track_id
from tracktour._viz_util import mask_by_id


def _get_parents(sol):
    """Get napari-like parent graph for sol.

    Parent connections can occur at merges,
    divisions and skip edges.

    Parameters
    ----------
    sol : networkx.DiGraph
        solution networkx graph

    Returns
    -------
    Dict[int, list[int]]
        dictionary of track_id to list of parent track_ids
    """
    parents_graph = defaultdict(set)
    for node_id, node_info in sol.nodes(data=True):
        # merge node has multiple parents
        if sol.in_degree(node_id) > 1:
            parent_tids = {
                sol.nodes[parent]["track_id"] for parent in sol.predecessors(node_id)
            }
            parents_graph[node_info["track_id"]].update(parent_tids)
        # dividing node's children have it as parent
        if sol.out_degree(node_id) > 1:
            for child in sol.successors(node_id):
                parents_graph[sol.nodes[child]["track_id"]].add(node_info["track_id"])
        # skip edges have different track ID so
        # dest node gets source added as parent
        elif sol.out_degree(node_id) == 1:
            for child in sol.successors(node_id):
                child_id = sol.nodes[child]["track_id"]
                if child_id != node_info["track_id"]:
                    parents_graph[child_id].add(node_info["track_id"])
    # don't want defaultdict behaviour anymore
    return {k: list(v) for k, v in parents_graph.items()}


def get_tracks_from_nxg(nxg: "nx.DiGraph"):
    """Take solution nxg and convert to napari tracks layer

    Args:
        nxg (nx.DiGraph): networkx DirectedGraph of solution

    Returns:
        tracks (napari.layers.Tracks): track layer of solution
    """
    if any("track_id" not in nxg.nodes[node] for node in nxg.nodes):
        assign_track_id(nxg)
    sol_node_df = pd.DataFrame.from_dict(nxg.nodes, orient="index")
    parent_connections = _get_parents(nxg)
    location_cols = ["y", "x"]
    if "z" in sol_node_df.columns:
        location_cols = ["z"] + location_cols
    track_df = sol_node_df[sol_node_df["track_id"] != -1].sort_values(
        ["track_id", "t"]
    )[["track_id", "t"] + location_cols]
    track_layer = Tracks(
        track_df, graph=parent_connections, tail_length=1, name="tracks"
    )
    return track_layer


def get_napari_graph(tracked, location_keys, frame_key, value_key):
    from napari.layers import Graph
    from napari_graph import DirectedGraph

    nodes = tracked.tracked_detections
    mig_edges = tracked.tracked_edges
    pos_keys = [frame_key] + list(location_keys)
    mig_graph = nx.from_pandas_edgelist(
        mig_edges, source="u", target="v", edge_attr=["flow"], create_using=nx.DiGraph
    )
    mig_graph.add_nodes_from(
        nodes[pos_keys + [value_key]].to_dict(orient="index").items()
    )
    pos = {
        tpl.Index: tuple([getattr(tpl, k) for k in pos_keys])
        for tpl in nodes.itertuples()
    }
    nx.set_node_attributes(mig_graph, pos, "pos")
    napari_graph = DirectedGraph.from_networkx(mig_graph)
    graph_layer = Graph(
        napari_graph,
        name="Solution Graph",
        out_of_slice_display=True,
        ndim=3,
        size=5,
        metadata={"nxg": mig_graph, "subgraph": mig_graph},
    )
    return graph_layer


def get_coloured_solution_layers(tracked, scale, segmentation):
    frame_key = tracked.frame_key
    value_key = tracked.value_key
    location_keys = tracked.location_keys
    layer_scale = (1,) + tuple(scale)

    subgraph = tracked.as_nx_digraph()
    tracks_layer = get_tracks_from_nxg(subgraph)
    tracks_layer.scale = layer_scale
    tracks_layer.metadata = {"nxg": subgraph}

    # recolor segmentation and graph points by track_id
    sol_node_df = pd.DataFrame.from_dict(subgraph.nodes, orient="index")
    masks = mask_by_id(sol_node_df, segmentation, frame_key, value_key)
    masked_seg = Labels(masks, name="Track Coloured Seg", visible=False)
    masked_seg.scale = layer_scale
    # subgraph, tracks layer and graph layer **all** need to know about colour :<<<<
    color_dict = {
        node_id: (node_info["track_id"], masked_seg.get_color(node_info["track_id"]))
        for node_id, node_info in subgraph.nodes(data=True)
    }
    nx.set_node_attributes(subgraph, {k: v[1] for k, v in color_dict.items()}, "color")
    coloured_points = Points(
        sol_node_df[[frame_key] + list(location_keys)], name="Track Coloured Points"
    )
    coloured_points.face_color = [val[1] for val in color_dict.values()]
    coloured_points.scale = layer_scale
    coloured_points.size = 1

    return coloured_points, masked_seg, tracks_layer


def get_nxg_from_tracks(tracks_layer: "napari.layers.Tracks"):
    """Get networkx graph from napari tracks layer

    Args:
        tracks_layer (napari.layers.Tracks): tracks layer

    Returns:
        nxg (nx.DiGraph): networkx DirectedGraph of tracks
    """
    # numpy array will be (track_id, t, z, y, x) or (track_id, t, y, x)
    loc_cols = ["z", "y", "x"] if tracks_layer.data.shape[1] == 5 else ["y", "x"]
    node_info = pd.DataFrame(tracks_layer.data, columns=["track_id", "t"] + loc_cols)
    parent_info = tracks_layer.graph
    node_container = []
    edges = []
    tid_first_last = defaultdict(dict)
    for track_id, track_nodes in node_info.groupby("track_id"):
        track_nodes = track_nodes.sort_values(by="t")
        node_container.extend(track_nodes.to_dict(orient="index").items())
        edges.extend(zip(track_nodes.index[:-1], track_nodes.index[1:]))
        tid_first_last[track_id]["first"] = int(track_nodes.index[0])
        tid_first_last[track_id]["last"] = int(track_nodes.index[-1])
    for tid, parents in parent_info.items():
        for parent in parents:
            edges.append((tid_first_last[parent]["last"], tid_first_last[tid]["first"]))
    nxg = nx.DiGraph()
    nxg.add_nodes_from(node_container)
    nxg.add_edges_from(edges)
    return nxg
