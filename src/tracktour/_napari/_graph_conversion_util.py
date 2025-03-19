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
        dictionary of track-id to list of parent track-ids
    """
    parents_graph = defaultdict(list)
    for node_id, node_info in sol.nodes(data=True):
        # merge node has multiple parents
        if sol.in_degree(node_id) > 1:
            parent_tids = [
                sol.nodes[parent]["track-id"] for parent in sol.predecessors(node_id)
            ]
            parents_graph[node_info["track-id"]].extend(parent_tids)
        # dividing node's children have it as parent
        if sol.out_degree(node_id) > 1:
            for child in sol.successors(node_id):
                parents_graph[sol.nodes[child]["track-id"]].append(
                    node_info["track-id"]
                )
        # skip edges have different track ID so
        # dest node gets source added as parent
        elif sol.out_degree(node_id) == 1:
            for child in sol.successors(node_id):
                child_id = sol.nodes[child]["track-id"]
                if child_id != node_info["track-id"]:
                    parents_graph[child_id].append(node_info["track-id"])
    # don't want defaultdict behaviour anymore
    return dict(parents_graph)


def get_tracks_from_nxg(nxg: "nx.DiGraph"):
    """Take solution nxg and convert to napari tracks layer

    Args:
        nxg (nx.DiGraph): networkx DirectedGraph of solution

    Returns:
        tracks (napari.layers.Tracks): track layer of solution
    """
    if any("track-id" not in nxg.nodes[node] for node in nxg.nodes):
        assign_track_id(nxg)
    sol_node_df = pd.DataFrame.from_dict(nxg.nodes, orient="index")
    parent_connections = _get_parents(nxg)
    location_cols = ["y", "x"]
    if "z" in sol_node_df.columns:
        location_cols = ["z"] + location_cols
    track_df = sol_node_df[sol_node_df["track-id"] != -1].sort_values(
        ["track-id", "t"]
    )[["track-id", "t"] + location_cols]
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


def get_coloured_solution_layers(
    tracked, location_keys, frame_key, value_key, scale, segmentation
):
    layer_scale = (1,) + tuple(scale)
    subgraph = tracked.as_nx_digraph()
    tracks_layer = get_tracks_from_nxg(subgraph)
    tracks_layer.scale = layer_scale

    # recolor segmentation and graph points by track-id
    sol_node_df = pd.DataFrame.from_dict(subgraph.nodes, orient="index")
    masks = mask_by_id(sol_node_df, segmentation, frame_key, value_key)
    masked_seg = Labels(masks, name="Track Coloured Seg", visible=False)
    masked_seg.scale = layer_scale
    # subgraph, tracks layer and graph layer **all** need to know about colour :<<<<
    color_dict = {
        node_id: (node_info["track-id"], masked_seg.get_color(node_info["track-id"]))
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


def get_detections_from_napari_graph(graph, segmentation):
    """Get detections dataframe from a napari_graph object and segmentation.

    Segmentation is required for the pixel values of each detection,
    and supports recolouring after the detections are solved.

    Parameters
    ----------
    graph : napari_graph.DirectedGraph
        graph to extract detections from. edges are ignored
    segmentation: np.ndarray
        2D+T or 3D+T array of segmentation labels
    """
    node_ids = graph.get_nodes()
    node_coords = graph.coords_buffer[node_ids]
    node_labels = segmentation[tuple(node_coords.astype(int).T)]
    all_node_info = np.hstack([node_coords, node_labels.reshape(-1, 1)])
    detections_df = pd.DataFrame(all_node_info, columns=["t", "y", "x", "label"])
    detections_df["label"] = detections_df["label"].astype(int)
    detections_df["t"] = detections_df["t"].astype(int)
    detections_df = detections_df.sort_values(["t"]).reset_index()
    return detections_df
