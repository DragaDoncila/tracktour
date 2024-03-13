from collections import defaultdict

import networkx as nx
import numpy as np
import pandas as pd
from napari.layers import Graph, Labels, Tracks
from napari_graph import DirectedGraph

from tracktour._graph_util import assign_track_id
from tracktour._viz_util import mask_by_id


def _get_parents(node_df, max_track_id, sol):
    parents = defaultdict(set)
    for tid in range(1, max_track_id):
        track_nodes = node_df[node_df["track-id"] == tid]
        if track_nodes.empty:
            continue
        # get the first occurrence of this tid
        node_id = track_nodes["t"].idxmin()
        for pred in sol.predecessors(node_id):
            if (p_tid := sol.nodes[pred]["track-id"]) != tid:
                parents[tid].add(p_tid)
    return parents


def get_tracks_from_nxg(nxg: "nx.DiGraph"):
    """Take solution nxg and convert to napari tracks layer

    Args:
        nxg (nx.DiGraph): networkx DirectedGraph of solution

    Returns:
        tracks (napari.layers.Tracks): track layer of solution
    """
    if tids := nx.get_node_attributes(nxg, "track-id"):
        max_id = max(list(tids.values()))
    else:
        max_id = assign_track_id(nxg)
    sol_node_df = pd.DataFrame.from_dict(nxg.nodes, orient="index")
    parent_connections = _get_parents(sol_node_df, max_id, nxg)
    # TODO: 3D BREAK!
    track_df = sol_node_df[sol_node_df["track-id"] != -1].sort_values(
        ["track-id", "t"]
    )[["track-id", "t", "y", "x"]]
    parent_connections = {k: list(v) for k, v in parent_connections.items()}
    track_layer = Tracks(
        track_df, graph=parent_connections, tail_length=1, name="tracks"
    )
    track_layer.display_id = True
    return track_layer


def get_napari_graph(tracked, location_keys, frame_key, value_key):
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


def get_coloured_graph_labels(
    tracked, location_keys, frame_key, value_key, segmentation
):
    napari_graph_layer = get_napari_graph(tracked, location_keys, frame_key, value_key)
    subgraph = napari_graph_layer.metadata["subgraph"]
    tracks_layer = get_tracks_from_nxg(subgraph)
    napari_graph_layer.metadata["tracks"] = tracks_layer

    # recolor segmentation and graph points by track-id
    sol_node_df = pd.DataFrame.from_dict(subgraph.nodes, orient="index")
    masks = mask_by_id(sol_node_df, segmentation, frame_key, value_key)
    masked_seg = Labels(masks, name="Track Coloured Seg", visible=False)

    # subgraph, tracks layer and graph layer **all** need to know about colour :<<<<
    color_dict = {
        node_id: (node_info["track-id"], masked_seg.get_color(node_info["track-id"]))
        for node_id, node_info in subgraph.nodes(data=True)
    }
    napari_graph_layer.face_color = [val[1] for val in color_dict.values()]
    napari_graph_layer.metadata["tracks"].metadata = {
        "colors": dict([(val[0], val[1]) for val in color_dict.values()])
    }
    nx.set_node_attributes(subgraph, {k: v[1] for k, v in color_dict.items()}, "color")
    return napari_graph_layer, masked_seg


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
