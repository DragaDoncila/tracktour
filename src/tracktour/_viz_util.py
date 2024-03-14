import networkx as nx
import numpy as np
import pandas as pd


def mask_by_id(nodes: pd.DataFrame, seg: np.ndarray, frame_key: str, value_key: str):
    masks = np.zeros_like(seg)
    max_id = nodes["track-id"].max()
    for i in range(1, max_id + 1):
        track_nodes = nodes[nodes["track-id"] == i]
        for row in track_nodes.itertuples():
            t = getattr(row, frame_key)
            orig_label = getattr(row, value_key)
            mask = seg[t] == orig_label
            masks[t][mask] = i

    # colour weird vertices with 1
    unassigned = nodes[nodes["track-id"] == -1]
    for row in unassigned.itertuples():
        t = row.t
        # TODO: breaks for brand new detections that aren't present in the segmentation!
        orig_label = getattr(row, value_key)
        mask = seg[t] == orig_label
        masks[t][mask] = 1

    return masks


def store_colour_vs_of_interest(
    graph, vs, pred_colour, v_colour, succ_colour, orig_colour=False
):
    if not orig_colour:
        nx.set_node_attributes(graph, "white", "color")
    for node in vs:
        parents = [edge[0] for edge in graph.in_edges(node)]
        children = [edge[1] for edge in graph.out_edges(node)]
        # colour the parents orange
        for parent in parents:
            graph.nodes[parent]["color"] = pred_colour
        # colour the merge node red
        graph.nodes[node]["color"] = v_colour
        # colour the children yellow
        for child in children:
            graph.nodes[child]["color"] = succ_colour
