import networkx as nx
import numpy as np
import pandas as pd


def mask_by_id(nodes: pd.DataFrame, seg: np.ndarray, frame_key: str, value_key: str):
    if (nodes["track_id"] == -1).any():
        raise ValueError("Unassigned track_id for nodes!")

    masks = np.zeros_like(seg)
    for t, frame_nodes in nodes.groupby(frame_key):
        frame_seg = seg[t]
        max_label = int(frame_seg.max())
        label_to_tid = np.zeros(max_label + 1, dtype=masks.dtype)
        label_to_tid[frame_nodes[value_key].values] = frame_nodes["track_id"].values
        masks[t] = label_to_tid[frame_seg]

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
