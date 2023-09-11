from napari_graph import DirectedGraph
import networkx as nx
from collections import defaultdict

import pandas as pd
from napari.layers import Tracks, Graph
from .._flow_graph import assign_track_id

def _get_parents(node_df, max_track_id, sol):
    parents = defaultdict(set)
    for tid in range(1, max_track_id):
        track_nodes = node_df[node_df['track-id'] == tid]
        if track_nodes.empty:
            continue
        # get the first occurrence of this tid
        node_id = track_nodes['t'].idxmin()
        for pred in sol.predecessors(node_id):
            if (p_tid := sol.nodes[pred]['track-id']) != tid:
                parents[tid].add(p_tid)
    return parents

def get_tracks_from_nxg(nxg: 'nx.DiGraph'):
    """Take solution nxg and convert to napari tracks layer

    Args:
        nxg (nx.DiGraph): networkx DirectedGraph of solution
        
    Returns:
        tracks (napari.layers.Tracks): track layer of solution
    """
    if tids := nx.get_node_attributes(nxg, 'track-id'):
        max_id = max(list(tids.values()))
    else:
        max_id = assign_track_id(nxg)
    sol_node_df = pd.DataFrame.from_dict(nxg.nodes, orient='index')
    parent_connections = _get_parents(sol_node_df, max_id, nxg)
    track_df = sol_node_df[sol_node_df['track-id'] != -1].sort_values(['track-id', 't'])[['track-id', 't', 'y', 'x']]
    parent_connections = {k: list(v) for k, v in parent_connections.items()}
    track_layer = Tracks(track_df, graph=parent_connections, tail_length=1, name='tracks')
    track_layer.display_id = True
    return track_layer

# def get_napari_graph_from_nxg(nxg: 'nx.DiGraph', seg_layer: 'napari.layers.Labels') -> 'napari.layers.Graph':
#     for _, node_info in nxg.nodes(data=True):
#         label = node_info["track-id"]
#         node_info["color"] = seg_layer.get_color(label)
#     sol_node_df = pd.DataFrame.from_dict(nxg.nodes, orient='index')
#     sol_napari_graph = DirectedGraph(edges=nxg.edges, coords=sol_node_df[["t", "y", "x"]])
#     sol_tracks = get_tracks_from_nxg(nxg)
#     sol_napari_layer = Graph(
#         sol_napari_graph,
#         name="Solution Graph",
#         out_of_slice_display=True,
#         ndim=3,
#         size=5,
#         properties=sol_node_df.drop(columns=['color']),
#         face_color=list(nx.get_node_attributes(nxg, "color").values()),
#         metadata={
#             'nxg': nxg,
#             'tracks': sol_tracks,
#         },
#     )
#     return sol_napari_layer
