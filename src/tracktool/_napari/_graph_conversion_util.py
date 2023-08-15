import networkx as nx
from collections import defaultdict

import pandas as pd
from napari.layers import Tracks

def _assign_all_track_id(sol):
    """Assign unique integer track ID to each node. 

    Nodes that have more than one incoming edge, or more than
    two children trigger a new track ID.

    Args:
        sol (nx.DiGraph): directed solution graph
    """
    roots = [node for node in sol.nodes if sol.in_degree(node) == 0]
    nx.set_node_attributes(sol, -1, 'track-id')
    track_id = 1
    for root in roots:
        for edge_key in nx.edge_dfs(sol, root):
            source, dest = edge_key[0], edge_key[1]
            source_out = sol.out_degree(source)
            # true root
            if sol.in_degree(source) == 0 and sol.nodes[source]['track-id'] == -1:
                sol.nodes[source]['track-id'] = track_id
            # source is splitting or destination has multiple parents
            if source_out > 1:
                track_id += 1
            elif sol.in_degree(dest) > 1:
                if sol.nodes[dest]['track-id'] != -1:
                    continue
                else:
                    track_id += 1
            sol.nodes[dest]['track-id'] = track_id
        track_id += 1
    return track_id

def _get_parents(node_df, max_track_id, sol):
    parents = defaultdict(set)
    for tid in range(1, max_track_id):
        track_nodes = node_df[node_df['track-id'] == tid]
        if track_nodes.empty:
            print(f'No nodes in track {tid}')
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
    max_id = _assign_all_track_id(nxg)
    sol_node_df = pd.DataFrame.from_dict(nxg.nodes, orient='index')
    parent_connections = _get_parents(sol_node_df, max_id, nxg)
    track_df = sol_node_df[sol_node_df['track-id'] != -1].sort_values(['track-id', 't'])[['track-id', 't', 'y', 'x']]
    parent_connections = {k: list(v) for k, v in parent_connections.items()}
    track_layer = Tracks(track_df, graph=parent_connections, tail_length=1, name='tracks')
    track_layer.display_id = True
    return track_layer
