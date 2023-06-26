import networkx as nx
import numpy as np

def assign_track_id(sol):
    """Assign unique integer track ID to each node. 

    Nodes that have more than one incoming edge, or more than
    two children get assigned track ID -1.

    Args:
        sol (nx.DiGraph): directed solution graph
    """
    roots = [node for node in sol.nodes if sol.in_degree(node) == 0]
    nx.set_node_attributes(sol, -1, 'track-id')
    track_id = 1
    for root in roots:
        for edge_key in nx.dfs_edges(sol, root):
            source, dest = edge_key[0], edge_key[1]
            source_out = sol.out_degree(source)
            # true root
            if sol.in_degree(source) == 0:
                sol.nodes[source]['track-id'] = track_id
            # merge into dest or triple split from source
            elif sol.in_degree(dest) > 1 or source_out > 2:
                sol.nodes[source]['track-id'] = -1
                sol.nodes[dest]['track-id'] = -1
                continue
            # double parent_split
            elif source_out == 2:
                track_id += 1
            sol.nodes[dest]['track-id'] = track_id
        track_id += 1

def mask_by_id(nodes, seg):
    masks = np.zeros_like(seg)
    max_id = nodes['track-id'].max()
    for i in range(1, max_id+1):
        track_nodes = nodes[nodes['track-id'] == i]
        for row in track_nodes.itertuples():
            t = row.t
            orig_label = row.pixel_value
            mask = seg[t] == orig_label
            masks[t][mask] = i
    
    # colour weird vertices with 1
    unassigned = nodes[nodes['track-id'] == -1]
    for row in unassigned.itertuples():
        t = row.t
        orig_label = row.pixel_value
        mask = seg[t] == orig_label
        masks[t][mask] = 1

    return masks

def get_point_colour(sol, merges, bad_parents):
    merges = set(merges)
    bad_parents = set(bad_parents)
    colours = ['white' for _ in range(sol.number_of_nodes())]
    for node in merges:
        parents = [edge[0] for edge in sol.in_edges(node)]
        children = [edge[1] for edge in sol.out_edges(node)]

        # colour the parents orange
        for parent in parents:
            colours[parent] = 'orange'
        # colour the merge node red
        colours[node] = 'red'
        # colour the children yellow
        for child in children:
            colours[child] = 'yellow'

    for node in bad_parents:
        children = [edge[1] for edge in sol.out_edges(node)]
        # colour children pink
        for child in children:
            colours[child] =  'pink'
        # colour parent purple
        colours[node] = 'purple'
    return colours

def store_colour_vs_of_interest(graph, vs, pred_colour, v_colour, succ_colour, orig_colour=False):
    if not orig_colour:
        nx.set_node_attributes(graph, 'white', 'color')
    for node in vs:
        parents = [edge[0] for edge in graph.in_edges(node)]
        children = [edge[1] for edge in graph.out_edges(node)]
        # colour the parents orange
        for parent in parents:
            graph.nodes[parent]['color'] = pred_colour
        # colour the merge node red
        graph.nodes[node]['color'] = v_colour
        # colour the children yellow
        for child in children:
            graph.nodes[child]['color'] = succ_colour
