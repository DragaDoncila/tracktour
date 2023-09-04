import os
import igraph
import networkx as nx
import pandas as pd
from traccuracy import TrackingGraph, TrackingData
from ._io_util import load_tiff_frames, get_im_centers
from ._flow_graph import FlowGraph

def assign_intertrack_edges(nx_g: 'nx.DiGraph'):
    """Currently assigns is_intertrack_edge=True for all edges 
    leaving a division vertex

    Args:
        g (nx.DiGraph): directed tracking graph
    """
    nx.set_edge_attributes(nx_g, 0, name='is_intertrack_edge')
    for e in nx_g.edges:
        src, dest = e
        # source has two children
        if len(nx_g.out_edges(src)) > 1:
            nx_g.edges[e]['is_intertrack_edge'] = 1
        # destination has two parents
        if len(nx_g.in_edges(dest)) > 1:
            nx_g.edges[e]['is_intertrack_edge'] = 1
            
def filter_to_migration_sol(nx_sol: 'nx.DiGraph'):
    unused_es = [e for e in nx_sol.edges if nx_sol.edges[e]['flow'] == 0]
    nx_sol.remove_edges_from(unused_es)
    delete_vs = []
    for v in nx_sol.nodes:
        v_info = nx_sol.nodes[v]
        if v_info['is_appearance'] or\
            v_info['is_target'] or\
                v_info['is_division'] or\
                    v_info['is_source']:
                    delete_vs.append(v)
    nx_sol.remove_nodes_from(delete_vs)
    return nx_sol

def get_traccuracy_graph(sol_igraph: 'FlowGraph', seg_ims: 'np.ndarray') -> 'TrackingGraph':
    nx_g = filter_to_migration_sol(sol_igraph.convert_sol_igraph_to_nx())
    assign_intertrack_edges(nx_g)
    track_graph = TrackingGraph(nx_g, label_key='pixel_value')  
    track_data = TrackingData(track_graph, seg_ims)
    return track_data

def get_traccuracy_graph_nx(sol_nx: 'nx.DiGraph', seg_ims: 'np.ndarray'):
    nx_g = filter_to_migration_sol(sol_nx)
    assign_intertrack_edges(nx_g)
    track_graph = TrackingGraph(nx_g, label_key='pixel_value')  
    track_data = TrackingData(track_graph, seg_ims)
    return track_data


def load_sol_flow_graph(sol_pth, seg_pth):
    sol = nx.read_graphml(sol_pth, node_type=int)
    sol_ims = load_tiff_frames(seg_pth)
    oracle_node_df = pd.DataFrame.from_dict(sol.nodes, orient='index')
    oracle_node_df.rename(columns={'pixel_value':'label'}, inplace=True)
    oracle_node_df.drop(oracle_node_df.tail(4).index, inplace = True)
    im_dim =  [(0, 0), sol_ims.shape[1:]]
    min_t = 0
    max_t = sol_ims.shape[0] - 1
    sol_g = FlowGraph(im_dim, oracle_node_df, min_t, max_t)
    store_flow(sol, sol_g)
    return sol_g, sol_ims, oracle_node_df

def store_flow(nx_sol, ig_sol):
    ig_sol._g.es.set_attribute_values('flow', 0)
    flow_es = nx.get_edge_attributes(nx_sol, 'flow')
    for e_id, flow in flow_es.items():
        src, target = e_id
        ig_sol._g.es[ig_sol._g.get_eid(src, target)]['flow'] = flow
        
def load_gt_graph(gt_path, return_ims=False):
    ims, coords, min_t, max_t, corners = get_im_centers(gt_path)
    srcs = []
    dests = []
    is_parent = []
    for label_val in range(coords['label'].min(), coords['label'].max()):
        gt_points = coords[coords.label == label_val].sort_values(by='t')
        track_edges = [(gt_points.index.values[i], gt_points.index.values[i+1]) for i in range(0, len(gt_points)-1)]
        if len(track_edges):
            sources, targets = zip(*track_edges)
            srcs.extend(sources)
            dests.extend(targets)
            is_parent.extend([0 for _ in range(len(sources))])

    man_track = pd.read_csv(os.path.join(gt_path, 'man_track.txt'), sep=' ', header=None)
    man_track.columns = ['current', 'start_t', 'end_t', 'parent']
    child_tracks = man_track[man_track.parent != 0]
    for index, row in child_tracks.iterrows():
        parent_id = row['parent']
        parent_end_t = man_track[man_track.current == parent_id]['end_t'].values[0]
        parent_coords = coords[(coords.label == parent_id)][coords.t == parent_end_t]
        child_coords = coords[(coords.label == row['current']) & (coords.t == row['start_t'])]
        srcs.append(parent_coords.index.values[0])
        dests.append(child_coords.index.values[0])
        is_parent.append(1)

    edges = pd.DataFrame({
        'sources': srcs,
        'dests': dests,
        'is_parent': is_parent
    })    
    graph = igraph.Graph.DataFrame(edges, directed=True, vertices=coords, use_vids=True)
    if not return_ims:
        return graph, coords
    return ims, graph, coords
