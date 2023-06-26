import networkx as nx
from traccuracy import TrackingGraph, TrackingData

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
