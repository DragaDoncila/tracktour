import os
import networkx as nx
from tracktour._io_util import load_tiff_frames

def get_graphml_reader(path):
    if can_read(path):
        return read_graphml
    
def get_ctc_tiff_reader(path):
    if not os.path.isdir(path):
        return None
    return read_ctc_tiffs

def can_read(path):
    if isinstance(path, list):
        return False
    if not path.endswith('graphml'):
        return False
    return True

def read_graphml(path):
    nxg = nx.read_graphml(path)
    pos = {
        n: (nxg.nodes[n]['t'], nxg.nodes[n]['x'], nxg.nodes[n]['y']) 
        if 'z' not in nxg.nodes[n] else 
        (nxg.nodes[n]['t'], nxg.nodes[n]['x'], nxg.nodes[n]['y'], nxg.nodes[n]['z']) 
        for n in nxg.nodes
    }
    nx.set_node_attributes(nxg, pos, 'pos')
    int_nx = nx.convert_node_labels_to_integers(nxg)

    layer_kwargs = {
        'name': 'Solution Graph',
        'size': 5,
        'out_of_slice_display': True,
        'metadata': {'nxg': nxg}
    }
    layer_type = 'graph'
    return [(int_nx, layer_kwargs, layer_type)]


def read_ctc_tiffs(path):
    path = os.path.join(path, '')
    ims = load_tiff_frames(path)
    return [(ims, {})]