from tifffile import TiffFile
from skimage.measure import regionprops
from skimage.graph import pixel_graph, central_pixel
import glob
import networkx as nx
import numpy as np
import pandas as pd
from ._flow_graph import FlowGraph

try: 
    from napari.utils import progress as tqdm
except ImportError:
    from tqdm import tqdm

def load_graph(seg_path, n_neighbours=10):
    ims, coords, min_t, max_t, corners = get_im_centers(seg_path)
    graph = FlowGraph(corners, coords, n_neighbours=n_neighbours, min_t=min_t, max_t=max_t)
    return ims, graph

def peek(im_file):
    with TiffFile(im_file) as im:
        im_shape = im.pages[0].shape
        im_dtype = im.pages[0].dtype
    return im_shape, im_dtype

def load_tiff_frames(im_dir):
    all_tiffs = list(sorted(glob.glob(f'{im_dir}*.tif')))
    n_frames = len(all_tiffs)
    frame_shape, im_dtype = peek(all_tiffs[0])
    im_array = np.zeros((n_frames, *frame_shape), dtype=im_dtype)
    for i, tiff_pth in enumerate(all_tiffs):
        with TiffFile(tiff_pth) as im:
            im_array[i] = im.pages[0].asarray()
    return im_array

def get_im_centers(im_pth):
    im_arr = load_tiff_frames(im_pth)
    coords_df, min_t, max_t, corners = extract_im_centers(im_arr)
    return im_arr, coords_df, min_t, max_t, corners

def extract_im_centers(im_arr):
    centers, labels = [], []
    for cent, lab in get_centers(im_arr):
        centers.extend(cent)
        labels.extend(lab)
    return get_im_info(centers, labels, im_arr)


def get_im_info(centers, labels, im_arr):
    center_coords = np.asarray(centers)
    coords_df = pd.DataFrame(center_coords, columns=['t', 'y', 'x'])
    coords_df['t'] = coords_df['t'].astype(int)
    coords_df['label'] = labels
    min_t = 0
    max_t = im_arr.shape[0]-1
    corners = [tuple([0 for _ in range(len(im_arr.shape[1:]))]), im_arr.shape[1:]]
    return coords_df, min_t, max_t, corners

def get_medoid(prop):
    region = prop.image
    g, nodes = pixel_graph(region, connectivity=2)
    medoid_offset, _ = central_pixel(
            g, nodes=nodes, shape=region.shape, partition_size=100
            )
    medoid_offset = np.asarray(medoid_offset)
    top_left = np.asarray(prop.bbox[:region.ndim])
    medoid = tuple(top_left + medoid_offset)
    return medoid    

def get_centers(segmentation):
    n_frames = segmentation.shape[0]
    for i in tqdm(range(n_frames), desc='Extracting Centroids'):
        current_frame = segmentation[i]
        props = regionprops(current_frame)
        if props:
            current_centers = [(i, *prop.centroid) for prop in props]
            frame_labels = segmentation[tuple(np.asarray(current_centers, dtype=int).T)]
            label_center_mapping = dict(zip(frame_labels, current_centers))
            # we haven't found centers for these labels, we need to medoid them
            unfound_labels = set(np.unique(current_frame)) - set(label_center_mapping.keys()) - set([0])
            for prop in props:
                if prop.label in unfound_labels:
                    label_center_mapping[prop.label] = (i, *get_medoid(prop))
            # 0 is not a valid label and would only exist in the dictionary
            # if some labels required the medoid treatment.
            label_center_mapping.pop(0, None)
            labels, centers = zip(*label_center_mapping.items())
        yield centers, labels

def store_flow(nx_sol, ig_sol):
    ig_sol._g.es.set_attribute_values('flow', 0)
    flow_es = nx.get_edge_attributes(nx_sol, 'flow')
    for e_id, flow in flow_es.items():
        src, target = e_id
        ig_sol._g.es[ig_sol._g.get_eid(src, target)]['flow'] = flow
        
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
