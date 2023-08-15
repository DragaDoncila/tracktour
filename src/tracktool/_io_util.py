from tifffile import TiffFile
from skimage.measure import regionprops
from skimage.graph import pixel_graph, central_pixel
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
from ._flow_graph import FlowGraph

def load_graph(seg_path):
    ims, coords, min_t, max_t, corners = get_im_centers(seg_path)
    graph = FlowGraph(corners, coords, min_t=min_t, max_t=max_t)
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
    centers, labels = get_centers(im_arr)
    center_coords = np.asarray(get_point_coords(centers))
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
    centers_of_mass = []
    all_labels = []
    for i in tqdm(range(n_frames), desc='Processing frames'):
        current_frame = segmentation[i]
        props = regionprops(current_frame)
        if props:
            current_centers = [prop.centroid for prop in props]
            frame_labels = current_frame[tuple(np.asarray(current_centers, dtype=int).T)]
            label_center_mapping = dict(zip(frame_labels, current_centers))
            # we haven't found centers for these labels, we need to medoid them
            unfound_labels = set(np.unique(current_frame)) - set(label_center_mapping.keys()) - set([0])
            for prop in props:
                if prop.label in unfound_labels:
                    label_center_mapping[prop.label] = get_medoid(prop)
            # 0 is not a valid label and would only exist in the dictionary
            # if some labels required the medoid treatment.
            label_center_mapping.pop(0, None)
            labels, centers = zip(*label_center_mapping.items())
            centers_of_mass.append(centers)
            all_labels.extend(labels)
    return centers_of_mass, all_labels

def get_point_coords(centers_of_mass):
    points_list = []
    for i, frame_centers in enumerate(centers_of_mass):
        points = [(i, *center) for center in frame_centers]
        points_list.extend(points)
    return points_list
