import glob
import logging
import os
import sys

import numpy as np
import pandas as pd
from skimage.graph import central_pixel, pixel_graph
from skimage.measure import regionprops
from skimage.morphology import skeletonize
from tifffile import imread

from tracktour._graph_util import assign_track_id, get_ctc_tracks, remove_merges
from tracktour._viz_util import mask_by_id

try:
    from napari.utils import progress as tqdm
except ImportError:
    from tqdm import tqdm

logger = logging.getLogger("tracktour")


def load_tiff_frames(im_dir):
    all_tiffs = list(sorted(glob.glob(os.path.join(im_dir, "*.tif"))))
    n_frames = len(all_tiffs)
    if not n_frames:
        raise FileNotFoundError(f"Couldn't find any .tif files in {im_dir}")

    stack = []
    for f in tqdm(all_tiffs, "Loading TIFFs"):
        stack.append(imread(f))

    im = np.stack(stack)
    return im


def get_ctc_output(original_seg, tracked_nx, frame_key, value_key, location_keys):
    # remove merges from tracked_nx - keep closest node
    mergeless = remove_merges(tracked_nx, location_keys)
    max_id = assign_track_id(mergeless)
    node_df = pd.DataFrame.from_dict(mergeless.nodes, orient="index")
    relabelled_seg = mask_by_id(node_df, original_seg, frame_key, value_key)
    track_df = get_ctc_tracks(mergeless)
    return relabelled_seg, track_df


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
    if center_coords.shape[1] == 3:
        cols = ["t", "y", "x"]
    elif center_coords.shape[1] == 4:
        cols = ["t", "z", "y", "x"]
    else:
        raise ValueError(
            f"Only 2D+T and 3D+T images are supported. Centroids have {center_coords.shape[1]} coordinates."
        )
    coords_df = pd.DataFrame(center_coords, columns=cols)
    coords_df["t"] = coords_df["t"].astype(int)
    coords_df["label"] = labels
    min_t = 0
    max_t = im_arr.shape[0] - 1
    corners = [tuple([0 for _ in range(len(im_arr.shape[1:]))]), im_arr.shape[1:]]
    return coords_df, min_t, max_t, corners


def get_medoid(prop):
    region = prop.image
    region_skeleton = skeletonize(region).astype(bool)
    skeleton_sum = np.sum(region_skeleton)
    if skeleton_sum == 1:
        logger.info(
            f"Region skeleton for {prop.label} is a single pixel, using it as medoid."
        )
        medoid_offset = np.unravel_index(
            np.argmax(region_skeleton), region_skeleton.shape
        )
    else:
        if skeleton_sum == 0:
            logger.warning(f"Region skeleton for {prop.label} is empty.")
        g, nodes = pixel_graph(region_skeleton, connectivity=2)
        medoid_offset, _ = central_pixel(
            g, nodes=nodes, shape=region_skeleton.shape, partition_size=100
        )
    medoid_offset = np.asarray(medoid_offset)
    top_left = np.asarray(prop.bbox[: region.ndim])
    medoid = tuple(top_left + medoid_offset)
    return medoid


def get_centers(segmentation):
    n_frames = segmentation.shape[0]
    for i in tqdm(range(n_frames), desc="Extracting Centroids"):
        current_frame = segmentation[i]
        centers = []
        labels = []
        props = regionprops(current_frame)
        if props:
            current_centers = [(i, *prop.centroid) for prop in props]
            frame_labels = segmentation[tuple(np.asarray(current_centers, dtype=int).T)]
            label_center_mapping = dict(zip(frame_labels, current_centers))
            # we haven't found centers for these labels, we need to medoid them
            unfound_labels = (
                set(np.unique(current_frame))
                - set(label_center_mapping.keys())
                - set([0])
            )
            for prop in props:
                if prop.label in unfound_labels:
                    logger.info(f"Medoiding label {prop.label} on frame {i}.")
                    label_center_mapping[prop.label] = (i, *get_medoid(prop))
            # 0 is not a valid label and would only exist in the dictionary
            # if some labels required the medoid treatment.
            label_center_mapping.pop(0, None)
            labels, centers = zip(*label_center_mapping.items())
        yield centers, labels
