import glob
import logging
import os
import sys
import warnings

import numpy as np
import pandas as pd
import yaml
from skimage.graph import central_pixel, pixel_graph
from skimage.measure import regionprops_table
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


def get_ctc_ds_name(pth):
    """Finds CTC dataset name '{name}_{seq#}' from path.

    Looks for sequence number, then finds parent directory with
    two hyphens in the name.
    """
    dirs = os.path.normpath(pth).split(os.path.sep)
    seq_index = [index for index, comp in enumerate(dirs) if comp[:2].isdigit()]
    ds_name_index = [index - 1 for index in seq_index]
    ds_name = ""
    for i, idx in enumerate(ds_name_index):
        if idx >= 0 and len(dirs[idx].split("-")) == 3:
            ds_name = dirs[idx] + "_" + dirs[seq_index[i]][:2]
            break
    return ds_name


def read_scale(ds_name):
    """Reads scale for CTC dataset. Warns if name is not found, and returns None."""
    scale_path = os.path.join(os.path.dirname(__file__), "scales.yaml")
    scales = {}
    with open(scale_path, "r") as f:
        scales = yaml.safe_load(f)
    if ds_name not in scales:
        warnings.warn(f"Scale for {ds_name} not found.")
        return None
    return scales[ds_name]["pixel_scale"]


def get_im_centers(im_pth):
    im_arr = load_tiff_frames(im_pth)
    coords_df, min_t, max_t, corners = extract_im_centers(im_arr)
    return im_arr, coords_df, min_t, max_t, corners


def extract_im_centers(im_arr):
    props = []
    for frame_prop in get_centers(im_arr):
        props.append(frame_prop)
    props = pd.concat(props, ignore_index=True)
    return props, *get_im_info(im_arr)


def get_im_info(im_arr):
    min_t = 0
    max_t = im_arr.shape[0] - 1
    corners = [tuple([0 for _ in range(len(im_arr.shape[1:]))]), im_arr.shape[1:]]
    return min_t, max_t, corners


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
            logger.warning(
                f"Region skeleton for {prop.label} is empty. Using whole region."
            )
            region_skeleton = region
        g, nodes = pixel_graph(region_skeleton, connectivity=2)
        medoid_offset, _ = central_pixel(
            g, nodes=nodes, shape=region_skeleton.shape, partition_size=100
        )
    medoid_offset = np.asarray(medoid_offset)
    bbox = prop[[col for col in prop.index if "bbox" in col]].values
    top_left = np.asarray(bbox[: region.ndim])
    medoid = tuple(top_left + medoid_offset)
    return medoid


def get_centers(segmentation):
    n_frames = segmentation.shape[0]
    if len(segmentation.shape) == 3:
        centroid_cols = ["y", "x"]
    elif len(segmentation.shape) == 4:
        centroid_cols = ["z", "y", "x"]
    else:
        raise ValueError(
            f"Only 2D+T and 3D+T images are supported. Segmentation has shape {segmentation.shape}."
        )
    for i in tqdm(range(n_frames), desc="Extracting Centroids"):
        current_frame = segmentation[i]
        props = pd.DataFrame(
            regionprops_table(
                current_frame,
                properties=(
                    "label",
                    "centroid",
                    "area",
                    "bbox",
                    "eccentricity",
                    "image",
                    "solidity",
                    "slice",
                ),
            )
        )
        props = props.rename(
            columns={
                **{
                    f"centroid-{i}": centroid_cols[i] for i in range(len(centroid_cols))
                },
            }
        )
        props["t"] = i
        new_col = props.apply(
            lambda row: row[centroid_cols]
            if current_frame[tuple(row[centroid_cols].values.astype(int))]
            == row["label"]
            else get_medoid(row),
            axis=1,
        )
        props[centroid_cols] = new_col
        yield props
