import glob
import logging
import os
import sys
import warnings

import networkx as nx
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

PROPS_OF_INTEREST = [
    "label",
    "centroid",
    "area",
    "bbox",
    "image",
    "slice",
]
# these attributes (PLUS SPATIAL ATTRIBUTES t, [z], y, x and bbox)
PROPS_OF_EXPORT = [
    "label",
    "area",
]


def load_tiff_frames(im_dir):
    all_tiffs = list(sorted(glob.glob(os.path.join(im_dir, "*.tif"))))
    n_frames = len(all_tiffs)
    if not n_frames:
        raise FileNotFoundError(f"Couldn't find any .tif files in {im_dir}")

    first_im = imread(all_tiffs[0])
    shape = (len(all_tiffs), *first_im.shape)
    dtype = first_im.dtype
    stack = np.zeros(shape=shape, dtype=dtype)
    stack[0] = first_im

    for i, f in enumerate(tqdm(all_tiffs[1:], "Loading TIFFs")):
        imread(f, out=stack[i + 1])

    return stack


def get_ctc_output(original_seg, tracked_nx, frame_key, value_key, location_keys):
    # remove merges (if any) from tracked_nx - keep closest node
    mergeless = remove_merges(tracked_nx, location_keys)
    max_id = assign_track_id(mergeless)
    node_df = pd.DataFrame.from_dict(mergeless.nodes, orient="index")
    relabelled_seg = mask_by_id(node_df, original_seg, frame_key, value_key)
    nx.set_node_attributes(
        mergeless, nx.get_node_attributes(mergeless, "track-id"), name=value_key
    )
    track_df = get_ctc_tracks(mergeless)
    return relabelled_seg, track_df, max_id


def get_ctc_ds_name(pth):
    """Finds CTC dataset name '{name}_{seq#}' from path.

    Looks for sequence number, then finds parent directory with
    two or three hyphens in the name.
    """
    dirs = os.path.normpath(pth).split(os.path.sep)
    seq_index = [index for index, comp in enumerate(dirs) if comp[:2].isdigit()]
    ds_name_index = [index - 1 for index in seq_index]
    ds_name = ""
    for i, idx in enumerate(ds_name_index):
        if idx >= 0 and len(dirs[idx].split("-")) >= 3:
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
    centroid_cols = None
    for frame_prop, centroid_cols in get_centers(im_arr):
        props.append(frame_prop)
        if centroid_cols is None:
            centroid_cols = centroid_cols
    needed_props = (
        ["t"]
        + centroid_cols
        + PROPS_OF_EXPORT
        + [f"bbox-{i}" for i in range(len(centroid_cols) * 2)]
    )
    props = pd.concat(props, ignore_index=True)[needed_props]
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
                properties=PROPS_OF_INTEREST,
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

        def medoid_or_centroid(row):
            if (
                current_frame[tuple(row[centroid_cols].values.astype(int))]
                == row["label"]
            ):
                row[centroid_cols] = row[centroid_cols]
            else:
                row[centroid_cols] = get_medoid(row)
            return row

        props = props.apply(
            medoid_or_centroid,
            axis=1,
        )
        yield props, centroid_cols
