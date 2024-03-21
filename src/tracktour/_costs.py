import math

import numpy as np
from scipy.spatial.distance import pdist


# Following four functions reproduced from https://stackoverflow.com/a/36867493
def calc_row_idx(k, n):
    return int(
        math.ceil(
            (1 / 2.0) * (-((-8 * k + 4 * n**2 - 4 * n - 7) ** 0.5) + 2 * n - 1) - 1
        )
    )


def elem_in_i_rows(i, n):
    return i * (n - 1 - i) + (i * (i + 1)) // 2


def calc_col_idx(k, i, n):
    return int(n - elem_in_i_rows(i + 1, n) + k)


def condensed_to_square(pdist_i, num_children):
    """Given condensed index, get children indices in the square form.

    Parameters
    ----------
    pdist_i : int
        index of distance in condensed form
    num_children : int
        number of original points for pdist

    Returns
    -------
    int, int
        indices of original points that gave distance
    """
    i = calc_row_idx(pdist_i, num_children)
    j = calc_col_idx(pdist_i, i, num_children)
    return i, j


def closest_neighbour_child_cost(detections, location_keys, edge_df):
    # TODO: assumes groups in groupby are positioned by order in detections
    location_col_indices = [detections.columns.get_loc(key) for key in location_keys]
    edges_by_source = edge_df.groupby("u")
    divisible_detections = detections[detections.t < detections.t.max()]
    det_outgoing_edges = zip(divisible_detections.itertuples(), edges_by_source)
    min_dists = [-1 for _ in range(len(detections))]
    for det_row, (group_source, edge_group) in det_outgoing_edges:
        assert det_row.Index == group_source, "Detections and edges are not aligned."
        src_coords = np.asarray([getattr(det_row, key) for key in location_keys])
        child_coord_array = np.asarray(
            [
                [detections.iat[v, l] for l in location_col_indices]
                for v in edge_group["v"].values
            ]
        )
        # if there's only one child, node cannot divide so cost is infinite
        # TODO: just don't have these edges at all
        if len(child_coord_array) == 1:
            min_dists[det_row.Index] = math.inf
            continue

        dists_to_child = np.linalg.norm(src_coords - child_coord_array, axis=1)
        inter_child_dists = pdist(child_coord_array)
        div_costs = inter_child_dists + np.asarray(
            [
                dists_to_child[
                    list(condensed_to_square(i, len(child_coord_array)))
                ].min()
                for i in range(len(inter_child_dists))
            ]
        )
        min_dists[det_row.Index] = div_costs.min()
    return min_dists


# TODO: Do we want to support negative coords
def dist_to_edge_cost_func(im_shape, detections, location_keys):
    # distance from 0 is just the coordinate itself
    dist_to_top_left = detections[list(location_keys)].values
    dist_to_bottom_right = np.asarray(im_shape) - dist_to_top_left
    dist_to_borders = np.concatenate((dist_to_top_left, dist_to_bottom_right), axis=1)
    min_dists = np.min(dist_to_borders, axis=1)
    return min_dists
