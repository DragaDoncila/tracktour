import math
from itertools import combinations

import numpy as np
from scipy.spatial.distance import pdist


def euclidean_cost_func(source_node, dest_node):
    return np.linalg.norm(
        np.asarray(dest_node["coords"]) - np.asarray(source_node["coords"])
    )


def min_pairwise_distance_cost(child_distances):
    """Get the smallest sum of distance from parent to two potential children.

    Parameters
    ----------
    child_distances : List[Tuple[float]]
        list of distances to k closest children
    """
    min_dist = math.inf
    for i, j in combinations(range(len(child_distances)), 2):
        # TODO: no concept of angles
        distance_first_child = child_distances[i]
        distance_second_child = child_distances[j]
        distance_sum = distance_first_child + distance_second_child
        if distance_sum < min_dist:
            min_dist = distance_sum
    return min_dist


# def closest_neighbour_child_cost(parent_coords, child_coords):
#     min_dist = math.inf
#     for i, j in combinations(range(len(child_coords)), 2):
#         coords_i = child_coords[i]
#         coords_j = child_coords[j]
#         inter_child_dist = np.linalg.norm(coords_j - coords_i)
#         dist_i = np.linalg.norm(parent_coords - coords_i)
#         dist_j = np.linalg.norm(parent_coords - coords_j)
#         total_dist = inter_child_dist + min(dist_i, dist_j)
#         if total_dist < min_dist:
#             min_dist = total_dist
#     return min_dist

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


# def dist_to_edge_cost_func(bounding_box_dimensions, node_coords):
#     min_to_edge = math.inf
#     box_mins = [dim[0] for dim in bounding_box_dimensions]
#     # skip time coords as not relevant
#     for i in range(len(node_coords)):
#         box_dim_max = bounding_box_dimensions[1][i]
#         box_dim_min = bounding_box_dimensions[0][i]

#         node_val = node_coords[i]
#         distance_to_min = node_val - box_dim_min
#         distance_to_max = box_dim_max - node_val
#         smallest = (
#             distance_to_min if distance_to_min < distance_to_max else distance_to_max
#         )
#         min_to_edge = min_to_edge if min_to_edge < smallest else smallest
#     return min_to_edge


# TODO: Do we want to support negative coords
def dist_to_edge_cost_func(im_shape, detections, location_keys):
    # distance from 0 is just the coordinate itself
    dist_to_top_left = detections[list(location_keys)].values
    dist_to_bottom_right = np.asarray(im_shape) - dist_to_top_left
    dist_to_borders = np.concatenate((dist_to_top_left, dist_to_bottom_right), axis=1)
    min_dists = np.min(dist_to_borders, axis=1)
    return min_dists
