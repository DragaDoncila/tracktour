import math
from itertools import combinations

import numpy as np


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


def closest_neighbour_child_cost(detections, location_keys, edge_df):
    min_dists = []
    for det_row in detections.itertuples():
        potential_children = edge_df[edge_df["u"] == det_row.Index]
        if potential_children.empty:
            min_dists.append(-1)
            continue
        src_coords = np.asarray([getattr(det_row, key) for key in location_keys])
        child_coords = {
            v: np.asarray(detections.loc[v, list(location_keys)])
            for v in potential_children["v"]
        }
        # let's also do closest child in here
        interchild_distances = [
            # distance between potential children in the next frame
            np.linalg.norm(child_coords[u] - child_coords[v]) +
            # distance from parent to closest potential child
            min(
                np.linalg.norm(src_coords - child_coords[u]),
                np.linalg.norm(src_coords - child_coords[v]),
            )
            for u, v in combinations(child_coords.keys(), 2)
        ]
        min_dists.append(min(interchild_distances))
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
