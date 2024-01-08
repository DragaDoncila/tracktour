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


def closest_neighbour_child_cost(parent_coords, child_coords):
    min_dist = math.inf
    for i, j in combinations(range(len(child_coords)), 2):
        coords_i = child_coords[i]
        coords_j = child_coords[j]
        inter_child_dist = np.linalg.norm(coords_j - coords_i)
        dist_i = np.linalg.norm(parent_coords - coords_i)
        dist_j = np.linalg.norm(parent_coords - coords_j)
        total_dist = inter_child_dist + min(dist_i, dist_j)
        if total_dist < min_dist:
            min_dist = total_dist
    return min_dist


def dist_to_edge_cost_func(bounding_box_dimensions, node_coords):
    min_to_edge = math.inf
    box_mins = [dim[0] for dim in bounding_box_dimensions]
    # skip time coords as not relevant
    for i in range(len(node_coords)):
        box_dim_max = bounding_box_dimensions[1][i]
        box_dim_min = bounding_box_dimensions[0][i]

        node_val = node_coords[i]
        distance_to_min = node_val - box_dim_min
        distance_to_max = box_dim_max - node_val
        smallest = (
            distance_to_min if distance_to_min < distance_to_max else distance_to_max
        )
        min_to_edge = min_to_edge if min_to_edge < smallest else smallest
    return min_to_edge
