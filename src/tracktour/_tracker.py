from collections import defaultdict
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from tqdm import tqdm

from ._costs import (
    closest_neighbour_child_cost,
    dist_to_edge_cost_func,
    euclidean_cost_func,
)


class Tracker:
    def __init__(self, im_shape: Tuple[int, int], k_neighbours: int = 10) -> None:
        self.im_shape = im_shape
        self.k_neighbours = k_neighbours

    def solve(
        self,
        detections: pd.DataFrame,
        time_key: str = "t",
        location_keys: Tuple[str] = ("y", "x"),
        value_key: Optional[str] = None,
    ):
        # build kd-trees
        kd_dict = self._build_trees(detections, time_key, location_keys)

        # get candidate edges
        edge_df = self._get_candidate_edges(detections, time_key, kd_dict)

        # migration cost (on edges) is just
        edge_df["migration_cost"] = edge_df["distance"]

        # compute costs for division and appearance/exit - on nodes
        detections["enter_exit_cost"] = dist_to_edge_cost_func(
            self.im_shape, detections, location_keys
        )
        detections["div_cost"] = closest_neighbour_child_cost(
            detections, location_keys, edge_df
        )
        # build model

        # solve, edge_df

    def _build_trees(
        self, detections: pd.DataFrame, time_key: str, location_keys: Tuple[str]
    ):
        """Build dictionary of KDTrees for each frame in detections

        Parameters
        ----------
        detections : pd.DataFrame
            _description_
        time_key : str
            _description_
        location_keys : Tuple[str]
            _description_

        Returns
        -------
        dict[int, KDTree]
            dictionary of KDTree objects for each frame in detections
        """
        kd_dict = {}
        sorted_ts = sorted(detections[time_key].unique())
        for t in sorted_ts:
            frame_detections = detections[detections[time_key] == t][
                list(location_keys)
            ]
            kd_dict[t] = KDTree(frame_detections)

        return kd_dict

    def _get_candidate_edges(self, detections: pd.DataFrame, time_key: str, kd_dict):
        all_edges = defaultdict(list)
        sorted_ts = sorted(detections[time_key].unique())
        for source_t in tqdm(
            sorted_ts[:-1], total=len(sorted_ts) - 1, desc="Computing candidate edges"
        ):
            dest_t = source_t + 1
            source_frame_tree = kd_dict[source_t]
            dest_frame_tree = kd_dict[dest_t]
            if source_frame_tree.n == 0:
                raise ValueError(f"KDTree for frame {source_t} is empty.")
            if dest_frame_tree.n == 0:
                raise ValueError(f"KDTree for frame {dest_t} is empty.")
            # not querying for more neighbours than are in the frame
            k = (
                self.k_neighbours
                if dest_frame_tree.n > self.k_neighbours
                else dest_frame_tree.n
            )
            # passing 1 makes query return an int so we pass a list containing 1 instead
            if k == 1:
                k = [1]
            k_closest_distances, k_closest_indices = dest_frame_tree.query(
                source_frame_tree.data, k=k
            )
            all_edges["u"].append(
                np.repeat(detections[detections.t == source_t].index.values, k)
            )
            # grab the vertex indices of closest neighbours using position indexing into detections
            # TODO: dangerous? maybe...
            all_edges["v"].append(
                detections[detections.t == dest_t]
                .iloc[k_closest_indices.ravel()]
                .index.values
            )
            all_edges["distance"].append(k_closest_distances.ravel())

        all_edges["u"] = np.concatenate(all_edges["u"])
        all_edges["v"] = np.concatenate(all_edges["v"])
        all_edges["distance"] = np.concatenate(all_edges["distance"])
        edge_df = pd.DataFrame(all_edges)
        return edge_df
