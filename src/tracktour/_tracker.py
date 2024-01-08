import pandas as pd

from scipy.spatial import KDTree
from typing import Optional, Tuple

from ._costs import euclidean_cost_func, dist_to_edge_cost_func, closest_neighbour_child_cost

class Tracker:

    def __init__(self, k_neighbours=10) -> None:
        self._k_neighbours = k_neighbours

        # if we want to make these public, we can add kwargs to init
        self._enter_exit_cost = dist_to_edge_cost_func
        self._migration_cost = euclidean_cost_func
        self._div_cost = closest_neighbour_child_cost


    def solve(self, detections: pd.DataFrame, time_key: str ='t', location_keys: Tuple[str]=('y', 'x'), value_key: Optional[str]=None):
        # build kd-trees
        kd_dict = self._build_trees(detections, time_key, location_keys)

        # compute costs and store in node and edge dataframes
        edge_df = self._compute_costs(detections, time_key, location_keys, kd_dict))
        # build model

        # solve

    def _build_trees(self, detections: pd.DataFrame, time_key: str, location_keys: Tuple[str]):
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
        sorted_ts = detections[time_key].sort_values()
        for t in sorted_ts:
            frame_detections = detections[detections[time_key] == t][list(location_keys)]
            kd_dict[t] = KDTree(frame_detections)

        return kd_dict

    def _compute_costs(self, detections: pd.DataFrame, time_key: str, location_keys: Tuple[str], kd_dict):
        pass
