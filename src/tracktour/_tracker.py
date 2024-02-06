import math
import time
from collections import defaultdict
from enum import Enum
from typing import Dict, Optional, Tuple

import gurobipy as gp
import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from tqdm import tqdm

from ._costs import (
    closest_neighbour_child_cost,
    dist_to_edge_cost_func,
    euclidean_cost_func,
)


class VirtualVertices(Enum):
    SOURCE = -1
    APP = -2
    DIV = -3
    TARGET = -4


class Tracker:
    MIGRATION_EDGE_CAPACITY = 2
    VIRTUAL_EDGE_CAPACITY = 1
    MINIMAL_VERTEX_DEMAND = 1
    VIRTUAL_INDEX_TO_LABEL = {
        VirtualVertices.SOURCE.value: "s",
        VirtualVertices.APP.value: "a",
        VirtualVertices.DIV.value: "d",
        VirtualVertices.TARGET.value: "t",
    }
    VIRTUAL_LABEL_TO_INDEX = {v: k for k, v in VIRTUAL_INDEX_TO_LABEL.items()}

    @staticmethod
    def index_to_label(index):
        if index in Tracker.VIRTUAL_INDEX_TO_LABEL:
            return Tracker.VIRTUAL_INDEX_TO_LABEL[index]
        return index

    @staticmethod
    def label_to_index(label):
        if label in Tracker.VIRTUAL_LABEL_TO_INDEX:
            return Tracker.VIRTUAL_LABEL_TO_INDEX[label]
        return label

    def __init__(self, im_shape: Tuple[int, int], k_neighbours: int = 10) -> None:
        self.im_shape = im_shape
        self.k_neighbours = k_neighbours

    def solve(
        self,
        detections: pd.DataFrame,
        time_key: str = "t",
        location_keys: Tuple[str] = ("y", "x"),
        # TODO: is this needed/useful?
        value_key: Optional[str] = None,
        # TODO: ?
        # migration_only: bool = False,
    ):
        """_summary_

        Parameters
        ----------
        detections : pd.DataFrame
            dataframe where each row is a detection, with coordinates at location_keys and time at time_key. Index must be sequential integers from 0. Detections
            must be sorted by time.
        time_key : str, optional
            _description_, by default "t"
        location_keys : Tuple[str], optional
            _description_, by default ("y", "x")
        value_key : Optional[str], optional
            _description_, by default None

        Returns
        -------
        _type_
            _description_
        """
        start = time.time()

        # build kd-trees
        kd_dict = self._build_trees(detections, time_key, location_keys)

        # get candidate edges
        edge_df = self._get_candidate_edges(detections, time_key, kd_dict)

        # migration cost (on edges) is just euclidean distance
        edge_df["cost"] = edge_df["distance"]

        # compute costs for division and appearance/exit - on vertices
        detections["enter_exit_cost"] = dist_to_edge_cost_func(
            self.im_shape, detections, location_keys
        )
        detections["div_cost"] = closest_neighbour_child_cost(
            detections, location_keys, edge_df
        )

        # build model
        model, vars, all_edges, all_vertices = self._to_gurobi_model(
            detections, edge_df, location_keys
        )

        dur = time.time() - start

        # solve and store solution on edges
        model.optimize()
        self._store_solution(model, all_edges)

        print(f"Building took {dur} seconds")
        # filter down to just migration edges
        migration_edges = all_edges[
            (all_edges.u >= 0) & (all_edges.v >= 0) & (all_edges.flow > 0)
        ].copy()
        return migration_edges

    def _build_trees(
        self, detections: pd.DataFrame, time_key: str, location_keys: Tuple[str]
    ) -> Dict[int, KDTree]:
        """Build dictionary of KDTrees for each frame in detections

        Parameters
        ----------
        detections : pd.DataFrame
            dataframe of real detections for generating tracks
        time_key : str
            column in `detections` denoting the frame number or time index
        location_keys : Tuple[str, str, str]
            tuple of columns in `detections` denoting spatial coordinates

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

    def _get_candidate_edges(
        self, detections: pd.DataFrame, time_key: str, kd_dict: Dict[int, KDTree]
    ):
        """Get edges joining vertices in frame t to candidate vertices in frame t+1

        Parameters
        ----------
        detections : pd.DataFrame
            dataframe of real detections for generating tracks
        time_key : str
            column in `detections` denoting the frame number or time index
        kd_dict : Dict[int, KDTree]
            dictionary of KDTree objects for each frame in detections

        Returns
        -------
        pd.DataFrame
            dataframe containing source, target and distance information for all
            candidate edges. Columns `u` and `v` index into `detections`.

        Raises
        ------
        ValueError
            if a frame is empty and contains no detections
        """
        all_edges = defaultdict(list)
        sorted_ts = sorted(detections[time_key].unique())
        for source_t in tqdm(
            sorted_ts[:-1], total=len(sorted_ts) - 1, desc="Computing candidate edges"
        ):
            dest_t = source_t + 1
            source_frame_tree = kd_dict[source_t]
            dest_frame_tree = kd_dict[dest_t]
            # TODO: I don't know that we actually want this do we...?
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
            # this will break if detections are intermingled in time (should copy detections and return our "tracked_detections" or similar)
            all_edges["v"].append(
                detections[detections.t == dest_t]
                .iloc[k_closest_indices.ravel()]
                .index.values
            )
            all_edges["distance"].append(k_closest_distances.ravel())

        all_edges["u"] = np.concatenate(all_edges["u"])
        all_edges["v"] = np.concatenate(all_edges["v"])
        all_edges["distance"] = np.concatenate(all_edges["distance"])
        all_edges["capacity"] = Tracker.MIGRATION_EDGE_CAPACITY
        edge_df = pd.DataFrame(all_edges)
        return edge_df

    def _to_gurobi_model(self, detections, edge_df, location_keys):
        """Takes detections and candidate edges and builds gurobi flow model.

        The model will contain virtual vertices source, appearance, division
        and target, and edges adjacent to these. Constraints are added to the
        model to ensure flow conservation is preserved, vertex demand is met,
        and division only occurs after appearance or migration.

        Parameters
        ----------
        detections : pd.DataFrame
            dataframe of real detections for generating tracks
        edge_df : pd.DataFrame
            dataframe of candidate edges for migration flow
        location_keys : Tuple[str, str, str]
            tuple of columns in `detections` denoting spatial coordinates

        Returns
        -------
        _type_
            _description_
        """
        full_det = self._get_all_vertices(detections, location_keys)
        all_edges = self._get_all_edges(edge_df, detections)

        model_var_info = {
            (
                e.Index,
                Tracker.index_to_label(e.u),
                Tracker.index_to_label(e.v),
            ): e.cost
            for e in all_edges.itertuples()
        }
        edge_capacities = all_edges.capacity.values
        var_names = gp.tuplelist(list(model_var_info.keys()))

        src_label = Tracker.index_to_label(VirtualVertices.SOURCE.value)
        app_label = Tracker.index_to_label(VirtualVertices.APP.value)
        div_label = Tracker.index_to_label(VirtualVertices.DIV.value)
        target_label = Tracker.index_to_label(VirtualVertices.TARGET.value)

        m = gp.Model("tracks")
        flow = m.addVars(
            var_names, obj=model_var_info, lb=0, ub=edge_capacities, name="flow"
        )
        # flow = (edge_id, source_label, target_label)
        src_out_edges = flow.select("*", src_label, "*")
        target_in_edges = flow.select("*", "*", target_label)
        # whole network flow
        m.addConstr(sum(src_out_edges) == sum(target_in_edges), "conserv_network")

        # dummy vertex flow conservation
        app_out = flow.select("*", app_label, "*")
        app_in = flow.select("*", src_label, app_label)
        m.addConstr(sum(app_in) == sum(app_out), "conserv_app")

        div_out = flow.select("*", div_label, "*")
        div_in = flow.select("*", src_label, div_label)
        m.addConstr(sum(div_in) == sum(div_out), "conserv_div")

        for vertex_id in detections.index:
            lbl = Tracker.index_to_label(vertex_id)
            outgoing = flow.select("*", lbl, "*")
            incoming = flow.select("*", "*", lbl)
            # node-wise flow conservation
            m.addConstr(sum(incoming) == sum(outgoing), f"conserv_{lbl}")
            # minimal flow into each node
            m.addConstr(sum(incoming) >= Tracker.MINIMAL_VERTEX_DEMAND, f"demand_{lbl}")

            # TODO: div optional?
            div_incoming = flow.select(
                "*", Tracker.index_to_label(VirtualVertices.DIV.value), lbl
            )
            # division only occurs after appearance or migration
            m.addConstr(
                sum(incoming) - sum(div_incoming) >= sum(div_incoming), f"div_{lbl}"
            )

        return m, flow, all_edges, full_det

    def _get_all_vertices(self, detections, location_keys):
        """Adds virtual vertices to copy of detections and returns.

        Any columns for which virtual vertices have no value are given value -1.
        Appearance and division cost of virtual vertices is 0.

        Parameters
        ----------
        detections : pd.DataFrame
            dataframe of real detections with indices of the natural numbers
        location_keys : Tuple[str, str, str]
            tuple of columns in `detections` denoting spatial coordinates

        Returns
        -------
        pd.DataFrame
            full dataframe of vertices including virtual vertices
        """
        v_vertices = {
            # s, a, d, t
            "Index": [
                VirtualVertices.SOURCE.value,
                VirtualVertices.APP.value,
                VirtualVertices.DIV.value,
                VirtualVertices.TARGET.value,
            ],
            "t": -1,
            **{key: -1 for key in location_keys},
            "enter_exit_cost": 0,
            "div_cost": 0,
        }
        v_vertices_df = pd.DataFrame(v_vertices).set_index("Index")
        full_det = pd.concat([detections.copy(), v_vertices_df])
        return full_det

    def _get_all_edges(self, edge_df, detections):
        """Adds virtual edges to copy of edge_df and returns.

        Connects source vertex to appearance and division. Connects
        all real vertices to appearance and target with appropriate costs (0
        for vertices in first frame and final frame respectively). `distance`
        values for edges adjacent to virtual vertices are -1.

        Parameters
        ----------
        edge_df : pd.DataFrame
            dataframe of candidate edges for migration flow
        detections : pd.DataFrame
            dataframe of real detections for generating tracks

        Returns
        -------
        pd.DataFrame
            dataframe of all edges forming the network
        """
        min_t = detections.t.min()
        max_t = detections.t.max()

        # add SA and SD
        v_edges = {
            "u": VirtualVertices.SOURCE.value,
            "v": [VirtualVertices.APP.value, VirtualVertices.DIV.value],
            "capacity": math.inf,
            "cost": 0,
            "distance": -1,
        }
        # add appearance edges to all vertices
        app_edges = {
            "u": VirtualVertices.APP.value,
            "v": detections.index.values,
            "capacity": Tracker.MIGRATION_EDGE_CAPACITY,
            "cost": [
                r.enter_exit_cost if r.t > min_t else 0 for r in detections.itertuples()
            ],
            "distance": -1,
        }
        # add disappearance edges to all vertices
        exit_edges = {
            "u": detections.index.values,
            "v": VirtualVertices.TARGET.value,
            "capacity": Tracker.MIGRATION_EDGE_CAPACITY,
            "cost": [
                r.enter_exit_cost if r.t < max_t else 0 for r in detections.itertuples()
            ],
            "distance": -1,
        }
        # add div edges to all vertices (not in final frame)
        divisible_detections = detections[detections.t < max_t]
        div_edges = {
            "u": VirtualVertices.DIV.value,
            "v": divisible_detections.index.values,
            "capacity": Tracker.VIRTUAL_EDGE_CAPACITY,
            "cost": divisible_detections.div_cost.values,
            "distance": -1,
        }
        all_edges = pd.concat(
            [
                pd.DataFrame(dct)
                for dct in [edge_df.copy(), v_edges, app_edges, exit_edges, div_edges]
            ]
        )
        return all_edges.reset_index()

    def _store_solution(self, model, all_edges):
        """Stores solution from gurobi model in edge dataframe.

        Parameters
        ----------
        model : gurobipy.Model
            model with optimal solution used to extract flow
        all_edges : pd.DataFrame
            dataframe of all edges forming the network
        """
        sol_dict = {
            tuple(
                [
                    int(Tracker.label_to_index(idx))
                    for idx in v.VarName.strip("flow[").strip("]").split(",")
                ]
            ): v.X
            for v in model.getVars()
        }
        edge_index, flow = zip(*[(k[0], v) for k, v in sol_dict.items()])
        all_edges.loc[list(edge_index), "flow"] = list(flow)
        return all_edges
