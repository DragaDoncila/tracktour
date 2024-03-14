import math
import time
import warnings
from collections import defaultdict
from enum import Enum
from typing import Dict, List, Optional, Tuple

import gurobipy as gp
import networkx as nx
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field
from scipy.spatial import KDTree
from tqdm import tqdm

from ._costs import closest_neighbour_child_cost, dist_to_edge_cost_func


class VirtualVertices(Enum):
    SOURCE = -1
    APP = -2
    DIV = -3
    TARGET = -4


class Tracked(BaseModel):
    # user facing solution
    tracked_edges: pd.DataFrame = Field(
        description="Dataframe of edges in solution. Columns u and v are indices into `tracked_detections`"
    )
    tracked_detections: pd.DataFrame = Field(
        description="Datafarme of detections with indices of the natural numbers. Indices of `tracked_edges` are indices into this dataframe. Coordinates of detections are available at location_keys."
    )
    frame_key: str = Field(description="Key in detections denoting frame.")
    location_keys: List[str] = Field(
        description="Ordered list of keys in detections denoting coordinates."
    )
    value_key: str = Field(
        description="Key in detections denoting integer value of the object."
    )

    # only available in debug mode
    all_edges: Optional[pd.DataFrame] = Field(
        default=None,
        description="Dataframe of all edges in the network with cost and flow information. Columns u and v are indices into `all_vertices`. Virtual vertices are labelled with negative indices. Only available in debug mode.",
    )
    all_vertices: Optional[pd.DataFrame] = Field(
        default=None,
        description="Dataframe of all vertices in the network with cost and flow information. Virtual vertices are labelled with negative indices. Only available in debug mode.",
    )
    model: Optional[gp.Model] = Field(
        default=None,
        description="Solved gurobi model of instance. Only available in debug mode.",
    )
    build_time: float = Field(
        default=-1,
        description="Time taken to build the candidate graph of the instance including cost computation. Only available in debug mode.",
    )
    gp_model_time: float = Field(
        default=-1,
        description="Time taken to build the gurobi model from the candidate graph. Only available in debug mode.",
    )
    solve_time: float = Field(
        default=-1,
        description="Time taken to solve the gurobi model as reported by gurobipy. Only available in debug mode.",
    )
    wall_time: float = Field(
        default=-1,
        description="Total time taken to build and solve the instance. Only available in debug mode.",
    )

    class Config:
        arbitrary_types_allowed = True

    def as_nx_digraph(self):
        edges = self.tracked_edges
        nodes = self.tracked_detections
        sol_graph = nx.from_pandas_edgelist(
            edges,
            "u",
            "v",
            ["flow"],  # optional but may be useful for debugging or finding merge edges
            create_using=nx.DiGraph,
        )
        det_keys = [self.frame_key] + self.location_keys + [self.value_key]
        sol_graph.add_nodes_from(nodes[det_keys].to_dict(orient="index").items())
        return sol_graph


# TODO: should also be pydantic?
"""
Assumptions that need to be enforced/documented/removed:

- detections are grouped by frame and sorted (can't have interleaved detections)
- the frame key is a direct index into the segmentation
- ALL COORDINATES ARE POSITIVE and within the given `im_shape`
"""


class Tracker:
    MERGE_EDGE_CAPACITY = 2
    # can be set to 2 when we are allowing "cheat" appearance
    APPEARANCE_EDGE_CAPACITY = 1
    DIVISION_EDGE_CAPACITY = 1
    MINIMAL_VERTEX_DEMAND = 1
    VIRTUAL_INDEX_TO_LABEL = {
        VirtualVertices.SOURCE.value: "s",
        VirtualVertices.APP.value: "a",
        VirtualVertices.DIV.value: "d",
        VirtualVertices.TARGET.value: "t",
    }
    VIRTUAL_LABEL_TO_INDEX = {v: k for k, v in VIRTUAL_INDEX_TO_LABEL.items()}
    DEBUG_MODE = False

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

        self.location_keys = None
        self.frame_key = None
        self.value_key = None

    def solve(
        self,
        detections: pd.DataFrame,
        frame_key: str = "t",
        location_keys: Tuple[str] = ("y", "x"),
        # TODO: we should be able to make this optional
        value_key: Optional[str] = "label",
        # TODO: ?
        # migration_only: bool = False,
    ):
        """_summary_

        Parameters
        ----------
        detections : pd.DataFrame
            dataframe where each row is a detection, with coordinates at location_keys and time at frame_key. Index must be sequential integers from 0. Detections
            must be sorted by frame.
        frame_key : str, optional
            _description_, by default "t"
        location_keys : Tuple[str], optional
            _description_, by default ("y", "x")
        value_key : Optional[str], optional
            _description_, by default None

        Returns
        -------
        _type_
            _description_d
        """
        self.location_keys = location_keys
        self.frame_key = frame_key
        self.value_key = value_key

        # TODO: copy/validate detections
        start = time.time()

        # build kd-trees
        # TODO: stop passing stuff around now that you store it as an attribute
        kd_dict = self._build_trees(detections, frame_key, location_keys)

        # get candidate edges
        edge_df = self._get_candidate_edges(detections, frame_key, kd_dict)

        # migration cost (on edges) is just euclidean distance
        edge_df["cost"] = edge_df["distance"]

        # compute costs for division and appearance/exit - on vertices
        enter_exit_cost, div_cost = self._compute_detection_costs(
            detections, location_keys, edge_df
        )
        detections["enter_exit_cost"] = enter_exit_cost
        detections["div_cost"] = div_cost

        # build model
        model, all_edges, all_vertices, gb_time = self._to_gurobi_model(
            detections, edge_df, frame_key, location_keys
        )

        build_duration = time.time() - start

        # solve and store solution on edges
        model.optimize()
        self._store_solution(model, all_edges)
        solve_duration = model.Runtime

        print(f"Building took {build_duration} seconds")

        # filter down to just migration edges
        migration_edges = all_edges[
            (all_edges.u >= 0) & (all_edges.v >= 0) & (all_edges.flow > 0)
        ].copy()

        wall_time = time.time() - start

        tracked_info = {
            "tracked_edges": migration_edges,
            "tracked_detections": detections,
            "frame_key": self.frame_key,
            "location_keys": self.location_keys,
            "value_key": self.value_key,
        }
        if self.DEBUG_MODE:
            tracked_info.update(
                {
                    "all_edges": all_edges,
                    "all_vertices": all_vertices,
                    "model": model,
                    "build_time": build_duration,
                    "gp_model_time": gb_time,
                    "solve_time": solve_duration,
                    "wall_time": wall_time,
                }
            )
        tracked = Tracked(**tracked_info)
        return tracked

    def _build_trees(
        self, detections: pd.DataFrame, frame_key: str, location_keys: Tuple[str]
    ) -> Dict[int, KDTree]:
        """Build dictionary of KDTrees for each frame in detections

        Parameters
        ----------
        detections : pd.DataFrame
            dataframe of real detections for generating tracks
        frame_key : str
            column in `detections` denoting the frame number or time index
        location_keys : Tuple[str, str, str]
            tuple of columns in `detections` denoting spatial coordinates

        Returns
        -------
        dict[int, KDTree]
            dictionary of KDTree objects for each frame in detections
        """
        kd_dict = {}
        sorted_ts = sorted(detections[frame_key].unique())
        for t in sorted_ts:
            frame_detections = detections[detections[frame_key] == t][
                list(location_keys)
            ]
            kd_dict[t] = KDTree(frame_detections)

        return kd_dict

    def _get_candidate_edges(
        self, detections: pd.DataFrame, frame_key: str, kd_dict: Dict[int, KDTree]
    ):
        """Get edges joining vertices in frame t to candidate vertices in frame t+1

        Parameters
        ----------
        detections : pd.DataFrame
            dataframe of real detections for generating tracks
        frame_key : str
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
        sorted_ts = sorted(detections[frame_key].unique())
        for i in tqdm(
            range(len(sorted_ts) - 1),
            total=len(sorted_ts) - 1,
            desc="Computing candidate edges",
        ):
            source_t = sorted_ts[i]
            dest_t = sorted_ts[i + 1]
            # TODO: I don't know that we actually want this do we...?
            if dest_t != source_t + 1:
                warnings.warn(
                    UserWarning(
                        f"Connecting frames {source_t} and {dest_t}. They are not consecutive. Are you missing detections?"
                    )
                )
            source_frame_tree = kd_dict[source_t]
            dest_frame_tree = kd_dict[dest_t]
            # not querying for more neighbours than are in the frame
            k = (
                self.k_neighbours
                if dest_frame_tree.n >= self.k_neighbours
                else dest_frame_tree.n
            )
            # passing 1 makes query return an int so we pass a list containing 1 instead
            if k == 1:
                k = [1]
            k_closest_distances, k_closest_indices = dest_frame_tree.query(
                source_frame_tree.data, k=k
            )
            all_edges["u"].append(
                np.repeat(detections[detections[frame_key] == source_t].index.values, k)
            )
            # grab the vertex indices of closest neighbours using position indexing into detections
            # TODO: dangerous? maybe...
            # this will break if detections are intermingled in time (should copy detections and return our "tracked_detections" or similar)
            all_edges["v"].append(
                detections[detections[frame_key] == dest_t]
                .iloc[k_closest_indices.ravel()]
                .index.values
            )
            all_edges["distance"].append(k_closest_distances.ravel())

        all_edges["u"] = np.concatenate(all_edges["u"])
        all_edges["v"] = np.concatenate(all_edges["v"])
        all_edges["distance"] = np.concatenate(all_edges["distance"])
        all_edges["capacity"] = self.MERGE_EDGE_CAPACITY
        edge_df = pd.DataFrame(all_edges)
        return edge_df

    def _to_gurobi_model(self, detections, edge_df, frame_key, location_keys):
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
        frame_key : str
            column in `detections` denoting the frame number or time index
        location_keys : Tuple[str, str, str]
            tuple of columns in `detections` denoting spatial coordinates

        Returns
        -------
        _type_
            _description_
        """
        start = time.time()
        full_det = self._get_all_vertices(detections, frame_key, location_keys)
        all_edges = self._get_all_edges(edge_df, detections, frame_key)

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
            if len(div_incoming):
                # division only occurs after appearance or migration
                m.addConstr(
                    sum(incoming) - sum(div_incoming) >= sum(div_incoming), f"div_{lbl}"
                )
        m.update()
        build_time = time.time() - start
        return m, all_edges, full_det, build_time

    def _get_all_vertices(self, detections, frame_key, location_keys):
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
            frame_key: -1,
            **{key: -1 for key in location_keys},
            "enter_exit_cost": 0,
            "div_cost": 0,
        }
        v_vertices_df = pd.DataFrame(v_vertices).set_index("Index")
        full_det = pd.concat([detections.copy(), v_vertices_df])
        return full_det

    def _compute_detection_costs(self, detections, location_keys, edge_df):
        enter_exit_cost = dist_to_edge_cost_func(
            self.im_shape, detections, location_keys
        )
        div_cost = closest_neighbour_child_cost(detections, location_keys, edge_df)
        return enter_exit_cost, div_cost

    def _get_all_edges(self, edge_df, detections, frame_key):
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
        frame_key : str
            column in `detections` denoting the frame number or time index

        Returns
        -------
        pd.DataFrame
            dataframe of all edges forming the network
        """
        min_t = detections[frame_key].min()
        max_t = detections[frame_key].max()

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
            "capacity": self.APPEARANCE_EDGE_CAPACITY,
            "cost": [
                r.enter_exit_cost if getattr(r, frame_key) > min_t else 0
                for r in detections.itertuples()
            ],
            "distance": -1,
        }
        # add disappearance edges to all vertices
        exit_edges = {
            "u": detections.index.values,
            "v": VirtualVertices.TARGET.value,
            "capacity": self.MERGE_EDGE_CAPACITY,
            "cost": [
                r.enter_exit_cost if getattr(r, frame_key) < max_t else 0
                for r in detections.itertuples()
            ],
            "distance": -1,
        }
        # add div edges to all vertices (not in final frame)
        divisible_detections = detections[detections[frame_key] < max_t]
        div_edges = {
            "u": VirtualVertices.DIV.value,
            "v": divisible_detections.index.values,
            "capacity": self.DIVISION_EDGE_CAPACITY,
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
