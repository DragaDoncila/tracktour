import math
import time
import warnings
from collections import defaultdict
from enum import Enum, StrEnum, auto
from typing import Dict, List, Literal, Optional, Tuple

import gurobipy as gp
import networkx as nx
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field
from scipy.spatial import KDTree
from tqdm import tqdm

from ._costs import closest_neighbour_child_cost, dist_to_edge_cost_func


class Cost(StrEnum):
    DISTANCE = auto()
    OVERLAP = auto()


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
        det_keys = [self.frame_key] + self.location_keys
        if self.value_key in self.tracked_detections.columns:
            det_keys += [self.value_key]
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
    FLOW_PENALTY_COEFFICIENT = 50

    USE_DIV_CONSTRAINT = False
    PENALIZE_FLOW = False
    ALLOW_MERGES = True

    DEBUG_MODE = False

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

    def __init__(
        self,
        im_shape: Optional[Tuple[int, int]] = None,
        seg: Optional[np.ndarray] = None,
        scale: Tuple[float, float] = (1.0, 1.0),
    ) -> None:
        """Create a Tracker instance for a given segmentation.

        Parameters
        ----------
        im_shape : Optional[Tuple[int, int]], optional
            shape of a single frame of the segmentation in pixels, by default None
        seg : Optional[np.ndarray], optional
            segmentation array, by default None
        scale : Tuple[float, float], optional
            scale of a single frame of the segmentation in pixels, by default (1.0, 1.0)
        """
        if im_shape is None and seg is None:
            raise ValueError("Either im_shape or seg must be provided.")
        if seg is not None:
            if im_shape is not None and im_shape != seg.shape[1:]:
                warnings.warn(
                    "Provided im_shape does not match seg shape. Using seg shape.",
                    UserWarning,
                )
            im_shape = seg.shape[1:]
        self._seg = seg
        self._im_shape = im_shape
        # will scale im_shape
        self.scale = scale

        self.k_neighbours = 10
        self.location_keys = None
        self.frame_key = None
        self.value_key = None

    @property
    def scale(self):
        return self._scale

    @scale.setter
    def scale(self, scale):
        self._scale = scale
        self._im_shape = tuple(
            self._im_shape[i] * self._scale[i] for i in range(len(self._im_shape))
        )

    @property
    def im_shape(self):
        return self._im_shape

    @im_shape.setter
    def im_shape(self, im_shape):
        self._im_shape = im_shape
        if self._scale != (1, 1):
            warnings.warn(
                "Setting im_shape will reset scale to (1, 1). Please set scale again if needed."
            )
            self._scale = (1, 1)

    @property
    def seg(self):
        return self._seg

    @seg.setter
    def seg(self, seg):
        self._seg = seg
        self._im_shape = seg.shape[1:]

    def solve(
        self,
        detections: pd.DataFrame = None,
        frame_key: str = "t",
        location_keys: Tuple[str, str] = ("y", "x"),
        # TODO: we should be able to make this optional
        value_key: Optional[str] = "label",
        k_neighbours: int = 10,
        # TODO: split these into migration, division, appearance, exit choices
        costs: Cost = Cost.DISTANCE,
        # TODO: ?
        # migration_only: bool = False,
    ):
        """_summary_

        Parameters
        ----------
        detections : pd.DataFrame, optional
            dataframe where each row is a detection, with coordinates at location_keys and time at frame_key. Index must be sequential integers from 0. Detections
            must be sorted by frame.
        frame_key : str, optional
            dataframe column denoting the image frame, by default "t"
        location_keys : Tuple[str], optional
            dataframe columns denoting the pixel coordinates, by default ("y", "x")
        value_key : Optional[str], optional
            dataframe column denoting the value of the pixel at the given coordinates, by default None
        k_neighbours : int, optional
            number of nearest neighbours to consider for migration, by default 10
        costs: str, optional
            use 'distance' for distance-based costs or 'overlap' for overlap-based costs

        Returns
        -------
        Tracked
            tracked object with tracked_detections and tracked_edges dataframes
        """
        if costs not in ["distance", "overlap"]:
            raise ValueError("Costs must be either 'distance' or 'overlap'.")
        if len(self._im_shape) == 3 and len(location_keys) == 2:
            warnings.warn(
                f"Segmentation frame is 3D but location keys are {location_keys}. Using ('z', 'y', 'x') as location keys."
            )
            location_keys = ("z", "y", "x")

        self.location_keys = location_keys
        self.frame_key = frame_key
        self.value_key = value_key
        self.k_neighbours = k_neighbours

        # scale detections (keeping original columns)
        self._scaled_location_keys = [f"{key}_scaled" for key in location_keys]
        for i in range(len(self.location_keys)):
            detections[self._scaled_location_keys[i]] = (
                detections[self.location_keys[i]] * self.scale[i]
            )

        # TODO: copy/validate detections
        start = time.time()

        # build kd-trees
        # TODO: stop passing stuff around now that you store it as an attribute
        kd_dict = self._build_trees(detections, frame_key, self._scaled_location_keys)

        # get candidate edges
        edge_df = self._get_candidate_edges(detections, frame_key, kd_dict)

        if costs == "distance":
            # migration cost (on edges) is just euclidean distance
            edge_df["cost"] = edge_df["distance"]

            # compute costs for division and appearance/exit - on vertices
            enter_exit_cost, div_cost = self._compute_detection_costs(
                detections, self._scaled_location_keys, edge_df
            )
            detections["enter_cost"] = enter_exit_cost
            detections["exit_cost"] = enter_exit_cost
            detections["div_cost"] = div_cost
        else:
            # migration cost on edges is 1-IOU of two detected objects
            overlaps = self._compute_edge_overlaps(detections, edge_df)
            edge_df["iou_overlap"] = overlaps
            edge_df["cost"] = 1 - edge_df["iou_overlap"]

            # division cost is 1 - IOU sum of all overlapping children
            div_cost = self._compute_overlap_division_cost(detections, edge_df)
            detections["div_cost"] = div_cost

            # appearance is infinite
            detections["enter_cost"] = math.inf

            # disappearance is proportional to the area of the object
            exit_cost = self._compute_area_exit_cost(detections)
            detections["exit_cost"] = exit_cost

        # build model
        model, all_edges, all_vertices, gb_time = self._to_gurobi_model(
            detections, edge_df, frame_key, self._scaled_location_keys
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

    def get_lb_ub_for_edge(self, edge):
        # we lower bound this edge by 1 - edge is part of the solution
        if edge["oracle_is_correct"] == 1:
            current_flow = edge["flow"]
            if current_flow > 0:
                return 1, edge["capacity"]
            # this edge has 0 current flow or haven't solved, set it to 1
            return 1, 1
        # we fix this edge to 0 - edge is not part of the solution
        elif edge["oracle_is_correct"] == 0:
            return 0, 0
        # oracle hasn't told us anything about this edge
        else:
            return 0, edge["capacity"]

    def solve_from_existing_edges(
        self,
        all_vertices,
        all_edges,
        frame_key: str = "t",
        location_keys: Tuple[str] = ("y", "x"),
        # TODO: we should be able to make this optional
        value_key: Optional[str] = "label",
        k_neighbours: int = 10,
        # TODO: ?
        # migration_only: bool = False,
    ):
        assert (
            "cost" in all_edges.columns
        ), "No cost column in edge dataframe. Try solving from scratch?"
        assert all(all_edges[all_edges.cost >= 0]), "Some costs are negative."
        assert (
            "learned_migration_cost" in all_edges.columns
        ), "No learned cost to use. Try classifying?"

        self.location_keys = location_keys
        self.frame_key = frame_key
        self.value_key = value_key
        self.k_neighbours = k_neighbours

        start = time.time()

        all_edges["model_cost"] = all_edges.apply(
            lambda row: row["learned_migration_cost"]
            if row["learned_migration_cost"] >= 0
            else row["cost"],
            axis=1,
        )
        all_edges[["model_lb", "model_ub"]] = all_edges.apply(
            self.get_lb_ub_for_edge, axis=1, result_type="expand"
        )
        detections = all_vertices[all_vertices.t >= 0]
        detections.index = detections.index.astype(int)

        model = self._make_gurobi_model_from_edges(
            all_edges,
            detections,
            "model_cost",
            all_edges["model_lb"].values,
            all_edges["model_ub"].values,
        )

        build_duration = time.time() - start

        # solve and store solution on edges
        model.optimize()
        if model.status != 2:
            print("Model infeasible. Terminating.")
            return None
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
                    "gp_model_time": -1,
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

        m = self._make_gurobi_model_from_edges(all_edges, detections)

        build_time = time.time() - start
        return m, all_edges, full_det, build_time

    def _make_gurobi_model_from_edges(
        self,
        all_edges,
        detections,
        cost_col_name: Optional[str] = None,
        lb_col: Optional[np.ndarray] = None,
        ub_col: Optional[np.ndarray] = None,
    ):
        if cost_col_name is None:
            cost_col_name = "cost"
        if lb_col is None:
            lb_col = [0 for _ in range(len(all_edges))]
        if ub_col is None:
            ub_col = all_edges.capacity.values

        flow_var_info = {
            (
                e.Index,
                Tracker.index_to_label(e.u),
                Tracker.index_to_label(e.v),
            ): getattr(e, cost_col_name)
            for e in all_edges.itertuples()
        }
        edge_capacities = all_edges.capacity.values
        flow_var_names = gp.tuplelist(list(flow_var_info.keys()))

        src_label = Tracker.index_to_label(VirtualVertices.SOURCE.value)
        app_label = Tracker.index_to_label(VirtualVertices.APP.value)
        div_label = Tracker.index_to_label(VirtualVertices.DIV.value)
        target_label = Tracker.index_to_label(VirtualVertices.TARGET.value)

        m = gp.Model("tracks")
        try:
            flow = m.addVars(
                flow_var_names, obj=flow_var_info, lb=lb_col, ub=ub_col, name="flow"
            )
        except gp.GurobiError as e:
            print(e)
        if self.PENALIZE_FLOW:
            warnings.warn(
                "Penalizing flow! This is not the default behavior and may lead to unexpected results."
            )
            penalty_var_info = {
                (
                    e.Index,
                    Tracker.index_to_label(e.u),
                    Tracker.index_to_label(e.v),
                ): self.FLOW_PENALTY_COEFFICIENT
                * getattr(e, cost_col_name)
                # only real edges can be penalized
                for e in all_edges[(all_edges.u >= 0) & (all_edges.v >= 0)].itertuples()
            }
            penalty_var_names = gp.tuplelist(list(penalty_var_info.keys()))
            penalty_flow = m.addVars(
                penalty_var_names, obj=penalty_var_info, lb=0, name="penalty"
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
            self._add_constraints_for_vertex(lbl, m, flow)

        if self.PENALIZE_FLOW:
            # for each penalty var, find its flow var and constrain them
            for var in penalty_flow:
                penalty_var = penalty_flow[var]
                flow_var = flow.select(var)[0]
                m.addConstr(penalty_var >= flow_var - 1, f"penalty_{var[1]}-{var[2]}")

        m.update()
        return m

    def _add_constraints_for_vertex(self, v, model, flow_vars):
        outgoing = flow_vars.select("*", v, "*")
        incoming = flow_vars.select("*", "*", v)
        div_incoming = flow_vars.select(
            "*", Tracker.index_to_label(VirtualVertices.DIV.value), v
        )
        if self.USE_DIV_CONSTRAINT:
            if not self.ALLOW_MERGES:
                raise NotImplementedError(
                    "Disallowing merges while using explicit division constraint is not yet implemented."
                )
            self._add_constraints_with_div(v, outgoing, incoming, div_incoming, model)
        else:
            self._add_constraints_without_div(
                v, outgoing, incoming, div_incoming, model
            )

    def _add_constraints_with_div(self, v, outgoing, incoming, div_incoming, model):
        # node-wise flow conservation
        model.addConstr(sum(incoming) == sum(outgoing), f"conserv_{v}")
        # minimal flow into each node
        model.addConstr(sum(incoming) >= Tracker.MINIMAL_VERTEX_DEMAND, f"demand_{v}")

        # TODO: div optional?
        if len(div_incoming):
            # division only occurs after appearance or migration
            model.addConstr(
                sum(incoming) - sum(div_incoming) >= sum(div_incoming), f"div_{v}"
            )

    def _add_constraints_without_div(self, v, outgoing, incoming, div_incoming, model):
        # node-wise flow conservation
        model.addConstr(sum(incoming) == sum(outgoing), f"conserv_{v}")
        # exclude incoming flow from division
        # TODO: div optional?
        if len(div_incoming):
            div_incoming = div_incoming[0]
            incoming = [var for var in incoming if not var.sameAs(div_incoming)]
        # minimal flow into each node
        if self.ALLOW_MERGES:
            model.addConstr(
                sum(incoming) >= Tracker.MINIMAL_VERTEX_DEMAND, f"demand_{v}"
            )
        else:
            model.addConstr(
                sum(incoming) == Tracker.MINIMAL_VERTEX_DEMAND, f"demand_{v}"
            )

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
            "enter_cost": 0,
            "exit_cost": 0,
            "div_cost": 0,
        }
        v_vertices_df = pd.DataFrame(v_vertices).set_index("Index")
        full_det = pd.concat([detections.copy(), v_vertices_df])
        return full_det

    def _compute_detection_costs(self, detections, location_keys, edge_df):
        # TODO: should verify there's no negatives here
        # if there are, we should raise that im_shape was probs wrong
        enter_exit_cost = dist_to_edge_cost_func(
            self._im_shape, detections, location_keys
        )
        div_cost = closest_neighbour_child_cost(detections, location_keys, edge_df)
        return enter_exit_cost, div_cost

    def _compute_edge_overlaps(self, detections, edge_df):
        from traccuracy.matchers._compute_overlap import get_labels_with_overlap

        if self.seg is None:
            raise ValueError(
                "Segmentation must be provided for overlap costs. You must initialize the tracker with a segmentation?"
            )
        overlap_dict = defaultdict(lambda: defaultdict(lambda: 0))
        for i in range(self.seg.shape[0] - 1):
            next_i = i + 1
            frame = self.seg[i]
            next_frame = self.seg[next_i]
            # (frame_label, next_frame_label, iou)
            overlaps = get_labels_with_overlap(frame, next_frame)
            for frame_label, next_frame_label, iou in overlaps:
                overlap_dict[i][(int(frame_label), int(next_frame_label))] = iou
        overlaps = [1 for _ in range(len(edge_df))]
        for i, row in enumerate(edge_df.itertuples()):
            src = detections.loc[row.u]
            dest = detections.loc[row.v]
            src_t = src.t
            iou = overlap_dict[src_t][(src.label, dest.label)]
            overlaps[i] = iou
        return overlaps

    def _compute_overlap_division_cost(self, detections, edge_df):
        div_costs = [1 for _ in range(len(detections))]
        for det_row in detections.itertuples():
            src_v = det_row.Index
            # find all potential children
            children = edge_df[edge_df.u == src_v]
            # get overlap of all children
            child_overlaps = children["iou_overlap"].sum()
            cost = 1 - child_overlaps
            div_costs[src_v] = cost
        return div_costs

    def _compute_area_exit_cost(self, detections):
        lower_quartile_area = detections.area.quantile(0.25)
        area_cost = detections.area / lower_quartile_area
        return area_cost

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
                r.enter_cost if getattr(r, frame_key) > min_t else 0
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
                r.exit_cost if getattr(r, frame_key) < max_t else 0
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
            if "flow" in v.VarName
        }
        edge_index, flow = zip(*[(k[0], v) for k, v in sol_dict.items()])
        all_edges.loc[list(edge_index), "flow"] = list(flow)
        return all_edges
