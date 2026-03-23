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

from ._costs import (
    closest_neighbour_child_cost,
    closest_neighbour_child_cost_single,
    dist_to_edge_cost_func,
    dist_to_edge_cost_single,
)


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

    def as_nx_digraph(self, include_all_attrs: bool = False):
        edges = self.tracked_edges
        nodes = self.tracked_detections
        if include_all_attrs:
            edge_attrs = [c for c in edges.columns if c not in ("u", "v")]
        else:
            edge_attrs = [
                "flow"
            ]  # optional but may be useful for debugging or finding merge edges
        sol_graph = nx.from_pandas_edgelist(
            edges,
            "u",
            "v",
            edge_attrs,
            create_using=nx.DiGraph,
        )
        if include_all_attrs:
            node_attrs = nodes.to_dict(orient="index")
        else:
            det_keys = [self.frame_key] + self.location_keys
            if self.value_key in self.tracked_detections.columns:
                det_keys += [self.value_key]
            node_attrs = nodes[det_keys].to_dict(orient="index")
        sol_graph.add_nodes_from(node_attrs.items())
        return sol_graph

    def as_candidate_nx_digraph(self, include_all_attrs: bool = False):
        """Build a networkx DiGraph from the full candidate graph.

        Only available when the Tracker was run with DEBUG_MODE=True.

        Parameters
        ----------
        include_all_attrs : bool, optional
            If True, all columns from all_edges and all_vertices are stored as
            edge and node properties. If False (default), only flow is stored
            on edges and frame_key + location_keys on nodes.

        Raises
        ------
        ValueError
            If all_edges or all_vertices is None (not produced in DEBUG_MODE).
        """
        if self.all_edges is None or self.all_vertices is None:
            raise ValueError(
                "Candidate graph is only available when the Tracker was run "
                "with DEBUG_MODE=True. Re-solve with tracker.DEBUG_MODE = True."
            )
        # 'index' is an artifact of reset_index() in _get_all_edges, not a
        # meaningful graph attribute
        _skip = {"u", "v", "index"}
        if include_all_attrs:
            edge_attrs = [c for c in self.all_edges.columns if c not in _skip]
        else:
            edge_attrs = ["flow"]
        cand_graph = nx.from_pandas_edgelist(
            self.all_edges,
            "u",
            "v",
            edge_attrs,
            create_using=nx.DiGraph,
        )
        if include_all_attrs:
            node_attrs = self.all_vertices.to_dict(orient="index")
        else:
            det_keys = [self.frame_key] + list(self.location_keys)
            node_attrs = self.all_vertices[det_keys].to_dict(orient="index")
        cand_graph.add_nodes_from(node_attrs.items())
        return cand_graph

    def assign_features(self):
        """Compute and assign all available edge features to all_edges in-place.

        Assigns three groups of features:

        - **Probability features** (always): softmax, softmax_entropy,
          parental_softmax.
        - **Migration features** (always): distance, chosen_neighbour_rank.
        - **Sensitivity features** (requires ``model`` from DEBUG_MODE):
          sa_obj_low, sa_obj_up, sensitivity_diff.  A warning is printed if
          the model is absent and these features are skipped.

        Returns
        -------
        list[str]
            Names of the columns added to ``all_edges``.

        Raises
        ------
        ValueError
            If ``all_edges`` is None (Tracker was not run in DEBUG_MODE).
        """
        import warnings

        from tracktour._features import (
            assign_migration_features,
            assign_probability_features,
            assign_sensitivity_features,
        )

        if self.all_edges is None:
            raise ValueError(
                "all_edges is only available when the Tracker was run with "
                "DEBUG_MODE=True. Re-solve with tracker.DEBUG_MODE = True."
            )

        cols = []
        cols += assign_probability_features(self.all_edges)
        cols += assign_migration_features(self.all_edges)

        if self.model is not None:
            cols += assign_sensitivity_features(self.all_edges, self.model)
        else:
            warnings.warn(
                "Sensitivity features require a live Gurobi model (DEBUG_MODE). "
                "Skipping sa_obj_low, sa_obj_up, sensitivity_diff.",
                UserWarning,
                stacklevel=2,
            )

        self._propagate_features_to_tracked_edges(cols)
        return cols

    def _propagate_features_to_tracked_edges(self, feature_cols):
        """Copy feature values from all_edges solution rows into tracked_edges.

        Matches rows by (u, v). Only copies columns not already present in
        tracked_edges. Mutates tracked_edges in-place.
        """
        import pandas as pd

        new_cols = [c for c in feature_cols if c not in self.tracked_edges.columns]
        if not new_cols:
            return

        sol_in_all = self.all_edges[
            (self.all_edges.u >= 0)
            & (self.all_edges.v >= 0)
            & (self.all_edges.flow > 0)
        ].set_index(["u", "v"])[new_cols]

        uv_index = pd.MultiIndex.from_arrays(
            [self.tracked_edges.u, self.tracked_edges.v]
        )
        for col in new_cols:
            self.tracked_edges[col] = sol_in_all[col].reindex(uv_index).values

    def write_solution_geff(self, path, overwrite=False):
        """Write the solution graph to a GEFF file. See _geff_io.write_solution_geff."""
        from tracktour._geff_io import write_solution_geff

        write_solution_geff(self, path, overwrite=overwrite)

    def write_candidate_geff(self, path, overwrite=False):
        """Write the candidate graph to a GEFF file. See _geff_io.write_candidate_geff."""
        from tracktour._geff_io import write_candidate_geff

        write_candidate_geff(self, path, overwrite=overwrite)


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

        self._kd_dict = None
        self._edge_df = None
        self._model = None
        self._model_flow_vars = None
        self._all_edges = None
        self._all_vertices = None

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
        self._kd_dict = self._build_trees(
            detections, frame_key, self._scaled_location_keys
        )

        # get candidate edges
        self._edge_df = self._get_candidate_edges(detections, frame_key, self._kd_dict)

        if costs == "distance":
            # migration cost (on edges) is just euclidean distance
            self._edge_df["cost"] = self._edge_df["distance"]

            # compute costs for division and appearance/exit - on vertices
            enter_exit_cost, div_cost = self._compute_detection_costs(
                detections, self._scaled_location_keys, self._edge_df
            )
            detections["enter_cost"] = enter_exit_cost
            detections["exit_cost"] = enter_exit_cost
            detections["div_cost"] = div_cost
        else:
            # migration cost on edges is 1-IOU of two detected objects
            overlaps = self._compute_edge_overlaps(detections, self._edge_df)
            self._edge_df["iou_overlap"] = overlaps
            self._edge_df["cost"] = 1 - self._edge_df["iou_overlap"]

            # division cost is 1 - IOU sum of all overlapping children
            div_cost = self._compute_overlap_division_cost(detections, self._edge_df)
            detections["div_cost"] = div_cost

            # appearance is infinite
            detections["enter_cost"] = math.inf

            # disappearance is proportional to the area of the object
            exit_cost = self._compute_area_exit_cost(detections)
            detections["exit_cost"] = exit_cost

        # build model
        model, all_edges, all_vertices, gb_time = self._to_gurobi_model(
            detections, self._edge_df, frame_key, self._scaled_location_keys
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

    def _fix_virtual_edge(self, all_edges, real_node_id, virtual_vertex, direction, lb):
        """Find and fix the lower bound on a virtual edge adjacent to a real node.

        Parameters
        ----------
        all_edges : pd.DataFrame
        real_node_id : int
        virtual_vertex : int
            One of the VirtualVertices int values (APP, TARGET, DIV, etc.)
        direction : str
            ``"in"`` if the edge runs virtual_vertex → real_node_id,
            ``"out"`` if it runs real_node_id → virtual_vertex.
        lb : float
        """
        if direction == "in":
            u, v = virtual_vertex, real_node_id
        else:
            u, v = real_node_id, virtual_vertex
        matches = all_edges[(all_edges.u == u) & (all_edges.v == v)]
        if not matches.empty:
            self.fix_edge_in_model(matches.index[0], u, v, lb=lb)

    def warm_start_from_solution_graph(
        self,
        nxg: "nx.DiGraph",
        frame_key: str = "t",
        location_keys: Tuple[str, ...] = ("y", "x"),
        value_key: Optional[str] = "label",
        scale: Optional[Tuple[float, ...]] = None,
    ) -> "Tracked":
        """Build a live Gurobi model from a solution graph without re-solving.

        Constructs a minimal candidate graph using only the solution edges,
        then fixes lb=1 on every edge that is active in the solution:

        - Migration edges (real node → real node present in nxg)
        - Appearance edges for nodes with no incoming migration (in_degree == 0)
        - Exit edges for nodes with no outgoing migration (out_degree == 0)
        - Division edges for nodes with more than one outgoing migration

        After fixing bounds, optimizes once (trivially fast) to populate flow
        values, then returns a ``Tracked`` object with ``all_edges`` populated
        so that ``fix_edge_in_model`` / ``_model.optimize()`` can be used for
        partial re-solves.

        Parameters
        ----------
        nxg : nx.DiGraph
            Solution graph with node attributes at ``frame_key`` and
            ``location_keys``.
        frame_key : str
            Node attribute name for the time/frame index.
        location_keys : tuple of str
            Node attribute names for spatial coordinates.
        value_key : str, optional
            Node attribute name for the detection label value.
        scale : tuple of float, optional
            Spatial scale per dimension. Defaults to all-ones if not provided.

        Returns
        -------
        Tracked
        """
        if scale is not None:
            self.scale = scale

        self.frame_key = frame_key
        self.location_keys = list(location_keys)
        self.value_key = value_key

        # Build detections DataFrame from real nodes (node_id >= 0)
        real_nodes = {nid: attrs for nid, attrs in nxg.nodes(data=True) if nid >= 0}
        det_records = [
            {
                "Index": nid,
                frame_key: attrs[frame_key],
                **{k: attrs[k] for k in location_keys},
                "enter_cost": 1.0,
                "exit_cost": 1.0,
                "div_cost": 1.0,
            }
            for nid, attrs in real_nodes.items()
        ]
        detections = pd.DataFrame(det_records).set_index("Index").sort_values(frame_key)

        # Build edge_df from solution migration edges only
        sol_edges = [(u, v) for u, v in nxg.edges() if u >= 0 and v >= 0]
        if sol_edges:
            edge_df = pd.DataFrame(
                {
                    "u": [e[0] for e in sol_edges],
                    "v": [e[1] for e in sol_edges],
                    "capacity": Tracker.MERGE_EDGE_CAPACITY,
                    "cost": [nxg.edges[u, v].get("cost", 1.0) for u, v in sol_edges],
                    "distance": -1,
                }
            )
        else:
            edge_df = pd.DataFrame(columns=["u", "v", "capacity", "cost", "distance"])

        # Build Gurobi model — sets _model, _all_edges, _model_flow_vars
        _, all_edges, all_vertices, _ = self._to_gurobi_model(
            detections, edge_df, frame_key, list(location_keys)
        )

        APP = VirtualVertices.APP.value
        TARGET = VirtualVertices.TARGET.value
        DIV = VirtualVertices.DIV.value

        # Migration edges present in the solution
        for u, v in nxg.edges():
            if u < 0 or v < 0:
                continue
            matches = all_edges[(all_edges.u == u) & (all_edges.v == v)]
            if not matches.empty:
                self.fix_edge_in_model(matches.index[0], u, v, lb=1)

        # Virtual edges — one pass through real nodes
        for node_id in real_nodes:
            if nxg.in_degree(node_id) == 0:
                self._fix_virtual_edge(all_edges, node_id, APP, "in", lb=1)
            if nxg.out_degree(node_id) == 0:
                self._fix_virtual_edge(all_edges, node_id, TARGET, "out", lb=1)
            if nxg.out_degree(node_id) > 1:
                self._fix_virtual_edge(all_edges, node_id, DIV, "in", lb=1)

        self._model.optimize()
        if self._model.status != 2:
            raise RuntimeError(
                "Warm-start optimization failed — model may be infeasible."
            )
        self._store_solution(self._model, all_edges)

        migration_edges = all_edges[
            (all_edges.u >= 0) & (all_edges.v >= 0) & (all_edges.flow > 0)
        ].copy()
        return Tracked(
            tracked_edges=migration_edges,
            tracked_detections=detections,
            frame_key=frame_key,
            location_keys=list(location_keys),
            value_key=value_key,
            all_edges=all_edges,
            all_vertices=all_vertices,
        )

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
        self,
        detections: pd.DataFrame,
        frame_key: str,
        location_keys: Tuple[str],
        frames=None,
        scale=None,
    ) -> Dict[int, KDTree]:
        """Build dictionary of KDTrees for each frame in detections.

        Parameters
        ----------
        detections : pd.DataFrame
            dataframe of real detections for generating tracks
        frame_key : str
            column in `detections` denoting the frame number or time index
        location_keys : Tuple[str, str, str]
            tuple of columns in `detections` denoting spatial coordinates
        frames : iterable, optional
            if provided, only build trees for these frames; otherwise build
            trees for all frames present in detections
        scale : array-like, optional
            if provided, multiply position values by scale before building each
            tree so that distances are in physical rather than pixel units

        Returns
        -------
        dict[int, KDTree]
            dictionary of KDTree objects for each frame in detections
        """
        if frames is None:
            sorted_ts = sorted(detections[frame_key].unique())
        else:
            sorted_ts = sorted(frames)
        kd_dict = {}
        for t in sorted_ts:
            frame_detections = detections[detections[frame_key] == t][
                list(location_keys)
            ]
            if not frame_detections.empty:
                values = frame_detections.values
                if scale is not None:
                    values = values * np.asarray(scale)
                kd_dict[t] = KDTree(values)

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
        self._model = m
        self._all_edges = all_edges
        self._all_vertices = full_det

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
            lb_col = np.zeros(len(all_edges))
        if ub_col is None:
            ub_col = all_edges.capacity.values

        edge_keys = []
        edge_costs = {}
        for e in all_edges.itertuples():
            key = (e.Index, Tracker.index_to_label(e.u), Tracker.index_to_label(e.v))
            edge_keys.append(key)
            edge_costs[key] = getattr(e, cost_col_name)

        m = gp.Model("tracks")

        flow = m.addVars(
            edge_keys,
            obj=edge_costs,
            lb=lb_col,
            ub=ub_col,
            name="flow",
        )
        self._model_flow_vars = flow

        src_label = Tracker.index_to_label(VirtualVertices.SOURCE.value)
        app_label = Tracker.index_to_label(VirtualVertices.APP.value)
        div_label = Tracker.index_to_label(VirtualVertices.DIV.value)
        target_label = Tracker.index_to_label(VirtualVertices.TARGET.value)

        # If needed, build penalty vars
        penalty_flow = {}
        if self.PENALIZE_FLOW:
            warnings.warn(
                "Penalizing flow! This is not the default behavior and may lead to unexpected results."
            )
            penalty_keys = []
            penalty_costs = {}
            for e in all_edges[(all_edges.u >= 0) & (all_edges.v >= 0)].itertuples():
                key = (
                    e.Index,
                    Tracker.index_to_label(e.u),
                    Tracker.index_to_label(e.v),
                )
                penalty_keys.append(key)
                penalty_costs[key] = self.FLOW_PENALTY_COEFFICIENT * getattr(
                    e, cost_col_name
                )
            penalty_flow = m.addVars(
                penalty_keys, obj=penalty_costs, lb=0, name="penalty"
            )

        out_edges = defaultdict(list)
        in_edges = defaultdict(list)
        div_edges = defaultdict(list)
        for k in edge_keys:
            _, u, v = k
            out_edges[u].append(flow[k])
            in_edges[v].append(flow[k])
            if u == div_label:
                div_edges[v].append(flow[k])

        src_label = Tracker.index_to_label(VirtualVertices.SOURCE.value)
        app_label = Tracker.index_to_label(VirtualVertices.APP.value)
        div_label = Tracker.index_to_label(VirtualVertices.DIV.value)
        target_label = Tracker.index_to_label(VirtualVertices.TARGET.value)

        # whole network flow
        m.addConstr(
            sum(out_edges[src_label]) == sum(in_edges[target_label]), "conserv_network"
        )

        # dummy vertex flow conservation
        m.addConstr(
            sum(in_edges[app_label]) == sum(out_edges[app_label]), "conserv_app"
        )
        m.addConstr(
            sum(in_edges[div_label]) == sum(out_edges[div_label]), "conserv_div"
        )

        # constraints for real detections
        for vertex_id in detections.index:
            lbl = Tracker.index_to_label(vertex_id)
            incoming = in_edges.get(lbl, [])
            outgoing = out_edges.get(lbl, [])
            div_incoming = div_edges.get(lbl, [])
            # Pass vars directly instead of using .select
            self._add_constraints_for_vertex(lbl, m, incoming, outgoing, div_incoming)

        # penalty constraints
        if self.PENALIZE_FLOW:
            m.addConstrs(
                (penalty_flow[k] >= flow[k] - 1 for k in penalty_flow.keys()),
                name="penalty",
            )

        return m

    def _add_constraints_for_vertex(self, v, model, incoming, outgoing, div_incoming):
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

    def fix_edge_in_model(self, edge_index, u, v, lb=None, ub=None):
        if self._model is None or self._all_edges is None:
            raise ValueError("No existing model. Cannot fix edge.")
        key = (edge_index, Tracker.index_to_label(u), Tracker.index_to_label(v))
        var = self._model_flow_vars[key]
        if lb is not None:
            var.LB = lb
        if ub is not None:
            var.UB = ub

    def ensure_candidate_edge(
        self, all_edges: pd.DataFrame, u: int, v: int, cost: float = 1.0
    ) -> tuple:
        """Return (updated all_edges, edge_idx), adding (u, v) as a candidate if absent.

        New edges have lb=0, ub=capacity, flow=0.  Call ``fix_edge_in_model``
        afterwards to constrain specific edges.
        """
        matches = all_edges[(all_edges.u == u) & (all_edges.v == v)]
        if not matches.empty:
            return all_edges, int(matches.index[0])

        APP = VirtualVertices.APP.value
        DIV = VirtualVertices.DIV.value

        if u == APP:
            capacity = Tracker.APPEARANCE_EDGE_CAPACITY
        elif u == DIV:
            capacity = Tracker.DIVISION_EDGE_CAPACITY
        else:
            capacity = Tracker.MERGE_EDGE_CAPACITY

        new_idx = int(all_edges.index.max()) + 1 if not all_edges.empty else 0
        new_row = pd.DataFrame(
            {
                "u": [u],
                "v": [v],
                "capacity": [capacity],
                "cost": [cost],
                "distance": [-1],
                "flow": [0.0],
            },
            index=[new_idx],
        )
        all_edges = pd.concat([all_edges, new_row])

        key = (new_idx, Tracker.index_to_label(u), Tracker.index_to_label(v))
        var = self._model.addVar(
            obj=cost,
            lb=0.0,
            ub=float(capacity),
            name=f"flow[{key[0]},{key[1]},{key[2]}]",
        )
        self._model.update()
        self._model_flow_vars[key] = var

        return all_edges, new_idx

    def find_knn_edges_for_node(
        self,
        tracked_detections: pd.DataFrame,
        node_id: int,
        frame: int,
        pos: tuple,
        k: int = 10,
    ) -> list:
        """Return (u, v, dist) candidate edges to k nearest neighbours in adjacent frames.

        Uses and lazily populates ``_kd_dict`` keyed by frame, with ``self.scale``
        applied to coordinates before querying.
        """
        if self._kd_dict is None:
            self._kd_dict = {}
        scale = (
            np.array(self.scale)
            if self.scale is not None
            else np.ones(len(self.location_keys))
        )
        scaled_pos = np.array(pos) * scale
        # Use pre-scaled columns if they exist (regular solve path), otherwise
        # build with explicit scale applied to unscaled location_keys values.
        lazy_loc_keys = (
            getattr(self, "_scaled_location_keys", None) or self.location_keys
        )
        lazy_scale = None if getattr(self, "_scaled_location_keys", None) else scale
        edges = []
        for delta in (-1, 1):
            neighbour_frame = frame + delta
            if neighbour_frame not in self._kd_dict:
                self._kd_dict.update(
                    self._build_trees(
                        tracked_detections,
                        self.frame_key,
                        lazy_loc_keys,
                        frames=[neighbour_frame],
                        scale=lazy_scale,
                    )
                )
            if neighbour_frame not in self._kd_dict:
                continue
            tree = self._kd_dict[neighbour_frame]
            frame_dets = tracked_detections[
                tracked_detections[self.frame_key] == neighbour_frame
            ]
            k_actual = min(k, tree.n)
            dists, idxs = tree.query(scaled_pos, k=k_actual if k_actual > 1 else [1])
            dists = np.atleast_1d(dists)
            idxs = np.atleast_1d(idxs)
            for dist, idx in zip(dists, idxs):
                neighbour_id = int(frame_dets.iloc[idx].name)
                if delta == -1:
                    edges.append((neighbour_id, node_id, float(dist)))
                else:
                    edges.append((node_id, neighbour_id, float(dist)))
        return edges

    def update_node_constraints(self, node_id: int):
        """Remove and re-add flow conservation constraints for an existing node.

        Must be called after new edges involving ``node_id`` are added via
        ``ensure_candidate_edge`` so that the Gurobi solver can route flow
        through those new edges.
        """
        lbl = Tracker.index_to_label(node_id)
        for prefix in ("conserv", "demand", "div"):
            constr = self._model.getConstrByName(f"{prefix}_{lbl}")
            if constr is not None:
                self._model.remove(constr)
        self._model.update()

        flow_vars = self._model_flow_vars
        div_lbl = Tracker.index_to_label(VirtualVertices.DIV.value)
        in_vars = [var for key, var in flow_vars.items() if key[2] == lbl]
        out_vars = [var for key, var in flow_vars.items() if key[1] == lbl]
        div_vars = [
            var for key, var in flow_vars.items() if key[2] == lbl and key[1] == div_lbl
        ]
        self._add_constraints_for_vertex(lbl, self._model, in_vars, out_vars, div_vars)
        self._model.update()

    def _set_virtual_edge_costs(self, node_id: int, enter_cost: float) -> None:
        """Set APP→node and node→TARGET Gurobi var.Obj to *enter_cost*.

        ``ensure_candidate_edge`` sets obj only when the var is newly created.
        Call this explicitly to update pre-existing virtual edges (e.g. after
        moving a node) using an enter_cost already computed by the caller.
        """
        app_lbl = Tracker.index_to_label(VirtualVertices.APP.value)
        target_lbl = Tracker.index_to_label(VirtualVertices.TARGET.value)
        node_lbl = Tracker.index_to_label(node_id)
        for key, var in self._model_flow_vars.items():
            if (key[1] == app_lbl and key[2] == node_lbl) or (
                key[1] == node_lbl and key[2] == target_lbl
            ):
                var.Obj = enter_cost
        self._model.update()

    def _prepare_move_node(
        self, tracked: "Tracked", node_id: int, new_frame: int, new_pos: tuple
    ) -> None:
        """Phase 1 of a node move: reset Gurobi edge bounds and update position.

        Disables old migration edges (ub=0) and restores virtual edge capacities,
        then updates the node's position in ``tracked_detections``.  Does not
        rebuild KD-trees or recompute edges — call ``rebuild_kd_trees`` once for
        all affected frames after all ``_prepare_*`` calls, then
        ``_apply_node_edges``.
        """
        APP = VirtualVertices.APP.value
        TARGET = VirtualVertices.TARGET.value
        DIV = VirtualVertices.DIV.value
        app_lbl = Tracker.index_to_label(APP)
        target_lbl = Tracker.index_to_label(TARGET)
        div_lbl = Tracker.index_to_label(DIV)
        node_lbl = Tracker.index_to_label(node_id)

        for key, var in self._model_flow_vars.items():
            u_lbl, v_lbl = key[1], key[2]
            if u_lbl != node_lbl and v_lbl != node_lbl:
                continue
            var.LB = 0.0
            if not isinstance(u_lbl, str) and not isinstance(v_lbl, str):
                var.UB = 0.0  # disable old migration edge
            elif u_lbl == app_lbl and v_lbl == node_lbl:
                var.UB = float(Tracker.APPEARANCE_EDGE_CAPACITY)
            elif u_lbl == node_lbl and v_lbl == target_lbl:
                var.UB = float(Tracker.MERGE_EDGE_CAPACITY)
            elif u_lbl == div_lbl and v_lbl == node_lbl:
                var.UB = float(Tracker.DIVISION_EDGE_CAPACITY)

        if (
            tracked.tracked_detections is not None
            and node_id in tracked.tracked_detections.index
        ):
            tracked.tracked_detections.at[node_id, self.frame_key] = new_frame
            for k, val in zip(self.location_keys, new_pos):
                tracked.tracked_detections.at[node_id, k] = val
        if tracked.all_vertices is not None and node_id in tracked.all_vertices.index:
            tracked.all_vertices.at[node_id, self.frame_key] = new_frame
            for k, val in zip(self.location_keys, new_pos):
                tracked.all_vertices.at[node_id, k] = val

    def _prepare_add_node(
        self, tracked: "Tracked", node_id: int, frame: int, pos: tuple
    ) -> None:
        """Phase 1 of adding a node: append the detection row to tracked state.

        Uses ``enter_cost`` as a placeholder for ``div_cost`` (recomputed in
        ``_apply_node_edges``).  Does not rebuild KD-trees or add Gurobi edges.
        Call ``rebuild_kd_trees`` once after all ``_prepare_*`` calls, then
        ``_apply_node_edges``.
        """
        scale = np.array(self.scale)
        src_scaled = np.array(pos) * scale
        enter_cost = float(dist_to_edge_cost_single(self._im_shape, src_scaled))

        new_det = pd.DataFrame(
            [
                {
                    self.frame_key: frame,
                    **dict(zip(self.location_keys, pos)),
                    "enter_cost": enter_cost,
                    "exit_cost": enter_cost,
                    "div_cost": enter_cost,  # placeholder; updated in _apply_node_edges
                }
            ],
            index=[node_id],
        )
        tracked.tracked_detections = pd.concat([tracked.tracked_detections, new_det])
        if tracked.all_vertices is not None:
            all_cols = {col: -1 for col in tracked.all_vertices.columns}
            all_cols.update(
                {
                    self.frame_key: frame,
                    **dict(zip(self.location_keys, pos)),
                    "enter_cost": enter_cost,
                    "exit_cost": enter_cost,
                    "div_cost": enter_cost,
                }
            )
            tracked.all_vertices = pd.concat(
                [tracked.all_vertices, pd.DataFrame([all_cols], index=[node_id])]
            )

    def rebuild_kd_trees(self, tracked: "Tracked", frames: list) -> None:
        """Rebuild KD-trees for *frames* from current ``tracked_detections``.

        Call this once after all ``_prepare_move_node`` / ``_prepare_add_node``
        invocations so that a single rebuild reflects every node's final position
        before the k-NN searches in ``_apply_node_edges`` run.
        """
        if self._kd_dict is None:
            self._kd_dict = {}
        lazy_loc_keys = (
            getattr(self, "_scaled_location_keys", None) or self.location_keys
        )
        lazy_scale = (
            None
            if getattr(self, "_scaled_location_keys", None)
            else np.array(self.scale)
        )
        self._kd_dict.update(
            self._build_trees(
                tracked.tracked_detections,
                self.frame_key,
                lazy_loc_keys,
                frames=frames,
                scale=lazy_scale,
            )
        )

    def _apply_node_edges(
        self, tracked: "Tracked", node_id: int, frame: int, pos: tuple
    ) -> None:
        """Phase 2 (shared by moves and adds): recompute costs, edges, constraints.

        Must be called after ``rebuild_kd_trees`` so that the k-NN search sees
        every node at its final position.  For moved nodes the virtual edges
        (APP/TARGET/DIV) already exist in the model and are re-enabled via
        ``ensure_candidate_edge``; their objective is updated by
        ``_set_virtual_edge_costs``.
        """
        APP = VirtualVertices.APP.value
        TARGET = VirtualVertices.TARGET.value
        DIV = VirtualVertices.DIV.value

        scale = np.array(self.scale)
        src_scaled = np.array(pos) * scale
        enter_cost = max(
            float(dist_to_edge_cost_single(self._im_shape, src_scaled)), 1.0
        )

        knn_edges = self.find_knn_edges_for_node(
            tracked.tracked_detections, node_id, frame, pos
        )

        knn_out = [(u, v, dist) for (u, v, dist) in knn_edges if u == node_id]
        child_scaled = [
            tracked.tracked_detections.loc[v][self.location_keys].values * scale
            for _, v, _ in knn_out
        ]
        div_cost = closest_neighbour_child_cost_single(src_scaled, child_scaled)
        if not math.isfinite(div_cost):
            div_cost = 1e9
        tracked.tracked_detections.at[node_id, "div_cost"] = div_cost

        tracked.all_edges, _ = self.ensure_candidate_edge(
            tracked.all_edges, APP, node_id, cost=enter_cost
        )
        tracked.all_edges, _ = self.ensure_candidate_edge(
            tracked.all_edges, node_id, TARGET, cost=enter_cost
        )
        tracked.all_edges, _ = self.ensure_candidate_edge(
            tracked.all_edges, DIV, node_id, cost=div_cost
        )
        self._set_virtual_edge_costs(node_id, enter_cost)

        neighbor_ids = set()
        for u, v, dist in knn_edges:
            matches = tracked.all_edges[
                (tracked.all_edges.u == u) & (tracked.all_edges.v == v)
            ]
            if not matches.empty:
                edge_idx = int(matches.index[0])
                ekey = (edge_idx, Tracker.index_to_label(u), Tracker.index_to_label(v))
                if ekey in self._model_flow_vars:
                    evar = self._model_flow_vars[ekey]
                    evar.LB = 0.0
                    evar.UB = float(Tracker.MERGE_EDGE_CAPACITY)
                    evar.Obj = dist
                tracked.all_edges.at[edge_idx, "cost"] = dist
            else:
                tracked.all_edges, _ = self.ensure_candidate_edge(
                    tracked.all_edges, u, v, cost=dist
                )
            if u >= 0 and u != node_id:
                neighbor_ids.add(u)
            if v >= 0 and v != node_id:
                neighbor_ids.add(v)
        self._model.update()

        self.update_node_constraints(node_id)
        for nbr_id in neighbor_ids:
            self.update_node_constraints(nbr_id)

    def move_node_in_model(
        self, tracked: "Tracked", node_id: int, new_frame: int, new_pos: tuple
    ) -> None:
        """Update an existing node's position in the live Gurobi model.

        Single-node convenience wrapper around ``_prepare_move_node``,
        ``rebuild_kd_trees``, and ``_apply_node_edges``.  When moving multiple
        nodes, call ``_prepare_move_node`` for each, then ``rebuild_kd_trees``
        once for the union of affected frames, then ``_apply_node_edges`` for
        each — so that every k-NN search sees all nodes at their final positions.
        """
        self._prepare_move_node(tracked, node_id, new_frame, new_pos)
        self.rebuild_kd_trees(tracked, [new_frame])
        self._apply_node_edges(tracked, node_id, new_frame, new_pos)

    def add_node_to_model(
        self, tracked: "Tracked", node_id: int, frame: int, pos: tuple
    ):
        """Add a new detection to tracked state and the live Gurobi model.

        Single-node convenience wrapper around ``_prepare_add_node``,
        ``rebuild_kd_trees``, and ``_apply_node_edges``.  When adding multiple
        nodes, call ``_prepare_add_node`` for each, then ``rebuild_kd_trees``
        once for the union of affected frames, then ``_apply_node_edges`` for
        each — so that every k-NN search sees all nodes at their final positions.
        """
        self._prepare_add_node(tracked, node_id, frame, pos)
        self.rebuild_kd_trees(tracked, [frame])
        self._apply_node_edges(tracked, node_id, frame, pos)
