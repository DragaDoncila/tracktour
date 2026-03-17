"""Shared pytest fixtures for tracktour tests."""

import types

import numpy as np
import pandas as pd
import pytest

from tracktour import Tracker
from tracktour._tracker import Tracked, VirtualVertices

# ---------------------------------------------------------------------------
# Detection fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def get_detections(n_frames=10, n_cells_per_frame=10):
    def _get_detections():
        np.random.seed(0)
        total_size = n_frames * n_cells_per_frame
        detections = pd.DataFrame(
            {
                "t": np.repeat(np.arange(n_frames), n_cells_per_frame),
                "y": np.random.randint(1, 19, size=total_size),
                "x": np.random.randint(1, 39, size=total_size),
            }
        )
        return detections

    return _get_detections


@pytest.fixture
def human_detections():
    im_shape = (10, 10)
    detection_dict = {
        "t": [0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2],
        "y": [1, 3, 9, 1, 1, 3, 8, 2, 1, 2, 7],
        "x": [1, 4, 9, 2, 3, 5, 9, 2, 3, 4, 9],
    }
    detections = pd.DataFrame(detection_dict)
    return detections, im_shape


# ---------------------------------------------------------------------------
# Synthetic Tracked fixture (no Gurobi required)
#
# 3 frames, 3 real nodes per frame, simple chain solution:
#   0→3→6,  1→4→7,  2→5→8
# Virtual nodes: source=-1, appearance=-2, division=-3, target=-4
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_tracked():
    """A Tracked object built from synthetic DataFrames without requiring Gurobi.

    Suitable for testing GEFF I/O, feature assignment, and anything else that
    only needs a structurally valid Tracked — not a real solve.

    Layout: 3 frames × 3 nodes, solution is a simple chain (0→3→6 etc.).
    """
    frame_key = "t"
    location_keys = ("y", "x")

    # Real nodes: ids 0-8
    tracked_detections = pd.DataFrame(
        {
            "t": [0, 0, 0, 1, 1, 1, 2, 2, 2],
            "y": [1, 3, 7, 1, 3, 7, 1, 3, 7],
            "x": [1, 1, 1, 2, 2, 2, 3, 3, 3],
            "enter_cost": [0.5] * 9,
            "exit_cost": [0.5] * 9,
            "div_cost": [1.0] * 9,
            "label": list(range(1, 10)),
        }
    )

    # Solution edges: three chains
    tracked_edges = pd.DataFrame(
        {
            "u": [0, 3, 1, 4, 2, 5],
            "v": [3, 6, 4, 7, 5, 8],
            "flow": [1, 1, 1, 1, 1, 1],
            "cost": [0.1, 0.1, 0.2, 0.2, 0.3, 0.3],
            "distance": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        }
    )

    # Virtual nodes (placeholder coords -1)
    virtual_rows = pd.DataFrame(
        {
            "t": [-1, -1, -1, -1],
            "y": [-1, -1, -1, -1],
            "x": [-1, -1, -1, -1],
            "enter_cost": [0.0, 0.0, 0.0, 0.0],
            "exit_cost": [0.0, 0.0, 0.0, 0.0],
            "div_cost": [0.0, 0.0, 0.0, 0.0],
            "label": [-1, -1, -1, -1],
        },
        index=[
            VirtualVertices.SOURCE.value,
            VirtualVertices.APP.value,
            VirtualVertices.DIV.value,
            VirtualVertices.TARGET.value,
        ],
    )
    all_vertices = pd.concat([tracked_detections, virtual_rows])

    # Candidate edges: migration + virtual edges, with integer index column
    # matching the reset_index() shape produced by _get_all_edges
    migration_edges = pd.DataFrame(
        {
            "u": [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
            "v": [3, 4, 3, 4, 5, 4, 6, 7, 6, 7, 7, 8],
            "cost": [0.1, 0.5, 0.5, 0.2, 0.3, 0.9, 0.1, 0.5, 0.5, 0.2, 0.5, 0.3],
            "distance": [1.0, 2.0, 2.0, 1.0, 1.0, 3.0, 1.0, 2.0, 2.0, 1.0, 2.0, 1.0],
            "capacity": [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
            "flow": [1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1],
        }
    )
    s = VirtualVertices.SOURCE.value
    a = VirtualVertices.APP.value
    d = VirtualVertices.DIV.value
    t = VirtualVertices.TARGET.value
    virtual_edges = pd.DataFrame(
        {
            "u": [
                s,
                s,
                a,
                a,
                a,
                a,
                a,
                a,
                a,
                a,
                a,
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                d,
                d,
                d,
                d,
                d,
                d,
            ],
            "v": [
                a,
                d,
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                t,
                t,
                t,
                t,
                t,
                t,
                t,
                t,
                t,
                3,
                4,
                5,
                6,
                7,
                8,
            ],
            "cost": [0.0] * 26,
            "distance": [-1.0] * 26,
            "capacity": [np.inf] * 2 + [1] * 9 + [2] * 9 + [1] * 6,
            "flow": [0.0] * 26,
        }
    )
    all_edges = pd.concat([migration_edges, virtual_edges], ignore_index=True)
    all_edges = all_edges.reset_index()  # adds 'index' column matching _get_all_edges

    return Tracked(
        tracked_edges=tracked_edges,
        tracked_detections=tracked_detections,
        frame_key=frame_key,
        location_keys=list(location_keys),
        value_key="label",
        all_edges=all_edges,
        all_vertices=all_vertices,
    )


# ---------------------------------------------------------------------------
# Real-solve fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def debug_tracked(human_detections):
    """A Tracked object produced in DEBUG_MODE with all_edges, all_vertices, and model."""
    detections, im_shape = human_detections
    tracker = Tracker(im_shape=im_shape)
    tracker.DEBUG_MODE = True
    tracked = tracker.solve(
        detections, k_neighbours=2, frame_key="t", location_keys=("y", "x")
    )
    return tracked


# ---------------------------------------------------------------------------
# Mock Gurobi model
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_gurobi_model():
    """Factory fixture returning a mock Gurobi model for a given all_edges DataFrame.

    Each variable has:
      - varName: "flow[idx,src,dst]" matching the row index and u/v columns.
        Virtual vertex IDs are mapped to their Gurobi string labels
        (-1→s, -2→a, -3→d, -4→t) to match real Gurobi output.
      - X: 1.0 if flow > 0 else 0.0
      - SAObjLow: cost - 0.1
      - SAObjUp: cost + 0.1
    """
    _virtual_labels = {-1: "s", -2: "a", -3: "d", -4: "t"}

    def _node_label(n):
        return _virtual_labels.get(n, n)

    def _make_model(all_edges):
        vars_ = []
        for row in all_edges.itertuples():
            var = types.SimpleNamespace(
                varName=f"flow[{row.Index},{_node_label(row.u)},{_node_label(row.v)}]",
                X=float(row.flow) if hasattr(row, "flow") else 0.0,
                SAObjLow=row.cost - 0.1,
                SAObjUp=row.cost + 0.1,
            )
            vars_.append(var)

        model = types.SimpleNamespace(getVars=lambda: vars_)
        return model

    return _make_model
