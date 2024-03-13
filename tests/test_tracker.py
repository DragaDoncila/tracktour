import math
import re

import numpy as np
import pandas as pd
import pytest
from scipy.spatial import KDTree

from tracktour import Tracker
from tracktour._tracker import VirtualVertices


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


def test_build_trees(get_detections):
    detections = get_detections()
    tracker = Tracker(im_shape=(20, 40))
    kd_dict = tracker._build_trees(detections, "t", ("y", "x"))
    assert len(kd_dict) == 10
    for t in range(0, 10):
        assert t in kd_dict
        assert isinstance(kd_dict[t], KDTree)
    # only checking structure of first tree
    one_dict = kd_dict[0]
    assert one_dict.m == 2
    assert one_dict.n == 10
    np.testing.assert_allclose(detections[detections.t == 0][["y", "x"]], one_dict.data)


def test_get_candidate_edges(get_detections):
    """Test that each detection gets k=10 edges to the next frame."""
    detections = get_detections()
    tracker = Tracker(im_shape=(20, 40))
    kd_dict = tracker._build_trees(detections, "t", ("y", "x"))
    edges = tracker._get_candidate_edges(detections, "t", kd_dict)
    # 10 edges for each node, 10 nodes per frame, 9 frames (not last one)
    assert len(edges) == 100 * 9

    # check that each source has 10 edges into next frame
    for u in detections.index.values:
        if detections.iloc[u].t < 9:
            assert len(edges[edges.u == u]) == 10


def test_get_candidate_edges_different_keys(get_detections):
    """Test that passing different frame and location keys works"""
    detections = get_detections()
    detections = detections.rename(columns={"t": "frame", "y": "row", "x": "col"})
    tracker = Tracker(im_shape=(20, 40))
    kd_dict = tracker._build_trees(detections, "frame", ("row", "col"))
    edges = tracker._get_candidate_edges(detections, "frame", kd_dict)
    # 10 edges for each node, 10 nodes per frame, 9 frames (not last one)
    assert len(edges) == 100 * 9

    # check that each source has 10 edges into next frame
    for u in detections.index.values:
        if detections.iloc[u].frame < 9:
            assert len(edges[edges.u == u]) == 10


def test_get_candidate_edges_too_many_neighbours(get_detections):
    """Test that when neighbours exceeds detections per frame all are connected."""
    detections = get_detections()
    tracker = Tracker(im_shape=(20, 40), k_neighbours=20)
    kd_dict = tracker._build_trees(detections, "t", ("y", "x"))
    edges = tracker._get_candidate_edges(detections, "t", kd_dict)
    # even though we passed k=20, we should only get 10 edges per node
    assert len(edges) == 100 * 9


def test_get_candidate_edges_empty_frame(get_detections):
    """Test that empty frames are handled gracefully."""
    detections = get_detections()
    # empty frame 2
    detections = detections.drop(detections[detections.t == 2].index)
    tracker = Tracker(im_shape=(20, 40))
    kd_dict = tracker._build_trees(detections, "t", ("y", "x"))
    with pytest.warns(
        UserWarning, match="Connecting frames 1 and 3. They are not consecutive."
    ):
        tracker._get_candidate_edges(detections, "t", kd_dict)


def test_get_candidate_edges_single_detection(get_detections):
    """Test that a single detection in a frame is handled gracefully."""
    detections = get_detections()
    # leave just one detection in frame 2
    detections = detections.drop(detections[detections.t == 2].index[:-1])
    tracker = Tracker(im_shape=(20, 40), k_neighbours=3)
    kd_dict = tracker._build_trees(detections, "t", ("y", "x"))
    edges = tracker._get_candidate_edges(detections, "t", kd_dict)
    singleton_index = detections[detections.t == 2].index.values[0]
    # 10 edges coming into single detection (from all nodes in previous frame)
    assert len(edges[edges.v == singleton_index]) == 10
    # 3 edges going out
    assert len(edges[edges.u == singleton_index]) == 3


def test_get_candidate_edges_single_neighbour(get_detections):
    """Test that closest-neighbour is handled gracefully"""
    detections = get_detections()
    tracker = Tracker(im_shape=(20, 40), k_neighbours=1)
    kd_dict = tracker._build_trees(detections, "t", ("y", "x"))
    edges = tracker._get_candidate_edges(detections, "t", kd_dict)

    for u in detections.index.values:
        if detections.iloc[u].t < 9:
            assert len(edges[edges.u == u]) == 1


def test_get_candidate_edges_correct_distance(human_detections):
    detections, im_shape = human_detections
    tracker = Tracker(im_shape=im_shape, k_neighbours=2)
    kd_dict = tracker._build_trees(detections, "t", ("y", "x"))
    edges = tracker._get_candidate_edges(detections, "t", kd_dict)
    # checking a couple of edges physically
    # distance (1, 1) -> (1, 2) = 1
    assert edges[(edges.u == 0) & (edges.v == 3)].distance.values[0] == 1.0
    # distance (1, 1) -> (1, 3) = 2
    assert edges[(edges.u == 0) & (edges.v == 4)].distance.values[0] == 2.0
    # distance (3, 5) -> (2, 4) = sqrt(2)
    assert edges[(edges.u == 5) & (edges.v == 9)].distance.values[0] == np.sqrt(2)
    # distance (3, 5) -> (1, 3) == sqrt((3-1)**2 + (5-3)**2)
    assert edges[(edges.u == 5) & (edges.v == 8)].distance.values[0] == np.sqrt(
        (3 - 1) ** 2 + (5 - 3) ** 2
    )


def test_get_candidate_edges_correct_capacity(get_detections):
    detections = get_detections()
    tracker = Tracker(im_shape=(20, 40), k_neighbours=1)
    kd_dict = tracker._build_trees(detections, "t", ("y", "x"))
    edges = tracker._get_candidate_edges(detections, "t", kd_dict)
    assert len(edges.capacity.unique()) == 1
    assert edges.capacity.unique()[0] == tracker.MERGE_EDGE_CAPACITY


def test_get_all_vertices(get_detections):
    detections = get_detections()
    tracker = Tracker(im_shape=(20, 40))
    vertices = tracker._get_all_vertices(detections, "t", ("y", "x"))
    assert len(vertices) == len(detections) + 4
    # negative indexed
    assert np.all(vertices.sort_index().index.values[:3] < 0)


def test_get_all_vertices_different_keys(get_detections):
    detections = get_detections()
    detections = detections.rename(columns={"t": "frame", "y": "row", "x": "col"})
    tracker = Tracker(im_shape=(20, 40))
    vertices = tracker._get_all_vertices(detections, "frame", ("row", "col"))
    assert len(vertices) == len(detections) + 4
    assert vertices.iloc[0].frame == 0
    assert vertices.iloc[0].row == detections.iloc[0].row
    assert vertices.iloc[0].col == detections.iloc[0].col


def test_get_all_edges(get_detections):
    detections = get_detections()
    tracker = Tracker(im_shape=(20, 40))
    kd_dict = tracker._build_trees(detections, "t", ("y", "x"))
    edges = tracker._get_candidate_edges(detections, "t", kd_dict)
    enter_exit_cost, div_cost = tracker._compute_detection_costs(
        detections, ("y", "x"), edges
    )
    detections["enter_exit_cost"] = enter_exit_cost
    detections["div_cost"] = div_cost
    all_edges = tracker._get_all_edges(edges, detections, "t")

    # all vertices connected to appearance
    assert len(all_edges[all_edges.u == VirtualVertices.APP.value]) == len(detections)
    # first frame appears free
    first_frame_det = detections[detections.t == 0].index
    assert np.all(
        all_edges[
            (all_edges.u == VirtualVertices.APP.value)
            & all_edges.v.isin(first_frame_det)
        ]["cost"]
        == 0
    )
    # last frame exits free
    last_frame_det = detections[detections.t == 9].index
    assert np.all(
        all_edges[
            (all_edges.u.isin(last_frame_det)) & all_edges.v
            == VirtualVertices.TARGET.value
        ]["cost"]
        == 0
    )
    # all but final frame can divide
    divisible_det = detections[detections.t < 9].index
    assert len(all_edges[all_edges.u == VirtualVertices.DIV.value]) == len(
        divisible_det
    )


def test_to_gurobi_model(human_detections):
    detections, im_shape = human_detections
    tracker = Tracker(im_shape=im_shape, k_neighbours=2)
    kd_dict = tracker._build_trees(detections, "t", ("y", "x"))
    edges = tracker._get_candidate_edges(detections, "t", kd_dict)
    edges["cost"] = edges["distance"]
    enter_exit_cost, div_cost = tracker._compute_detection_costs(
        detections, ("y", "x"), edges
    )
    detections["enter_exit_cost"] = enter_exit_cost
    detections["div_cost"] = div_cost
    m, _, _, _ = tracker._to_gurobi_model(detections, edges, "t", ("y", "x"))
    # assert that we have a flow conserv & demand constraint for each detection
    constr_names = [constr.ConstrName for constr in m.getConstrs()]
    num_detections = len(detections)
    num_div_detections = len(detections[detections.t < 2])
    conserv_pattern = re.compile(r"conserv_[0-9]+")
    demand_pattern = re.compile(r"demand_[0-9]+")
    div_pattern = re.compile(r"div_[0-9]+")
    assert (
        len(list(filter(lambda n: conserv_pattern.match(n), constr_names)))
        == num_detections
    )
    assert (
        len(list(filter(lambda n: demand_pattern.match(n), constr_names)))
        == num_detections
    )

    # assert that we have a div constraint for divisible detections
    assert (
        len(list(filter(lambda n: div_pattern.match(n), constr_names)))
        == num_div_detections
    )

    # assert that we have the network constraint and dummmy constraints
    assert "conserv_network" in constr_names
    assert "conserv_app" in constr_names
    assert "conserv_div" in constr_names

    # assert on variable capacities
    for var in m.getVars():
        assert var.LB == 0
        # appearance and division edges
        if ",a," in var.VarName or ",d," in var.VarName:
            assert var.UB == 1
        # dummy-dummy edges
        elif "s" in var.VarName:
            assert var.UB == math.inf
        # migration and exit edges
        else:
            assert var.UB == 2


def test_solve_public(human_detections):
    detections, im_shape = human_detections
    tracker = Tracker(im_shape=im_shape, k_neighbours=2)
    tracked = tracker.solve(detections, "t", ("y", "x"))
    solution_edges = np.asarray(
        [[0, 3], [1, 5], [2, 6], [3, 7], [4, 8], [5, 9], [6, 10]], dtype=int
    )
    np.testing.assert_array_equal(
        tracked.tracked_edges[["u", "v"]].values, solution_edges
    )
    np.testing.assert_equal(tracked.tracked_edges["flow"].values, 1)


def test_solve_with_divisions(human_detections):
    pass


def test_solve_debug(human_detections):
    detections, im_shape = human_detections
    tracker = Tracker(im_shape=im_shape, k_neighbours=2)
    tracker.DEBUG_MODE = True
    tracker.solve(detections, "t", ("y", "x"))


def test_big_instance_unchanged():
    # generate and write out a big set of detections
    # solve it and write out the solution
    # add everything to git and assert on dataframes being the same

    pass


# test 3d da
# test cost functions specifically
