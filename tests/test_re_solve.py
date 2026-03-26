"""Tests for partial re-solve operations: move node, add node, mark exit.

These tests exercise Tracker.move_node_in_model, add_node_to_model, and
fix_edge_in_model on a warm-started model so we can verify both model state
and the resulting solution without needing the napari widget.

Graph layout (all tests share the same fixture):
    Frame 0: node 0 at (5, 2)   node 1 at (5, 18)
    Frame 1: node 2 at (5, 2)   node 3 at (5, 18)
    Solution: 0→2, 1→3
    im_shape = (10, 20)
"""

import networkx as nx
import pytest

from tracktour import Tracker

# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def warm_started():
    """Warm-started tracker on a simple 2-frame, 2-node-per-frame graph."""
    nxg = nx.DiGraph()
    nxg.add_node(0, t=0, y=5.0, x=2.0)
    nxg.add_node(1, t=0, y=5.0, x=18.0)
    nxg.add_node(2, t=1, y=5.0, x=2.0)
    nxg.add_node(3, t=1, y=5.0, x=18.0)
    nxg.add_edge(0, 2)
    nxg.add_edge(1, 3)

    tracker = Tracker(im_shape=(10, 20))
    tracker.DEBUG_MODE = True
    tracked = tracker.warm_start_from_solution_graph(
        nxg, frame_key="t", location_keys=("y", "x"), scale=(1.0, 1.0)
    )
    return tracker, tracked


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _migration_flow_edges(tracked):
    """Set of (u, v) real migration edges with flow > 0."""
    ae = tracked.all_edges
    used = ae[(ae.u >= 0) & (ae.v >= 0) & (ae.flow > 0)]
    return set(zip(used.u.tolist(), used.v.tolist()))


def _re_solve(tracker, tracked):
    """Optimize model and store solution. Returns True if feasible."""
    tracker._model.optimize()
    if tracker._model.status != 2:
        return False
    tracker._store_solution(tracker._model, tracked.all_edges)
    return True


# ---------------------------------------------------------------------------
# Baseline
# ---------------------------------------------------------------------------


def test_warm_start_initial_solution(warm_started):
    """Warm-start should reproduce the original 0→2 and 1→3 solution."""
    _, tracked = warm_started
    edges = _migration_flow_edges(tracked)
    assert (0, 2) in edges
    assert (1, 3) in edges


# ---------------------------------------------------------------------------
# Move node
# ---------------------------------------------------------------------------


def test_move_node_updates_tracked_detections(warm_started):
    """move_node_in_model should update tracked_detections in place."""
    tracker, tracked = warm_started
    tracker.move_node_in_model(tracked, node_id=2, new_frame=1, new_pos=(5.0, 18.0))

    assert tracked.tracked_detections.at[2, "x"] == pytest.approx(18.0)
    assert tracked.tracked_detections.at[2, "y"] == pytest.approx(5.0)
    assert tracked.tracked_detections.at[2, "t"] == 1


def test_move_node_re_solve_feasible(warm_started):
    """Re-solve after moving a node must be feasible."""
    tracker, tracked = warm_started
    tracker.move_node_in_model(tracked, node_id=2, new_frame=1, new_pos=(5.0, 18.0))
    assert _re_solve(tracker, tracked)


def test_move_node_disables_old_edge(warm_started):
    """After moving node 2, the old edge 0→2 should carry no flow."""
    tracker, tracked = warm_started
    tracker.move_node_in_model(tracked, node_id=2, new_frame=1, new_pos=(5.0, 18.0))
    assert _re_solve(tracker, tracked)

    edges = _migration_flow_edges(tracked)
    assert (0, 2) not in edges


# ---------------------------------------------------------------------------
# Move node then add node — exercises the pd.concat revert bug
# ---------------------------------------------------------------------------


def test_move_then_add_preserves_moved_position_in_tracked_detections(warm_started):
    """Moving a node then adding a new node must not revert the moved node's position.

    add_node_to_model uses pd.concat to append to tracked_detections.  In some
    pandas versions this can create a new DataFrame that drops earlier in-place
    mutations from move_node_in_model, reverting the position to its original value.
    """
    tracker, tracked = warm_started

    tracker.move_node_in_model(tracked, node_id=2, new_frame=1, new_pos=(5.0, 18.0))
    # Adding a new node triggers pd.concat inside add_node_to_model
    tracker.add_node_to_model(tracked, node_id=4, frame=0, pos=(5.0, 10.0))

    # Node 2's new position must survive the concat
    assert tracked.tracked_detections.at[2, "x"] == pytest.approx(18.0)


def test_move_then_add_re_solve_feasible(warm_started):
    """Re-solve after moving a node and adding a new node must be feasible."""
    tracker, tracked = warm_started
    tracker.move_node_in_model(tracked, node_id=2, new_frame=1, new_pos=(5.0, 18.0))
    tracker.add_node_to_model(tracked, node_id=4, frame=0, pos=(5.0, 10.0))
    assert _re_solve(tracker, tracked)


def test_move_then_add_old_edge_absent(warm_started):
    """After move + add + re-solve, the old edge 0→2 should not be in the solution."""
    tracker, tracked = warm_started
    tracker.move_node_in_model(tracked, node_id=2, new_frame=1, new_pos=(5.0, 18.0))
    tracker.add_node_to_model(tracked, node_id=4, frame=0, pos=(5.0, 10.0))
    assert _re_solve(tracker, tracked)

    assert (0, 2) not in _migration_flow_edges(tracked)


# ---------------------------------------------------------------------------
# Add node only
# ---------------------------------------------------------------------------


def test_add_node_appears_in_tracked_detections(warm_started):
    """add_node_to_model should insert the new node into tracked_detections."""
    tracker, tracked = warm_started
    tracker.add_node_to_model(tracked, node_id=4, frame=0, pos=(5.0, 10.0))

    assert 4 in tracked.tracked_detections.index
    assert tracked.tracked_detections.at[4, "x"] == pytest.approx(10.0)
    assert tracked.tracked_detections.at[4, "y"] == pytest.approx(5.0)
    assert tracked.tracked_detections.at[4, "t"] == 0


def test_add_node_re_solve_feasible(warm_started):
    """Re-solve with an added node must be feasible."""
    tracker, tracked = warm_started
    tracker.add_node_to_model(tracked, node_id=4, frame=0, pos=(5.0, 10.0))
    assert _re_solve(tracker, tracked)


def test_add_node_has_flow_in_solution(warm_started):
    """The added node must participate in the solution (has flow on some edge)."""
    tracker, tracked = warm_started
    tracker.add_node_to_model(tracked, node_id=4, frame=0, pos=(5.0, 10.0))
    assert _re_solve(tracker, tracked)

    used = tracked.all_edges[tracked.all_edges.flow > 0]
    assert (used.u == 4).any() or (used.v == 4).any()


# ---------------------------------------------------------------------------
# Mark exit (oracle correction: edge forced to 0)
# ---------------------------------------------------------------------------


def _find_edge_idx(tracked, u, v):
    match = tracked.all_edges[(tracked.all_edges.u == u) & (tracked.all_edges.v == v)]
    assert not match.empty, f"Edge {u}→{v} not found in all_edges"
    return int(match.index[0])


def test_batch_move_nodes_see_each_other(warm_started):
    """When two nodes are moved simultaneously, each k-NN search should see the other
    node at its new position, not its old one.

    We swap nodes 2 and 3: 2 goes to (5,18) and 3 goes to (5,2).  The two-phase
    protocol (_prepare_* then rebuild_kd_trees then _apply_node_edges) means the
    k-NN search for node 2 sees node 3 already at (5,2) and vice versa.
    """
    tracker, tracked = warm_started

    # Phase 1: update positions only
    tracker._prepare_move_node(tracked, node_id=2, new_frame=1, new_pos=(5.0, 18.0))
    tracker._prepare_move_node(tracked, node_id=3, new_frame=1, new_pos=(5.0, 2.0))

    # Single KD-tree rebuild for all affected frames
    tracker.rebuild_kd_trees(tracked, [1])

    # Verify both positions are updated before edges are searched
    assert tracked.tracked_detections.at[2, "x"] == pytest.approx(18.0)
    assert tracked.tracked_detections.at[3, "x"] == pytest.approx(2.0)

    # Phase 2: recompute edges with all nodes at final positions
    tracker._apply_node_edges(tracked, node_id=2, frame=1, pos=(5.0, 18.0))
    tracker._apply_node_edges(tracked, node_id=3, frame=1, pos=(5.0, 2.0))

    assert _re_solve(tracker, tracked)
    # Old edges 0→2 and 1→3 must be gone (ub=0 from prepare step)
    edges = _migration_flow_edges(tracked)
    assert (0, 2) not in edges
    assert (1, 3) not in edges


def test_mark_exit_removes_edge_from_solution(warm_started):
    """Fixing edge 0→2 to lb=0, ub=0 should remove it from the re-solved solution."""
    tracker, tracked = warm_started

    edge_idx = _find_edge_idx(tracked, 0, 2)
    tracker.fix_edge_in_model(edge_idx, 0, 2, lb=0, ub=0)
    assert _re_solve(tracker, tracked)

    assert (0, 2) not in _migration_flow_edges(tracked)


def test_mark_exit_re_solve_feasible(warm_started):
    """Re-solve must succeed even when an edge is forced to zero flow."""
    tracker, tracked = warm_started

    edge_idx = _find_edge_idx(tracked, 0, 2)
    tracker.fix_edge_in_model(edge_idx, 0, 2, lb=0, ub=0)
    assert _re_solve(tracker, tracked)


def test_mark_exit_other_edges_unaffected(warm_started):
    """Fixing 0→2 should not force 1→3 out of the solution."""
    tracker, tracked = warm_started

    edge_idx = _find_edge_idx(tracked, 0, 2)
    tracker.fix_edge_in_model(edge_idx, 0, 2, lb=0, ub=0)
    assert _re_solve(tracker, tracked)

    # 1→3 was not touched and should still be active (lb=1 from warm-start)
    assert (1, 3) in _migration_flow_edges(tracked)


# ---------------------------------------------------------------------------
# Combined: move + mark exit
# ---------------------------------------------------------------------------


def test_move_and_mark_exit_feasible(warm_started):
    """Moving a node and marking another edge as exit together must remain feasible."""
    tracker, tracked = warm_started

    tracker.move_node_in_model(tracked, node_id=2, new_frame=1, new_pos=(5.0, 18.0))

    edge_idx = _find_edge_idx(tracked, 1, 3)
    tracker.fix_edge_in_model(edge_idx, 1, 3, lb=0, ub=0)

    assert _re_solve(tracker, tracked)
    assert (0, 2) not in _migration_flow_edges(tracked)
    assert (1, 3) not in _migration_flow_edges(tracked)
