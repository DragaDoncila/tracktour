"""Tests for edge sampling strategies."""

import networkx as nx
import pandas as pd
import pytest

from tracktour._napari.track_annotator.samplers import (
    DUCBEdgeSampler,
    RandomEdgeSampler,
    TrajectoryEdgeSampler,
)

# ---------------------------------------------------------------------------
# Shared fixtures for DUCBEdgeSampler
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_edges_df():
    """Four solution edges with one feature column 'score'."""
    return pd.DataFrame(
        {
            "u": [0, 1, 2, 3],
            "v": [4, 5, 6, 7],
            "score": [0.9, 0.1, 0.5, 0.3],
        }
    )


@pytest.fixture
def two_arm_edges_df():
    """Four edges with two feature columns."""
    return pd.DataFrame(
        {
            "u": [0, 1, 2, 3],
            "v": [4, 5, 6, 7],
            "score": [0.9, 0.1, 0.5, 0.3],
            "rank": [3, 0, 2, 1],
        }
    )


# ---------------------------------------------------------------------------
# RandomEdgeSampler
# ---------------------------------------------------------------------------


class TestRandomEdgeSampler:
    """Tests for RandomEdgeSampler."""

    @pytest.fixture
    def edges(self):
        return [(0, 1), (1, 2), (2, 3), (3, 4)]

    def test_current_returns_tuple(self, edges):
        sampler = RandomEdgeSampler(edges, seed=0)
        edge = sampler.current()
        assert isinstance(edge, tuple)
        assert len(edge) == 2

    def test_total_count(self, edges):
        sampler = RandomEdgeSampler(edges, seed=0)
        assert sampler.total_count() == 4

    def test_current_index_starts_at_zero(self, edges):
        sampler = RandomEdgeSampler(edges, seed=0)
        assert sampler.current_index() == 0

    def test_next_advances_and_returns_edge(self, edges):
        sampler = RandomEdgeSampler(edges, seed=0)
        assert sampler.current_index() == 0
        next_edge = sampler.next()
        assert sampler.current_index() == 1
        assert next_edge == sampler.current()

    def test_next_returns_none_at_end(self, edges):
        sampler = RandomEdgeSampler(edges, seed=0)
        for _ in range(len(edges) - 1):
            assert sampler.next() is not None
        assert sampler.next() is None

    def test_previous_goes_back_and_returns_edge(self, edges):
        sampler = RandomEdgeSampler(edges, seed=0)
        sampler.next()
        sampler.next()
        assert sampler.current_index() == 2
        prev_edge = sampler.previous()
        assert sampler.current_index() == 1
        assert prev_edge == sampler.current()

    def test_previous_returns_none_at_start(self, edges):
        sampler = RandomEdgeSampler(edges, seed=0)
        assert sampler.previous() is None
        assert sampler.current_index() == 0

    def test_at_start(self, edges):
        sampler = RandomEdgeSampler(edges, seed=0)
        assert sampler.at_start()
        sampler.next()
        assert not sampler.at_start()
        sampler.previous()
        assert sampler.at_start()

    def test_at_end(self, edges):
        sampler = RandomEdgeSampler(edges, seed=0)
        assert not sampler.at_end()
        for _ in range(len(edges) - 1):
            sampler.next()
        assert sampler.at_end()

    def test_handles_numpy_arrays(self):
        import numpy as np

        edges = np.array([[0, 1], [1, 2], [2, 3]])
        sampler = RandomEdgeSampler(edges, seed=0)
        edge = sampler.current()
        assert isinstance(edge, tuple)
        assert isinstance(edge[0], int)
        assert isinstance(edge[1], int)

    def test_single_edge(self):
        sampler = RandomEdgeSampler([(0, 1)], seed=0)
        assert sampler.at_start()
        assert sampler.at_end()
        assert sampler.current() == (0, 1)
        assert sampler.next() is None
        assert sampler.previous() is None


# ---------------------------------------------------------------------------
# DUCBEdgeSampler
# ---------------------------------------------------------------------------


def test_ducb_current_returns_tuple(simple_edges_df):
    sampler = DUCBEdgeSampler(simple_edges_df, {"score": True})
    edge = sampler.current()
    assert isinstance(edge, tuple)
    assert len(edge) == 2


def test_ducb_total_count(simple_edges_df):
    sampler = DUCBEdgeSampler(simple_edges_df, {"score": True})
    assert sampler.total_count() == 4


def test_ducb_current_index_starts_at_zero(simple_edges_df):
    sampler = DUCBEdgeSampler(simple_edges_df, {"score": True})
    assert sampler.current_index() == 0


def test_ducb_at_start_initially(simple_edges_df):
    sampler = DUCBEdgeSampler(simple_edges_df, {"score": True})
    assert sampler.at_start()


def test_ducb_not_at_end_initially_with_multiple_edges(simple_edges_df):
    sampler = DUCBEdgeSampler(simple_edges_df, {"score": True})
    assert not sampler.at_end()


def test_ducb_single_edge_at_start_and_end():
    df = pd.DataFrame({"u": [0], "v": [1], "score": [0.5]})
    sampler = DUCBEdgeSampler(df, {"score": True})
    assert sampler.at_start()
    assert sampler.at_end()
    assert sampler.next() is None
    assert sampler.previous() is None


def test_ducb_ascending_first_edge_has_smallest_score(simple_edges_df):
    # ascending=True: smallest score first (row with score=0.1, u=1, v=5)
    sampler = DUCBEdgeSampler(simple_edges_df, {"score": True})
    assert sampler.current() == (1, 5)


def test_ducb_descending_first_edge_has_largest_score(simple_edges_df):
    # ascending=False: largest score first (row with score=0.9, u=0, v=4)
    sampler = DUCBEdgeSampler(simple_edges_df, {"score": False})
    assert sampler.current() == (0, 4)


def test_ducb_next_advances_index(simple_edges_df):
    sampler = DUCBEdgeSampler(simple_edges_df, {"score": True})
    sampler.next()
    assert sampler.current_index() == 1


def test_ducb_next_returns_edge_tuple(simple_edges_df):
    sampler = DUCBEdgeSampler(simple_edges_df, {"score": True})
    edge = sampler.next()
    assert isinstance(edge, tuple)
    assert len(edge) == 2


def test_ducb_previous_returns_none_at_start(simple_edges_df):
    sampler = DUCBEdgeSampler(simple_edges_df, {"score": True})
    assert sampler.previous() is None
    assert sampler.current_index() == 0


def test_ducb_previous_goes_back(simple_edges_df):
    sampler = DUCBEdgeSampler(simple_edges_df, {"score": True})
    first_edge = sampler.current()
    sampler.next()
    prev = sampler.previous()
    assert prev == first_edge
    assert sampler.current_index() == 0


def test_ducb_previous_replays_history_not_rerun_ucb(simple_edges_df):
    sampler = DUCBEdgeSampler(simple_edges_df, {"score": True})
    first_edge = sampler.current()
    sampler.next()
    second_edge = sampler.current()
    # go back then forward — must return the same edges
    sampler.previous()
    assert sampler.current() == first_edge
    sampler.next()
    assert sampler.current() == second_edge


def test_ducb_exhaust_all_edges_returns_none(simple_edges_df):
    sampler = DUCBEdgeSampler(simple_edges_df, {"score": True})
    for _ in range(len(simple_edges_df) - 1):
        result = sampler.next()
        assert result is not None
    assert sampler.next() is None


def test_ducb_at_end_after_exhausting_all_edges(simple_edges_df):
    sampler = DUCBEdgeSampler(simple_edges_df, {"score": True})
    for _ in range(len(simple_edges_df) - 1):
        sampler.next()
    assert sampler.at_end()


def test_ducb_visits_all_edges_exactly_once(simple_edges_df):
    sampler = DUCBEdgeSampler(simple_edges_df, {"score": True})
    seen = {sampler.current()}
    while not sampler.at_end():
        seen.add(sampler.next())
    expected = set(zip(simple_edges_df.u, simple_edges_df.v))
    assert seen == expected


def test_ducb_n_incremented_at_selection_not_at_reward(simple_edges_df):
    # N is incremented when an arm is selected (first edge picked in __init__)
    sampler = DUCBEdgeSampler(simple_edges_df, {"score": True})
    assert sampler.discounted_arm_played["score"] == 1.0  # from the initial pick
    # provide_reward does not change N
    sampler.provide_reward(1.0)
    assert sampler.discounted_arm_played["score"] == 1.0


def test_ducb_provide_reward_updates_s(simple_edges_df):
    sampler = DUCBEdgeSampler(simple_edges_df, {"score": True})
    sampler.provide_reward(1.0)
    assert sampler.discounted_arm_rewards["score"] == 1.0


def test_ducb_provide_reward_zero_updates_s_to_zero(simple_edges_df):
    sampler = DUCBEdgeSampler(simple_edges_df, {"score": True})
    sampler.provide_reward(0.0)
    assert sampler.discounted_arm_rewards["score"] == 0.0
    assert (
        sampler.discounted_arm_played["score"] == 1.0
    )  # play count from selection, unchanged by reward


def test_ducb_n_and_s_discounted_on_next_call(simple_edges_df):
    sampler = DUCBEdgeSampler(simple_edges_df, {"score": True})
    gamma = sampler._gamma
    assert sampler.discounted_arm_played["score"] == 1.0
    sampler.provide_reward(1.0)
    sampler.next()
    # discount applied before next UCB round: N * gamma, then +1 for new selection
    assert sampler.discounted_arm_played["score"] == pytest.approx(gamma + 1.0)
    assert sampler.discounted_arm_rewards["score"] == pytest.approx(gamma)


def test_ducb_two_arms_each_played_before_ucb(two_arm_edges_df):
    # Both arms have N=0 at start, so each is played once before UCB kicks in
    sampler = DUCBEdgeSampler(two_arm_edges_df, {"score": True, "rank": True})
    _, arm0, _ = sampler._history[0]
    sampler.next()
    _, arm1, _ = sampler._history[1]
    assert {arm0, arm1} == {"score", "rank"}


def test_ducb_provide_reward_stores_in_history(simple_edges_df):
    sampler = DUCBEdgeSampler(simple_edges_df, {"score": True})
    sampler.provide_reward(1.0)
    assert sampler._history[0][2] == 1.0


def test_ducb_provide_reward_same_value_is_noop(simple_edges_df):
    sampler = DUCBEdgeSampler(simple_edges_df, {"score": True})
    sampler.provide_reward(1.0)
    rewards_before = sampler.discounted_arm_rewards["score"]
    sampler.provide_reward(1.0)
    assert sampler.discounted_arm_rewards["score"] == rewards_before


def test_ducb_retroactive_reward_sets_dirty_flag(simple_edges_df):
    sampler = DUCBEdgeSampler(simple_edges_df, {"score": True})
    sampler.provide_reward(1.0)
    sampler.next()
    sampler.provide_reward(0.0)
    sampler.previous()
    # back at position 0, which is not at the frontier
    sampler.provide_reward(0.0)  # change from 1.0 to 0.0
    assert sampler._history_dirty


def test_ducb_retroactive_reward_same_value_does_not_set_dirty(simple_edges_df):
    sampler = DUCBEdgeSampler(simple_edges_df, {"score": True})
    sampler.provide_reward(1.0)
    sampler.next()
    sampler.previous()
    sampler.provide_reward(1.0)  # same value — no change
    assert not sampler._history_dirty


def test_ducb_recompute_clears_dirty_flag(simple_edges_df):
    sampler = DUCBEdgeSampler(simple_edges_df, {"score": True})
    sampler.provide_reward(1.0)
    sampler.next()
    sampler.previous()
    sampler.provide_reward(0.0)
    assert sampler._history_dirty
    sampler.next()  # advances to frontier, triggering recompute
    sampler.next()  # now at new frontier: recompute should have run
    assert not sampler._history_dirty


def test_ducb_recompute_bandit_state_matches_incremental(simple_edges_df):
    # Build a sampler the normal incremental way
    sampler = DUCBEdgeSampler(simple_edges_df, {"score": True})
    sampler.provide_reward(1.0)
    sampler.next()
    sampler.provide_reward(0.0)
    incremental_played = dict(sampler.discounted_arm_played)
    incremental_rewards = dict(sampler.discounted_arm_rewards)

    # Force a full recompute and check it matches
    sampler._recompute_bandit_state()
    assert sampler.discounted_arm_played == pytest.approx(incremental_played)
    assert sampler.discounted_arm_rewards == pytest.approx(incremental_rewards)


def test_ducb_empty_bandit_arms_raises():
    df = pd.DataFrame({"u": [0], "v": [1], "score": [0.5]})
    with pytest.raises(ValueError):
        DUCBEdgeSampler(df, {})


# ---------------------------------------------------------------------------
# TrajectoryEdgeSampler
# ---------------------------------------------------------------------------

# Shared graphs for trajectory tests


def _linear_graph():
    """0→1→2→3 (single trajectory, no divisions)."""
    g = nx.DiGraph()
    g.add_edges_from([(0, 1), (1, 2), (2, 3)])
    return g


def _division_graph():
    """0→1→2 and 1→3 (one division at node 1)."""
    g = nx.DiGraph()
    g.add_edges_from([(0, 1), (1, 2), (1, 3)])
    return g


def _two_root_graph():
    """Two independent chains: 0→1→2 and 3→4→5."""
    g = nx.DiGraph()
    g.add_edges_from([(0, 1), (1, 2), (3, 4), (4, 5)])
    return g


def test_trajectory_visits_all_edges():
    sampler = TrajectoryEdgeSampler(_linear_graph(), seed=0)
    assert sampler.total_count() == 3


def test_trajectory_linear_order_is_root_to_leaf():
    sampler = TrajectoryEdgeSampler(_linear_graph(), seed=0)
    edges = [sampler.current()]
    for _ in range(sampler.total_count() - 1):
        sampler.next()
        edges.append(sampler.current())
    assert edges == [(0, 1), (1, 2), (2, 3)]


def test_trajectory_division_first_edge_is_root():
    sampler = TrajectoryEdgeSampler(_division_graph(), seed=0)
    assert sampler.current()[0] == 0


def test_trajectory_division_second_edge_continues_from_first_child():
    # After (0,1), the next edge should start from 1 (the child).
    sampler = TrajectoryEdgeSampler(_division_graph(), seed=0)
    first = sampler.current()
    sampler.next()
    second = sampler.current()
    assert first == (0, 1)
    assert second[0] == 1  # continues down from node 1


def test_trajectory_division_deferred_branch_visited_last():
    # After the first branch (0→1→child_a), the deferred child_b is visited.
    sampler = TrajectoryEdgeSampler(_division_graph(), seed=0)
    edges = [sampler.current()]
    for _ in range(sampler.total_count() - 1):
        sampler.next()
        edges.append(sampler.current())
    # All three edges must appear; first two must share a source node
    assert len(edges) == 3
    assert edges[0] == (0, 1)
    assert edges[1][0] == 1
    assert edges[2][0] == 1
    assert edges[1][1] != edges[2][1]  # two different children of node 1


def test_trajectory_two_roots_each_trajectory_is_contiguous():
    # One root's chain should appear consecutively, then the other's.
    sampler = TrajectoryEdgeSampler(_two_root_graph(), seed=0)
    edges = [sampler.current()]
    for _ in range(sampler.total_count() - 1):
        sampler.next()
        edges.append(sampler.current())
    # Edges from each root form contiguous blocks
    root_a = edges[0][0]  # whichever root came first
    root_b = 3 if root_a == 0 else 0
    first_half = [e for e in edges if e[0] in (root_a, root_a + 1)]
    second_half = [e for e in edges if e[0] in (root_b, root_b + 1)]
    assert len(first_half) == 2
    assert len(second_half) == 2
    assert edges.index(first_half[0]) < edges.index(second_half[0])


def test_trajectory_navigation_previous():
    sampler = TrajectoryEdgeSampler(_linear_graph(), seed=0)
    first = sampler.current()
    sampler.next()
    sampler.previous()
    assert sampler.current() == first


def test_trajectory_at_start_initially():
    sampler = TrajectoryEdgeSampler(_linear_graph(), seed=0)
    assert sampler.at_start()


def test_trajectory_at_end_after_full_traversal():
    sampler = TrajectoryEdgeSampler(_linear_graph(), seed=0)
    for _ in range(sampler.total_count() - 1):
        sampler.next()
    assert sampler.at_end()


def test_trajectory_different_seeds_may_differ_on_division():
    # With a division, different seeds should (usually) produce different orders.
    s0 = TrajectoryEdgeSampler(_division_graph(), seed=0)
    s1 = TrajectoryEdgeSampler(_division_graph(), seed=42)
    # Both orderings are valid; we just check they cover the same edges
    assert set(tuple(e) for e in s0._edges) == set(tuple(e) for e in s1._edges)
