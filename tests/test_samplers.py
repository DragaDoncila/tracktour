"""Tests for edge sampling strategies."""

import pandas as pd
import pytest

from tracktour._napari.track_annotator.samplers import (
    DUCBEdgeSampler,
    RandomEdgeSampler,
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
    gamma = 0.9
    sampler = DUCBEdgeSampler(simple_edges_df, {"score": True}, gamma=gamma)
    assert sampler.discounted_arm_played["score"] == 1.0
    sampler.provide_reward(1.0)
    sampler.next()
    # discount applied before next UCB round: N * gamma, then +1 for new selection
    assert sampler.discounted_arm_played["score"] == pytest.approx(gamma + 1.0)
    assert sampler.discounted_arm_rewards["score"] == pytest.approx(gamma)


def test_ducb_two_arms_each_played_before_ucb(two_arm_edges_df):
    # Both arms have N=0 at start, so each is played once before UCB kicks in
    sampler = DUCBEdgeSampler(two_arm_edges_df, {"score": True, "rank": True})
    _, arm0 = sampler._history[0]
    sampler.next()
    _, arm1 = sampler._history[1]
    assert {arm0, arm1} == {"score", "rank"}


def test_ducb_empty_bandit_arms_raises():
    df = pd.DataFrame({"u": [0], "v": [1], "score": [0.5]})
    with pytest.raises(ValueError):
        DUCBEdgeSampler(df, {})
