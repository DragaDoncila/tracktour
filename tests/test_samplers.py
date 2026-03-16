"""Tests for edge sampling strategies."""

import pytest

from tracktour._napari.track_annotator.samplers import RandomEdgeSampler


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
