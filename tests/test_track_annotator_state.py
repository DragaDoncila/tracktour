"""Tests for AnnotationState class."""

import networkx as nx
import pytest

from tracktour._napari.track_annotator.state import (
    FN_EDGE_ATTR,
    FN_NODE_ATTR,
    FP_EDGE_ATTR,
    FP_NODE_ATTR,
    TP_EDGE_ATTR,
    TP_NODE_VOTES,
    AnnotationState,
)


@pytest.fixture
def simple_graph():
    """Create a simple test graph."""
    G = nx.DiGraph()
    G.add_node(0, t=0, y=10, x=20)
    G.add_node(1, t=1, y=15, x=25)
    G.add_node(2, t=2, y=20, x=30)
    G.add_edge(0, 1)
    G.add_edge(1, 2)
    return G


class TestAnnotationStateInit:
    """Tests for AnnotationState initialization."""

    def test_initializes_with_graph(self, simple_graph):
        """Test that state initializes correctly."""
        state = AnnotationState(simple_graph)
        assert state.original_graph is simple_graph
        assert isinstance(state.gt_graph, nx.DiGraph)
        assert len(state.gt_graph) == 0

    def test_initial_sets_are_empty(self, simple_graph):
        """Test that all tracking sets start empty."""
        state = AnnotationState(simple_graph)
        assert len(state.tp_objects) == 0
        assert len(state.fp_objects) == 0
        assert len(state.fn_objects) == 0
        assert len(state.tp_edges) == 0
        assert len(state.fp_edges) == 0
        assert len(state.fn_edges) == 0


class TestTPNodeOperations:
    """Tests for true positive node operations."""

    def test_add_tp_node_creates_gt_node(self, simple_graph):
        """Test that adding TP node creates it in GT graph."""
        state = AnnotationState(simple_graph)
        location = {"t": 0, "y": 10, "x": 20}

        state.add_or_update_tp_node(
            orig_node_id=0, gt_node_id=0, location_attrs=location
        )

        assert 0 in state.tp_objects
        assert 0 in state.gt_graph.nodes
        assert state.gt_graph.nodes[0]["orig_idx"] == 0
        assert state.gt_graph.nodes[0][TP_NODE_VOTES] == 1
        assert state.gt_graph.nodes[0]["t"] == 0

    def test_add_tp_node_multiple_times_increments_votes(self, simple_graph):
        """Test that adding same TP node increments vote count."""
        state = AnnotationState(simple_graph)
        location = {"t": 0, "y": 10, "x": 20}

        state.add_or_update_tp_node(0, 0, location)
        state.add_or_update_tp_node(0, 0, location)
        state.add_or_update_tp_node(0, 0, location)

        assert state.gt_graph.nodes[0][TP_NODE_VOTES] == 3
        assert 0 in state.tp_objects

    def test_add_tp_node_with_different_gt_id(self, simple_graph):
        """Test TP node where GT ID differs from original ID."""
        state = AnnotationState(simple_graph)
        location = {"t": 0, "y": 10, "x": 20}

        state.add_or_update_tp_node(
            orig_node_id=0, gt_node_id=100, location_attrs=location
        )

        assert 0 in state.tp_objects  # Original ID
        assert 100 in state.gt_graph.nodes  # GT ID
        assert state.gt_graph.nodes[100]["orig_idx"] == 0

    def test_remove_tp_node_decrements_votes(self, simple_graph):
        """Test that removing TP node decrements votes."""
        state = AnnotationState(simple_graph)
        location = {"t": 0, "y": 10, "x": 20}

        state.add_or_update_tp_node(0, 0, location)
        state.add_or_update_tp_node(0, 0, location)
        assert state.gt_graph.nodes[0][TP_NODE_VOTES] == 2

        state.remove_tp_node(0, 0)
        assert state.gt_graph.nodes[0][TP_NODE_VOTES] == 1

    def test_remove_tp_node_removes_when_votes_zero(self, simple_graph):
        """Test that node is removed from GT graph when votes reach zero."""
        state = AnnotationState(simple_graph)
        location = {"t": 0, "y": 10, "x": 20}

        state.add_or_update_tp_node(0, 0, location)
        state.remove_tp_node(0, 0)

        assert 0 not in state.tp_objects
        assert 0 not in state.gt_graph.nodes

    def test_properties_return_copies(self, simple_graph):
        """Test that properties return copies, not references."""
        state = AnnotationState(simple_graph)
        location = {"t": 0, "y": 10, "x": 20}
        state.add_or_update_tp_node(0, 0, location)

        tp_set = state.tp_objects
        tp_set.add(999)  # Modify the returned set

        # Original should be unchanged
        assert 999 not in state.tp_objects


class TestFPNodeOperations:
    """Tests for false positive node operations."""

    def test_add_fp_node_marks_in_original_graph(self, simple_graph):
        """Test that FP node is marked in original graph."""
        state = AnnotationState(simple_graph)

        state.add_fp_node(1)

        assert 1 in state.fp_objects
        assert state.original_graph.nodes[1][FP_NODE_ATTR] is True
        assert 1 not in state.gt_graph.nodes  # Should not be in GT

    def test_remove_fp_node(self, simple_graph):
        """Test removing FP node marking."""
        state = AnnotationState(simple_graph)

        state.add_fp_node(1)
        state.remove_fp_node(1)

        assert 1 not in state.fp_objects
        assert FP_NODE_ATTR not in state.original_graph.nodes[1]


class TestFNNodeOperations:
    """Tests for false negative node operations."""

    def test_add_fn_node_creates_in_gt_graph(self, simple_graph):
        """Test that FN node is created in GT graph."""
        state = AnnotationState(simple_graph)
        location = {"t": 5, "y": 50, "x": 60}

        state.add_fn_node(gt_node_id=100, location_attrs=location)

        assert 100 in state.fn_objects
        assert 100 in state.gt_graph.nodes
        assert state.gt_graph.nodes[100]["orig_idx"] == -1  # No original
        assert state.gt_graph.nodes[100][FN_NODE_ATTR] is True
        assert state.gt_graph.nodes[100]["t"] == 5

    def test_fn_node_not_in_original_graph(self, simple_graph):
        """Test that FN nodes don't affect original graph."""
        state = AnnotationState(simple_graph)
        location = {"t": 5, "y": 50, "x": 60}

        state.add_fn_node(100, location)

        # Original graph should be unchanged
        assert 100 not in state.original_graph.nodes

    def test_remove_fn_node(self, simple_graph):
        """Test removing FN node."""
        state = AnnotationState(simple_graph)
        location = {"t": 5, "y": 50, "x": 60}

        state.add_fn_node(100, location)
        state.remove_fn_node(100)

        assert 100 not in state.fn_objects
        assert 100 not in state.gt_graph.nodes


class TestTPEdgeOperations:
    """Tests for true positive edge operations."""

    def test_add_tp_edge_marks_both_graphs(self, simple_graph):
        """Test that TP edge is marked in both graphs."""
        state = AnnotationState(simple_graph)
        orig_edge = (0, 1)
        gt_edge = (0, 1)

        state.add_tp_edge(orig_edge, gt_edge)

        # Check original graph
        assert orig_edge in state.tp_edges
        assert state.original_graph.edges[orig_edge][TP_EDGE_ATTR] is True
        assert state.original_graph.edges[orig_edge]["gt_src"] == 0
        assert state.original_graph.edges[orig_edge]["gt_tgt"] == 1

        # Check GT graph
        assert state.gt_graph.has_edge(*gt_edge)
        assert state.gt_graph.edges[gt_edge][TP_EDGE_ATTR] is True

    def test_add_tp_edge_with_different_gt_ids(self, simple_graph):
        """Test TP edge where GT IDs differ from original."""
        state = AnnotationState(simple_graph)
        orig_edge = (0, 1)
        gt_edge = (100, 101)

        state.add_tp_edge(orig_edge, gt_edge)

        assert orig_edge in state.tp_edges
        assert state.original_graph.edges[orig_edge]["gt_src"] == 100
        assert state.original_graph.edges[orig_edge]["gt_tgt"] == 101
        assert state.gt_graph.has_edge(100, 101)

    def test_remove_tp_edge(self, simple_graph):
        """Test removing TP edge."""
        state = AnnotationState(simple_graph)
        orig_edge = (0, 1)
        gt_edge = (0, 1)

        state.add_tp_edge(orig_edge, gt_edge)
        state.remove_tp_edge(orig_edge, gt_edge)

        # Check original graph
        assert orig_edge not in state.tp_edges
        assert TP_EDGE_ATTR not in state.original_graph.edges[orig_edge]
        assert "gt_src" not in state.original_graph.edges[orig_edge]

        # Check GT graph
        assert not state.gt_graph.has_edge(*gt_edge)


class TestFPEdgeOperations:
    """Tests for false positive edge operations."""

    def test_add_fp_edge(self, simple_graph):
        """Test adding FP edge."""
        state = AnnotationState(simple_graph)
        edge = (0, 1)

        state.add_fp_edge(edge)

        assert edge in state.fp_edges
        assert state.original_graph.edges[edge][FP_EDGE_ATTR] is True
        # Should not affect GT graph
        assert not state.gt_graph.has_edge(*edge)

    def test_remove_fp_edge(self, simple_graph):
        """Test removing FP edge."""
        state = AnnotationState(simple_graph)
        edge = (0, 1)

        state.add_fp_edge(edge)
        state.remove_fp_edge(edge)

        assert edge not in state.fp_edges
        assert FP_EDGE_ATTR not in state.original_graph.edges[edge]


class TestFNEdgeOperations:
    """Tests for false negative edge operations."""

    def test_add_fn_edge(self, simple_graph):
        """Test adding FN edge."""
        state = AnnotationState(simple_graph)
        # First add nodes to GT graph
        state.add_fn_node(100, {"t": 0, "y": 10, "x": 20})
        state.add_fn_node(101, {"t": 1, "y": 15, "x": 25})

        edge = (100, 101)
        state.add_fn_edge(edge)

        assert edge in state.fn_edges
        assert state.gt_graph.has_edge(*edge)
        assert state.gt_graph.edges[edge][FN_EDGE_ATTR] is True

    def test_add_fn_edge_creates_edge_if_not_exists(self, simple_graph):
        """Test that FN edge is created if it doesn't exist."""
        state = AnnotationState(simple_graph)
        edge = (100, 101)

        state.add_fn_edge(edge)

        assert state.gt_graph.has_edge(*edge)

    def test_remove_fn_edge(self, simple_graph):
        """Test removing FN edge."""
        state = AnnotationState(simple_graph)
        edge = (100, 101)

        state.add_fn_edge(edge)
        state.remove_fn_edge(edge)

        assert edge not in state.fn_edges
        assert not state.gt_graph.has_edge(*edge)


class TestEdgeMetadata:
    """Tests for edge metadata operations."""

    def test_set_and_get_edge_metadata(self, simple_graph):
        """Test setting and getting edge metadata."""
        state = AnnotationState(simple_graph)
        edge = (0, 1)

        state.set_edge_metadata(edge, "seen", True)
        state.set_edge_metadata(edge, "src_present", False)

        assert state.get_edge_metadata(edge, "seen") is True
        assert state.get_edge_metadata(edge, "src_present") is False

    def test_get_edge_metadata_default(self, simple_graph):
        """Test getting metadata with default value."""
        state = AnnotationState(simple_graph)
        edge = (0, 1)

        result = state.get_edge_metadata(edge, "nonexistent", default="default_val")
        assert result == "default_val"

    def test_remove_edge_metadata(self, simple_graph):
        """Test removing edge metadata."""
        state = AnnotationState(simple_graph)
        edge = (0, 1)

        state.set_edge_metadata(edge, "test_key", "test_value")
        state.remove_edge_metadata(edge, "test_key")

        assert state.get_edge_metadata(edge, "test_key") is None


class TestComplexScenarios:
    """Tests for complex multi-operation scenarios."""

    def test_mixed_node_types(self, simple_graph):
        """Test tracking different node types simultaneously."""
        state = AnnotationState(simple_graph)

        # Add TP node
        state.add_or_update_tp_node(0, 0, {"t": 0, "y": 10, "x": 20})
        # Add FP node
        state.add_fp_node(1)
        # Add FN node
        state.add_fn_node(100, {"t": 5, "y": 50, "x": 60})

        assert 0 in state.tp_objects
        assert 1 in state.fp_objects
        assert 100 in state.fn_objects

        # Check graphs
        assert 0 in state.gt_graph.nodes  # TP in GT
        assert 100 in state.gt_graph.nodes  # FN in GT
        assert 1 in state.original_graph.nodes  # FP in original
        assert state.original_graph.nodes[1][FP_NODE_ATTR] is True

    def test_mixed_edge_types(self, simple_graph):
        """Test tracking different edge types simultaneously."""
        state = AnnotationState(simple_graph)

        # Add TP edge
        state.add_tp_edge((0, 1), (0, 1))
        # Add FP edge
        state.add_fp_edge((1, 2))

        assert (0, 1) in state.tp_edges
        assert (1, 2) in state.fp_edges

        assert state.gt_graph.has_edge(0, 1)
        assert not state.gt_graph.has_edge(1, 2)

    def test_node_referenced_multiple_times(self, simple_graph):
        """Test node that participates in multiple edges."""
        state = AnnotationState(simple_graph)
        location = {"t": 1, "y": 15, "x": 25}

        # Node 1 is in two edges
        state.add_or_update_tp_node(1, 1, location)  # From edge (0,1)
        state.add_or_update_tp_node(1, 1, location)  # From edge (1,2)

        assert state.gt_graph.nodes[1][TP_NODE_VOTES] == 2

        # Remove once, should still exist
        state.remove_tp_node(1, 1)
        assert 1 in state.gt_graph.nodes
        assert state.gt_graph.nodes[1][TP_NODE_VOTES] == 1

        # Remove again, should be gone
        state.remove_tp_node(1, 1)
        assert 1 not in state.gt_graph.nodes
