"""Tests for EdgeAnnotationCommand classes."""

import networkx as nx
import pytest

from tracktour._napari.track_annotator.commands import (
    CompositeCommand,
    MarkEdgeFPCommand,
    MarkEdgeFPWithCorrectionCommand,
    MarkEdgeFPWithSingleNodeCommand,
    MarkEdgeTPCommand,
    MarkNodeFPCommand,
)
from tracktour._napari.track_annotator.state import AnnotationState


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


@pytest.fixture
def state(simple_graph):
    """Create annotation state."""
    return AnnotationState(simple_graph)


class TestMarkEdgeTPCommand:
    """Tests for MarkEdgeTPCommand."""

    def test_execute_marks_edge_and_nodes_tp(self, state):
        """Test that execute marks edge and nodes as TP."""
        cmd = MarkEdgeTPCommand(
            orig_edge=(0, 1),
            gt_edge=(0, 1),
            src_location={"t": 0, "y": 10, "x": 20},
            tgt_location={"t": 1, "y": 15, "x": 25},
        )

        cmd.execute(state)

        assert (0, 1) in state.tp_edges
        assert 0 in state.tp_objects
        assert 1 in state.tp_objects
        assert state.get_edge_metadata((0, 1), "seen") is True

    def test_undo_reverses_execute(self, state):
        """Test that undo completely reverses execute."""
        cmd = MarkEdgeTPCommand(
            orig_edge=(0, 1),
            gt_edge=(0, 1),
            src_location={"t": 0, "y": 10, "x": 20},
            tgt_location={"t": 1, "y": 15, "x": 25},
        )

        # Execute then undo
        cmd.execute(state)
        cmd.undo(state)

        # Should be back to initial state
        assert (0, 1) not in state.tp_edges
        assert 0 not in state.tp_objects
        assert 1 not in state.tp_objects
        assert state.get_edge_metadata((0, 1), "seen") is None
        assert not state.gt_graph.has_edge(0, 1)

    def test_multiple_execute_undo_cycles(self, state):
        """Test that command can be executed and undone multiple times."""
        cmd = MarkEdgeTPCommand(
            orig_edge=(0, 1),
            gt_edge=(0, 1),
            src_location={"t": 0, "y": 10, "x": 20},
            tgt_location={"t": 1, "y": 15, "x": 25},
        )

        for _ in range(3):
            cmd.execute(state)
            assert (0, 1) in state.tp_edges

            cmd.undo(state)
            assert (0, 1) not in state.tp_edges


class TestMarkEdgeFPWithCorrectionCommand:
    """Tests for MarkEdgeFPWithCorrectionCommand."""

    def test_execute_with_two_tp_nodes(self, state):
        """Test execute when both nodes are TP (moved locations)."""
        cmd = MarkEdgeFPWithCorrectionCommand(
            orig_edge=(0, 1),
            gt_edge=(10, 11),
            src_location={"t": 0, "y": 12, "x": 22},  # Moved from original
            tgt_location={"t": 1, "y": 17, "x": 27},
            src_is_fn=False,
            tgt_is_fn=False,
            src_orig_id=0,
            tgt_orig_id=1,
        )

        cmd.execute(state)

        # Original edge marked FP
        assert (0, 1) in state.fp_edges

        # GT edge added as FN
        assert (10, 11) in state.fn_edges

        # Both nodes marked TP in original graph
        assert 0 in state.tp_objects
        assert 1 in state.tp_objects

        # GT nodes created
        assert 10 in state.gt_graph.nodes
        assert 11 in state.gt_graph.nodes

    def test_execute_with_fn_nodes(self, state):
        """Test execute when nodes are FN (new detections)."""
        cmd = MarkEdgeFPWithCorrectionCommand(
            orig_edge=(0, 1),
            gt_edge=(100, 101),
            src_location={"t": 0, "y": 50, "x": 60},
            tgt_location={"t": 1, "y": 55, "x": 65},
            src_is_fn=True,
            tgt_is_fn=True,
            src_orig_id=None,
            tgt_orig_id=None,
        )

        cmd.execute(state)

        # Original edge marked FP
        assert (0, 1) in state.fp_edges

        # FN nodes created
        assert 100 in state.fn_objects
        assert 101 in state.fn_objects

        # FN edge created
        assert (100, 101) in state.fn_edges

    def test_execute_with_mixed_nodes(self, state):
        """Test execute with one TP and one FN node."""
        cmd = MarkEdgeFPWithCorrectionCommand(
            orig_edge=(0, 1),
            gt_edge=(0, 100),  # Keep src, new tgt
            src_location={"t": 0, "y": 10, "x": 20},
            tgt_location={"t": 1, "y": 55, "x": 65},
            src_is_fn=False,
            tgt_is_fn=True,
            src_orig_id=0,
            tgt_orig_id=None,
        )

        cmd.execute(state)

        assert (0, 1) in state.fp_edges
        assert 0 in state.tp_objects
        assert 100 in state.fn_objects
        assert (0, 100) in state.fn_edges

    def test_undo_reverses_execute(self, state):
        """Test that undo completely reverses execute."""
        cmd = MarkEdgeFPWithCorrectionCommand(
            orig_edge=(0, 1),
            gt_edge=(10, 11),
            src_location={"t": 0, "y": 12, "x": 22},
            tgt_location={"t": 1, "y": 17, "x": 27},
            src_is_fn=False,
            tgt_is_fn=False,
            src_orig_id=0,
            tgt_orig_id=1,
        )

        cmd.execute(state)
        cmd.undo(state)

        # Everything should be reverted
        assert (0, 1) not in state.fp_edges
        assert (10, 11) not in state.fn_edges
        assert 0 not in state.tp_objects
        assert 1 not in state.tp_objects
        assert 10 not in state.gt_graph.nodes
        assert 11 not in state.gt_graph.nodes


class TestMarkEdgeFPWithSingleNodeCommand:
    """Tests for MarkEdgeFPWithSingleNodeCommand."""

    def test_execute_with_src_present(self, state):
        """Test execute when source node remains."""
        cmd = MarkEdgeFPWithSingleNodeCommand(
            orig_edge=(0, 1),
            gt_node_id=0,
            remaining_orig_id=0,
            location={"t": 0, "y": 10, "x": 20},
            src_present=True,
        )

        cmd.execute(state)

        assert (0, 1) in state.fp_edges
        assert 0 in state.tp_objects
        assert 1 not in state.tp_objects
        assert state.get_edge_metadata((0, 1), "src_present") is True
        assert state.get_edge_metadata((0, 1), "tgt_present") is False

    def test_execute_with_tgt_present(self, state):
        """Test execute when target node remains."""
        cmd = MarkEdgeFPWithSingleNodeCommand(
            orig_edge=(0, 1),
            gt_node_id=1,
            remaining_orig_id=1,
            location={"t": 1, "y": 15, "x": 25},
            src_present=False,
        )

        cmd.execute(state)

        assert (0, 1) in state.fp_edges
        assert 1 in state.tp_objects
        assert 0 not in state.tp_objects
        assert state.get_edge_metadata((0, 1), "src_present") is False
        assert state.get_edge_metadata((0, 1), "tgt_present") is True

    def test_undo_reverses_execute(self, state):
        """Test that undo completely reverses execute."""
        cmd = MarkEdgeFPWithSingleNodeCommand(
            orig_edge=(0, 1),
            gt_node_id=0,
            remaining_orig_id=0,
            location={"t": 0, "y": 10, "x": 20},
            src_present=True,
        )

        cmd.execute(state)
        cmd.undo(state)

        assert (0, 1) not in state.fp_edges
        assert 0 not in state.tp_objects
        assert state.get_edge_metadata((0, 1), "src_present") is None
        assert state.get_edge_metadata((0, 1), "tgt_present") is None


class TestMarkEdgeFPCommand:
    """Tests for MarkEdgeFPCommand."""

    def test_execute_marks_edge_fp(self, state):
        """Test that execute marks edge as FP."""
        cmd = MarkEdgeFPCommand(orig_edge=(0, 1))

        cmd.execute(state)

        assert (0, 1) in state.fp_edges
        assert state.get_edge_metadata((0, 1), "seen") is True

    def test_undo_reverses_execute(self, state):
        """Test that undo completely reverses execute."""
        cmd = MarkEdgeFPCommand(orig_edge=(0, 1))

        cmd.execute(state)
        cmd.undo(state)

        assert (0, 1) not in state.fp_edges
        assert state.get_edge_metadata((0, 1), "seen") is None


class TestMarkNodeFPCommand:
    """Tests for MarkNodeFPCommand."""

    def test_execute_marks_node_fp(self, state):
        """Test that execute marks node as FP."""
        cmd = MarkNodeFPCommand(node_id=0)

        cmd.execute(state)

        assert 0 in state.fp_objects

    def test_undo_reverses_execute(self, state):
        """Test that undo completely reverses execute."""
        cmd = MarkNodeFPCommand(node_id=0)

        cmd.execute(state)
        cmd.undo(state)

        assert 0 not in state.fp_objects


class TestCompositeCommand:
    """Tests for CompositeCommand."""

    def test_execute_runs_all_subcommands(self, state):
        """Test that all subcommands are executed."""
        cmd = CompositeCommand(
            [
                MarkEdgeFPCommand((0, 1)),
                MarkNodeFPCommand(0),
                MarkNodeFPCommand(1),
            ]
        )

        cmd.execute(state)

        assert (0, 1) in state.fp_edges
        assert 0 in state.fp_objects
        assert 1 in state.fp_objects

    def test_undo_reverses_all_subcommands(self, state):
        """Test that undo reverses all subcommands in reverse order."""
        cmd = CompositeCommand(
            [
                MarkEdgeFPCommand((0, 1)),
                MarkNodeFPCommand(0),
                MarkNodeFPCommand(1),
            ]
        )

        cmd.execute(state)
        cmd.undo(state)

        assert (0, 1) not in state.fp_edges
        assert 0 not in state.fp_objects
        assert 1 not in state.fp_objects

    def test_undo_order_matters(self, state):
        """Test that undo happens in reverse order."""
        # Track order of operations
        operations = []

        class TrackingCommand(MarkNodeFPCommand):
            def __init__(self, node_id, label):
                super().__init__(node_id)
                self.label = label

            def execute(self, state):
                operations.append(f"execute-{self.label}")
                super().execute(state)

            def undo(self, state):
                operations.append(f"undo-{self.label}")
                super().undo(state)

        cmd = CompositeCommand(
            [
                TrackingCommand(0, "A"),
                TrackingCommand(1, "B"),
                TrackingCommand(2, "C"),
            ]
        )

        cmd.execute(state)
        cmd.undo(state)

        # Execute should be A, B, C
        # Undo should be C, B, A (reverse)
        assert operations == [
            "execute-A",
            "execute-B",
            "execute-C",
            "undo-C",
            "undo-B",
            "undo-A",
        ]


class TestCommandIdempotency:
    """Tests for command execute/undo idempotency."""

    def test_multiple_execute_undo_cycles_tp_edge(self, state):
        """Test TP edge command can be cycled multiple times."""
        cmd = MarkEdgeTPCommand(
            orig_edge=(0, 1),
            gt_edge=(0, 1),
            src_location={"t": 0, "y": 10, "x": 20},
            tgt_location={"t": 1, "y": 15, "x": 25},
        )

        for _ in range(5):
            cmd.execute(state)
            assert (0, 1) in state.tp_edges
            cmd.undo(state)
            assert (0, 1) not in state.tp_edges

    def test_state_unchanged_after_undo(self, state):
        """Test that state is identical before and after execute/undo."""
        # Capture initial state
        initial_tp_edges = state.tp_edges.copy()
        initial_fp_edges = state.fp_edges.copy()
        initial_tp_objects = state.tp_objects.copy()

        # Execute and undo command
        cmd = MarkEdgeFPCommand((0, 1))
        cmd.execute(state)
        cmd.undo(state)

        # State should be identical
        assert state.tp_edges == initial_tp_edges
        assert state.fp_edges == initial_fp_edges
        assert state.tp_objects == initial_tp_objects
