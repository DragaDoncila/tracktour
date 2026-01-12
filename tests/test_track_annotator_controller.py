"""Tests for AnnotationController."""

import networkx as nx
import pytest

from tracktour._napari.track_annotator.commands import (
    MarkEdgeFPCommand,
    MarkEdgeTPCommand,
    MarkNodeFPCommand,
)
from tracktour._napari.track_annotator.controller import (
    AnnotationController,
    AnnotationError,
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
def controller(simple_graph):
    """Create annotation controller."""
    state = AnnotationState(simple_graph)
    return AnnotationController(state)


class TestControllerInit:
    """Tests for controller initialization."""

    def test_initializes_with_state(self, controller):
        """Test that controller initializes correctly."""
        assert isinstance(controller.state, AnnotationState)
        assert len(controller.edge_commands) == 0

    def test_no_edges_annotated_initially(self, controller):
        """Test that no edges are marked as annotated initially."""
        assert not controller.is_edge_annotated((0, 1))
        assert len(controller.get_annotated_edges()) == 0


class TestAnnotateEdge:
    """Tests for annotate_edge method."""

    def test_annotate_edge_executes_command(self, controller):
        """Test that annotating an edge executes the command."""
        cmd = MarkEdgeTPCommand(
            orig_edge=(0, 1),
            gt_edge=(0, 1),
            src_location={"t": 0, "y": 10, "x": 20},
            tgt_location={"t": 1, "y": 15, "x": 25},
        )

        controller.annotate_edge((0, 1), cmd)

        assert (0, 1) in controller.state.tp_edges
        assert controller.is_edge_annotated((0, 1))

    def test_annotate_edge_stores_command(self, controller):
        """Test that command is stored for later retrieval."""
        cmd = MarkEdgeTPCommand(
            orig_edge=(0, 1),
            gt_edge=(0, 1),
            src_location={"t": 0, "y": 10, "x": 20},
            tgt_location={"t": 1, "y": 15, "x": 25},
        )

        controller.annotate_edge((0, 1), cmd)

        retrieved_cmd = controller.get_edge_command((0, 1))
        assert retrieved_cmd is cmd

    def test_reannotate_edge_undoes_previous(self, controller):
        """Test that re-annotating undoes the previous annotation."""
        # First annotation
        cmd1 = MarkEdgeTPCommand(
            orig_edge=(0, 1),
            gt_edge=(0, 1),
            src_location={"t": 0, "y": 10, "x": 20},
            tgt_location={"t": 1, "y": 15, "x": 25},
        )
        controller.annotate_edge((0, 1), cmd1)
        assert (0, 1) in controller.state.tp_edges

        # Second annotation (different)
        cmd2 = MarkEdgeFPCommand((0, 1))
        controller.annotate_edge((0, 1), cmd2)

        # First should be undone, second should be active
        assert (0, 1) not in controller.state.tp_edges
        assert (0, 1) in controller.state.fp_edges
        assert controller.get_edge_command((0, 1)) is cmd2

    def test_annotate_multiple_edges(self, controller):
        """Test annotating multiple different edges."""
        cmd1 = MarkEdgeTPCommand(
            orig_edge=(0, 1),
            gt_edge=(0, 1),
            src_location={"t": 0, "y": 10, "x": 20},
            tgt_location={"t": 1, "y": 15, "x": 25},
        )
        cmd2 = MarkEdgeFPCommand((1, 2))

        controller.annotate_edge((0, 1), cmd1)
        controller.annotate_edge((1, 2), cmd2)

        assert controller.is_edge_annotated((0, 1))
        assert controller.is_edge_annotated((1, 2))
        assert len(controller.get_annotated_edges()) == 2


class TestResetEdgeToOriginal:
    """Tests for reset_edge_to_original method."""

    def test_reset_undoes_annotation(self, controller):
        """Test that reset undoes the annotation."""
        cmd = MarkEdgeTPCommand(
            orig_edge=(0, 1),
            gt_edge=(0, 1),
            src_location={"t": 0, "y": 10, "x": 20},
            tgt_location={"t": 1, "y": 15, "x": 25},
        )
        controller.annotate_edge((0, 1), cmd)

        result = controller.reset_edge_to_original((0, 1))

        assert result is True
        assert (0, 1) not in controller.state.tp_edges
        assert not controller.is_edge_annotated((0, 1))

    def test_reset_removes_command(self, controller):
        """Test that reset removes the command from tracking."""
        cmd = MarkEdgeTPCommand(
            orig_edge=(0, 1),
            gt_edge=(0, 1),
            src_location={"t": 0, "y": 10, "x": 20},
            tgt_location={"t": 1, "y": 15, "x": 25},
        )
        controller.annotate_edge((0, 1), cmd)

        controller.reset_edge_to_original((0, 1))

        assert controller.get_edge_command((0, 1)) is None
        assert (0, 1) not in controller.get_annotated_edges()

    def test_reset_unannotated_edge_returns_false(self, controller):
        """Test that resetting an unannotated edge returns False."""
        result = controller.reset_edge_to_original((0, 1))
        assert result is False

    def test_reset_doesnt_affect_other_edges(self, controller):
        """Test that resetting one edge doesn't affect others."""
        cmd1 = MarkEdgeTPCommand(
            orig_edge=(0, 1),
            gt_edge=(0, 1),
            src_location={"t": 0, "y": 10, "x": 20},
            tgt_location={"t": 1, "y": 15, "x": 25},
        )
        cmd2 = MarkEdgeFPCommand((1, 2))

        controller.annotate_edge((0, 1), cmd1)
        controller.annotate_edge((1, 2), cmd2)

        controller.reset_edge_to_original((0, 1))

        # Edge (1, 2) should still be annotated
        assert controller.is_edge_annotated((1, 2))
        assert (1, 2) in controller.state.fp_edges


class TestIsEdgeAnnotated:
    """Tests for is_edge_annotated method."""

    def test_unannotated_edge_returns_false(self, controller):
        """Test that unannotated edges return False."""
        assert not controller.is_edge_annotated((0, 1))

    def test_annotated_edge_returns_true(self, controller):
        """Test that annotated edges return True."""
        cmd = MarkEdgeFPCommand((0, 1))
        controller.annotate_edge((0, 1), cmd)
        assert controller.is_edge_annotated((0, 1))

    def test_reset_edge_returns_false(self, controller):
        """Test that reset edges return False."""
        cmd = MarkEdgeFPCommand((0, 1))
        controller.annotate_edge((0, 1), cmd)
        controller.reset_edge_to_original((0, 1))
        assert not controller.is_edge_annotated((0, 1))


class TestGetAnnotatedEdges:
    """Tests for get_annotated_edges method."""

    def test_returns_empty_set_initially(self, controller):
        """Test that initially returns empty set."""
        assert len(controller.get_annotated_edges()) == 0

    def test_returns_all_annotated_edges(self, controller):
        """Test that all annotated edges are returned."""
        cmd1 = MarkEdgeFPCommand((0, 1))
        cmd2 = MarkEdgeFPCommand((1, 2))

        controller.annotate_edge((0, 1), cmd1)
        controller.annotate_edge((1, 2), cmd2)

        edges = controller.get_annotated_edges()
        assert edges == {(0, 1), (1, 2)}

    def test_excludes_reset_edges(self, controller):
        """Test that reset edges are not included."""
        cmd1 = MarkEdgeFPCommand((0, 1))
        cmd2 = MarkEdgeFPCommand((1, 2))

        controller.annotate_edge((0, 1), cmd1)
        controller.annotate_edge((1, 2), cmd2)
        controller.reset_edge_to_original((0, 1))

        edges = controller.get_annotated_edges()
        assert edges == {(1, 2)}


class TestClearAllAnnotations:
    """Tests for clear_all_annotations method."""

    def test_clears_all_annotations(self, controller):
        """Test that all annotations are cleared."""
        cmd1 = MarkEdgeTPCommand(
            orig_edge=(0, 1),
            gt_edge=(0, 1),
            src_location={"t": 0, "y": 10, "x": 20},
            tgt_location={"t": 1, "y": 15, "x": 25},
        )
        cmd2 = MarkEdgeFPCommand((1, 2))

        controller.annotate_edge((0, 1), cmd1)
        controller.annotate_edge((1, 2), cmd2)

        controller.clear_all_annotations()

        assert len(controller.get_annotated_edges()) == 0
        assert (0, 1) not in controller.state.tp_edges
        assert (1, 2) not in controller.state.fp_edges

    def test_clears_command_tracking(self, controller):
        """Test that command tracking is cleared."""
        cmd = MarkEdgeFPCommand((0, 1))
        controller.annotate_edge((0, 1), cmd)

        controller.clear_all_annotations()

        assert controller.get_edge_command((0, 1)) is None
        assert len(controller.edge_commands) == 0


class TestComplexScenarios:
    """Tests for complex annotation workflows."""

    def test_annotate_reset_reannotate(self, controller):
        """Test annotating, resetting, and re-annotating the same edge."""
        # First annotation
        cmd1 = MarkEdgeTPCommand(
            orig_edge=(0, 1),
            gt_edge=(0, 1),
            src_location={"t": 0, "y": 10, "x": 20},
            tgt_location={"t": 1, "y": 15, "x": 25},
        )
        controller.annotate_edge((0, 1), cmd1)
        assert (0, 1) in controller.state.tp_edges

        # Reset
        controller.reset_edge_to_original((0, 1))
        assert (0, 1) not in controller.state.tp_edges

        # Re-annotate differently
        cmd2 = MarkEdgeFPCommand((0, 1))
        controller.annotate_edge((0, 1), cmd2)
        assert (0, 1) in controller.state.fp_edges

    def test_non_sequential_edge_annotation(self, controller):
        """Test annotating edges in non-sequential order."""
        # Annotate edge (1, 2) first
        cmd1 = MarkEdgeFPCommand((1, 2))
        controller.annotate_edge((1, 2), cmd1)

        # Then annotate edge (0, 1)
        cmd2 = MarkEdgeTPCommand(
            orig_edge=(0, 1),
            gt_edge=(0, 1),
            src_location={"t": 0, "y": 10, "x": 20},
            tgt_location={"t": 1, "y": 15, "x": 25},
        )
        controller.annotate_edge((0, 1), cmd2)

        # Then reset (1, 2)
        controller.reset_edge_to_original((1, 2))

        # Edge (0, 1) should still be annotated
        assert controller.is_edge_annotated((0, 1))
        assert not controller.is_edge_annotated((1, 2))

    def test_multiple_modifications_same_edge(self, controller):
        """Test modifying the same edge multiple times."""
        edge = (0, 1)

        # Annotation 1: TP
        cmd1 = MarkEdgeTPCommand(
            orig_edge=edge,
            gt_edge=edge,
            src_location={"t": 0, "y": 10, "x": 20},
            tgt_location={"t": 1, "y": 15, "x": 25},
        )
        controller.annotate_edge(edge, cmd1)
        assert (0, 1) in controller.state.tp_edges

        # Annotation 2: FP
        cmd2 = MarkEdgeFPCommand(edge)
        controller.annotate_edge(edge, cmd2)
        assert (0, 1) not in controller.state.tp_edges
        assert (0, 1) in controller.state.fp_edges

        # Annotation 3: TP again
        cmd3 = MarkEdgeTPCommand(
            orig_edge=edge,
            gt_edge=edge,
            src_location={"t": 0, "y": 10, "x": 20},
            tgt_location={"t": 1, "y": 15, "x": 25},
        )
        controller.annotate_edge(edge, cmd3)
        assert (0, 1) not in controller.state.fp_edges
        assert (0, 1) in controller.state.tp_edges

        # Final command should be cmd3
        assert controller.get_edge_command(edge) is cmd3
