"""Integration tests for TrackAnnotator widget.

These tests focus on napari integration and widget behavior, not the underlying
business logic which is tested separately in the component tests.
"""

import networkx as nx
import numpy as np
import pytest

from tracktour._napari.track_annotator import TrackAnnotator


@pytest.fixture
def simple_segmentation():
    """Create a simple 3D segmentation with a few labeled regions.

    Shape: (3 timepoints, 5, 10, 10) - (t, z, y, x)
    Each timepoint has 2 labeled regions.
    """
    seg = np.zeros((3, 5, 10, 10), dtype=np.uint16)

    # Timepoint 0: two objects
    seg[0, 2, 2:4, 2:4] = 1  # object 1 at t=0
    seg[0, 2, 6:8, 6:8] = 2  # object 2 at t=0

    # Timepoint 1: two objects (slightly shifted)
    seg[1, 2, 2:4, 3:5] = 3  # object 3 at t=1
    seg[1, 2, 6:8, 7:9] = 4  # object 4 at t=1

    # Timepoint 2: two objects (slightly shifted again)
    seg[2, 2, 2:4, 4:6] = 5  # object 5 at t=2
    seg[2, 2, 6:8, 8:10] = 6  # object 6 at t=2

    return seg


@pytest.fixture
def simple_graph():
    """Create a simple track graph matching the segmentation.

    Two tracks:
    - Track 1: nodes 1 -> 3 -> 5
    - Track 2: nodes 2 -> 4 -> 6
    """
    G = nx.DiGraph()

    # Track 1
    G.add_node(1, t=0, z=2, y=3, x=3)
    G.add_node(3, t=1, z=2, y=3, x=4)
    G.add_node(5, t=2, z=2, y=3, x=5)
    G.add_edge(1, 3)
    G.add_edge(3, 5)

    # Track 2
    G.add_node(2, t=0, z=2, y=7, x=7)
    G.add_node(4, t=1, z=2, y=7, x=8)
    G.add_node(6, t=2, z=2, y=7, x=9)
    G.add_edge(2, 4)
    G.add_edge(4, 6)

    return G


@pytest.fixture
def tracks_data(simple_graph):
    """Convert graph to napari tracks format.

    Returns tracks array with columns: [track_id, t, z, y, x]
    """
    tracks = []

    # Group nodes by track (using connected components)
    for track_id, nodes in enumerate(nx.weakly_connected_components(simple_graph)):
        for node in sorted(nodes):
            node_data = simple_graph.nodes[node]
            tracks.append(
                [
                    track_id,
                    node_data["t"],
                    node_data["z"],
                    node_data["y"],
                    node_data["x"],
                ]
            )

    return np.array(tracks)


@pytest.fixture
def viewer_with_layers(
    make_napari_viewer, simple_segmentation, tracks_data, simple_graph
):
    """Create viewer with segmentation and tracks layers."""
    viewer = make_napari_viewer()

    # Add segmentation layer
    seg_layer = viewer.add_labels(simple_segmentation, name="Segmentation")

    # Add tracks layer
    tracks_layer = viewer.add_tracks(tracks_data, name="Tracks")

    # Store the graph in layer metadata (this is what the widget expects)
    tracks_layer.metadata["nxg"] = simple_graph

    return viewer, seg_layer, tracks_layer, simple_graph


@pytest.fixture
def widget_with_setup(viewer_with_layers):
    """Create TrackAnnotator widget with layers set up and annotation layers initialized."""
    viewer, seg_layer, tracks_layer, graph = viewer_with_layers
    widget = TrackAnnotator(viewer)

    # Explicitly set the combo box values to trigger layer setup
    # Setting value triggers the changed signal which calls _setup_annotation_layers
    widget._seg_combo.value = seg_layer
    widget._track_combo.value = tracks_layer

    # Verify that annotation layers were created
    assert widget._state is not None
    assert widget._controller is not None

    return widget


class TestWidgetInitialization:
    """Tests for widget initialization and setup."""

    def test_widget_initializes_state_when_layers_selected(self, widget_with_setup):
        """Test that selecting both layers initializes state and controller."""
        widget = widget_with_setup

        assert widget._state is not None
        assert widget._controller is not None
        assert widget._state.original_graph is not None

        # Check that the graph has the expected structure
        assert len(widget._state.original_graph.nodes) == 6
        assert len(widget._state.original_graph.edges) == 4

    def test_widget_creates_annotation_layers(self, widget_with_setup):
        """Test that annotation layers are created in the viewer."""
        widget = widget_with_setup
        viewer = widget._viewer

        layer_names = [layer.name for layer in viewer.layers]

        assert "Source Target" in layer_names
        assert "Current Edge" in layer_names


class TestEdgeNavigation:
    """Tests for edge navigation functionality."""

    def test_can_navigate_to_first_edge(self, widget_with_setup):
        """Test that widget can display the first edge."""
        widget = widget_with_setup

        # The widget should be showing an edge after initialization
        assert widget._current_display_idx >= 0
        assert widget._edge_sample_order is not None
        assert len(widget._edge_sample_order) == 4  # 4 edges in our graph

    def test_next_button_advances_edge(self, widget_with_setup):
        """Test that clicking next button advances to next edge."""
        widget = widget_with_setup
        initial_idx = widget._current_display_idx

        # Simulate clicking next button
        widget._next_edge_button.clicked()

        assert widget._current_display_idx == initial_idx + 1

    def test_previous_button_goes_back(self, widget_with_setup):
        """Test that clicking previous button goes to previous edge."""
        widget = widget_with_setup

        # First advance to second edge
        widget._next_edge_button.clicked()
        current_idx = widget._current_display_idx

        # Then go back
        widget._previous_edge_button.clicked()

        assert widget._current_display_idx == current_idx - 1

    def test_navigation_buttons_disabled_at_boundaries(self, widget_with_setup):
        """Test that navigation buttons are properly disabled at boundaries."""
        widget = widget_with_setup
        num_edges = len(widget._edge_sample_order)

        # Widget should start at first edge - previous should be disabled
        assert widget._current_display_idx == 0
        assert not widget._previous_edge_button.enabled

        # Navigate to second-to-last edge by clicking next repeatedly
        for _ in range(num_edges - 2):
            widget._next_edge_button.clicked()

        # Should now be at second-to-last edge, button still says "Save && Next"
        assert widget._current_display_idx == num_edges - 2
        assert widget._next_edge_button.text == "Save && Next"

        # Click next once more to reach last edge
        widget._next_edge_button.clicked()

        # Button text should change to "Save && Finish" at last edge
        assert widget._current_display_idx == num_edges - 1
        assert widget._next_edge_button.text == "Save && Finish"

        # Click "Save && Finish" button
        widget._next_edge_button.clicked()

        # After clicking at last edge, button should be disabled
        assert not widget._next_edge_button.enabled
