"""Integration tests for TrackAnnotator widget.

These tests focus on napari integration and widget behavior, not the underlying
business logic which is tested separately in the component tests.
"""

import networkx as nx
import numpy as np
import pytest

from tracktour._napari.track_annotator import TrackAnnotator
from tracktour._napari.track_annotator.widget import (
    EDGE_FOCUS_POINT_NAME,
    EDGE_FOCUS_VECTOR_NAME,
)


@pytest.fixture
def simple_segmentation():
    """Create a simple 3D segmentation with a few labeled regions.

    Shape: (3 timepoints, 5, 100, 100) - (t, z, y, x)
    Each timepoint has 2 labeled regions.
    """
    seg = np.zeros((3, 5, 100, 100), dtype=np.uint16)

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

    # Track 1 - labels match segmentation IDs
    G.add_node(1, t=0, z=2, y=3, x=3, label=1)
    G.add_node(3, t=1, z=2, y=3, x=4, label=3)
    G.add_node(5, t=2, z=2, y=3, x=5, label=5)
    G.add_edge(1, 3)
    G.add_edge(3, 5)

    # Track 2
    G.add_node(2, t=0, z=2, y=7, x=7, label=2)
    G.add_node(4, t=1, z=2, y=7, x=8, label=4)
    G.add_node(6, t=2, z=2, y=7, x=9, label=6)
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

    yield viewer, seg_layer, tracks_layer, simple_graph

    viewer.close()


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

        assert EDGE_FOCUS_POINT_NAME in layer_names
        assert EDGE_FOCUS_VECTOR_NAME in layer_names


class TestEdgeNavigation:
    """Tests for edge navigation functionality."""

    def test_can_navigate_to_first_edge(self, widget_with_setup):
        """Test that widget can display the first edge."""
        widget = widget_with_setup

        assert widget._current_display_idx >= 0
        assert widget._edge_sample_order is not None
        assert len(widget._edge_sample_order) == 4

    def test_navigation_updates_camera_center(self, widget_with_setup, simple_graph):
        """Test that camera centers on the edge when navigating."""
        widget = widget_with_setup
        viewer = widget._viewer

        first_edge = widget._edge_sample_order[0]
        src_node, tgt_node = first_edge
        src_data = simple_graph.nodes[src_node]
        tgt_data = simple_graph.nodes[tgt_node]

        expected_center_y = (src_data["y"] + tgt_data["y"]) / 2
        expected_center_x = (src_data["x"] + tgt_data["x"]) / 2

        camera_center = viewer.camera.center
        assert abs(camera_center[-2] - expected_center_y) < 1.0
        assert abs(camera_center[-1] - expected_center_x) < 1.0

        widget._next_edge_button.clicked()

        second_edge = widget._edge_sample_order[1]
        src_node2, tgt_node2 = second_edge
        src_data2 = simple_graph.nodes[src_node2]
        tgt_data2 = simple_graph.nodes[tgt_node2]

        expected_center_y2 = (src_data2["y"] + tgt_data2["y"]) / 2
        expected_center_x2 = (src_data2["x"] + tgt_data2["x"]) / 2

        new_camera_center = viewer.camera.center
        assert abs(new_camera_center[-2] - expected_center_y2) < 1.0
        assert abs(new_camera_center[-1] - expected_center_x2) < 1.0

    def test_navigation_updates_points_layer(self, widget_with_setup, simple_graph):
        """Test that points layer displays correct source and target positions."""
        widget = widget_with_setup
        points_layer = widget._viewer.layers[EDGE_FOCUS_POINT_NAME]

        current_edge = tuple(widget._edge_sample_order[widget._current_display_idx])
        src_node, tgt_node = current_edge
        src_data = simple_graph.nodes[src_node]
        tgt_data = simple_graph.nodes[tgt_node]

        assert len(points_layer.data) == 2

        symbols = list(points_layer.symbol)
        assert "disc" in symbols
        assert "ring" in symbols

        disc_idx = symbols.index("disc")
        ring_idx = symbols.index("ring")

        src_point = points_layer.data[disc_idx]
        tgt_point = points_layer.data[ring_idx]

        # Points drop time dimension, so they're 3D [z, y, x]
        assert src_point[0] == src_data["z"]
        assert src_point[1] == src_data["y"]
        assert src_point[2] == src_data["x"]

        assert tgt_point[0] == tgt_data["z"]
        assert tgt_point[1] == tgt_data["y"]
        assert tgt_point[2] == tgt_data["x"]

        widget._next_edge_button.clicked()

        next_edge = tuple(widget._edge_sample_order[widget._current_display_idx])
        src_node_next, tgt_node_next = next_edge
        src_data_next = simple_graph.nodes[src_node_next]
        tgt_data_next = simple_graph.nodes[tgt_node_next]

        assert len(points_layer.data) == 2

        symbols_next = list(points_layer.symbol)
        disc_idx_next = symbols_next.index("disc")
        ring_idx_next = symbols_next.index("ring")

        src_point_next = points_layer.data[disc_idx_next]
        tgt_point_next = points_layer.data[ring_idx_next]

        assert src_point_next[0] == src_data_next["z"]
        assert src_point_next[1] == src_data_next["y"]
        assert src_point_next[2] == src_data_next["x"]

        assert tgt_point_next[0] == tgt_data_next["z"]
        assert tgt_point_next[1] == tgt_data_next["y"]
        assert tgt_point_next[2] == tgt_data_next["x"]

    def test_navigation_updates_vectors_layer(self, widget_with_setup, simple_graph):
        """Test that vectors layer displays edge from source to target."""
        widget = widget_with_setup
        vectors_layer = widget._viewer.layers[EDGE_FOCUS_VECTOR_NAME]

        current_edge = tuple(widget._edge_sample_order[widget._current_display_idx])
        src_node, tgt_node = current_edge
        src_data = simple_graph.nodes[src_node]
        tgt_data = simple_graph.nodes[tgt_node]

        assert len(vectors_layer.data) == 1

        vector_data = vectors_layer.data[0]
        vector_start = vector_data[0]
        vector_direction = vector_data[1]

        # Vectors drop the time dimension, so they're 3D [z, y, x]
        assert vector_start[0] == src_data["z"]
        assert vector_start[1] == src_data["y"]
        assert vector_start[2] == src_data["x"]

        expected_direction = [
            tgt_data["z"] - src_data["z"],
            tgt_data["y"] - src_data["y"],
            tgt_data["x"] - src_data["x"],
        ]

        assert vector_direction[0] == expected_direction[0]
        assert vector_direction[1] == expected_direction[1]
        assert vector_direction[2] == expected_direction[2]

    def test_next_button_advances_edge(self, widget_with_setup):
        """Test that clicking next button advances to next edge."""
        widget = widget_with_setup
        initial_idx = widget._current_display_idx

        widget._next_edge_button.clicked()

        assert widget._current_display_idx == initial_idx + 1

    def test_previous_button_goes_back(self, widget_with_setup):
        """Test that clicking previous button goes to previous edge."""
        widget = widget_with_setup

        widget._next_edge_button.clicked()
        current_idx = widget._current_display_idx

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


class TestPointEditing:
    """Tests for user editing points in the Source Target layer."""

    def test_deleting_one_point_sets_changed_flag(self, widget_with_setup):
        """Test that deleting one point sets the points_layer_changed flag."""
        widget = widget_with_setup
        points_layer = widget._viewer.layers[EDGE_FOCUS_POINT_NAME]

        assert not widget._points_layer_changed

        # Delete one point by setting data to only one point
        original_data = points_layer.data.copy()
        points_layer.data = original_data[0:1]

        # Flag should be set after layer modification
        assert widget._points_layer_changed

    def test_deleting_one_point_creates_single_node_command(self, widget_with_setup):
        """Test that deleting one point and saving creates single node FP command."""
        widget = widget_with_setup
        points_layer = widget._viewer.layers[EDGE_FOCUS_POINT_NAME]

        current_edge = tuple(widget._edge_sample_order[widget._current_display_idx])

        # Keep only target point (ring)
        symbols = list(points_layer.symbol)
        ring_idx = symbols.index("ring")
        points_layer.data = points_layer.data[ring_idx : ring_idx + 1]
        points_layer.symbol = [symbols[ring_idx]]

        widget._next_edge_button.clicked()

        # Edge should be marked as FP with single node
        assert current_edge in widget._state.fp_edges
        assert widget._controller.is_edge_annotated(current_edge)

    def test_deleting_both_points_marks_edge_fp(self, widget_with_setup):
        """Test that deleting both points marks edge as FP without correction."""
        widget = widget_with_setup
        points_layer = widget._viewer.layers[EDGE_FOCUS_POINT_NAME]

        current_edge = tuple(widget._edge_sample_order[widget._current_display_idx])

        # Remove all points (3D data: [z, y, x])
        points_layer.data = np.empty((0, 3))
        points_layer.symbol = []

        widget._next_edge_button.clicked()

        # Edge should be marked as FP
        assert current_edge in widget._state.fp_edges
        assert widget._controller.is_edge_annotated(current_edge)

    def test_moving_point_sets_changed_flag(self, widget_with_setup):
        """Test that moving a point sets the points_layer_changed flag."""
        widget = widget_with_setup
        points_layer = widget._viewer.layers[EDGE_FOCUS_POINT_NAME]

        assert not widget._points_layer_changed

        # Modify one point's position
        new_data = points_layer.data.copy()
        new_data[0, 1] += 5  # Move y coordinate (3D data: [z, y, x])
        points_layer.data = new_data

        assert widget._points_layer_changed

    def test_moving_point_creates_fp_with_correction(self, widget_with_setup):
        """Test that moving a point and saving creates FP with correction command."""
        widget = widget_with_setup
        points_layer = widget._viewer.layers[EDGE_FOCUS_POINT_NAME]

        current_edge = tuple(widget._edge_sample_order[widget._current_display_idx])

        # Move target point to a new location
        new_data = points_layer.data.copy()
        symbols = list(points_layer.symbol)
        ring_idx = symbols.index("ring")

        new_y = 50
        new_x = 50
        # Points are 3D [z, y, x]
        new_data[ring_idx, 1] = new_y
        new_data[ring_idx, 2] = new_x
        points_layer.data = new_data

        widget._next_edge_button.clicked()

        # Edge should be marked as FP
        assert current_edge in widget._state.fp_edges
        assert widget._controller.is_edge_annotated(current_edge)

        # GT graph should have a corrected node at the new location
        corrected_node_found = False
        for _, node_data in widget._state.gt_graph.nodes(data=True):
            if abs(node_data["y"] - new_y) < 0.1 and abs(node_data["x"] - new_x) < 0.1:
                corrected_node_found = True
                break

        assert corrected_node_found

    def test_moving_point_to_blank_location_creates_fn_node(self, widget_with_setup):
        """Test that moving a point to a blank location creates an FN node."""
        widget = widget_with_setup
        points_layer = widget._viewer.layers[EDGE_FOCUS_POINT_NAME]

        current_edge = tuple(widget._edge_sample_order[widget._current_display_idx])

        # Move target point to a blank location (no segmentation label)
        new_data = points_layer.data.copy()
        symbols = list(points_layer.symbol)
        ring_idx = symbols.index("ring")

        blank_y = 0
        blank_x = 0
        # Points are 3D [z, y, x]
        new_data[ring_idx, 1] = blank_y
        new_data[ring_idx, 2] = blank_x
        points_layer.data = new_data

        widget._next_edge_button.clicked()

        # Edge should be marked as FP
        assert current_edge in widget._state.fp_edges

        # GT graph should have an FN node at the blank location
        fn_node_found = False
        for node_id, node_data in widget._state.gt_graph.nodes(data=True):
            if (
                abs(node_data["y"] - blank_y) < 0.1
                and abs(node_data["x"] - blank_x) < 0.1
            ):
                fn_node_found = True
                assert node_id in widget._state.fn_objects
                break

        assert fn_node_found

    def test_keeping_both_original_points_marks_edge_tp(self, widget_with_setup):
        """Test that keeping both original points marks edge as TP."""
        widget = widget_with_setup

        current_edge = tuple(widget._edge_sample_order[widget._current_display_idx])

        # Don't modify points layer - click next directly
        widget._next_edge_button.clicked()

        # Edge should be marked as TP
        assert current_edge in widget._state.tp_edges
        assert widget._controller.is_edge_annotated(current_edge)
