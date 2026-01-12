import glob
import os
import warnings

import networkx as nx
import numpy as np
from magicgui.widgets import FileEdit, PushButton, create_widget
from napari.utils.notifications import show_info
from qtpy.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ._graph_conversion_util import get_nxg_from_tracks, get_tracks_from_nxg
from .track_annotator.commands import (
    CompositeCommand,
    EdgeAnnotationCommand,
    MarkEdgeFPCommand,
    MarkEdgeFPWithCorrectionCommand,
    MarkEdgeFPWithSingleNodeCommand,
    MarkEdgeTPCommand,
    MarkNodeFPCommand,
)
from .track_annotator.controller import AnnotationController
from .track_annotator.state import AnnotationState
from .track_annotator.utils import (
    get_count_label_from_grid,
    get_counts_grid_layout,
    get_int_loc,
    get_loc_array,
    get_loc_dict,
    get_region_center,
    get_separator_widget,
    get_src_tgt_idx,
    split_coords,
)

EDGE_FOCUS_POINT_NAME = "Source Target"
EDGE_FOCUS_VECTOR_NAME = "Current Edge"
GT_TRACKS_NAME = "Ground Truth Tracks"

POINT_IN_FRAME_COLOR = [0.816, 0.337, 0.933, 1]
POINT_OUT_FRAME_COLOR = [0.816, 0.337, 0.933, 0.3]
VECTOR_COLOR = [1, 1, 1, 1]

FP_EDGE_ATTR = "tracktour_annotated_fp"
FN_EDGE_ATTR = "tracktour_annotated_fn"
TP_EDGE_ATTR = "tracktour_annotated_tp"

FP_NODE_ATTR = "tracktour_annotated_fp"
FN_NODE_ATTR = "tracktour_annotated_fn"
TP_NODE_VOTES = "tracktour_annotated_tp_votes"

OUT_GT_GRAPH_NAME = "tracktour_gt_graph"
OUT_SOL_GRAPH_NAME = "tracktour_original_graph"


class TrackAnnotator(QWidget):
    def __init__(
        self,
        viewer: "napari.viewer.Viewer",
    ) -> None:
        super().__init__()

        # initialize bits and pieces for tracking state
        self._viewer = viewer
        self._edge_sample_order = None
        self._current_display_idx = 0
        self._points_layer_changed = False

        # Annotation state and controller (created when layers are selected)
        self._state: AnnotationState = None
        self._controller: AnnotationController = None

        # layer selection widgets
        self._seg_combo = create_widget(
            annotation="napari.layers.Labels", label="Segmentation"
        )
        self._track_combo = create_widget(
            annotation="napari.layers.Tracks", label="Tracks"
        )
        self._seg_combo.changed.connect(self._setup_annotation_layers)
        self._track_combo.changed.connect(self._setup_annotation_layers)
        self._viewer.layers.events.inserted.connect(self._seg_combo.reset_choices)
        self._viewer.layers.events.removed.connect(self._seg_combo.reset_choices)
        self._viewer.layers.events.inserted.connect(self._track_combo.reset_choices)
        self._viewer.layers.events.removed.connect(self._track_combo.reset_choices)

        # widgets for navigating edges
        self.base_layout = QVBoxLayout()

        self._edge_control_layout = QHBoxLayout()
        self._previous_edge_button = PushButton(text="Previous")
        self._previous_edge_button.enabled = False

        self._reset_edge_button = PushButton(text="Reset Current Edge")
        self._next_edge_button = PushButton(text="Save && Next")

        self._next_edge_button.clicked.connect(self._display_next_edge)
        self._previous_edge_button.clicked.connect(self._display_previous_edge)
        self._reset_edge_button.clicked.connect(self._reset_current_edge)

        self._edge_control_layout.addWidget(self._previous_edge_button.native)
        self._edge_control_layout.addWidget(self._reset_edge_button.native)
        self._edge_control_layout.addWidget(self._next_edge_button.native)

        # widgets for current edge status and reset button
        self._edge_status_layout = QHBoxLayout()
        self._edge_status_layout.addWidget(QLabel("Edge Status: "))

        # widgets for behaviour config
        self._warn_on_orphan_node_checkbox = create_widget(
            annotation=bool, label="Warn on orphan node"
        )

        self._edge_status_label = QLabel("New")
        self._edge_status_layout.addWidget(self._edge_status_label)
        self._reset_to_original_button = PushButton(text="Reset to Original")
        self._reset_to_original_button.enabled = False
        self._reset_to_original_button.clicked.connect(self._reset_to_original_edge)
        self._edge_status_layout.addWidget(self._reset_to_original_button.native)

        # widgets for counts
        self._counts_grid_layout = get_counts_grid_layout()

        # widgets for export
        self._view_ground_truth_button = PushButton(text="View Ground Truth")

        self._export_layout = QHBoxLayout()
        self._export_path = FileEdit(
            mode="d", filter="*.tracktour", label="Export Path"
        )
        self._save_annotations_button = PushButton(text="Save Annotations")
        self._save_annotations_button.clicked.connect(self._save_annotated_graphs)
        self._save_project_button = PushButton(text="Save Project")
        self._view_ground_truth_button.clicked.connect(self._add_ground_truth_tracks)

        self._export_layout.addWidget(self._save_annotations_button.native)
        self._export_layout.addWidget(self._save_project_button.native)

        # add everything to the layout
        self.base_layout.addWidget(self._seg_combo.native)
        self.base_layout.addWidget(self._track_combo.native)
        self.base_layout.addWidget(get_separator_widget())
        self.base_layout.addLayout(self._edge_control_layout)
        self.base_layout.addLayout(self._edge_status_layout)
        self.base_layout.addWidget(self._warn_on_orphan_node_checkbox.native)
        self.base_layout.addWidget(get_separator_widget())
        self.base_layout.addLayout(self._counts_grid_layout)
        self.base_layout.addWidget(self._view_ground_truth_button.native)
        self.base_layout.addWidget(get_separator_widget())
        self.base_layout.addWidget(self._export_path.native)
        self.base_layout.addLayout(self._export_layout)
        self.setLayout(self.base_layout)

        self._setup_annotation_layers()

    def _setup_annotation_layers(self):
        if self._seg_combo.value is None or self._track_combo.value is None:
            self._next_edge_button.enabled = False
            self._previous_edge_button.enabled = False
            return

        # get original graph and create state/controller
        nxg = self._get_original_nxg()
        self._state = AnnotationState(nxg)
        self._controller = AnnotationController(self._state)

        # get a sample order going
        self._edge_sample_order = np.random.RandomState(seed=0).permutation(
            np.asarray(nxg.edges)
        )

        # reset display index
        self._current_display_idx = 0
        self._display_edge(self._current_display_idx)

        self._next_edge_button.enabled = True

        # connect current step event to colour the points layer appropriately
        self._viewer.dims.events.current_step.connect(self._handle_current_step_change)

    def _check_valid_layers(self):
        if self._seg_combo.value is None:
            raise ValueError("No segmentation layer selected")
        if self._track_combo.value is None:
            raise ValueError("No tracks layer selected")

    def _get_original_nxg(self):
        tracks_layer = self._track_combo.value
        if "nxg" in tracks_layer.metadata:
            self._original_nxg = tracks_layer.metadata["nxg"]
            return tracks_layer.metadata["nxg"]
        nxg = get_nxg_from_tracks(tracks_layer)
        seg = self._seg_combo.value.data
        node_labels = {
            node_id: seg.data[tuple(get_int_loc(get_loc_array(node_info)))]
            for node_id, node_info in nxg.nodes(data=True)
        }
        nx.set_node_attributes(nxg, node_labels, "label")
        self._original_nxg = nxg
        return nxg

    def _get_region_bbox(self, loc1, loc2):
        """
        Get the bounding box containing the region of both segments
        """
        seg = self._seg_combo.value.data
        loc1_index = np.round(loc1).astype(int)
        loc2_index = np.round(loc2).astype(int)
        first_label = seg[tuple(loc1_index)]
        second_label = seg[tuple(loc2_index)]
        frame_with_label1 = seg[loc1_index[0]] == first_label
        frame_with_label2 = seg[loc2_index[0]] == second_label

        min1, max1 = np.argwhere(frame_with_label1).min(axis=0), np.argwhere(
            frame_with_label1
        ).max(axis=0)
        min2, max2 = np.argwhere(frame_with_label2).min(axis=0), np.argwhere(
            frame_with_label2
        ).max(axis=0)

        bounding_min = np.minimum(min1, min2) - 10
        bounding_max = np.maximum(max1, max2) + 10
        return bounding_min, bounding_max

    def _handle_current_step_change(self, event):
        if EDGE_FOCUS_POINT_NAME not in self._viewer.layers:
            return
        points_layer = self._viewer.layers[EDGE_FOCUS_POINT_NAME]
        src_loc, tgt_loc = self._get_edge_locs()
        slider_dims = self._viewer.dims.not_displayed
        src_slider_points = np.round(src_loc[list(slider_dims)]).astype(int)
        tgt_slider_points = np.round(tgt_loc[list(slider_dims)]).astype(int)
        current_slider_step = np.asarray([event.value[i] for i in slider_dims])
        current_symbols = points_layer.symbol
        face_colors = []
        for symbol in current_symbols:
            if symbol == "disc" and np.allclose(src_slider_points, current_slider_step):
                face_colors.append(POINT_IN_FRAME_COLOR)
            elif symbol == "ring" and np.allclose(
                tgt_slider_points, current_slider_step
            ):
                face_colors.append(POINT_IN_FRAME_COLOR)
            else:
                face_colors.append(POINT_OUT_FRAME_COLOR)
        if len(face_colors):
            points_layer.face_color = face_colors

    def _add_current_edge_focus_point(self, src, tgt):
        # we drop the first dimension of the points, which is the frame
        src_proj = src[1:]
        tgt_proj = tgt[1:]
        points_data = np.vstack([src_proj, tgt_proj])
        points_symbols = ["disc", "ring"]
        points_face_color = [POINT_IN_FRAME_COLOR, POINT_OUT_FRAME_COLOR]

        point_focus_layer = self._display_points_layer(
            points_data, points_symbols, points_face_color
        )

        vectors_data = [[src_proj, tgt_proj - src_proj]]
        vectors_style = "arrow"
        vectors_color = VECTOR_COLOR
        vectors_width = 0.3
        vectors_scale = self._seg_combo.value.scale[1:]
        if EDGE_FOCUS_VECTOR_NAME in self._viewer.layers:
            edge_focus_layer = self._viewer.layers[EDGE_FOCUS_VECTOR_NAME]
            edge_focus_layer.data = vectors_data
        else:
            edge_focus_layer = self._viewer.add_vectors(
                vectors_data,
                name=EDGE_FOCUS_VECTOR_NAME,
                edge_width=vectors_width,
                edge_color=vectors_color,
                vector_style=vectors_style,
                scale=vectors_scale,
            )
        point_focus_layer.selected_data.clear()
        point_focus_layer.mode = "SELECT"
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="Public access to Window.qt_viewer is deprecated"
            )
            self._viewer.window.qt_viewer.dims.setFocus()
        self._viewer.layers.selection.active = point_focus_layer

    def _get_edge_locs(self, current_edge_idx=None):
        if current_edge_idx is None:
            current_edge_idx = self._current_display_idx
        current_edge = self._edge_sample_order[current_edge_idx]
        nxg = self._state.original_graph
        src_idx = current_edge[0]
        tgt_idx = current_edge[1]
        # locating the source and target nodes, ignoring track_id
        src_loc = get_loc_array(nxg.nodes[src_idx])
        tgt_loc = get_loc_array(nxg.nodes[tgt_idx])
        return src_loc, tgt_loc

    def _display_points_layer(self, points_data, points_symbols, points_face_color):
        self._points_layer_changed = False
        # need to match the scale of the segmentation layer
        current_scale = self._seg_combo.value.scale[1:]
        if EDGE_FOCUS_POINT_NAME in self._viewer.layers:
            self._viewer.layers[EDGE_FOCUS_POINT_NAME].events.data.disconnect(
                self._handle_points_change
            )
            point_focus_layer = self._viewer.layers[EDGE_FOCUS_POINT_NAME]
            point_focus_layer.data = points_data
            # point_focus_layer.scale = current_scale
            if len(points_data):
                point_focus_layer.symbol = points_symbols
                point_focus_layer.face_color = points_face_color

        else:
            point_focus_layer = self._viewer.add_points(
                points_data,
                name=EDGE_FOCUS_POINT_NAME,
                size=2,
                symbol=points_symbols,
                face_color=points_face_color,
                scale=current_scale,
            )
        point_focus_layer.events.data.connect(self._handle_points_change)
        self._viewer.layers.selection.active = point_focus_layer
        point_focus_layer.selected_data.clear()
        point_focus_layer.mode = "SELECT"
        return point_focus_layer

    def _handle_points_change(self, event):
        if event.action == "changed" or event.action == "removed":
            self._points_layer_changed = True
        points_layer = self._viewer.layers[EDGE_FOCUS_POINT_NAME]
        if event.action == "changed" and len(points_layer.data) == 2:
            if EDGE_FOCUS_VECTOR_NAME not in self._viewer.layers:
                return
            # moved a point, need to update the vectors layer
            src_idx, tgt_idx = get_src_tgt_idx(points_layer.symbol)
            if src_idx is not None and tgt_idx is not None:
                src_loc = points_layer.data[src_idx]
                tgt_loc = points_layer.data[tgt_idx]
                vectors_data = [[src_loc, tgt_loc - src_loc]]
                self._viewer.layers[EDGE_FOCUS_VECTOR_NAME].data = vectors_data
        if event.action == "removed":
            if EDGE_FOCUS_VECTOR_NAME in self._viewer.layers:
                self._viewer.layers[EDGE_FOCUS_VECTOR_NAME].data = []

    def _display_edge(self, current_edge_idx):
        current_scale = self._seg_combo.value.scale[1:]
        src_loc, tgt_loc = self._get_edge_locs(current_edge_idx)
        # bbox = self._get_region_bbox(src_loc, tgt_loc)
        self._add_current_edge_focus_point(src_loc, tgt_loc)
        self._setup_display_options(src_loc, tgt_loc, current_scale)
        self._edge_status_label.setText("New")
        self._reset_to_original_button.enabled = False

    def _display_gt_edge(self, current_edge_idx):
        edge_info = self._state.original_graph.edges[
            self._edge_sample_order[current_edge_idx]
        ]
        original_src_loc, original_tgt_loc = self._get_edge_locs(current_edge_idx)
        current_scale = self._seg_combo.value.scale[1:]
        edge_label = "Seen"
        # there's a gt edge we can display here, let's do that
        if "gt_src" in edge_info and "gt_tgt" in edge_info:
            gt_edge_idx = (edge_info["gt_src"], edge_info["gt_tgt"])
            src_loc = get_loc_array(self._state.gt_graph.nodes[gt_edge_idx[0]])
            tgt_loc = get_loc_array(self._state.gt_graph.nodes[gt_edge_idx[1]])
            if not np.allclose(src_loc, original_src_loc) or not np.allclose(
                tgt_loc, original_tgt_loc
            ):
                edge_label = "Edited"
            self._add_current_edge_focus_point(src_loc, tgt_loc)
            self._setup_display_options(src_loc, tgt_loc, current_scale)
        # there's no edge, but we may want to display some points, with no edge
        else:
            edge_label = "Edited"
            points_data = []
            points_symbols = []
            points_face_colors = []
            camera_center = get_region_center(
                original_src_loc[1:], original_tgt_loc[1:]
            )
            current_step = original_src_loc
            if (src_present := edge_info.get("src_present", False)) or edge_info.get(
                "tgt_present", False
            ):
                loc = original_src_loc if src_present else original_tgt_loc
                symbol = "disc" if src_present else "ring"
                points_data.append(loc[1:])
                points_symbols.append(symbol)
                points_face_colors.append(POINT_IN_FRAME_COLOR)
                camera_center = loc[1:]
                current_step = loc
            else:
                # nothing to display...
                edge_label = "Deleted"
            self._display_points_layer(points_data, points_symbols, points_face_colors)
            self._setup_display_options(
                original_src_loc,
                original_tgt_loc,
                current_scale,
                current_step,
                camera_center,
            )
            # if there was a vector, we need to clear its data because there's no vector
            if EDGE_FOCUS_VECTOR_NAME in self._viewer.layers:
                self._viewer.layers[EDGE_FOCUS_VECTOR_NAME].data = []
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="Public access to Window.qt_viewer is deprecated"
            )
            self._viewer.window.qt_viewer.dims.setFocus()
        self._edge_status_label.setText(edge_label)
        self._reset_to_original_button.enabled = edge_label in {"Edited", "Deleted"}

    def _setup_display_options(
        self, src_loc, tgt_loc, current_scale, current_step=None, camera_center=None
    ):
        # we have a 3D + t layer, so we need to make sure
        # the user can see the edge in the frame
        if len(self._seg_combo.value.data.shape) == 4:
            self._setup_thick_slicing(src_loc, tgt_loc, current_scale)
        if current_step is None:
            current_step = src_loc
        if camera_center is None:
            # get center of the region for zooming
            camera_center = get_region_center(src_loc[1:], tgt_loc[1:])
        self._viewer.camera.center = np.multiply(camera_center, current_scale)
        # TODO: should be dynamic based on data...
        self._viewer.camera.zoom = 30
        self._viewer.dims.current_step = current_step

    def _setup_thick_slicing(self, src_loc, tgt_loc, current_scale):
        """Turn on out_of_slice display for points and vectors.

        Set up thick slicing for the image and labels.
        """
        # the thickness needs to be twice the scaled distance between the z
        # coordinate of the two points so that when
        # any of the points is in the "middle" of the slice, we still
        # see the other point
        required_z_thickness = (
            2 * (np.abs(src_loc[1] - tgt_loc[1]) + 1) * current_scale[0]
        )
        self._viewer.dims.thickness = (1, required_z_thickness, 1, 1)
        self._viewer.layers[EDGE_FOCUS_POINT_NAME].projection_mode = "ALL"
        self._viewer.layers[EDGE_FOCUS_VECTOR_NAME].projection_mode = "ALL"
        for layer in self._viewer.layers:
            if layer.visible:
                if type(layer).__name__ == "Image":
                    layer.projection_mode = "MEAN"

    def _display_next_edge(self):
        edge_saved = self._save_edge_annotation()
        if self._current_display_idx == len(self._edge_sample_order) - 1:
            self._finish_annotating()
            return

        if edge_saved:
            self._current_display_idx += 1
        if (
            "seen"
            in self._state.original_graph.edges[
                self._edge_sample_order[self._current_display_idx]
            ]
        ):
            self._display_gt_edge(self._current_display_idx)
        else:
            self._display_edge(self._current_display_idx)

        # next button needs to be enabled current edge is not the last edge
        self._next_edge_button.enabled = self._current_display_idx < len(
            self._edge_sample_order
        )
        if self._current_display_idx == len(self._edge_sample_order) - 1:
            self._next_edge_button.text = "Save && Finish"

        # previous button needs to be enabled if current edge is not the first edge
        self._previous_edge_button.enabled = self._current_display_idx > 0

    def _display_previous_edge(self):
        self._current_display_idx -= 1
        if (
            "seen"
            in self._state.original_graph.edges[
                self._edge_sample_order[self._current_display_idx]
            ]
        ):
            self._display_gt_edge(self._current_display_idx)
        else:
            self._display_edge(self._current_display_idx)
        self._next_edge_button.enabled = True

        # next button needs to be enabled current edge is not the last edge
        self._next_edge_button.enabled = (
            self._current_display_idx < len(self._edge_sample_order) - 1
        )
        # previous button needs to be enabled if current edge is not the first edge
        self._previous_edge_button.enabled = self._current_display_idx > 0

    def _finish_annotating(self):
        self._next_edge_button.enabled = False
        self._previous_edge_button.enabled = False
        self._reset_edge_button.enabled = False
        self._reset_to_original_button.enabled = False
        if EDGE_FOCUS_POINT_NAME in self._viewer.layers:
            self._viewer.layers[EDGE_FOCUS_POINT_NAME].events.data.disconnect(
                self._handle_points_change
            )
            self._viewer.dims.events.current_step.disconnect(
                self._handle_current_step_change
            )
            self._viewer.layers.remove(EDGE_FOCUS_POINT_NAME)
        if EDGE_FOCUS_VECTOR_NAME in self._viewer.layers:
            self._viewer.layers.remove(EDGE_FOCUS_VECTOR_NAME)
        self._points_layer_changed = False

    def _get_original_matching_node(self, frame, label):
        """
        Get the original node index that matches the frame and label
        """
        nxg = self._state.original_graph
        for node in nxg.nodes:
            node_info = nxg.nodes[node]
            if node_info["t"] == frame and node_info["label"] == label:
                return node
        raise ValueError(f"No node found with frame {frame} and label {label}")

    def _get_node_is_orphan(self, node_id):
        nxg = self._state.original_graph
        for in_edge in nxg.in_edges(node_id):
            if not nxg.edges[in_edge].get(FP_EDGE_ATTR, False):
                return False
        for out_edge in nxg.out_edges(node_id):
            if not nxg.edges[out_edge].get(FP_EDGE_ATTR, False):
                return False
        return True

    # === New helper methods for command-based refactoring ===

    def _get_next_gt_node_id(self) -> int:
        """Get the next available GT node ID."""
        if len(self._state.gt_graph.nodes) == 0:
            return 0
        return max(self._state.gt_graph.nodes) + 1

    def _find_existing_gt_node(self, orig_node_id: int) -> int | None:
        """Find GT node that corresponds to orig_node_id, if it exists."""
        for gt_node_id in self._state.gt_graph.nodes:
            if self._state.gt_graph.nodes[gt_node_id].get("orig_idx") == orig_node_id:
                return gt_node_id
        return None

    def _get_gt_node_info_for_command(
        self, node_points_idx: int
    ) -> tuple[int, dict, bool, int | None]:
        """Extract node information for command building.

        This method determines:
        1. What GT node ID to use
        2. The node's location attributes
        3. Whether it's an FN (new detection) or TP
        4. The original node ID (if TP)

        Parameters
        ----------
        node_points_idx : int
            Index into points layer data

        Returns
        -------
        tuple of (gt_node_id, location_attrs, is_fn, orig_node_id)
            gt_node_id: ID to use in GT graph
            location_attrs: dict with t, y, x, (optionally z)
            is_fn: True if this is a false negative (new detection)
            orig_node_id: Original node ID if TP, None if FN
        """
        points_layer = self._viewer.layers[EDGE_FOCUS_POINT_NAME]
        original_src_loc, original_tgt_loc = self._get_edge_locs()
        original_loc = (
            original_src_loc
            if points_layer.symbol[node_points_idx] == "disc"
            else original_tgt_loc
        )

        # Get annotated location
        annotated_loc = points_layer.data[node_points_idx]
        annotated_loc = np.concatenate([[original_loc[0]], annotated_loc])

        # Check what detection this point is over
        annotated_point_seg_idx = get_int_loc(annotated_loc)
        seg_layer = self._seg_combo.value
        annotated_label = seg_layer.data[tuple(annotated_point_seg_idx)]

        location_attrs = split_coords(annotated_loc)

        if annotated_label == 0:
            # FN: Point is over blank space (new detection)
            # Need new GT node ID
            gt_node_id = self._get_next_gt_node_id()
            return (gt_node_id, location_attrs, True, None)
        else:
            # TP: Point is over a detection
            node_frame = annotated_point_seg_idx[0]
            orig_node_id = self._get_original_matching_node(node_frame, annotated_label)

            # Check if this orig_node already has a GT node
            gt_node_id = self._find_existing_gt_node(orig_node_id)
            if gt_node_id is None:
                # TODO: shouldn't this be a brand new ID?
                gt_node_id = orig_node_id  # Use same ID for first occurrence

            # Get location from original graph
            nxg = self._state.original_graph
            location_attrs = get_loc_dict(nxg.nodes[orig_node_id])

            return (gt_node_id, location_attrs, False, orig_node_id)

    def _check_for_orphan_nodes(self, edge: tuple[int, int]) -> list[int]:
        """Check which nodes from edge are now orphaned.

        Returns list of orphaned node IDs.
        """
        orphans = []
        if self._get_node_is_orphan(edge[0]):
            orphans.append(edge[0])
        if self._get_node_is_orphan(edge[1]):
            orphans.append(edge[1])
        return orphans

    # === Command builder methods ===

    def _build_tp_edge_command(
        self,
        orig_edge: tuple[int, int],
        src_loc: np.ndarray,
        tgt_loc: np.ndarray,
    ) -> EdgeAnnotationCommand:
        """Build command for TP edge (both points unchanged)."""
        # Use same GT IDs as original IDs for TP
        gt_edge = orig_edge
        src_location = split_coords(src_loc)
        tgt_location = split_coords(tgt_loc)

        return MarkEdgeTPCommand(
            orig_edge=orig_edge,
            gt_edge=gt_edge,
            src_location=src_location,
            tgt_location=tgt_location,
        )

    def _build_fp_with_correction_command(
        self,
        orig_edge: tuple[int, int],
        src_idx: int,
        tgt_idx: int,
    ) -> EdgeAnnotationCommand:
        """Build command for FP edge with corrected FN edge."""
        # Extract GT node info (includes determining GT IDs)
        gt_src_id, src_loc, src_is_fn, src_orig_id = self._get_gt_node_info_for_command(
            src_idx
        )
        gt_tgt_id, tgt_loc, tgt_is_fn, tgt_orig_id = self._get_gt_node_info_for_command(
            tgt_idx
        )

        # Create main command
        main_cmd = MarkEdgeFPWithCorrectionCommand(
            orig_edge=orig_edge,
            gt_edge=(gt_src_id, gt_tgt_id),
            src_location=src_loc,
            tgt_location=tgt_loc,
            src_is_fn=src_is_fn,
            tgt_is_fn=tgt_is_fn,
            src_orig_id=src_orig_id,
            tgt_orig_id=tgt_orig_id,
        )

        # Check for orphan nodes
        orphans = self._check_for_orphan_nodes(orig_edge)

        if orphans:
            # Wrap in composite command
            commands = [main_cmd]
            for orphan_id in orphans:
                if self._warn_on_orphan_node_checkbox.value:
                    show_info("Orphan node detected. Marking as FP.")
                commands.append(MarkNodeFPCommand(orphan_id))
            return CompositeCommand(commands)
        else:
            return main_cmd

    def _build_single_point_command(
        self, orig_edge: tuple[int, int]
    ) -> EdgeAnnotationCommand:
        """Build command for single remaining point."""
        points_layer = self._viewer.layers[EDGE_FOCUS_POINT_NAME]
        src_present = points_layer.symbol[0] == "disc"

        # Get GT node info for remaining point
        gt_node_id, location, is_fn, orig_id = self._get_gt_node_info_for_command(0)

        # Note: Single point should never be FN (we validated it hasn't moved)
        assert not is_fn, "Single remaining point should be TP"

        main_cmd = MarkEdgeFPWithSingleNodeCommand(
            orig_edge=orig_edge,
            gt_node_id=gt_node_id,
            remaining_orig_id=orig_id,
            location=location,
            src_present=src_present,
        )

        # Check for orphan
        orphan_id = orig_edge[1] if src_present else orig_edge[0]
        if self._get_node_is_orphan(orphan_id):
            if self._warn_on_orphan_node_checkbox.value:
                show_info("Orphan node detected. Marking as FP.")
            return CompositeCommand([main_cmd, MarkNodeFPCommand(orphan_id)])
        else:
            return main_cmd

    def _build_no_points_command(
        self, orig_edge: tuple[int, int]
    ) -> EdgeAnnotationCommand:
        """Build command for deleted edge (no points)."""
        main_cmd = MarkEdgeFPCommand(orig_edge)

        # Check both nodes for orphans
        orphans = self._check_for_orphan_nodes(orig_edge)

        if orphans:
            commands = [main_cmd]
            for orphan_id in orphans:
                if self._warn_on_orphan_node_checkbox.value:
                    show_info("Orphan node detected. Marking as FP.")
                commands.append(MarkNodeFPCommand(orphan_id))
            return CompositeCommand(commands)
        else:
            return main_cmd

    def _save_edge_annotation(self):
        seg = self._seg_combo.value.data
        original_edge = self._edge_sample_order[self._current_display_idx]
        original_edge = (int(original_edge[0]), int(original_edge[1]))
        original_src_loc, original_tgt_loc = self._get_edge_locs()
        original_src_label = seg[tuple(get_int_loc(original_src_loc))]
        original_tgt_label = seg[tuple(get_int_loc(original_tgt_loc))]
        points_layer = self._viewer.layers[EDGE_FOCUS_POINT_NAME]

        has_two_original_points = False
        num_points = len(points_layer.data)
        # check if the state of the points layer is valid
        if num_points > 2:
            show_info("More than two points in the current edge. Resetting edge.")
            return False
        # points data is either 2 or 1
        if num_points == 2:
            src_idx, tgt_idx = get_src_tgt_idx(points_layer.symbol)
            if src_idx is None or tgt_idx is None:
                return False
            point_one_loc = np.concatenate(
                [[original_src_loc[0]], points_layer.data[src_idx]]
            )
            point_two_loc = np.concatenate(
                [[original_tgt_loc[0]], points_layer.data[tgt_idx]]
            )
            point_one_moved = (
                seg[tuple(get_int_loc(point_one_loc))] != original_src_label
            )
            point_two_moved = (
                seg[tuple(get_int_loc(point_two_loc))] != original_tgt_label
            )
            # if both points have changed, we need to warn
            if point_one_moved and point_two_moved:
                show_info("Both points have changed. Resetting edge.")
                return False
            if not point_one_moved and not point_two_moved:
                has_two_original_points = True
        # single point in the data
        elif num_points == 1:
            if points_layer.symbol[0] != "disc" and points_layer.symbol[0] != "ring":
                show_info(
                    "Remaining point is neither source nor target. Did you change symbols? Resetting edge."
                )
                return False
            if points_layer.symbol[0] == "disc":
                full_point_loc = np.concatenate(
                    [[original_src_loc[0]], points_layer.data[0]]
                )
                point_moved = (
                    seg[tuple(get_int_loc(full_point_loc))] != original_src_label
                )
            else:
                full_point_loc = np.concatenate(
                    [[original_tgt_loc[0]], points_layer.data[0]]
                )
                point_moved = (
                    seg[tuple(get_int_loc(full_point_loc))] != original_tgt_label
                )
            if point_moved:
                show_info("Both points have changed. Resetting edge.")
                return False

        # Early exit if no changes on seen edge
        if (not self._points_layer_changed) and self._controller.is_edge_annotated(
            original_edge
        ):
            return True

        # Build appropriate command based on scenario
        if num_points == 2:
            src_idx, tgt_idx = get_src_tgt_idx(points_layer.symbol)
            if has_two_original_points:
                # Scenario 1: TP Edge
                command = self._build_tp_edge_command(
                    original_edge, original_src_loc, original_tgt_loc
                )
            else:
                # Scenario 2: FP/FN Edge with correction
                command = self._build_fp_with_correction_command(
                    original_edge, src_idx, tgt_idx
                )
        elif num_points == 1:
            # Scenario 3: Single point
            command = self._build_single_point_command(original_edge)
        elif num_points == 0:
            # Scenario 4: No points
            command = self._build_no_points_command(original_edge)

        # Execute command via controller
        try:
            self._controller.annotate_edge(original_edge, command)
            self._update_label_displays()
            self._points_layer_changed = False
            return True
        except Exception as e:
            show_info(f"Failed to save annotation: {e}")
            return False

    def _update_label_displays(self):
        get_count_label_from_grid(self._counts_grid_layout, "TPO").setText(
            str(len(self._state.tp_objects))
        )
        get_count_label_from_grid(self._counts_grid_layout, "FPO").setText(
            str(len(self._state.fp_objects))
        )
        get_count_label_from_grid(self._counts_grid_layout, "FNO").setText(
            str(len(self._state.fn_objects))
        )
        get_count_label_from_grid(self._counts_grid_layout, "TPT").setText(
            str(len(self._state.tp_edges))
        )
        get_count_label_from_grid(self._counts_grid_layout, "FPT").setText(
            str(len(self._state.fp_edges))
        )
        get_count_label_from_grid(self._counts_grid_layout, "FNT").setText(
            str(len(self._state.fn_edges))
        )

    def _reset_current_edge(self):
        if (
            "seen"
            in self._state.original_graph.edges[
                self._edge_sample_order[self._current_display_idx]
            ]
        ):
            self._display_gt_edge(self._current_display_idx)
        else:
            self._display_edge(self._current_display_idx)

    def _reset_to_original_edge(self):
        original_edge = self._edge_sample_order[self._current_display_idx]
        original_edge = (int(original_edge[0]), int(original_edge[1]))
        # Use controller to reset the edge
        self._controller.reset_edge_to_original(original_edge)
        # Update display
        self._update_label_displays()
        self._display_edge(self._current_display_idx)
        self._reset_to_original_button.enabled = False
        self._edge_status_label.setText("New")

    def _add_ground_truth_tracks(self):
        if (
            self._state is None
            or len(self._state.gt_graph.nodes) == 0
            or len(self._state.gt_graph.edges) == 0
        ):
            show_info("No ground truth tracks to display. Try annotating some edges!")
            return
        seg_layer_scale = self._seg_combo.value.scale
        tracks = get_tracks_from_nxg(self._state.gt_graph)
        tracks.name = GT_TRACKS_NAME
        tracks.scale = seg_layer_scale
        if GT_TRACKS_NAME in self._viewer.layers:
            gt_tracks_layer = self._viewer.layers[GT_TRACKS_NAME]
            gt_tracks_layer.data = tracks.data
            gt_tracks_layer.graph = tracks.graph
            gt_tracks_layer.scale = seg_layer_scale
        else:
            self._viewer.add_layer(tracks)

    def _save_annotated_graphs(self):
        dir_path = self._export_path.value
        if (
            self._state is None
            or len(self._state.gt_graph.nodes) == 0
            or len(self._state.gt_graph.edges) == 0
        ):
            show_info("Nothing so save. Try annotating some edges!")
            return
        gt_path = os.path.join(dir_path, f"{OUT_GT_GRAPH_NAME}.graphml")
        sol_path = os.path.join(dir_path, f"{OUT_SOL_GRAPH_NAME}.graphml")
        if os.path.exists(gt_path):
            show_info("Ground truth graph already exists. Saving new graph.")
            already_saved = glob.glob(
                os.path.join(dir_path, f"{OUT_GT_GRAPH_NAME}*.graphml")
            )
            num_saved = len(already_saved)
            gt_path = os.path.join(dir_path, f"{OUT_GT_GRAPH_NAME}_{num_saved}.graphml")
            sol_path = os.path.join(
                dir_path, f"{OUT_SOL_GRAPH_NAME}_{num_saved}.graphml"
            )
        # delete and then restore color_array to avoid error in saving
        sol_nxg = self._state.original_graph
        colors = {
            node: sol_nxg.nodes[node].pop("color", None) for node in sol_nxg.nodes
        }

        nx.write_graphml(self._state.gt_graph, gt_path)
        nx.write_graphml(self._state.original_graph, sol_path)
        nx.set_node_attributes(sol_nxg, colors, "color")
        print("Saved")
