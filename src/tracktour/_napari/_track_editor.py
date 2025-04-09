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


def get_separator_widget():
    separator = QFrame()
    separator.setMinimumWidth(1)
    separator.setFixedHeight(5)
    separator.setLineWidth(2)
    separator.setMidLineWidth(2)
    separator.setFrameShape(QFrame.HLine)
    separator.setFrameShadow(QFrame.Sunken)
    return separator


def get_counts_grid_layout():
    text_labels = [
        ["TP Object: ", "FP Object: ", "FN Object: "],
        ["TP Track: ", "FP Track: ", "FN Track: "],
    ]
    grid_layout = QGridLayout()
    for row in range(2):
        for col in range(0, 6, 2):
            label = QLabel(text_labels[row][col // 2])
            label.setSizePolicy(
                QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding
            )
            grid_layout.addWidget(label, row, col)
            label = QLabel("0")
            label.setSizePolicy(
                QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding
            )
            grid_layout.addWidget(label, row, col + 1)
    return grid_layout


def get_count_label_from_grid(grid_layout, label):
    if label == "TPO":
        return grid_layout.itemAtPosition(0, 1).widget()
    if label == "FPO":
        return grid_layout.itemAtPosition(0, 3).widget()
    if label == "FNO":
        return grid_layout.itemAtPosition(0, 5).widget()
    if label == "TPT":
        return grid_layout.itemAtPosition(1, 1).widget()
    if label == "FPT":
        return grid_layout.itemAtPosition(1, 3).widget()
    if label == "FNT":
        return grid_layout.itemAtPosition(1, 5).widget()


class TrackAnnotator(QWidget):
    def __init__(
        self,
        viewer: "napari.viewer.Viewer",
    ) -> None:
        super().__init__()

        # initialize bits and pieces for tracking state
        self._viewer = viewer
        self._edge_sample_order = None
        self._original_nxg = None
        self._current_display_idx = 0

        self._gt_nxg = None
        self._points_layer_changed = False

        # IDs into original nxg
        self._tp_objects: set[int] = set()
        self._fp_objects: set[int] = set()
        self._tp_edges: set[tuple[int, int]] = set()
        self._fp_edges: set[tuple[int, int]] = set()

        # IDs into gt nxg
        self._fn_objects: set[int] = set()
        self._fn_edges: set[tuple[int, int]] = set()

        # dict to track actions for undo
        self._edge_actions: dict[
            tuple[int, int], dict[str, int | tuple[int, int]]
        ] = dict()

        # layer selection widgets
        self._seg_combo = create_widget(
            annotation="napari.layers.Labels", label="Segmentation"
        )
        self._track_combo = create_widget(
            annotation="napari.layers.Tracks", label="Tracks"
        )
        self._seg_combo.changed.connect(self._setup_annotation_layers)
        self._track_combo.changed.connect(self._setup_annotation_layers)

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
        self.base_layout.addWidget(get_separator_widget())
        self.base_layout.addLayout(self._edge_status_layout)
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

        # get a sample order going
        nxg = self._get_original_nxg()
        self._edge_sample_order = np.random.RandomState(seed=0).permutation(
            np.asarray(nxg.edges)
        )

        # start building a GT graph
        self._gt_nxg = nx.DiGraph()

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
        if self._original_nxg is not None:
            return self._original_nxg
        tracks_layer = self._track_combo.value
        if "nxg" in tracks_layer.metadata:
            self._original_nxg = tracks_layer.metadata["nxg"]
            return tracks_layer.metadata["nxg"]
        nxg = get_nxg_from_tracks(tracks_layer)
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
        src_t = src_loc[0]
        tgt_t = tgt_loc[0]
        current_step = event.value
        current_symbols = points_layer.symbol
        current_t = current_step[0]
        face_colors = []
        for symbol in current_symbols:
            if symbol == "disc" and src_t == current_t:
                face_colors.append(POINT_IN_FRAME_COLOR)
            elif symbol == "ring" and tgt_t == current_t:
                face_colors.append(POINT_IN_FRAME_COLOR)
            else:
                face_colors.append(POINT_OUT_FRAME_COLOR)
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
        self._viewer.layers.selection.active = point_focus_layer
        point_focus_layer.selected_data.clear()
        point_focus_layer.mode = "SELECT"
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="Public access to Window.qt_viewer is deprecated"
            )
            self._viewer.window.qt_viewer.dims.setFocus()

    def _get_edge_locs(self, current_edge_idx=None):
        if current_edge_idx is None:
            current_edge_idx = self._current_display_idx
        current_edge = self._edge_sample_order[current_edge_idx]
        nxg = self._get_original_nxg()
        src_idx = current_edge[0]
        tgt_idx = current_edge[1]
        # locating the source and target nodes, ignoring track-id
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
        # get center of the region for zooming
        center = get_region_center(src_loc[1:], tgt_loc[1:])
        # bbox = self._get_region_bbox(src_loc, tgt_loc)
        self._add_current_edge_focus_point(src_loc, tgt_loc)
        self._viewer.camera.center = np.multiply(center, current_scale)
        # TODO: should be dynamic based on data...
        self._viewer.camera.zoom = 70
        self._viewer.dims.current_step = (src_loc[0], 0, 0)
        self._edge_status_label.setText("New")
        self._reset_to_original_button.enabled = False

    def _display_gt_edge(self, current_edge_idx):
        edge_info = self._get_original_nxg().edges[
            self._edge_sample_order[current_edge_idx]
        ]
        original_src_loc, original_tgt_loc = self._get_edge_locs(current_edge_idx)
        current_scale = self._seg_combo.value.scale[1:]
        edge_label = "Seen"
        # there's a gt edge we can display here, let's do that
        if "gt_src" in edge_info and "gt_tgt" in edge_info:
            gt_edge_idx = (edge_info["gt_src"], edge_info["gt_tgt"])
            src_loc = get_loc_array(self._gt_nxg.nodes[gt_edge_idx[0]])
            tgt_loc = get_loc_array(self._gt_nxg.nodes[gt_edge_idx[1]])
            if not np.allclose(src_loc, original_src_loc) or not np.allclose(
                tgt_loc, original_tgt_loc
            ):
                edge_label = "Edited"
            center = get_region_center(src_loc[1:], tgt_loc[1:])
            self._add_current_edge_focus_point(src_loc, tgt_loc)
            self._viewer.camera.center = np.multiply(center, current_scale)
            self._viewer.camera.zoom = 70
            self._viewer.dims.current_step = (src_loc[0], 0, 0)
        # there's no edge, but we may want to display some points, with no edge
        else:
            edge_label = "Edited"
            points_data = []
            points_symbols = []
            points_face_colors = []
            camera_center = get_region_center(
                original_src_loc[1:], original_tgt_loc[1:]
            )
            if edge_info["src_present"]:
                points_data.append(original_src_loc[1:])
                points_symbols.append("disc")
                points_face_colors.append(POINT_IN_FRAME_COLOR)
                camera_center = original_src_loc[1:]
                current_step = original_src_loc[0]
            if edge_info["tgt_present"]:
                points_data.append(original_tgt_loc[1:])
                points_symbols.append("ring")
                points_face_colors.append(POINT_OUT_FRAME_COLOR)
                camera_center = original_tgt_loc[1:]
                current_step = original_tgt_loc[0]
            points_focus_layer = self._display_points_layer(
                points_data, points_symbols, points_face_colors
            )
            self._viewer.camera.center = np.multiply(camera_center, current_scale)
            self._viewer.camera.zoom = 70
            self._viewer.dims.current_step = (current_step, 0, 0)
            self._viewer.layers.selection.active = points_focus_layer
            points_focus_layer.selected_data.clear()
            points_focus_layer.mode = "SELECT"
            # if there was a vector, we need to clear its data because there's no vector
            if EDGE_FOCUS_VECTOR_NAME in self._viewer.layers:
                self._viewer.layers[EDGE_FOCUS_VECTOR_NAME].data = []
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="Public access to Window.qt_viewer is deprecated"
            )
            self._viewer.window.qt_viewer.dims.setFocus()
        self._edge_status_label.setText(edge_label)
        self._reset_to_original_button.enabled = edge_label == "Edited"

    def _display_next_edge(self):
        edge_saved = self._save_edge_annotation()
        if edge_saved:
            self._current_display_idx += 1
        if (
            "seen"
            in self._get_original_nxg().edges[
                self._edge_sample_order[self._current_display_idx]
            ]
        ):
            self._display_gt_edge(self._current_display_idx)
        else:
            self._display_edge(self._current_display_idx)

        # next button needs to be enabled current edge is not the last edge
        self._next_edge_button.enabled = (
            self._current_display_idx < len(self._edge_sample_order) - 1
        )
        # previous button needs to be enabled if current edge is not the first edge
        self._previous_edge_button.enabled = self._current_display_idx > 0

    def _display_previous_edge(self):
        self._current_display_idx -= 1
        if (
            "seen"
            in self._get_original_nxg().edges[
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

    def get_new_node_index(self, node_attrs):
        """
        Get new index for node, or index of existing node if it exists.
        """
        new_idx = max(self._gt_nxg.nodes) + 1 if len(self._gt_nxg.nodes) else 0
        # if it was an FN node, we don't want to use its original index
        if node_attrs.get(FN_NODE_ATTR, False):
            # TODO: need to check here if this FN node hasn't already been added...
            return new_idx
        for node in self._gt_nxg.nodes:
            if self._gt_nxg.nodes[node]["orig_idx"] == node_attrs["orig_idx"]:
                return node
        return new_idx

    def _get_original_matching_node(self, frame, label):
        """
        Get the original node index that matches the frame and label
        """
        nxg = self._get_original_nxg()
        for node in nxg.nodes:
            node_info = nxg.nodes[node]
            if node_info["t"] == frame and node_info["label"] == label:
                return node
        raise ValueError(f"No node found with frame {frame} and label {label}")

    def get_gt_node_attrs(self, src_points_idx, tgt_points_idx):
        nxg = self._get_original_nxg()
        gt_src_attrs = {}
        gt_tgt_attrs = {}

        points_layer = self._viewer.layers[EDGE_FOCUS_POINT_NAME]
        original_src_loc, original_tgt_loc = self._get_edge_locs()

        annotated_src_loc = points_layer.data[src_points_idx]
        annotated_tgt_loc = points_layer.data[tgt_points_idx]
        annotated_src_loc = np.concatenate([[original_src_loc[0]], annotated_src_loc])
        annotated_tgt_loc = np.concatenate([[original_tgt_loc[0]], annotated_tgt_loc])

        # check if either of the points are over blank space
        annotated_src_idx = get_int_loc(annotated_src_loc)
        annotated_tgt_idx = get_int_loc(annotated_tgt_loc)
        seg_layer = self._seg_combo.value
        src_label = seg_layer.data[tuple(annotated_src_idx)]
        tgt_label = seg_layer.data[tuple(annotated_tgt_idx)]
        if src_label == 0:
            # source was a previous FN
            gt_src_attrs[FN_NODE_ATTR] = True
            # mark the tgt TP
            gt_tgt_attrs[TP_NODE_VOTES] = 1
            # give src a non-existent original index
            gt_src_attrs["orig_idx"] = -1
            # give src the location of the "added" point
            gt_src_attrs.update(split_coords(annotated_src_loc))
            # give tgt the location and index of the original point
            tgt_frame = annotated_tgt_idx[0]
            orig_tgt_idx = self._get_original_matching_node(tgt_frame, tgt_label)
            gt_tgt_attrs["orig_idx"] = orig_tgt_idx
            gt_tgt_attrs.update(get_loc_dict(nxg.nodes[orig_tgt_idx]))
        elif tgt_label == 0:
            # target was a previous FN
            gt_tgt_attrs[FN_NODE_ATTR] = True
            # mark the src TP
            gt_src_attrs[TP_NODE_VOTES] = 1
            # give tgt a non-existent original index
            gt_tgt_attrs["orig_idx"] = -1
            # give tgt the location of the "added" point
            gt_tgt_attrs.update(split_coords(annotated_tgt_loc))
            # give src the location and index of the original point
            src_frame = annotated_src_idx[0]
            orig_src_idx = self._get_original_matching_node(src_frame, src_label)
            gt_src_attrs["orig_idx"] = orig_src_idx
            gt_src_attrs.update(get_loc_dict(nxg.nodes[orig_src_idx]))
        else:
            # both nodes already existed and still do, so they are TP
            gt_src_attrs[TP_NODE_VOTES] = 1
            gt_tgt_attrs[TP_NODE_VOTES] = 1
            src_frame = annotated_src_idx[0]
            orig_src_idx = self._get_original_matching_node(src_frame, src_label)
            tgt_frame = annotated_tgt_idx[0]
            orig_tgt_idx = self._get_original_matching_node(tgt_frame, tgt_label)
            gt_src_attrs["orig_idx"] = orig_src_idx
            gt_tgt_attrs["orig_idx"] = orig_tgt_idx
            gt_src_attrs.update(get_loc_dict(nxg.nodes[orig_src_idx]))
            gt_tgt_attrs.update(get_loc_dict(nxg.nodes[orig_tgt_idx]))
        return gt_src_attrs, gt_tgt_attrs

    def _save_edge_annotation(self):
        seg = self._seg_combo.value.data
        original_edge = self._edge_sample_order[self._current_display_idx]
        original_edge = (int(original_edge[0]), int(original_edge[1]))
        original_src_loc, original_tgt_loc = self._get_edge_locs()
        original_src_label = seg[tuple(get_int_loc(original_src_loc))]
        original_tgt_label = seg[tuple(get_int_loc(original_tgt_loc))]
        points_layer = self._viewer.layers[EDGE_FOCUS_POINT_NAME]
        nxg = self._get_original_nxg()

        has_two_original_points = False
        num_points = len(points_layer.data)
        # check if the state of the points layer is valid
        if num_points > 2:
            show_info("More than two points in the current edge. Resetting edge.")
            return False
        # TODO: is this actually ok?
        if num_points == 0:
            show_info("No points in the current edge. Resetting edge.")
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
        else:
            if points_layer.symbol[0] != "disc" and points_layer.symbol[0] != "ring":
                show_info(
                    "Remaining point is neither source nor target. Did you change symbols? Resetting edge."
                )
                return False

        # points layer didn't change and we've seen it before, nothing to do
        if (not self._points_layer_changed) and "seen" in nxg.edges[original_edge]:
            print("No changes on seen edge")
            return True
        # if we've seen the edge before and there's been changes, we need to undo
        if "seen" in nxg.edges[original_edge] and self._points_layer_changed:
            print("undoing edge actions")
            self._undo_edge_actions(original_edge)

        actions = {
            "added_to_fpo": [],
            "added_to_fno": [],
            "added_to_tpo": [],
            "added_to_tpe": [],
            "added_to_fne": [],
            "added_to_fpe": [],
        }
        # save the edge
        if num_points == 2:
            if has_two_original_points:
                # this is a TP edge
                gt_edge_attr = {TP_EDGE_ATTR: True}
                self._tp_edges.add(original_edge)
                actions["added_to_tpe"].append(original_edge)
                self._tp_objects.update({original_edge[0], original_edge[1]})
                actions["added_to_tpo"].extend([original_edge[0], original_edge[1]])
            else:
                # one of the points must have moved, we have an FP/FN edge
                nxg.edges[original_edge][FP_EDGE_ATTR] = True
                gt_edge_attr = {FN_EDGE_ATTR: True}
                self._fp_edges.add(original_edge)
                actions["added_to_fpe"].append(original_edge)
            # add the GT edge
            gt_src_attrs, gt_tgt_attrs = self.get_gt_node_attrs(src_idx, tgt_idx)
            # add nodes with new indices, add edge
            src_index = self.get_new_node_index(gt_src_attrs)
            if src_index in self._gt_nxg.nodes and TP_NODE_VOTES in gt_src_attrs:
                gt_src_attrs[TP_NODE_VOTES] = (
                    self._gt_nxg.nodes[src_index][TP_NODE_VOTES] + 1
                )
            self._gt_nxg.add_nodes_from([(src_index, gt_src_attrs)])
            tgt_index = self.get_new_node_index(gt_tgt_attrs)
            if tgt_index in self._gt_nxg.nodes and TP_NODE_VOTES in gt_tgt_attrs:
                gt_tgt_attrs[TP_NODE_VOTES] = (
                    self._gt_nxg.nodes[tgt_index][TP_NODE_VOTES] + 1
                )
            self._gt_nxg.add_nodes_from([(tgt_index, gt_tgt_attrs)])
            self._gt_nxg.add_edge(src_index, tgt_index, **gt_edge_attr)
            if FN_NODE_ATTR in gt_src_attrs:
                self._fn_objects.add(src_index)
                actions["added_to_fno"].append(src_index)
            elif TP_NODE_VOTES in gt_src_attrs:
                self._tp_objects.add(gt_src_attrs["orig_idx"])
                actions["added_to_tpo"].append(gt_src_attrs["orig_idx"])
            if FN_NODE_ATTR in gt_tgt_attrs:
                self._fn_objects.add(tgt_index)
                actions["added_to_fno"].append(tgt_index)
            elif TP_NODE_VOTES in gt_tgt_attrs:
                self._tp_objects.add(gt_tgt_attrs["orig_idx"])
                actions["added_to_tpo"].append(gt_tgt_attrs["orig_idx"])
            if FN_EDGE_ATTR in gt_edge_attr:
                self._fn_edges.add((src_index, tgt_index))
                actions["added_to_fne"].append((src_index, tgt_index))
            # update the original nxg with the GT edge index
            nxg.edges[original_edge]["gt_src"] = src_index
            nxg.edges[original_edge]["gt_tgt"] = tgt_index
        elif num_points == 1:
            # this is an FP edge and we only have a single point left
            # TODO: should we update TP votes?
            nxg.edges[original_edge][FP_EDGE_ATTR] = True
            self._fp_edges.add(original_edge)
            actions["added_to_fpe"].append(original_edge)
            # need to be able to restore this point/lack of points if we come back to this
            # edge, so store whether the source or the target remains
            nxg.edges[original_edge]["src_present"] = False
            nxg.edges[original_edge]["tgt_present"] = False
            if len(points_layer.data) == 1:
                if points_layer.symbol[0] == "disc":
                    nxg.edges[original_edge]["src_present"] = True
                else:
                    nxg.edges[original_edge]["tgt_present"] = True

        # mark this edge as seen
        self._edge_actions[original_edge] = actions
        self._update_label_displays()
        nxg.edges[original_edge]["seen"] = True
        self._points_layer_changed = False
        return True

    def _undo_edge_actions(self, edge):
        prior_actions = self._edge_actions[edge]
        nxg = self._get_original_nxg()
        if len(prior_actions["added_to_fpe"]):
            for prior_fp_edge in prior_actions["added_to_fpe"]:
                self._fp_edges.remove(prior_fp_edge)
                nxg.edges[prior_fp_edge].pop(FP_EDGE_ATTR, None)
                nxg.edges[prior_fp_edge].pop("gt_src", None)
                nxg.edges[prior_fp_edge].pop("gt_tgt", None)
                nxg.edges[prior_fp_edge].pop("src_present", None)
                nxg.edges[prior_fp_edge].pop("tgt_present", None)
        if len(prior_actions["added_to_fne"]):
            for prior_fn_edge in prior_actions["added_to_fne"]:
                self._fn_edges.remove(prior_fn_edge)
                self._gt_nxg.remove_edge(*prior_fn_edge)
        if len(prior_actions["added_to_tpe"]):
            for prior_tp_edge in prior_actions["added_to_tpe"]:
                self._tp_edges.remove(prior_tp_edge)
                nxg.edges[prior_tp_edge].pop(TP_EDGE_ATTR, None)
                gt_src = nxg.edges[prior_tp_edge].pop("gt_src", None)
                gt_tgt = nxg.edges[prior_tp_edge].pop("gt_tgt", None)
                if gt_src is not None and gt_tgt is not None:
                    self._gt_nxg.remove_edge(gt_src, gt_tgt)
        if len(prior_actions["added_to_fno"]):
            for prior_fn_node in prior_actions["added_to_fno"]:
                self._fn_objects.remove(prior_fn_node)
                self._gt_nxg.remove_node(prior_fn_node)
        if len(prior_actions["added_to_tpo"]):
            marked_tp_nodes = set(prior_actions["added_to_tpo"])
            # go through gt graph and remove any nodes matching
            # original indices in the tp set
            to_remove_gt = []
            to_remove_sol = set()
            for node in self._gt_nxg.nodes:
                if (
                    TP_NODE_VOTES in self._gt_nxg.nodes[node]
                    and self._gt_nxg.nodes[node].get("orig_idx", None)
                    in marked_tp_nodes
                ):
                    self._gt_nxg.nodes[node][TP_NODE_VOTES] -= 1
                    if self._gt_nxg.nodes[node][TP_NODE_VOTES] <= 0:
                        to_remove_gt.append(node)
                        to_remove_sol.add(self._gt_nxg.nodes[node]["orig_idx"])
            self._gt_nxg.remove_nodes_from(to_remove_gt)
            self._tp_objects.difference_update(to_remove_sol)
        self._update_label_displays()

        if len(prior_actions["added_to_fpo"]):
            for prior_fp_node in prior_actions["added_to_fpo"]:
                self._fp_objects.remove(prior_fp_node)
                nxg.nodes[prior_fp_node].pop(FP_NODE_ATTR, None)

    def _update_label_displays(self):
        get_count_label_from_grid(self._counts_grid_layout, "TPO").setText(
            str(len(self._tp_objects))
        )
        get_count_label_from_grid(self._counts_grid_layout, "FPO").setText(
            str(len(self._fp_objects))
        )
        get_count_label_from_grid(self._counts_grid_layout, "FNO").setText(
            str(len(self._fn_objects))
        )
        get_count_label_from_grid(self._counts_grid_layout, "TPT").setText(
            str(len(self._tp_edges))
        )
        get_count_label_from_grid(self._counts_grid_layout, "FPT").setText(
            str(len(self._fp_edges))
        )
        get_count_label_from_grid(self._counts_grid_layout, "FNT").setText(
            str(len(self._fn_edges))
        )

    def _reset_current_edge(self):
        if (
            "seen"
            in self._get_original_nxg().edges[
                self._edge_sample_order[self._current_display_idx]
            ]
        ):
            self._display_gt_edge(self._current_display_idx)
        else:
            self._display_edge(self._current_display_idx)

    def _reset_to_original_edge(self):
        original_edge = self._edge_sample_order[self._current_display_idx]
        original_edge = (int(original_edge[0]), int(original_edge[1]))
        self._get_original_nxg().edges[original_edge].pop("seen", None)
        self._undo_edge_actions(original_edge)
        self._display_edge(self._current_display_idx)
        self._reset_to_original_button.enabled = False
        self._edge_status_label.setText("New")

    def _add_ground_truth_tracks(self):
        if (
            self._gt_nxg is None
            or len(self._gt_nxg.nodes) == 0
            or len(self._gt_nxg.edges) == 0
        ):
            show_info("No ground truth tracks to display. Try annotating some edges!")
            return
        seg_layer_scale = self._seg_combo.value.scale
        tracks = get_tracks_from_nxg(self._gt_nxg)
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
            self._gt_nxg is None
            or len(self._gt_nxg.nodes) == 0
            or len(self._gt_nxg.edges) == 0
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
        sol_nxg = self._get_original_nxg()
        colors = {node: sol_nxg.nodes[node].pop("color") for node in sol_nxg.nodes}

        nx.write_graphml(self._gt_nxg, gt_path)
        nx.write_graphml(self._get_original_nxg(), sol_path)
        nx.set_node_attributes(sol_nxg, colors, "color")
        print("Saved")


def get_region_center(loc1, loc2):
    """
    Get the camera center-point
    """
    return (loc1 + loc2) / 2


def get_loc_array(node_info):
    """
    Get the location array from node info
    """
    loc = []
    loc.append(node_info["t"])
    if "z" in node_info:
        loc.append(node_info["z"])
    loc.append(node_info["y"])
    loc.append(node_info["x"])
    return np.asarray(loc)


def get_loc_dict(node_info):
    """
    Get the location dictionary from node info
    """
    loc = {}
    loc["t"] = node_info["t"]
    if "z" in node_info:
        loc["z"] = node_info["z"]
    loc["y"] = node_info["y"]
    loc["x"] = node_info["x"]
    return loc


def split_coords(loc):
    """
    Split the location into t, z, y, x
    """
    if len(loc) == 4:
        return {"t": int(loc[0]), "z": loc[1], "y": loc[2], "x": loc[3]}
    return {"t": int(loc[0]), "y": loc[1], "x": loc[2]}


def get_int_loc(loc):
    """
    Get the integer location
    """
    return np.round(loc).astype(int)


def get_src_tgt_idx(points_symbols):
    (src_idx,) = np.where(points_symbols == "disc")
    (tgt_idx,) = np.where(points_symbols == "ring")
    if len(src_idx) == 0 or len(tgt_idx) == 0:
        show_info(
            "Missing source disc or target ring. Did you change symbols? Resetting edge."
        )
        return None, None
    src_idx = int(src_idx[0])
    tgt_idx = int(tgt_idx[0])
    return src_idx, tgt_idx
