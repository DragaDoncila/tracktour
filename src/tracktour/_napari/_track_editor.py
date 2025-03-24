import warnings
from collections import defaultdict

import networkx as nx
import numpy as np
from magicgui.widgets import Container, PushButton, create_widget
from qtpy.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from tracktour._tracker import Tracker

from ._graph_conversion_util import get_nxg_from_tracks

EDGE_FOCUS_POINT_NAME = "Source Target"
EDGE_FOCUS_VECTOR_NAME = "Current Edge"
POINT_IN_FRAME_COLOR = [0.816, 0.337, 0.933, 1]
POINT_OUT_FRAME_COLOR = [0.816, 0.337, 0.933, 0.5]
VECTOR_COLOR = [1, 1, 1, 1]

FP_EDGE_ATTR = "tracktour_annotated_fp"
FN_EDGE_ATTR = "tracktour_annotated_fn"
TP_EDGE_ATTR = "tracktour_annotated_tp"

FP_NODE_ATTR = "tracktour_annotated_fp"
FN_NODE_ATTR = "tracktour_annotated_fn"
TP_NODE_ATTR = "tracktour_annotated_tp"


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
        self._viewer = viewer
        self._edge_sample_order = None
        self._original_nxg = None
        self._current_display_idx = 0

        self._gt_seg_layer = None
        self._gt_points_layer = None
        self._gt_tracks_layer = None
        self._gt_nxg = None

        self._seg_combo = create_widget(
            annotation="napari.layers.Labels", label="Segmentation"
        )
        self._track_combo = create_widget(
            annotation="napari.layers.Tracks", label="Tracks"
        )
        self._seg_combo.changed.connect(self._setup_annotation_layers)
        self._track_combo.changed.connect(self._setup_annotation_layers)

        self.base_layout = QVBoxLayout()

        self._edge_control_layout = QHBoxLayout()
        self._previous_edge_button = PushButton(text="Previous")
        self._previous_edge_button.enabled = False

        self._reset_edge_button = PushButton(text="Reset Current Edge")
        self._next_edge_button = PushButton(text="Next")

        self._edge_control_layout.addWidget(self._previous_edge_button.native)
        self._edge_control_layout.addWidget(self._reset_edge_button.native)
        self._edge_control_layout.addWidget(self._next_edge_button.native)

        self._counts_grid_layout = get_counts_grid_layout()
        self._tp_object_count = 0
        self._fp_object_count = 0
        self._fn_object_count = 0
        self._tp_track_count = 0
        self._fp_track_count = 0
        self._fn_track_count = 0

        self.base_layout.addWidget(self._seg_combo.native)
        self.base_layout.addWidget(self._track_combo.native)
        self.base_layout.addWidget(get_separator_widget())
        self.base_layout.addLayout(self._edge_control_layout)
        self.base_layout.addWidget(get_separator_widget())
        self.base_layout.addLayout(self._counts_grid_layout)
        self.setLayout(self.base_layout)

        self._next_edge_button.clicked.connect(self._display_next_edge)
        self._previous_edge_button.clicked.connect(self._display_previous_edge)
        self._reset_edge_button.clicked.connect(self._reset_current_edge)

        self._setup_annotation_layers()

    def _add_gt_layers(self):
        # we have two layers selected, setup a ground truth segmentation layer
        # a ground truth points layer and a ground truth tracks layer
        current_seg_layer = self._seg_combo.value

        scale = current_seg_layer.scale
        new_seg = np.zeros_like(
            current_seg_layer.data, dtype=current_seg_layer.data.dtype
        )

        self._gt_seg_layer = self._viewer.add_labels(
            new_seg, scale=scale, name="GT Segmentation", visible=False
        )
        self._gt_seg_layer.editable = False
        self._gt_points_layer = self._viewer.add_points(
            scale=scale, name="GT Centers", visible=False
        )
        self._gt_points_layer.editable = False

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
        tracks_layer = self._track_combo.value
        if "nxg" in tracks_layer.metadata:
            return tracks_layer.metadata["nxg"]
        nxg = get_nxg_from_tracks(tracks_layer)
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
        current_step = event.value
        src_loc, tgt_loc = self._get_edge_locs()
        src_t = src_loc[0]
        tgt_t = tgt_loc[0]
        current_t = current_step[0]
        if current_t == src_t:
            points_layer.face_color = [POINT_IN_FRAME_COLOR, POINT_OUT_FRAME_COLOR]
        elif current_t == tgt_t:
            points_layer.face_color = [POINT_OUT_FRAME_COLOR, POINT_IN_FRAME_COLOR]
        else:
            points_layer.face_color = [POINT_OUT_FRAME_COLOR, POINT_OUT_FRAME_COLOR]

    def _add_current_edge_focus_point(self, src, tgt):
        # we drop the first dimension of the points, which is the frame
        src_proj = src[1:]
        tgt_proj = tgt[1:]
        points_data = np.vstack([src_proj, tgt_proj])
        points_symbols = ["disc", "ring"]
        points_face_color = [POINT_IN_FRAME_COLOR, POINT_OUT_FRAME_COLOR]

        vectors_data = [[src_proj, tgt_proj - src_proj]]
        vectors_style = "arrow"
        vectors_color = VECTOR_COLOR
        vectors_width = 0.3
        if EDGE_FOCUS_POINT_NAME in self._viewer.layers:
            point_focus_layer = self._viewer.layers[EDGE_FOCUS_POINT_NAME]
            point_focus_layer.data = points_data
            point_focus_layer.symbol = points_symbols
            point_focus_layer.face_color = points_face_color
        else:
            point_focus_layer = self._viewer.add_points(
                points_data,
                name=EDGE_FOCUS_POINT_NAME,
                size=3,
                symbol=points_symbols,
                face_color=points_face_color,
            )

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
            )
        self._viewer.layers.selection.active = point_focus_layer
        point_focus_layer.mode = "SELECT"

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

    def _display_edge(self, current_edge_idx):
        src_loc, tgt_loc = self._get_edge_locs(current_edge_idx)
        # get bounding box containing region of both nodes, ignoring t
        center = get_region_center(src_loc[1:], tgt_loc[1:])
        # bbox = self._get_region_bbox(src_loc, tgt_loc)
        self._add_current_edge_focus_point(src_loc, tgt_loc)
        self._viewer.camera.center = center
        self._viewer.dims.current_step = (src_loc[0], 0, 0)

    def _display_next_edge(self):
        edge_saved = self._save_edge_annotation()
        if edge_saved:
            self._current_display_idx += 1
        self._display_edge(self._current_display_idx)

        # next button needs to be enabled current edge is not the last edge
        self._next_edge_button.enabled = (
            self._current_display_idx < len(self._edge_sample_order) - 1
        )
        # previous button needs to be enabled if current edge is not the first edge
        self._previous_edge_button.enabled = self._current_display_idx > 0

    def _display_previous_edge(self):
        edge_saved = self._save_edge_annotation()
        if edge_saved:
            self._current_display_idx -= 1
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
        new_idx = len(self._gt_nxg.nodes)
        # if it was an FN node, we don't want to use its original index
        if node_attrs.get(FN_NODE_ATTR, False):
            # TODO: need to check here if this FN node hasn't already been added...
            return new_idx
        for node in self._gt_nxg.nodes:
            if self._gt_nxg.nodes[node]["orig_idx"] == node_attrs["orig_idx"]:
                return node
        return new_idx

    def get_gt_node_attrs(self, src_points_idx, tgt_points_idx):
        gt_src_attrs = {}
        gt_tgt_attrs = {}

        points_layer = self._viewer.layers[EDGE_FOCUS_POINT_NAME]
        original_src_loc, original_tgt_loc = self._get_edge_locs()
        gt_src_attrs["t"] = int(original_src_loc[0])
        gt_tgt_attrs["t"] = int(original_tgt_loc[0])

        gt_src_loc = points_layer.data[src_points_idx]
        gt_tgt_loc = points_layer.data[tgt_points_idx]
        gt_src_loc = np.concatenate([[original_src_loc[0]], gt_src_loc])
        gt_tgt_loc = np.concatenate([[original_tgt_loc[0]], gt_tgt_loc])
        gt_src_attrs.update(split_coords(gt_src_loc))
        gt_tgt_attrs.update(split_coords(gt_tgt_loc))

        # check if either of the points are over blank space
        gt_src_idx = get_int_loc(gt_src_loc)
        gt_tgt_idx = get_int_loc(gt_tgt_loc)
        seg_layer = self._seg_combo.value
        if seg_layer.data[tuple(gt_src_idx)] == 0:
            # source was a previous FN
            gt_src_attrs[FN_NODE_ATTR] = True
        elif seg_layer.data[tuple(gt_tgt_idx)] == 0:
            # target was a previous FN
            gt_tgt_attrs[FN_NODE_ATTR] = True
        else:
            # both nodes already existed and still do, so they are TP
            gt_src_attrs[TP_NODE_ATTR] = True
            gt_tgt_attrs[TP_NODE_ATTR] = True

        original_edge = self._edge_sample_order[self._current_display_idx]
        gt_src_attrs["orig_idx"] = int(original_edge[0])
        gt_tgt_attrs["orig_idx"] = int(original_edge[1])

        return gt_src_attrs, gt_tgt_attrs

    def _save_edge_annotation(self):
        original_edge = self._edge_sample_order[self._current_display_idx]
        original_src_loc, original_tgt_loc = self._get_edge_locs()
        points_layer = self._viewer.layers[EDGE_FOCUS_POINT_NAME]
        nxg = self._get_original_nxg()
        if len(points_layer.data) > 2:
            warnings.warn("More than two points in the current edge. Resetting edge.")
            return False
        elif len(points_layer.data) == 2:
            point_one_loc = points_layer.data[0]
            point_two_loc = points_layer.data[1]
            point_one_moved = not np.allclose(point_one_loc, original_src_loc[1:])
            point_two_moved = not np.allclose(point_two_loc, original_tgt_loc[1:])
            # if both points have changed, we need to warn
            if point_one_moved and point_two_moved:
                warnings.warn("Both points have changed. Resetting edge.")
                return False
            # have either of the points moved?
            if point_one_moved or point_two_moved:
                print("Point moved!")
                # we have an FP/FN edge combo
                # mark the networkx edge as an FP edge
                nxg.edges[original_edge][FP_EDGE_ATTR] = True
                gt_edge_attr = {FN_EDGE_ATTR: True}
                self._fp_track_count += 1
                self._fn_track_count += 1
            else:
                # no points moved, we have a TP edge, add it to nxg
                nxg.edges[original_edge][TP_EDGE_ATTR] = True
                gt_edge_attr = {TP_EDGE_ATTR: True}
                # mark the nodes as TP
                nxg.nodes[original_edge[0]][TP_NODE_ATTR] = True
                nxg.nodes[original_edge[1]][TP_NODE_ATTR] = True
                print("TP edge")
                self._tp_track_count += 1
                self._tp_object_count += 2
            (src_idx,) = np.where(points_layer.symbol == "disc")
            (tgt_idx,) = np.where(points_layer.symbol == "ring")
            if len(src_idx) == 0 or len(tgt_idx) == 0:
                warnings.warn(
                    "Missing source disc or target ring. Did you change symbols? Resetting edge."
                )
                return False
            src_idx = int(src_idx[0])
            tgt_idx = int(tgt_idx[0])
            gt_src_attrs, gt_tgt_attrs = self.get_gt_node_attrs(src_idx, tgt_idx)
            if FN_NODE_ATTR in gt_src_attrs or FN_NODE_ATTR in gt_tgt_attrs:
                self._fn_object_count += 1
            # add nodes with new indices, add edge
            src_index = self.get_new_node_index(gt_src_attrs)
            self._gt_nxg.add_nodes_from([(src_index, gt_src_attrs)])
            tgt_index = self.get_new_node_index(gt_tgt_attrs)
            self._gt_nxg.add_nodes_from([(tgt_index, gt_tgt_attrs)])
            self._gt_nxg.add_edge(src_index, tgt_index, **gt_edge_attr)
        # we've lost one or both points, we have an FP edge
        # TODO: checking for orphaned nodes, we should track them as we go
        else:
            # points data is 0, we mark the edge as FP, do nothing with the nodes
            # nothing needs to be added to GT?
            nxg.edges[original_edge][FP_EDGE_ATTR] = True
            self._fp_track_count += 1
        self._update_label_displays()
        return True

    def _update_label_displays(self):
        get_count_label_from_grid(self._counts_grid_layout, "TPO").setText(
            str(self._tp_object_count)
        )
        get_count_label_from_grid(self._counts_grid_layout, "FPO").setText(
            str(self._fp_object_count)
        )
        get_count_label_from_grid(self._counts_grid_layout, "FNO").setText(
            str(self._fn_object_count)
        )
        get_count_label_from_grid(self._counts_grid_layout, "TPT").setText(
            str(self._tp_track_count)
        )
        get_count_label_from_grid(self._counts_grid_layout, "FPT").setText(
            str(self._fp_track_count)
        )
        get_count_label_from_grid(self._counts_grid_layout, "FNT").setText(
            str(self._fn_track_count)
        )

    def _reset_current_edge(self):
        self._check_valid_layers()


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
