import warnings
from collections import defaultdict

import numpy as np
from magicgui.widgets import Container, PushButton, create_widget
from qtpy.QtWidgets import QHBoxLayout, QVBoxLayout, QWidget

from tracktour._tracker import Tracker

from ._graph_conversion_util import get_nxg_from_tracks

EDGE_FOCUS_POINT_NAME = "Current Edge"


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

        self.base_layout.addWidget(self._seg_combo.native)
        self.base_layout.addWidget(self._track_combo.native)
        self.base_layout.addLayout(self._edge_control_layout)
        self.setLayout(self.base_layout)

        self._next_edge_button.clicked.connect(self._display_next_edge)
        self._previous_edge_button.clicked.connect(self._display_previous_edge)
        self._reset_edge_button.clicked.connect(self._reset_current_edge)

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
        self._gt_points_layer = self._viewer.add_points(
            scale=scale, name="GT Centers", visible=False
        )

        # reset display index
        self._current_display_idx = 0
        self._display_edge(self._current_display_idx)

        self._next_edge_button.enabled = True

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

    def _add_current_edge_focus_point(self, loc1, loc2):
        # project loc1 onto loc2 frame
        loc3 = loc1.copy()
        loc3[0] = loc2[0]
        # project loc2 onto loc1 frame
        loc4 = loc2.copy()
        loc4[0] = loc1[0]
        if EDGE_FOCUS_POINT_NAME in self._viewer.layers:
            edge_focus_layer = self._viewer.layers[EDGE_FOCUS_POINT_NAME]
            edge_focus_layer.data = np.vstack([loc1, loc2, loc3, loc4])
            edge_focus_layer.face_color = ["red", "red", "lightcoral", "lightcoral"]
        else:
            edge_focus_layer = self._viewer.add_points(
                np.vstack([loc1, loc2, loc3, loc4]),
                name=EDGE_FOCUS_POINT_NAME,
                size=3,
                symbol="diamond",
                face_color=["red", "red", "lightcoral", "lightcoral"],
            )

    def _display_edge(self, current_edge_idx):
        current_edge = self._edge_sample_order[current_edge_idx]
        nxg = self._get_original_nxg()
        src_idx = current_edge[0]
        tgt_idx = current_edge[1]
        # locating the source and target nodes, ignoring track-id
        src_loc = get_loc_array(nxg.nodes[src_idx])
        tgt_loc = get_loc_array(nxg.nodes[tgt_idx])
        # get bounding box containing region of both nodes, ignoring t
        center = get_region_center(src_loc[1:], tgt_loc[1:])
        # bbox = self._get_region_bbox(src_loc, tgt_loc)
        self._add_current_edge_focus_point(src_loc, tgt_loc)
        self._viewer.camera.center = center
        self._viewer.dims.current_step = (tgt_loc[0], 0, 0)

    def _display_next_edge(self):
        self._current_display_idx += 1
        self._display_edge(self._current_display_idx)

        # next button needs to be enabled current edge is not the last edge
        self._next_edge_button.enabled = (
            self._current_display_idx < len(self._edge_sample_order) - 1
        )
        # previous button needs to be enabled if current edge is not the first edge
        self._previous_edge_button.enabled = self._current_display_idx > 0

    def _display_previous_edge(self):
        self._current_display_idx -= 1
        self._display_edge(self._current_display_idx)
        self._next_edge_button.enabled = True

        # next button needs to be enabled current edge is not the last edge
        self._next_edge_button.enabled = (
            self._current_display_idx < len(self._edge_sample_order) - 1
        )
        # previous button needs to be enabled if current edge is not the first edge
        self._previous_edge_button.enabled = self._current_display_idx > 0

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
