import warnings
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from dask.array import Array
from magicgui.widgets import Container, PushButton, create_widget

if TYPE_CHECKING:
    import napari

from tracktour._geff_io import write_candidate_geff, write_solution_geff
from tracktour._io_util import extract_im_centers
from tracktour._napari._graph_conversion_util import (
    get_coloured_solution_layers,
    get_tracks_from_nxg,
)
from tracktour._tracker import Cost, Tracker


class TrackingSolver(Container):
    def __init__(
        self,
        viewer: "napari.viewer.Viewer",
        layout: str = "vertical",
        labels: bool = True,
    ) -> None:
        super().__init__(
            layout=layout,
            labels=labels,
        )
        self._viewer = viewer
        self._tracked = None

        self._input_layer_combo = create_widget(
            annotation="napari.layers.Layer",
            label="Input Layer",
            options={"choices": self._get_input_layer_choices},
        )
        self._n_neighbours_spin = create_widget(
            value=10, annotation="int", label="n Neighbours", options={"min": 2}
        )
        self._n_children_spin = create_widget(
            annotation="int", label="max Children", options={"min": 2}
        )
        self._cost_combo = create_widget(annotation=Cost, label="Cost Function")
        self._allow_merges_checkbox = create_widget(
            value=True, annotation=bool, label="Allow Merges"
        )

        self._solve_button = PushButton(text="Solve")
        self._solve_button.clicked.connect(self._solve_graph)

        self._export_solution_button = PushButton(text="Export Solution to GEFF")
        self._export_solution_button.clicked.connect(self._export_solution_geff)
        self._export_solution_button.enabled = False

        self._export_candidate_button = PushButton(
            text="Export Candidate Graph to GEFF"
        )
        self._export_candidate_button.clicked.connect(self._export_candidate_geff)
        self._export_candidate_button.enabled = False

        self.extend(
            [
                self._input_layer_combo,
                self._n_neighbours_spin,
                self._n_children_spin,
                self._cost_combo,
                self._allow_merges_checkbox,
                self._solve_button,
                self._export_solution_button,
                self._export_candidate_button,
            ]
        )

    def _get_input_layer_choices(self, widget):
        import napari.layers

        return [
            layer
            for layer in self._viewer.layers
            if isinstance(layer, (napari.layers.Labels, napari.layers.Points))
        ]

    def _solve_graph(self):
        import napari.layers

        input_layer = self._input_layer_combo.value
        if input_layer is None:
            warnings.warn("Please select an input layer.")
            return

        n_neighbours = self._n_neighbours_spin.value
        n_children = self._n_children_spin.value
        cost_choice = self._cost_combo.value
        allow_merges = self._allow_merges_checkbox.value

        if isinstance(input_layer, napari.layers.Labels):
            self._solve_from_labels(
                input_layer, n_neighbours, n_children, cost_choice, allow_merges
            )
        elif isinstance(input_layer, napari.layers.Points):
            self._solve_from_points(
                input_layer, n_neighbours, n_children, cost_choice, allow_merges
            )
        else:
            warnings.warn(f"Unsupported layer type: {type(input_layer)}")
            return

        self._export_solution_button.enabled = True
        self._export_candidate_button.enabled = True

    def _solve_from_labels(
        self, seg_layer, n_neighbours, n_children, cost_choice, allow_merges
    ):
        segmentation = seg_layer.data
        if isinstance(segmentation, Array):
            warnings.warn(
                "Your segmentation is a dask array which is not currently supported. Will attempt conversion to numpy array."
            )
            segmentation = np.asarray(segmentation)
        coords_df, *_ = extract_im_centers(segmentation)

        tracker = Tracker(
            im_shape=segmentation.shape[1:], seg=segmentation, scale=seg_layer.scale[1:]
        )
        tracker.DEBUG_MODE = True
        tracker.DIVISION_EDGE_CAPACITY = n_children - 1
        tracker.ALLOW_MERGES = allow_merges
        tracked = tracker.solve(
            coords_df, value_key="label", k_neighbours=n_neighbours, costs=cost_choice
        )
        tracked.assign_features()

        coloured_labels, tracks = get_coloured_solution_layers(
            tracked,
            tracker.scale,
            segmentation,
        )
        tracks.metadata["tracked"] = tracked
        tracks.metadata["tracker"] = tracker
        self._viewer.add_layer(coloured_labels)
        self._viewer.add_layer(tracks)
        self._tracked = tracked

        # turn off original segmentation layer
        seg_layer.visible = False

    def _solve_from_points(
        self, points_layer, n_neighbours, n_children, cost_choice, allow_merges
    ):
        points_data = points_layer.data  # shape (N, D): [t, y, x] or [t, z, y, x]
        ndim = points_data.shape[1] - 1
        loc_cols = ["z", "y", "x"] if ndim == 3 else ["y", "x"]

        coords_df = pd.DataFrame(points_data, columns=["t"] + loc_cols)
        coords_df["t"] = coords_df["t"].astype(int)
        coords_df = coords_df.sort_values("t").reset_index(drop=True)
        # synthetic label column required by Tracked; integers don't affect the solve
        coords_df["label"] = np.arange(len(coords_df))

        im_shape = tuple(int(np.ceil(coords_df[col].max())) + 1 for col in loc_cols)

        tracker = Tracker(im_shape=im_shape, scale=points_layer.scale[1:])
        tracker.DEBUG_MODE = True
        tracker.DIVISION_EDGE_CAPACITY = n_children - 1
        tracker.ALLOW_MERGES = allow_merges
        tracked = tracker.solve(
            coords_df, value_key="label", k_neighbours=n_neighbours, costs=cost_choice
        )
        tracked.assign_features()

        subgraph = tracked.as_nx_digraph(include_all_attrs=True)
        tracks_layer = get_tracks_from_nxg(subgraph)
        tracks_layer.scale = (1,) + tuple(tracker.scale)
        tracks_layer.metadata = {
            "nxg": subgraph,
            "tracked": tracked,
            "tracker": tracker,
        }
        self._viewer.add_layer(tracks_layer)
        self._tracked = tracked

    def _export_solution_geff(self):
        if self._tracked is None:
            return
        from qtpy.QtWidgets import QFileDialog

        path, _ = QFileDialog.getSaveFileName(
            None, "Export solution to GEFF", "", "GEFF files (*.geff)"
        )
        if path:
            write_solution_geff(self._tracked, path)

    def _export_candidate_geff(self):
        if self._tracked is None:
            return
        from qtpy.QtWidgets import QFileDialog

        path, _ = QFileDialog.getSaveFileName(
            None, "Export candidate graph to GEFF", "", "GEFF files (*.geff)"
        )
        if path:
            write_candidate_geff(self._tracked, path)
