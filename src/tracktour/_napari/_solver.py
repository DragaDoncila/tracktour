import warnings

import numpy as np
from dask.array import Array
from magicgui.widgets import Container, PushButton, create_widget

from tracktour._geff_io import write_candidate_geff, write_solution_geff
from tracktour._io_util import extract_im_centers
from tracktour._napari._graph_conversion_util import get_coloured_solution_layers
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

        self._seg_layer_combo = create_widget(
            annotation="napari.layers.Labels", label="Segmentation Layer"
        )
        self._n_neighbours_spin = create_widget(
            annotation="int", label="n Neighbours", options={"min": 2}
        )
        self._n_children_spin = create_widget(
            annotation="int", label="max Children", options={"min": 2}
        )
        self._cost_combo = create_widget(annotation=Cost, label="Cost Function")

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
                self._seg_layer_combo,
                self._n_neighbours_spin,
                self._n_children_spin,
                self._cost_combo,
                self._solve_button,
                self._export_solution_button,
                self._export_candidate_button,
            ]
        )

    def _solve_graph(self):
        seg_layer = self._seg_layer_combo.value
        segmentation = seg_layer.data
        n_neighbours = self._n_neighbours_spin.value
        n_children = self._n_children_spin.value
        cost_choice = self._cost_combo.value

        if isinstance(segmentation, Array):
            warnings.warn(
                "Your segmentation is a dask array which is not currently supported. Will attempt conversion to numpy array."
            )
            segmentation = np.asarray(segmentation)
        coords_df, min_t, max_t, corners = extract_im_centers(segmentation)

        tracker = Tracker(
            im_shape=segmentation.shape[1:], seg=segmentation, scale=seg_layer.scale[1:]
        )
        tracker.DEBUG_MODE = True
        tracker.DIVISION_EDGE_CAPACITY = n_children - 1
        tracked = tracker.solve(
            coords_df, value_key="label", k_neighbours=n_neighbours, costs=cost_choice
        )
        tracked.assign_features()

        coloured_points, coloured_labels, tracks = get_coloured_solution_layers(
            tracked,
            tracker.scale,
            segmentation,
        )
        tracks.metadata["tracked"] = tracked

        self._viewer.add_layer(coloured_points)
        self._viewer.add_layer(coloured_labels)
        self._viewer.add_layer(tracks)

        self._tracked = tracked
        self._export_solution_button.enabled = True
        self._export_candidate_button.enabled = True

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
