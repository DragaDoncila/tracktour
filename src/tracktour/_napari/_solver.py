import warnings

import networkx as nx
import numpy as np
import pandas as pd
from dask.array import Array
from magicgui.widgets import Container, PushButton, create_widget

from tracktour._io_util import extract_im_centers
from tracktour._napari._graph_conversion_util import get_coloured_graph_labels
from tracktour._tracker import Tracker

# from napari.qt.threading import create_worker


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

        self._seg_layer_combo = create_widget(
            annotation="napari.layers.Labels", label="Segmentation Layer"
        )
        self._n_neighbours_spin = create_widget(
            annotation="int", label="n Neighbours", options={"min": 2}
        )
        self._solve_button = PushButton(text="Solve")

        self._solve_button.clicked.connect(self._solve_graph)

        self.extend(
            [self._seg_layer_combo, self._n_neighbours_spin, self._solve_button]
        )

    def _solve_graph(self):
        seg_layer = self._seg_layer_combo.value
        segmentation = seg_layer.data
        n_neighbours = self._n_neighbours_spin.value
        # centers, labels = [], []

        # def yield_extend(yielded):
        #     cent, lab = yielded
        #     centers.extend(cent)
        #     labels.extend(lab)

        # center_worker = create_worker(
        #     get_centers,
        #     segmentation=segmentation,
        #     _progress={
        #         'total':len(segmentation),
        #         'desc':'Extracting Centers'
        #         },
        #     _connect={
        #         'yielded': yield_extend
        #         }
        #     )
        # coords_df, min_t, max_t, corners = get_im_info(centers, labels, segmentation)
        if isinstance(segmentation, Array):
            warnings.warn(
                "Your segmentation is a dask array which is not currently supported. Will attempt conversion to numpy array."
            )
            segmentation = np.asarray(segmentation)
        coords_df, min_t, max_t, corners = extract_im_centers(segmentation)

        tracker = Tracker(segmentation.shape[1:], k_neighbours=n_neighbours)
        tracked = tracker.solve(coords_df, value_key="label")

        # make graph layer and tracks layer from solution
        napari_graph_layer, coloured_seg_layer = get_coloured_graph_labels(
            tracked,
            tracker.location_keys,
            tracker.frame_key,
            tracker.value_key,
            segmentation,
        )
        napari_graph_layer.metadata["k_neighbours"] = n_neighbours
        self._viewer.add_layer(napari_graph_layer)
        self._viewer.add_layer(coloured_seg_layer)
        seg_layer.visible = False
