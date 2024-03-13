import warnings
from collections import defaultdict

from magicgui.widgets import Container, PushButton, create_widget

from tracktour._tracker import Tracker

from ._graph_conversion_util import (
    get_coloured_graph_labels,
    get_detections_from_napari_graph,
)


class TrackEditor(Container):
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

        self._graph_layer_combo = create_widget(
            annotation="napari.layers.Graph", label="Graph Layer"
        )
        self._seg_combo = create_widget(
            annotation="napari.layers.Labels", label="Segmentation"
        )

        self._resolve_btn = PushButton(text="Re-solve")
        self._resolve_btn.changed.connect(self._resolve_candidate_graph)

        self.extend([self._graph_layer_combo, self._seg_combo, self._resolve_btn])

    def _resolve_candidate_graph(self):
        if not self._graph_layer_combo.value:
            return
        if not self._seg_combo.value:
            return
        current_layer = self._graph_layer_combo.value
        seg_layer = self._seg_combo.value
        seg_ims = seg_layer.data
        detections_df = get_detections_from_napari_graph(current_layer.data, seg_ims)
        n_neighbours = current_layer.metadata.get("k_neighbours", None)
        if n_neighbours is None:
            warnings.warn(
                "No k_neighbours found in graph metadata. Using default value of 10."
            )
            n_neighbours = 10
        tracker = Tracker(seg_ims.shape[1:], k_neighbours=n_neighbours)
        tracked = tracker.solve(detections_df, value_key="label")
        napari_graph_layer, coloured_seg_layer = get_coloured_graph_labels(
            tracked,
            tracker.location_keys,
            tracker.frame_key,
            tracker.value_key,
            seg_ims,
        )

        with current_layer.events.data.blocker():
            current_layer.data = napari_graph_layer.data
            current_layer.metadata["nxg"] = napari_graph_layer.metadata["nxg"]
            current_layer.metadata["tracks"] = napari_graph_layer.metadata["tracks"]
            current_layer.metadata["subgraph"] = napari_graph_layer.metadata["subgraph"]
            current_layer.face_color = napari_graph_layer.face_color
        if coloured_seg_layer.name in self._viewer.layers:
            self._viewer.layers[coloured_seg_layer.name].data = coloured_seg_layer.data
