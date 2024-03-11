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
        # self._candidate_graph = None
        # self._changing_indices = set()
        # self._changes = defaultdict(list)

        self._graph_layer_combo = create_widget(
            annotation="napari.layers.Graph", label="Graph Layer"
        )
        self._seg_combo = create_widget(
            annotation="napari.layers.Labels", label="Segmentation"
        )
        # self._graph_layer_combo.changed.connect(self._connect_data_handlers)

        self._resolve_btn = PushButton(text="Re-solve")
        self._resolve_btn.changed.connect(self._resolve_candidate_graph)

        # if self._graph_layer_combo.value:
        #     self._connect_data_handlers()

        self.extend([self._graph_layer_combo, self._seg_combo, self._resolve_btn])

    # def _connect_data_handlers(self):
    #     """Connect to data event on layer and build candidate graph"""
    #     if not self._graph_layer_combo.value:
    #         return
    #     current_layer = self._graph_layer_combo.value
    #     current_layer.events.data.connect(self._on_graph_data_changed)

    # def _on_graph_data_changed(self, event):
    #     if event.action == "removed":
    #         warnings.warn("Cannot yet remove graph vertices.")
    #         return
    #     # need to track which indices are being changed
    #     if event.action == "changing":
    #         self._changing_indices.update(set(event.data_indices))
    #         return

    #     layer = self._graph_layer_combo.value
    #     layer_graph = layer.data
    #     if event.action == "added":
    #         if event.data_indices[0] == -1:
    #             added_node_idx = len(layer_graph.get_nodes()) - 1
    #         else:
    #             added_node_idx = layer_graph.get_nodes()[event.data_indices[0]]
    #         self._changes["add"].append(added_node_idx)
    #     elif event.action == "changed":
    #         changed_nodes = tuple(self._changing_indices)
    #         for changed_node in changed_nodes:
    #             self._changes["move"].append(changed_node)
    #         self._changing_indices = set()

    # def _introduce_changes_to_candidate_graph(self):
    #     layer = self._graph_layer_combo.value
    #     layer_graph = layer.data
    #     new_label = max(self._candidate_graph._g.vs["pixel_value"]) + 1
    #     new_nid = len(self._candidate_graph._g.vs)
    #     to_introduce = {}
    #     affected_ts = set()
    #     for node in self._changes["add"]:
    #         node_coords = layer_graph.coords_buffer[node]
    #         t = int(node_coords[0])
    #         node_coords = tuple(node_coords[1:])
    #         label = new_label
    #         to_introduce[new_nid] = (t, node_coords, label)
    #         new_label += 1
    #         new_nid += 1
    #         affected_ts.add(t)
    #     if len(to_introduce):
    #         self._candidate_graph.introduce_vertices(to_introduce, rebuild=False)
    #     for node in self._changes["move"]:
    #         t = int(layer_graph.coords_buffer[node][0])
    #         new_coords = tuple(layer_graph.coords_buffer[node][1:])
    #         self._candidate_graph.move_vertex(node, new_coords)
    #         affected_ts.add(t)
    #     self._candidate_graph.rebuild_frames(affected_ts)
    #     self._changes = defaultdict(list)

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
        # Here we would ideally just replace data of the existing layers, as well as the metadata for the graph layer
        # can we just set data, would be nice!
        with current_layer.events.data.blocker():
            current_layer.data = napari_graph_layer.data
            current_layer.metadata["nxg"] = napari_graph_layer.metadata["nxg"]
            current_layer.metadata["tracks"] = napari_graph_layer.metadata["tracks"]
            current_layer.metadata["subgraph"] = napari_graph_layer.metadata["subgraph"]
            current_layer.face_color = napari_graph_layer.face_color
        if coloured_seg_layer.name in self._viewer.layers:
            self._viewer.layers[coloured_seg_layer.name].data = coloured_seg_layer.data
