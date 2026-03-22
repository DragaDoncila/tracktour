import networkx as nx
from magicgui.widgets import ComboBox, Container, create_widget


class MergeExplorer(Container):
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
        self._nxg: "nx.DiGraph | None" = None

        self._tracks_layer_combo = create_widget(
            annotation="napari.layers.Tracks", label="Tracks Layer"
        )
        self._merge_id_combo = ComboBox(label="Merged Cells")

        self._tracks_layer_combo.changed.connect(self._update_merge_node_choices)
        self._merge_id_combo.changed.connect(self._show_selected_merge)

        self.extend([self._tracks_layer_combo, self._merge_id_combo])
        self._update_merge_node_choices()

    def _update_merge_node_choices(self):
        layer = self._tracks_layer_combo.value
        if layer is None:
            self._nxg = None
            self._merge_id_combo.choices = []
            return

        self._nxg = layer.metadata.get("nxg")
        if self._nxg is None:
            self._merge_id_combo.choices = []
            return

        nxg = self._nxg

        def get_choices(_combo):
            return [
                (f"Node {node}, Frame {nxg.nodes[node]['t']}", node)
                for node in nxg.nodes
                if nxg.in_degree(node) > 1
            ]

        self._merge_id_combo.choices = get_choices

    def _show_selected_merge(self):
        merge_id = self._merge_id_combo.value
        if merge_id is None or self._nxg is None:
            return

        node = self._nxg.nodes[merge_id]
        t = int(node["t"])
        loc_keys = ["z", "y", "x"] if "z" in node else ["y", "x"]
        spatial = tuple(float(node[k]) for k in loc_keys)

        self._viewer.dims.current_step = (t,) + (0,) * len(loc_keys)
        self._viewer.camera.center = spatial
        self._viewer.camera.zoom = 8
