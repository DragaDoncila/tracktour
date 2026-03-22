import networkx as nx
from magicgui.widgets import PushButton, create_widget
from qtpy.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout, QWidget


class MergeExplorer(QWidget):
    def __init__(
        self,
        viewer: "napari.viewer.Viewer",
    ) -> None:
        super().__init__()
        self._viewer = viewer
        self._nxg: "nx.DiGraph | None" = None
        self._merge_nodes: list = []
        self._merge_idx: int = 0

        self._tracks_layer_combo = create_widget(
            annotation="napari.layers.Tracks", label="Tracks Layer"
        )
        self._status_label = QLabel("No layer selected")

        self._prev_button = PushButton(text="Previous")
        self._prev_button.enabled = False
        self._nav_label = QLabel("0/0")
        self._next_button = PushButton(text="Next")
        self._next_button.enabled = False

        nav_layout = QHBoxLayout()
        nav_layout.addWidget(self._prev_button.native)
        nav_layout.addWidget(self._nav_label)
        nav_layout.addWidget(self._next_button.native)

        self._tracks_layer_combo.changed.connect(self._on_tracks_layer_changed)
        self._prev_button.clicked.connect(self._go_prev)
        self._next_button.clicked.connect(self._go_next)

        base_layout = QVBoxLayout()
        base_layout.addWidget(self._tracks_layer_combo.native)
        base_layout.addWidget(self._status_label)
        base_layout.addLayout(nav_layout)
        self.setLayout(base_layout)

        self._on_tracks_layer_changed()

    def _on_tracks_layer_changed(self):
        layer = self._tracks_layer_combo.value
        if layer is None:
            self._nxg = None
            self._merge_nodes = []
            self._merge_idx = 0
            self._status_label.setText("No layer selected")
            self._update_nav()
            return

        self._nxg = layer.metadata.get("nxg")
        if self._nxg is None:
            self._merge_nodes = []
            self._merge_idx = 0
            self._status_label.setText("No graph in layer metadata")
            self._update_nav()
            return

        self._merge_nodes = sorted(
            [n for n in self._nxg.nodes if self._nxg.in_degree(n) > 1]
        )
        self._merge_idx = 0
        n = len(self._merge_nodes)
        self._status_label.setText(f"Found {n} merge node{'s' if n != 1 else ''}")
        self._update_nav()
        if n > 0:
            self._navigate_to_current()

    def _update_nav(self):
        n = len(self._merge_nodes)
        if n == 0:
            self._nav_label.setText("0/0")
            self._prev_button.enabled = False
            self._next_button.enabled = False
        else:
            self._nav_label.setText(f"{self._merge_idx + 1}/{n}")
            self._prev_button.enabled = self._merge_idx > 0
            self._next_button.enabled = self._merge_idx < n - 1

    def _go_prev(self):
        if self._merge_idx == 0:
            return
        self._merge_idx -= 1
        self._update_nav()
        self._navigate_to_current()

    def _go_next(self):
        if self._merge_idx >= len(self._merge_nodes) - 1:
            return
        self._merge_idx += 1
        self._update_nav()
        self._navigate_to_current()

    def _navigate_to_current(self):
        if not self._merge_nodes or self._nxg is None:
            return
        merge_id = self._merge_nodes[self._merge_idx]
        node = self._nxg.nodes[merge_id]
        t = int(node["t"])
        loc_keys = ["z", "y", "x"] if "z" in node else ["y", "x"]
        spatial = tuple(float(node[k]) for k in loc_keys)

        self._viewer.dims.current_step = (t,) + (0,) * len(loc_keys)
        self._viewer.camera.center = spatial
        self._viewer.camera.zoom = 8
