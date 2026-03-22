import networkx as nx
import numpy as np
from magicgui.widgets import PushButton, create_widget
from qtpy.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout, QWidget
from superqt import QToggleSwitch

MERGE_CONTEXT_LAYER = "Merge Context"
MERGE_EDGES_LAYER = "Merge Edges"

# Okabe-Ito colourblind-friendly colours
COLOUR_MERGE = "#D55E00"  # Vermillion
COLOUR_PARENT_A = "#0072B2"  # Blue
COLOUR_PARENT_B = "#E69F00"  # Orange
COLOUR_CHILD = "#009E73"  # Bluish green
COLOUR_ACTIVE = "#FFFFFF"  # White — active parent
COLOUR_MARKED = "#808080"  # Grey — marked for exit


class MergeExplorer(QWidget):
    def __init__(
        self,
        viewer: "napari.viewer.Viewer",
    ) -> None:
        super().__init__()
        self._viewer = viewer
        self._nxg: "nx.DiGraph | None" = None
        self._tracked = None
        self._tracker = None
        self._merge_nodes: list = []
        self._merge_idx: int = 0
        self._loc_keys: list = ["y", "x"]
        self._frame_key: str = "t"
        self._active_parent: int = 0  # 0 = Parent A, 1 = Parent B
        self._oracle_corrections: dict = {}  # (parent_id, merge_id) -> 0

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

        self._parent_switch = QToggleSwitch()
        self._parent_switch.setChecked(False)  # off = Parent A, on = Parent B
        parent_layout = QHBoxLayout()
        parent_layout.addWidget(QLabel("Active parent:"))
        parent_layout.addWidget(QLabel("A"))
        parent_layout.addWidget(self._parent_switch)
        parent_layout.addWidget(QLabel("B"))
        parent_layout.addStretch()

        self._mark_exit_button = PushButton(text="Mark Active Parent as Exit")

        self._tracks_layer_combo.changed.connect(self._on_tracks_layer_changed)
        self._prev_button.clicked.connect(self._go_prev)
        self._next_button.clicked.connect(self._go_next)
        self._parent_switch.toggled.connect(self._on_parent_switched)
        self._mark_exit_button.clicked.connect(self._mark_exit)

        base_layout = QVBoxLayout()
        base_layout.addWidget(self._tracks_layer_combo.native)
        base_layout.addWidget(self._status_label)
        base_layout.addLayout(nav_layout)
        base_layout.addLayout(parent_layout)
        base_layout.addWidget(self._mark_exit_button.native)
        self.setLayout(base_layout)

        viewer.bind_key("p", self._toggle_parent)

        self._on_tracks_layer_changed()

    def _on_tracks_layer_changed(self):
        layer = self._tracks_layer_combo.value
        if layer is None:
            self._nxg = None
            self._tracked = None
            self._tracker = None
            self._merge_nodes = []
            self._merge_idx = 0
            self._active_parent = 0
            self._oracle_corrections = {}
            self._status_label.setText("No layer selected")
            self._update_nav()
            return

        self._nxg = layer.metadata.get("nxg")
        if self._nxg is None:
            self._merge_nodes = []
            self._merge_idx = 0
            self._active_parent = 0
            self._oracle_corrections = {}
            self._status_label.setText("No graph in layer metadata")
            self._update_nav()
            return

        self._tracked = layer.metadata.get("tracked")
        self._tracker = layer.metadata.get("tracker")

        if self._tracked is None:
            self._status_label.setText("Reconstructing model from solution graph…")
            try:
                from tracktour._tracker import Tracker

                # Infer keys from node attributes
                sample_attrs = next(iter(self._nxg.nodes(data=True)))[1]
                frame_key = "t"
                location_keys = ("z", "y", "x") if "z" in sample_attrs else ("y", "x")
                scale = tuple(float(s) for s in layer.scale[1:])

                tracker = Tracker(im_shape=tuple(1 for _ in location_keys), scale=scale)
                tracked = tracker.warm_start_from_solution_graph(
                    self._nxg,
                    frame_key=frame_key,
                    location_keys=location_keys,
                    scale=scale,
                )
                self._tracker = tracker
                self._tracked = tracked
                layer.metadata["tracker"] = tracker
                layer.metadata["tracked"] = tracked
            except Exception as e:
                self._status_label.setText(f"Reconstruction failed: {e}")
                self._update_nav()
                return

        self._loc_keys = list(self._tracked.location_keys)
        self._frame_key = self._tracked.frame_key
        self._active_parent = 0
        self._oracle_corrections = {}
        self._parent_switch.setChecked(False)

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
        merge_node = self._nxg.nodes[merge_id]
        spatial = tuple(float(merge_node[k]) for k in self._loc_keys)

        # Start at the earliest parent frame so the user can step forward to the merge
        predecessors = list(self._nxg.predecessors(merge_id))
        if predecessors:
            t = int(min(self._nxg.nodes[p][self._frame_key] for p in predecessors))
        else:
            t = int(merge_node[self._frame_key])

        self._viewer.dims.current_step = (t,) + (0,) * len(self._loc_keys)
        self._viewer.camera.center = spatial
        self._viewer.camera.zoom = 8

        self._show_merge_context(merge_id)

    def _toggle_parent(self, viewer=None):
        self._parent_switch.setChecked(not self._parent_switch.isChecked())

    def _on_parent_switched(self, checked: bool):
        self._active_parent = 1 if checked else 0
        if self._merge_nodes:
            self._show_merge_context(self._merge_nodes[self._merge_idx])

    def _mark_exit(self):
        if not self._merge_nodes or self._nxg is None:
            return
        merge_id = self._merge_nodes[self._merge_idx]
        predecessors = sorted(self._nxg.predecessors(merge_id))
        if self._active_parent >= len(predecessors):
            return
        parent_id = predecessors[self._active_parent]
        key = (parent_id, merge_id)
        if key in self._oracle_corrections:
            del self._oracle_corrections[key]
            self._status_label.setText(
                f"Unmarked exit: Parent {'A' if self._active_parent == 0 else 'B'} "
                f"→ merge {merge_id}"
            )
        else:
            self._oracle_corrections[key] = 0
            self._status_label.setText(
                f"Marked exit: Parent {'A' if self._active_parent == 0 else 'B'} "
                f"→ merge {merge_id}"
            )
        self._show_merge_context(merge_id)

    def _node_pos(self, node_id: int) -> np.ndarray:
        node = self._nxg.nodes[node_id]
        return np.array(
            [float(node[self._frame_key])] + [float(node[k]) for k in self._loc_keys]
        )

    def _parent_color(self, parent_id: int, merge_id: int, base_color: str) -> str:
        """Return display color for a parent node given active/correction state."""
        idx = sorted(self._nxg.predecessors(merge_id)).index(parent_id)
        if (parent_id, merge_id) in self._oracle_corrections:
            return COLOUR_MARKED
        if idx == self._active_parent:
            return COLOUR_ACTIVE
        return base_color

    def _show_merge_context(self, merge_id: int):
        nxg = self._nxg
        predecessors = sorted(nxg.predecessors(merge_id))
        successors = list(nxg.successors(merge_id))

        base_parent_colors = [COLOUR_PARENT_A, COLOUR_PARENT_B]

        # Build points: merge node, then parents (A=first, B=rest), then children
        point_ids = [merge_id] + predecessors + successors
        symbols = (
            ["star"]
            + ["disc"] * min(1, len(predecessors))
            + ["ring"] * max(0, len(predecessors) - 1)
            + ["square"] * len(successors)
        )
        colors = (
            [COLOUR_MERGE]
            + [
                self._parent_color(p, merge_id, base_parent_colors[i])
                for i, p in enumerate(predecessors)
            ]
            + [COLOUR_CHILD] * len(successors)
        )
        points_data = np.array([self._node_pos(n) for n in point_ids])

        if MERGE_CONTEXT_LAYER in self._viewer.layers:
            ctx = self._viewer.layers[MERGE_CONTEXT_LAYER]
            ctx.data = points_data
            ctx.symbol = symbols
            ctx.face_color = colors
        else:
            tracks_layer = self._tracks_layer_combo.value
            ctx = self._viewer.add_points(
                points_data,
                name=MERGE_CONTEXT_LAYER,
                symbol=symbols,
                face_color=colors,
                opacity=0.7,
                size=6,
            )
            if tracks_layer is not None:
                ctx.scale = tracks_layer.scale

        # Build vectors: shape (N, 2, D) — [start, direction]
        vectors = []
        edge_colors = []
        m = self._node_pos(merge_id)
        for i, parent in enumerate(predecessors):
            p = self._node_pos(parent)
            vectors.append([p, m - p])
            edge_colors.append(
                self._parent_color(parent, merge_id, base_parent_colors[i])
            )
        for child in successors:
            c = self._node_pos(child)
            vectors.append([m, c - m])
            edge_colors.append(COLOUR_CHILD)

        if not vectors:
            return

        vectors_data = np.array(vectors)

        if MERGE_EDGES_LAYER in self._viewer.layers:
            edges = self._viewer.layers[MERGE_EDGES_LAYER]
            edges.data = vectors_data
            edges.edge_color = edge_colors
        else:
            tracks_layer = self._tracks_layer_combo.value
            edges = self._viewer.add_vectors(
                vectors_data,
                name=MERGE_EDGES_LAYER,
                edge_color=edge_colors,
                vector_style="arrow",
                edge_width=0.3,
                out_of_slice_display=True,
            )
            if tracks_layer is not None:
                edges.scale = tracks_layer.scale
