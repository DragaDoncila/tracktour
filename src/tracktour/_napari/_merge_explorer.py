import networkx as nx
import numpy as np
from magicgui.widgets import PushButton, create_widget
from qtpy.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout, QWidget
from superqt import QToggleSwitch

TRACK_NODES_LAYER = "Track Nodes"

# Okabe-Ito colourblind-friendly colours as RGBA float tuples
COLOUR_DEFAULT = (0.667, 0.667, 0.667, 1.0)  # Grey — default node
COLOUR_MERGE = (0.835, 0.369, 0.0, 1.0)  # Vermillion (#D55E00)
COLOUR_PARENT_A = (0.0, 0.447, 0.698, 1.0)  # Blue (#0072B2)
COLOUR_PARENT_B = (0.902, 0.624, 0.0, 1.0)  # Orange (#E69F00)
COLOUR_CHILD = (0.0, 0.620, 0.451, 1.0)  # Bluish green (#009E73)
COLOUR_ACTIVE = (1.0, 1.0, 1.0, 1.0)  # White — active parent
COLOUR_MARKED = (0.502, 0.502, 0.502, 1.0)  # Grey — marked for exit


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
        self._next_node_idx: int = 0  # counter for new node IDs
        self._oracle_corrections: dict = {}  # (parent_id, merge_id) -> 0
        self._node_id_to_point_idx: dict = {}  # node_id -> index in TRACK_NODES_LAYER
        self._point_idx_to_node_id: dict = {}  # index in TRACK_NODES_LAYER -> node_id
        self._node_positions: dict = {}  # node_id -> np.ndarray snapshot
        self._moved_point_indices: set = set()  # point indices changed since last build

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
        self._re_solve_button = PushButton(text="Re-solve")

        self._tracks_layer_combo.changed.connect(self._on_tracks_layer_changed)
        self._prev_button.clicked.connect(self._go_prev)
        self._next_button.clicked.connect(self._go_next)
        self._parent_switch.toggled.connect(self._on_parent_switched)
        self._mark_exit_button.clicked.connect(self._mark_exit)
        self._re_solve_button.clicked.connect(self._re_solve)

        base_layout = QVBoxLayout()
        base_layout.addWidget(self._tracks_layer_combo.native)
        base_layout.addWidget(self._status_label)
        base_layout.addLayout(nav_layout)
        base_layout.addLayout(parent_layout)
        base_layout.addWidget(self._mark_exit_button.native)
        base_layout.addWidget(self._re_solve_button.native)
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
            self._node_id_to_point_idx = {}
            self._point_idx_to_node_id = {}
            self._node_positions = {}
            self._moved_point_indices = set()
            self._status_label.setText("No layer selected")
            self._update_nav()
            return

        self._nxg = layer.metadata.get("nxg")
        if self._nxg is None:
            self._merge_nodes = []
            self._merge_idx = 0
            self._active_parent = 0
            self._oracle_corrections = {}
            self._node_id_to_point_idx = {}
            self._point_idx_to_node_id = {}
            self._node_positions = {}
            self._moved_point_indices = set()
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

                # Infer im_shape from the spatial columns of the tracks layer
                # data (columns 2+ are spatial after id and time).  Use
                # ceil(max) + 1 so every node position sits strictly inside
                # the image boundary.
                spatial_data = layer.data[:, 2:]
                im_shape = tuple(
                    int(np.ceil(spatial_data[:, i].max())) + 1
                    for i in range(len(location_keys))
                )
                tracker = Tracker(im_shape=im_shape, scale=scale)
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
        self._node_id_to_point_idx = {}
        self._node_positions = {}
        self._moved_point_indices = set()
        self._parent_switch.setChecked(False)

        self._merge_nodes = sorted(
            [n for n in self._nxg.nodes if self._nxg.in_degree(n) > 1]
        )
        self._merge_idx = -1
        self._next_node_idx = self._nxg.number_of_nodes()
        n = len(self._merge_nodes)
        self._status_label.setText(f"Found {n} merge node{'s' if n != 1 else ''}")
        self._update_nav()
        self._build_nodes_layer()

    def _build_nodes_layer(self):
        """Create or refresh TRACK_NODES_LAYER with all graph nodes as grey discs."""
        if self._nxg is None:
            return

        node_ids = list(self._nxg.nodes)
        positions = np.array([self._node_pos(n) for n in node_ids])

        self._node_id_to_point_idx = {nid: i for i, nid in enumerate(node_ids)}
        self._point_idx_to_node_id = {i: nid for i, nid in enumerate(node_ids)}
        self._node_positions = {
            nid: positions[i].copy() for i, nid in enumerate(node_ids)
        }
        self._moved_point_indices = set()

        tracks_layer = self._tracks_layer_combo.value
        n_nodes = len(node_ids)

        if TRACK_NODES_LAYER in self._viewer.layers:
            nodes_layer = self._viewer.layers[TRACK_NODES_LAYER]
            # Disconnect old callback before replacing data
            try:
                nodes_layer.events.data.disconnect(self._on_nodes_data_changed)
            except Exception:
                pass
            nodes_layer.data = positions
            nodes_layer.symbol = ["disc"] * n_nodes
            nodes_layer.face_color = [COLOUR_DEFAULT] * n_nodes
        else:
            nodes_layer = self._viewer.add_points(
                positions,
                name=TRACK_NODES_LAYER,
                symbol=["disc"] * n_nodes,
                face_color=[COLOUR_DEFAULT] * n_nodes,
                opacity=0.7,
                size=6,
            )
            if tracks_layer is not None:
                nodes_layer.scale = tracks_layer.scale

        nodes_layer.events.data.connect(self._on_nodes_data_changed)

    def _on_nodes_data_changed(self, event):
        if event.action == "changed":
            for idx in event.data_indices:
                self._moved_point_indices.add(int(idx))

    def _update_nav(self):
        n = len(self._merge_nodes)
        if n == 0:
            self._nav_label.setText("0/0")
            self._prev_button.enabled = False
            self._next_button.enabled = False
        elif self._merge_idx == -1:
            self._nav_label.setText(f"—/{n}")
            self._prev_button.enabled = False
            self._next_button.enabled = True
        else:
            self._nav_label.setText(f"{self._merge_idx + 1}/{n}")
            self._prev_button.enabled = self._merge_idx > 0
            self._next_button.enabled = self._merge_idx < n - 1

    def _go_prev(self):
        if self._merge_idx <= 0:
            return
        self._merge_idx -= 1
        self._update_nav()
        self._navigate_to_current()

    def _go_next(self):
        if self._merge_idx >= len(self._merge_nodes) - 1:
            return
        self._merge_idx = max(0, self._merge_idx + 1)
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
        if TRACK_NODES_LAYER not in self._viewer.layers:
            return

        nodes_layer = self._viewer.layers[TRACK_NODES_LAYER]
        n_points = len(nodes_layer.data)

        # Reset all points to default style
        nodes_layer.symbol = ["disc"] * n_points
        nodes_layer.face_color = [COLOUR_DEFAULT] * n_points

        nxg = self._nxg
        predecessors = sorted(nxg.predecessors(merge_id))
        successors = list(nxg.successors(merge_id))
        base_parent_colors = [COLOUR_PARENT_A, COLOUR_PARENT_B]

        # Style merge node
        if merge_id in self._node_id_to_point_idx:
            idx = self._node_id_to_point_idx[merge_id]
            nodes_layer.symbol[idx] = "star"
            nodes_layer.face_color[idx] = COLOUR_MERGE

        # Style parents
        for i, parent_id in enumerate(predecessors):
            if parent_id not in self._node_id_to_point_idx:
                continue
            idx = self._node_id_to_point_idx[parent_id]
            symbol = "disc" if i == 0 else "ring"
            color = self._parent_color(parent_id, merge_id, base_parent_colors[i])
            nodes_layer.symbol[idx] = symbol
            nodes_layer.face_color[idx] = color

        # Style children
        for child_id in successors:
            if child_id not in self._node_id_to_point_idx:
                continue
            idx = self._node_id_to_point_idx[child_id]
            nodes_layer.symbol[idx] = "square"
            nodes_layer.face_color[idx] = COLOUR_CHILD

        # Trigger layer refresh
        nodes_layer.refresh()

        nodes_layer.visible = True

    def _re_solve(self):
        if self._tracker is None or self._tracked is None:
            self._status_label.setText("No live model available.")
            return

        tracker = self._tracker
        tracked = self._tracked

        model_changed = False
        # Collect all position changes before touching edges so that the single
        # KD-tree rebuild and the subsequent k-NN searches see every node at its
        # final position.
        moved_nodes: dict = {}  # node_id -> (frame, spatial)
        new_nodes: list = []  # [(point_idx, node_id, frame, spatial)]

        if TRACK_NODES_LAYER in self._viewer.layers:
            nodes_layer = self._viewer.layers[TRACK_NODES_LAYER]

            for point_idx in self._moved_point_indices:
                node_id = self._point_idx_to_node_id.get(point_idx)
                if node_id is None:
                    continue
                current_pos = nodes_layer.data[point_idx]
                stored_pos = self._node_positions[node_id]
                if not np.allclose(current_pos, stored_pos):
                    model_changed = True
                    frame = int(current_pos[0])
                    spatial = tuple(float(v) for v in current_pos[1:])
                    moved_nodes[node_id] = (frame, spatial)

            n_known = len(self._node_id_to_point_idx)
            for i in range(n_known, len(nodes_layer.data)):
                model_changed = True
                raw_pos = nodes_layer.data[i]
                frame = int(raw_pos[0])
                spatial = tuple(float(v) for v in raw_pos[1:])
                node_id = self._next_node_idx
                self._next_node_idx += 1
                new_nodes.append((i, node_id, frame, spatial))

        # Phase 1: update all positions in tracked_detections and reset edges.
        for node_id, (frame, spatial) in moved_nodes.items():
            tracker._prepare_move_node(tracked, node_id, frame, spatial)
        for _, node_id, frame, spatial in new_nodes:
            tracker._prepare_add_node(tracked, node_id, frame, spatial)

        # Single KD-tree rebuild for all affected frames.
        affected_frames = [frame for frame, _ in moved_nodes.values()] + [
            frame for _, _, frame, _ in new_nodes
        ]
        if affected_frames:
            tracker.rebuild_kd_trees(tracked, affected_frames)
            # Reset bounds on ALL migration edges touching the affected frames so
            # that nearby nodes (not just the moved/added ones) can be reassigned.
            tracker._reset_frame_edges(tracked, affected_frames)

        # Phase 2: recompute edges and constraints with all nodes at final positions.
        for node_id, (frame, spatial) in moved_nodes.items():
            tracker._apply_node_edges(tracked, node_id, frame, spatial)
        for i, node_id, frame, spatial in new_nodes:
            tracker._apply_node_edges(tracked, node_id, frame, spatial)
            self._nxg.add_node(
                node_id,
                **{self._frame_key: frame, **dict(zip(self._loc_keys, spatial))},
            )
            self._node_id_to_point_idx[node_id] = i
            self._point_idx_to_node_id[i] = node_id
            self._node_positions[node_id] = nodes_layer.data[i].copy()

        for (u, v), oracle_val in self._oracle_corrections.items():
            model_changed = True
            tracked.all_edges, edge_idx = tracker.ensure_candidate_edge(
                tracked.all_edges, u, v
            )
            tracker.fix_edge_in_model(edge_idx, u, v, lb=oracle_val, ub=oracle_val)

        if not model_changed:
            self._status_label.setText("No corrections to apply.")
            return

        tracker._model.optimize()
        if tracker._model.status != 2:
            self._status_label.setText("Re-solve failed — model infeasible.")
            return

        tracker._store_solution(tracker._model, tracked.all_edges)

        all_edges = tracked.all_edges
        migration_edges = all_edges[
            (all_edges.u >= 0) & (all_edges.v >= 0) & (all_edges.flow > 0)
        ].copy()
        tracked.tracked_edges = migration_edges
        new_nxg = tracked.as_nx_digraph()

        self._nxg = new_nxg
        self._oracle_corrections = {}

        layer = self._tracks_layer_combo.value
        if layer is not None:
            layer.metadata["nxg"] = new_nxg
            from ._graph_conversion_util import get_tracks_from_nxg

            new_tracks = get_tracks_from_nxg(new_nxg)
            layer.data = new_tracks.data
            layer.graph = new_tracks.graph

        self._merge_nodes = sorted(
            [n for n in new_nxg.nodes if new_nxg.in_degree(n) > 1]
        )
        n = len(self._merge_nodes)
        self._merge_idx = -1
        self._status_label.setText(
            f"Re-solved. Found {n} merge node{'s' if n != 1 else ''}"
        )

        # Rebuild TRACK_NODES_LAYER to reflect updated graph and reset tracking state
        self._build_nodes_layer()

        self._update_nav()
