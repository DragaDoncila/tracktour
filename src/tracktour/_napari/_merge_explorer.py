import warnings

import networkx as nx
import numpy as np
from magicgui.widgets import ComboBox, Container, create_widget

DIMGREY = np.array([0.4, 0.4, 0.4, 1])
BBOX_LAYER_NAME = "Focus Box"


class MockEvent:
    def __init__(self, position):
        self.position = position


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
        self._lineage_dc = None
        self._bbox_dc = None
        self._old_color = None

        self._graph_layer_combo = create_widget(
            annotation="napari.layers.Graph", label="Graph Layer"
        )
        self._merge_id_combo = ComboBox(label="Merged Cells")
        self._focus_mode_check = create_widget(annotation=bool, label="Focus Mode")

        self._graph_layer_combo.changed.connect(self._update_merge_node_choices)
        self._graph_layer_combo.changed.connect(self._update_lineage_callback)
        self._merge_id_combo.changed.connect(self._show_selected_merge)
        self._focus_mode_check.changed.connect(self._handle_focus_mode_change)

        dw, self._arboretum = viewer.window.add_plugin_dock_widget(
            "napari-arboretum", "Arboretum"
        )

        self.extend(
            [self._graph_layer_combo, self._merge_id_combo, self._focus_mode_check]
        )
        self._update_merge_node_choices()
        self._update_lineage_callback()

    def _update_merge_node_choices(self):
        chosen_layer = self._graph_layer_combo.value
        # TODO: cleanup on layer delete isn't really working
        if chosen_layer is None:
            self._merge_id_combo.choices = []
            self._focus_mode_check.value = False
            self._handle_focus_mode_change()
            return

        if "subgraph" in chosen_layer.metadata:
            nxg = chosen_layer.metadata["subgraph"]
        else:
            nxg = chosen_layer.data.to_networkx()
            chosen_layer.metadata["subgraph"] = nxg

        def get_choices(_merge_id_combo):
            merge_nodes = [
                (
                    f'Track {nxg.nodes[node]["track-id"]}, Frame {nxg.nodes[node]["t"]}',
                    node,
                )
                for node in nxg.nodes
                if nxg.in_degree(node) > 1
            ]
            if not len(merge_nodes):
                warnings.warn(
                    f"Graph {chosen_layer.name} contains no merges! Try a different layer."
                )
            return merge_nodes

        self._merge_id_combo.choices = get_choices

    def _update_lineage_callback(self):
        chosen_layer = self._graph_layer_combo.value
        if chosen_layer is None:
            return
        if self._lineage_dc is not None:
            chosen_layer.mouse_double_click_callbacks.remove(self._lineage_dc)
        self._lineage_dc = chosen_layer.mouse_double_click_callbacks.append(
            self._show_lineage_tree
        )

    def _show_lineage_tree(self, layer, event):
        if self._arboretum is None:
            return
        tracks_layer = layer.metadata["tracks"]
        position = event.position
        ev = MockEvent(position)
        self._arboretum.show_tree(tracks_layer, ev)

        # if focus mode is on
        if self._focus_mode_check.value:
            self.color_nodes_in_tree(layer)
            self.draw_current_bounding_box()

    def _show_selected_merge(self):
        merge_id = self._merge_id_combo.value
        graph_layer = self._graph_layer_combo.value
        if merge_id is None or graph_layer is None:
            return
        # make this the only visible graph layer
        for layer in self._viewer.layers:
            if layer._type_string == "graph":
                layer.visible = False
        graph_layer.visible = True

        # get nxg from graph_layer
        nxg = graph_layer.metadata["subgraph"]
        node_info = nxg.nodes[merge_id]

        # center camera
        self._viewer.dims.current_step = (node_info["t"], 0, 0)
        self._viewer.camera.center = (node_info["y"], node_info["x"])
        self._viewer.camera.zoom = 6

        # show lineage tree
        event = MockEvent((node_info["t"], node_info["y"], node_info["x"]))
        self._show_lineage_tree(graph_layer, event)
        self._viewer.layers.selection.select_only(graph_layer)

    def _handle_focus_mode_change(self):
        chosen_layer = self._graph_layer_combo.value
        if chosen_layer is None:
            return

        if self._focus_mode_check.value:
            self.color_nodes_in_tree(chosen_layer)
            self._bbox_dc = self._viewer.dims.events.current_step.connect(
                self.draw_current_bounding_box
            )
            self.draw_current_bounding_box()
        else:
            if self._old_color is not None:
                nxg = chosen_layer.metadata["subgraph"]
                nx.set_node_attributes(nxg, self._old_color, "color")
                chosen_layer.face_color = list(
                    nx.get_node_attributes(nxg, "color").values()
                )
            if self._bbox_dc is not None:
                self._viewer.layers.remove(BBOX_LAYER_NAME)
                self._viewer.dims.events.current_step.disconnect(self._bbox_dc)
                self._bbox_dc = None

    def color_nodes_in_tree(self, layer=None, ev=None):
        nxg = layer.metadata["subgraph"]
        if self._arboretum.plotter.has_plot:
            # restore face color
            if self._old_color is not None:
                nx.set_node_attributes(nxg, self._old_color, "color")
            # store restored color
            self._old_color = nx.get_node_attributes(nxg, "color")
            nx.set_node_attributes(nxg, DIMGREY, "color")

            shown_tids = set(
                [node.ID for node in self._arboretum.plotter._current_nodes]
            )
            tid_node_color = {}
            # recolor nodes and set face color
            for tid in shown_tids:
                tid_node_color.update(
                    {
                        node: self._old_color[node]
                        for node in nxg.nodes
                        if nxg.nodes[node]["track-id"] == tid
                    }
                )
            nx.set_node_attributes(nxg, tid_node_color, "color")
            layer.face_color = list(nx.get_node_attributes(nxg, "color").values())

    def draw_current_bounding_box(self, event=None):
        if not self._arboretum.plotter.has_plot:
            return
        z_value = self._viewer.dims.current_step[0]
        shown_at_t = [
            node for node in self._arboretum.plotter._current_nodes if z_value in node.t
        ]
        tdata = self._arboretum.tracks.data
        positions = [
            np.squeeze(
                tdata[np.logical_and(tdata[:, 0] == node.ID, tdata[:, 1] == z_value)][
                    :, 2:
                ]
            )
            for node in shown_at_t
        ]
        min_x, min_y = tuple(positions[0])
        max_x, max_y = min_x, min_y
        for position in positions[1:]:
            if (x := position[0]) < min_x:
                min_x = x
            elif (x := position[0]) > max_x:
                max_x = x

            if (y := position[1]) < min_y:
                min_y = y
            elif (y := position[1]) > max_y:
                max_y = y
        bbox_corners = [(min_x - 10, min_y - 10), (max_x + 10, max_y + 10)]
        if BBOX_LAYER_NAME in self._viewer.layers:
            bbox_layer = self._viewer.layers[BBOX_LAYER_NAME]
            bbox_layer.data = []
        else:
            bbox_layer = self._viewer.add_shapes(name=BBOX_LAYER_NAME)
        bbox_layer.add_rectangles(
            [bbox_corners], face_color=[(0, 0, 0, 0)], edge_color=[(1, 1, 1, 1)]
        )
