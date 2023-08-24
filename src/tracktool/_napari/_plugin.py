from enum import Enum, auto
from typing import Any, Sequence
import warnings
from magicgui import magic_factory
from magicgui.widgets import Container, ComboBox, create_widget
import numpy as np
import networkx as nx

# class MERGE_SET(Enum):
#     # pre-capacity change, pre oracle
#     OG_MERGE = auto()
#     # pre-capacity change, post oracle
#     # ORIGINAL_FINAL = auto()
#     # post-capacity change original merge vertices
#     OG_NEW = auto()
#     # post-capacity change, pre oracle
#     NEW_MERGE = auto()
#     # post-capacity change, post oracle
#     # NEW_FINAL = auto()
#     # post-capacity change new merges
#     NEW_NEW = auto()


# def get_layer_from_merge_set(merge_set: MERGE_SET):
#     if merge_set == MERGE_SET.OG_MERGE:
#         layer_name = ''

DIMGREY = np.array([0.4, 0.4, 0.4, 1])

class MockEvent:
    def __init__(self, position):
        self.position = position

class MergeExplorer(Container):
    def __init__(
        self,
        viewer: 'napari.viewer.Viewer',
        layout: str = "vertical",
        labels: bool = True,
    ) -> None:
        super().__init__(
            layout=layout,
            labels=labels,
        )
        self._viewer = viewer
        self._lineage_dc = None
        
        self._graph_layer_combo = create_widget(annotation="napari.layers.Graph", label='Graph Layer')
        self._merge_id_combo = ComboBox(label="Merged Cells")
        self._focus_mode_check = create_widget(annotation=bool, label="Focus Mode")
        
        self._graph_layer_combo.changed.connect(self._update_merge_node_choices)
        self._graph_layer_combo.changed.connect(self._update_lineage_callback)
        self._merge_id_combo.changed.connect(self._show_selected_merge)
        self._focus_mode_check.changed.connect(self._handle_focus_mode_change)
        
        dw, self._arboretum = viewer.window.add_plugin_dock_widget("napari-arboretum", "Arboretum")
        
        self.extend([self._graph_layer_combo, self._merge_id_combo, self._focus_mode_check])    
        self._update_merge_node_choices()
        self._update_lineage_callback()
        
    def _update_merge_node_choices(self):
        chosen_layer = self._graph_layer_combo.value
        if "nxg" in chosen_layer.metadata:
            nxg = chosen_layer.metadata["nxg"]
        else:
            nxg = chosen_layer.data.to_networkx()
            chosen_layer.metadata["nxg"] = nxg

        def get_choices(_merge_id_combo):
            merge_nodes = [node for node in nxg.nodes if nxg.in_degree(node) > 1]
            if not len(merge_nodes):
                warnings.warn(
                    f"Graph {chosen_layer.name} contains no merges! Try a different layer."
                )
            return merge_nodes

        self._merge_id_combo.choices = get_choices
    
    def _update_lineage_callback(self):
        if self._lineage_dc is not None:
            self._lineage_dc()
        chosen_layer = self._graph_layer_combo.value
        self._lineage_dc = chosen_layer.mouse_double_click_callbacks.append(self._show_lineage_tree)
        
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
            
    def _show_selected_merge(self):
        merge_id = self._merge_id_combo.value
        graph_layer = self._graph_layer_combo.value
        
        if merge_id is None:
            return
        # make this the only visible graph layer
        for layer in self._viewer.layers:
            if layer._type_string == "graph":
                layer.visible = False
        graph_layer.visible = True

        # get nxg from graph_layer
        nxg = graph_layer.metadata["nxg"]
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
        if self._focus_mode_check.value:
            self.color_nodes_in_tree(chosen_layer)
        else:
            if "old-color" in chosen_layer.metadata:
                nxg = chosen_layer.metadata['nxg']
                nx.set_node_attributes(nxg, chosen_layer.metadata["old-color"], "color")            
                chosen_layer.face_color = list(nx.get_node_attributes(nxg, "color").values())

    def color_nodes_in_tree(self, layer=None, ev=None):
        nxg = layer.metadata['nxg']
        if self._arboretum.plotter.has_nodes:
            # restore face color
            if "old-color" in layer.metadata:
                nx.set_node_attributes(nxg, layer.metadata["old-color"], "color")
            # store restored color
            layer.metadata["old-color"] = nx.get_node_attributes(nxg, "color")
            nx.set_node_attributes(nxg, DIMGREY, "color")

            shown_tids = set([node.ID for node in self._arboretum.plotter._current_nodes])
            tid_node_color = {}
            # recolor nodes and set face color
            for tid in shown_tids:
                tid_node_color.update(
                    {
                        node: layer.metadata["old-color"][node]
                        for node in nxg.nodes
                        if nxg.nodes[node]["track-id"] == tid
                    }
                )
            nx.set_node_attributes(nxg, tid_node_color, "color")
            layer.face_color = list(nx.get_node_attributes(nxg, "color").values())

