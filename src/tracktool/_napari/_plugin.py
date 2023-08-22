from enum import Enum, auto
import warnings
from magicgui import magic_factory
import numpy as np
from skimage.measure import points_in_poly
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

def connect_merge_node_updates(merge_selection_widget):
    merge_selection_widget.graph_layer.changed.connect(
        lambda _: _update_merge_nodes(merge_selection_widget)
    )
    _update_merge_nodes(merge_selection_widget)

def _update_merge_nodes(merge_selection_widget):
    chosen_layer = merge_selection_widget.graph_layer.value
    if 'nxg' in chosen_layer.metadata:
        nxg = chosen_layer.metadata['nxg']
    else:
        nxg = chosen_layer.data.to_networkx()
        chosen_layer.metadata['nxg'] = nxg
    
    def get_choices(merge_node_combo):
        merge_nodes = [node for node in nxg.nodes if nxg.in_degree(node) > 1]
        if not len(merge_nodes):
            warnings.warn(f'Graph {chosen_layer.name} contains no merges! Try a different layer.')
        return merge_nodes

    merge_selection_widget.merge_id.choices = get_choices


@magic_factory(
        widget_init=connect_merge_node_updates,
        merge_id = {"widget_type": "ComboBox"},
        auto_call = True
)
def filter_merge_sets(graph_layer: 'napari.layers.Graph', merge_id, viewer:'napari.Viewer'):
    if merge_id is None:
        return
    # make this the only visible graph layer
    for layer in viewer.layers:
        if layer._type_string == 'graph':
           layer.visible = False
    graph_layer.visible = True
     
    # get nxg from graph_layer
    nxg = graph_layer.metadata['nxg']
    node_info = nxg.nodes[merge_id]
    
    # center camera
    viewer.dims.current_step = (node_info['t'], 0, 0)
    viewer.camera.center = (node_info['y'], node_info['x'])
    viewer.camera.zoom = 6
    
    # add dock widget (or just retrieve) and show tree for selected merge
    tracks_layer = graph_layer.metadata['tracks']
    position = (node_info['t'], node_info['y'], node_info['x'])
    dw, widget = viewer.window.add_plugin_dock_widget("napari-arboretum", "Arboretum")
    ev = MockEvent(position)
    widget.show_tree(tracks_layer, ev)
    viewer.layers.selection.select_only(graph_layer)

    
    # if this is the first time we've launched the widget, also add the double-click callback
    if 'show_cb' not in graph_layer.metadata:
        def show_tree_on_double_click(layer, event):
            tracks_layer = layer.metadata['tracks']
            ev = MockEvent(event.position)
            widget.show_tree(tracks_layer, ev)
            viewer.layers.selection.select_only(graph_layer)
        dc = graph_layer.mouse_double_click_callbacks.append(show_tree_on_double_click)
        graph_layer.metadata['show_cb'] = dc
    

    if widget.plotter.has_nodes and 'lineage_bbox' in viewer.layers:
        # restore face color (where are we keeping it?)
        if 'old-color' in graph_layer.metadata:
            nx.set_node_attributes(nxg, graph_layer.metadata['old-color'], 'color')
        # store restored color
        graph_layer.metadata['old-color'] = nx.get_node_attributes(nxg, 'color')
        nx.set_node_attributes(nxg, DIMGREY, 'color')

        shown_tids = set([node.ID for node in widget.plotter._current_nodes])
        tid_node_color = {}
        # recolor nodes and set face color
        for tid in shown_tids:
            tid_node_color.update({node: graph_layer.metadata['old-color'][node] for node in nxg.nodes if nxg.nodes[node]['track-id'] == tid})
        nx.set_node_attributes(nxg, tid_node_color, 'color')
        graph_layer.face_color = list(nx.get_node_attributes(nxg, "color").values())
    # how/when do we get the original colors restored?
