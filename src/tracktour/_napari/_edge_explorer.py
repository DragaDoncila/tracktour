from enum import Enum
from magicgui.widgets import Container, ComboBox, create_widget
import networkx as nx
import numpy as np

class EdgeGroup(Enum):
    MERGES = 'merges'
    GAP_CLOSING = 'gap_closing'
    DIVISIONS = 'divisions'
    FALSE_NEGATIVE = 'false_negative'
    FALSE_POSITIVE = 'false_positive'
    WRONG_SEMANTIC = 'wrong_semantic'
    

class EdgeExplorer(Container):
    def __init__(
        self,
        viewer: 'napari.viewer.Viewer',
    ) -> None:
        super().__init__()
        self._viewer = viewer
        self._current_layer = None

        self._graph_layer_combo = create_widget(annotation="napari.layers.Graph", label='Graph Layer')
        self._edge_group_combo = create_widget(annotation=EdgeGroup, label='Edges of Interest')

        self._graph_layer_combo.changed.connect(self._update_current_layer)
        self.extend([
            self._graph_layer_combo, 
            self._edge_group_combo
        ])

        self._update_current_layer()
        self._update_edge_group()

    
    def _update_current_layer(self, ev=None):
        chosen_layer = self._graph_layer_combo.value
        if chosen_layer is None:
            self._edge_group_combo.value = None
            self._edge_group_combo.enabled = False
            return
        
        self._current_layer = chosen_layer
        self._edge_group_combo.enabled = True
        self._edge_group_combo.value = EdgeGroup.GAP_CLOSING
        self._edge_group_combo.changed.connect(self._update_edge_group)

    def _update_edge_group(self, ev=None):
        chosen_edge_group = self._edge_group_combo.value
        if chosen_edge_group is None:
            return
        
        if chosen_edge_group == EdgeGroup.MERGES:
            self._show_merges()
        elif chosen_edge_group == EdgeGroup.GAP_CLOSING:
            self._show_gap_closing()
        elif chosen_edge_group == EdgeGroup.DIVISIONS:
            self._show_divisions()
        elif chosen_edge_group == EdgeGroup.FALSE_NEGATIVE:
            self._show_false_negative()
        elif chosen_edge_group == EdgeGroup.FALSE_POSITIVE:
            self._show_false_positive()
        elif chosen_edge_group == EdgeGroup.WRONG_SEMANTIC:
            self._show_wrong_semantic()

    def _show_merges(self):
        raise NotImplementedError

    def _show_gap_closing(self):
        napari_graph = self._current_layer
        nxg = self._current_layer.metadata["nxg"]
        if len(nx.get_edge_attributes(nxg, 'GAP_CLOSING')) == 0:
            self.annotate_gap_closing_edges(nxg)
        napari_graph.edges_visible = np.asarray(list(nx.get_edge_attributes(nxg, 'GAP_CLOSING').values()))

    def _show_divisions(self):
        raise NotImplementedError

    def _show_false_negative(self):
        raise NotImplementedError

    def _show_false_positive(self):
        raise NotImplementedError

    def _show_wrong_semantic(self):
        raise NotImplementedError
    
    def annotate_gap_closing_edges(self, nxg):
        print("Annotating gap closing edges")
        nx.set_edge_attributes(nxg, False, 'GAP_CLOSING')
        for src, dest in nxg.edges:
            if nxg.nodes[src]['t'] < nxg.nodes[dest]['t'] - 1:
                nxg.edges[src, dest]['GAP_CLOSING'] = True
