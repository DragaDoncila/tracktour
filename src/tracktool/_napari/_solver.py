from magicgui.widgets import Container, PushButton, create_widget
import pandas as pd
from tracktool._io_util import extract_im_centers
from tracktool._flow_graph import FlowGraph
from tracktool._viz_util import mask_by_id
from tracktool._napari._graph_conversion_util import get_tracks_from_nxg
import networkx as nx
# from napari.qt.threading import create_worker

class TrackingSolver(Container):
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
        
        self._seg_layer_combo = create_widget(annotation="napari.layers.Labels", label='Segmentation Layer')
        self._n_neighbours_spin = create_widget(annotation="int", label="n Neighbours", options={'min':2})
        self._solve_button = PushButton(text='Solve')
        
        self._solve_button.clicked.connect(self._solve_graph)
        
        self.extend([
            self._seg_layer_combo,
            self._n_neighbours_spin,
            self._solve_button
        ])
        
    
    def _solve_graph(self):
        seg_layer = self._seg_layer_combo.value
        segmentation = seg_layer.data
        n_neighbours = self._n_neighbours_spin.value
        # centers, labels = [], []
        
        # def yield_extend(yielded):
        #     cent, lab = yielded
        #     centers.extend(cent)
        #     labels.extend(lab)
        
        # center_worker = create_worker(
        #     get_centers,
        #     segmentation=segmentation,
        #     _progress={
        #         'total':len(segmentation), 
        #         'desc':'Extracting Centers'
        #         }, 
        #     _connect={
        #         'yielded': yield_extend    
        #         }
        #     )
        # coords_df, min_t, max_t, corners = get_im_info(centers, labels, segmentation)
        coords_df, min_t, max_t, corners = extract_im_centers(segmentation)
        
        flow_graph = FlowGraph(corners, coords_df, n_neighbours=n_neighbours, min_t=min_t, max_t=max_t)
        flow_graph.solve()

        # make graph layer and tracks layer from solution        
        napari_graph_layer = flow_graph.to_napari_graph()
        subgraph = napari_graph_layer.metadata['subgraph']
        tracks_layer = get_tracks_from_nxg(subgraph)
        napari_graph_layer.metadata['tracks'] = tracks_layer
        
        # recolor segmentation and graph points by track-id
        sol_node_df = pd.DataFrame.from_dict(subgraph.nodes, orient="index")
        masks = mask_by_id(sol_node_df, segmentation)
        masked_seg = self._viewer.add_labels(masks, name='Solution Segmentation')
        
        # subgraph, tracks layer and graph layer **all** need to know about colour :<<<<
        color_dict = {node_id: (node_info['track-id'], masked_seg.get_color(node_info['track-id'])) for node_id, node_info in subgraph.nodes(data=True)}
        napari_graph_layer.face_color = [val[1] for val in color_dict.values()]
        napari_graph_layer.metadata['tracks'].metadata={'colors': dict([(val[0], val[1]) for val in color_dict.values()])}
        nx.set_node_attributes(subgraph, {k: v[1] for k, v in color_dict.items()}, 'color')

        self._viewer.add_layer(napari_graph_layer)
        seg_layer.visible = False
        
        
        
