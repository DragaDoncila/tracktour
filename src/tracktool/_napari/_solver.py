from magicgui.widgets import Container, PushButton, create_widget
import pandas as pd
from tracktool._io_util import get_centers, get_im_info, extract_im_centers
from tracktool._flow_graph import FlowGraph
from tracktool._napari._graph_conversion_util import _assign_all_track_id, get_napari_graph_from_nxg
from tracktool._viz_util import mask_by_id
from tracktool._graph_util import filter_to_migration_sol
from napari.qt.threading import create_worker

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
        
        # next three lines should probs be methods on FlowGraph?
        nx_g = flow_graph.convert_sol_igraph_to_nx()
        filter_to_migration_sol(nx_g)
        _assign_all_track_id(nx_g)
        sol_node_df = pd.DataFrame.from_dict(nx_g.nodes, orient="index")
        masks = mask_by_id(sol_node_df, segmentation)
        mask_layer = self._viewer.add_labels(masks, name='Solution Segmentation')
        napari_graph = get_napari_graph_from_nxg(nx_g, mask_layer)
        
        seg_layer.visible = False
        self._viewer.add_layer(napari_graph)
        
        
        
