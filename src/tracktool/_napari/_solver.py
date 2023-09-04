from magicgui.widgets import Container, ComboBox, create_widget


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
        
