import napari
from tracktool import load_tiff_frames

SEG_PATH = '/home/draga/PhD/data/cell_tracking_challenge/ST_Segmentations/Fluo-N2DL-HeLa/01_ST/SEG/'
IM_PATH = '/home/draga/PhD/data/cell_tracking_challenge/ST_Segmentations/Fluo-N2DL-HeLa/01/'

seg_ims = load_tiff_frames(SEG_PATH)
ims = load_tiff_frames(IM_PATH)

viewer = napari.Viewer()
viewer.add_image(ims, name='Image')
viewer.add_labels(seg_ims, name='Segmentation')
viewer.window.add_plugin_dock_widget('tracktool', 'Track Solver')

napari.run()
