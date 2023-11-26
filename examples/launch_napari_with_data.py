import napari
from tracktool import load_tiff_frames

SEG_PATH = '/home/ddon0001/PhD/data/cell_tracking_challenge/ST/Fluo-N2DL-HeLa/02_ST/SEG/'
IM_PATH = '/home/ddon0001/PhD/data/cell_tracking_challenge/ST/Fluo-N2DL-HeLa/02/'

seg_ims = load_tiff_frames(SEG_PATH)
ims = load_tiff_frames(IM_PATH)

viewer = napari.Viewer()
viewer.add_image(ims, name='Image')
viewer.add_labels(seg_ims, name='Segmentation')
viewer.window.add_plugin_dock_widget('tracktool', 'Track Solver')
# viewer.window.add_plugin_dock_widget('tracktool', 'Merge Explorer')
# viewer.window.add_plugin_dock_widget('tracktool', 'Track Editor')

napari.run()
