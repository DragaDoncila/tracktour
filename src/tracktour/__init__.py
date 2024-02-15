"""Network flow based tracker with guided error correction"""
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("tracktour")
except PackageNotFoundError:
    __version__ = "uninstalled"
__author__ = "Draga Doncila Pop"
__email__ = "ddoncila@gmail.com"

from ._flow_graph import FlowGraph
from ._graph_util import get_traccuracy_graph, get_traccuracy_graph_nx, load_gt_info
from ._io_util import get_im_centers, load_graph, load_sol_flow_graph, load_tiff_frames
from ._napari._graph_conversion_util import get_tracks_from_nxg
from ._tracker import Tracker

__all__ = [
    "FlowGraph",
    "Tracker",
    "get_im_centers",
    "load_gt_info",
    "load_graph",
    "load_sol_flow_graph",
    "load_tiff_frames",
    "get_traccuracy_graph",
    "get_traccuracy_graph_nx",
    "get_tracks_from_nxg",
]
