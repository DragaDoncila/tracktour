"""Network flow based tracker with guided error correction"""
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("tracktour")
except PackageNotFoundError:
    __version__ = "uninstalled"
__author__ = "Draga Doncila Pop"
__email__ = "ddoncila@gmail.com"

from ._io_util import get_im_centers, load_tiff_frames
from ._tracker import Tracker

__all__ = [
    "Tracker",
    "get_im_centers",
    "load_tiff_frames",
]
