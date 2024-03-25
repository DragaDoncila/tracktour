"""Network flow based tracker with guided error correction"""
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("tracktour")
except PackageNotFoundError:
    __version__ = "uninstalled"
__author__ = "Draga Doncila Pop"
__email__ = "ddoncila@gmail.com"

from ._io_util import (
    extract_im_centers,
    get_ctc_output,
    get_im_centers,
    load_tiff_frames,
)
from ._tracker import Tracker

__all__ = [
    "Tracker",
    "extract_im_centers",
    "get_ctc_output",
    "get_im_centers",
    "load_tiff_frames",
]
