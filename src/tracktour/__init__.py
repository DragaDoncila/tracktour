"""Network flow based tracker with guided error correction"""
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("tracktour")
except PackageNotFoundError:
    __version__ = "uninstalled"
__author__ = "Draga Doncila Pop"
__email__ = "ddoncila@gmail.com"

from ._features import (
    assign_all_features,
    assign_migration_features,
    assign_probability_features,
    assign_sensitivity_features,
)
from ._geff_io import (
    read_candidate_geff,
    read_geff,
    write_candidate_geff,
    write_solution_geff,
)
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
    "assign_migration_features",
    "assign_sensitivity_features",
    "assign_probability_features",
    "assign_all_features",
    "read_geff",
    "read_candidate_geff",
    "write_solution_geff",
    "write_candidate_geff",
]
