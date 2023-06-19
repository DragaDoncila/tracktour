"""Network flow based tracker with guided error correction"""

__version__ = "0.0.1"
__author__ = "Draga Doncila Pop"
__email__ = "ddoncila@gmail.com"

from ._flow_graph import FlowGraph
from ._io_util import load_graph

__all__ = [
    'FlowGraph',
    'load_graph',
]
