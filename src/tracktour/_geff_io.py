"""GEFF (Graph Exchange File Format) I/O for tracktour graphs.

Provides read/write for both the solution graph (tracked edges only) and the
full candidate graph (all edges and vertices, DEBUG_MODE only).

The candidate graph includes virtual vertices (source=-1, appearance=-2,
division=-3, target=-4) with placeholder coordinates of -1 for all spatial
and time axes. Because GEFF does not support negative node IDs, these are
remapped to positive integers on write and restored on read via a stored
node attribute.
"""

import geff
import networkx as nx

# Edge columns that are graph endpoints, not attributes
_EDGE_ENDPOINT_COLS = {"u", "v"}

# Internal index column added by reset_index() in _get_all_edges
_INTERNAL_EDGE_COLS = {"index"}

# Node attribute used to store the original (possibly negative) node ID
# so that virtual nodes can be restored after GEFF round-trip
_ORIGINAL_ID_ATTR = "tracktour_node_id"


def write_solution_geff(tracked, path, overwrite=False):
    """Write the solution graph to a GEFF file.

    Stores all columns from tracked_detections as node properties and all
    columns from tracked_edges (except u and v) as edge properties.

    Parameters
    ----------
    tracked : tracktour._tracker.Tracked
        Solved tracking result.
    path : str or Path
        Output path for the zarr-backed GEFF store.
    overwrite : bool, optional
        If True, overwrite an existing GEFF at path. Default False.
    """
    g = tracked.as_nx_digraph(include_all_attrs=True)
    axis_names = _axis_names(tracked)
    geff.write(g, path, axis_names=axis_names, overwrite=overwrite)


def write_candidate_geff(tracked, path, overwrite=False):
    """Write the full candidate graph to a GEFF file.

    Only available when the Tracked object was produced with DEBUG_MODE=True.
    Virtual vertices (source=-1, appearance=-2, division=-3, target=-4) are
    included with placeholder coordinates of -1 for all spatial and time axes.
    Because GEFF requires positive node IDs, virtual nodes are remapped to
    positive integers; the original IDs are stored as a node attribute and
    restored by read_candidate_geff.

    Parameters
    ----------
    tracked : tracktour._tracker.Tracked
        Solved tracking result produced in DEBUG_MODE.
    path : str or Path
        Output path for the zarr-backed GEFF store.
    overwrite : bool, optional
        If True, overwrite an existing GEFF at path. Default False.

    Raises
    ------
    ValueError
        If tracked.all_edges or tracked.all_vertices is None (not DEBUG_MODE).
    """
    g = tracked.as_candidate_nx_digraph(include_all_attrs=True)
    g = _remap_negative_node_ids(g)
    axis_names = _axis_names(tracked)
    geff.write(g, path, axis_names=axis_names, overwrite=overwrite)


def read_geff(path):
    """Read a GEFF file into a networkx DiGraph.

    For candidate GEFFs written by write_candidate_geff, use
    read_candidate_geff instead so that virtual node IDs are restored.

    Parameters
    ----------
    path : str or Path
        Path to the zarr-backed GEFF store.

    Returns
    -------
    graph : nx.DiGraph
    metadata : geff.GeffMetadata
    """
    return geff.read(path, backend="networkx")


def read_candidate_geff(path):
    """Read a candidate GEFF written by write_candidate_geff.

    Restores original node IDs (including negative virtual node IDs) using
    the stored tracktour_node_id attribute.

    Parameters
    ----------
    path : str or Path
        Path to the zarr-backed GEFF store.

    Returns
    -------
    graph : nx.DiGraph
        Graph with original node IDs restored.
    metadata : geff.GeffMetadata
    """
    g, meta = geff.read(path, backend="networkx")
    g = _restore_node_ids(g)
    return g, meta


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _axis_names(tracked):
    return [tracked.frame_key] + list(tracked.location_keys)


def _remap_negative_node_ids(g):
    """Remap negative node IDs to positive and store originals as an attribute.

    GEFF does not support negative node IDs, so virtual nodes (-1 to -4) are
    remapped to max_real_id + |virtual_id|. The original ID is stored on every
    node as _ORIGINAL_ID_ATTR so that _restore_node_ids can invert the mapping.
    """
    max_real_id = max((n for n in g.nodes() if n >= 0), default=0)
    mapping = {n: (max_real_id + abs(n) if n < 0 else n) for n in g.nodes()}
    nx.set_node_attributes(g, {n: n for n in g.nodes()}, name=_ORIGINAL_ID_ATTR)
    return nx.relabel_nodes(g, mapping)


def _restore_node_ids(g):
    """Remap geff node IDs back to original IDs using _ORIGINAL_ID_ATTR."""
    mapping = {
        n: int(attrs[_ORIGINAL_ID_ATTR])
        for n, attrs in g.nodes(data=True)
        if _ORIGINAL_ID_ATTR in attrs
    }
    if not mapping:
        return g
    return nx.relabel_nodes(g, mapping)
