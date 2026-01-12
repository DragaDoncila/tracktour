"""Utility functions for the TrackAnnotator widget."""

import numpy as np
from napari.utils.notifications import show_info
from qtpy.QtWidgets import QFrame, QGridLayout, QLabel, QSizePolicy


def get_separator_widget():
    """Create a horizontal separator line for UI layout.

    Returns
    -------
    QFrame
        A horizontal separator widget
    """
    separator = QFrame()
    separator.setMinimumWidth(1)
    separator.setFixedHeight(5)
    separator.setLineWidth(2)
    separator.setMidLineWidth(2)
    separator.setFrameShape(QFrame.HLine)
    separator.setFrameShadow(QFrame.Sunken)
    return separator


def get_counts_grid_layout():
    """Create a grid layout for displaying annotation counts.

    Creates a 2x6 grid with labels for TP/FP/FN counts for both
    objects and tracks.

    Returns
    -------
    QGridLayout
        Grid layout with count labels
    """
    text_labels = [
        ["TP Object: ", "FP Object: ", "FN Object: "],
        ["TP Track: ", "FP Track: ", "FN Track: "],
    ]
    grid_layout = QGridLayout()
    for row in range(2):
        for col in range(0, 6, 2):
            label = QLabel(text_labels[row][col // 2])
            label.setSizePolicy(
                QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding
            )
            grid_layout.addWidget(label, row, col)
            label = QLabel("0")
            label.setSizePolicy(
                QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding
            )
            grid_layout.addWidget(label, row, col + 1)
    return grid_layout


def get_count_label_from_grid(grid_layout, label):
    """Get a specific count label widget from the grid layout.

    Parameters
    ----------
    grid_layout : QGridLayout
        The grid layout containing count labels
    label : str
        Which count label to retrieve. One of: "TPO", "FPO", "FNO", "TPT", "FPT", "FNT"

    Returns
    -------
    QLabel
        The count label widget at the specified position
    """
    # TODO: make an enum or something so we're not using raw strings for each count label
    if label == "TPO":
        return grid_layout.itemAtPosition(0, 1).widget()
    if label == "FPO":
        return grid_layout.itemAtPosition(0, 3).widget()
    if label == "FNO":
        return grid_layout.itemAtPosition(0, 5).widget()
    if label == "TPT":
        return grid_layout.itemAtPosition(1, 1).widget()
    if label == "FPT":
        return grid_layout.itemAtPosition(1, 3).widget()
    if label == "FNT":
        return grid_layout.itemAtPosition(1, 5).widget()


def get_region_center(loc1, loc2):
    """Get the camera center-point between two locations.

    Parameters
    ----------
    loc1 : np.ndarray
        First location
    loc2 : np.ndarray
        Second location

    Returns
    -------
    np.ndarray
        Midpoint between the two locations
    """
    return (loc1 + loc2) / 2


def get_loc_array(node_info):
    """Get the location array from node info.

    Extracts the spatial-temporal coordinates from a node's attributes
    and returns them as an array in the order [t, z, y, x] or [t, y, x].

    Parameters
    ----------
    node_info : dict
        Node attributes from networkx graph

    Returns
    -------
    np.ndarray
        Location array with coordinates
    """
    loc = []
    loc.append(node_info["t"])
    if "z" in node_info:
        loc.append(node_info["z"])
    loc.append(node_info["y"])
    loc.append(node_info["x"])
    return np.asarray(loc)


def get_loc_dict(node_info):
    """Get the location dictionary from node info.

    Extracts the spatial-temporal coordinates from a node's attributes
    and returns them as a dictionary.

    Parameters
    ----------
    node_info : dict
        Node attributes from networkx graph

    Returns
    -------
    dict
        Dictionary with keys 't', 'z' (optional), 'y', 'x'
    """
    loc = {}
    loc["t"] = node_info["t"]
    if "z" in node_info:
        loc["z"] = node_info["z"]
    loc["y"] = node_info["y"]
    loc["x"] = node_info["x"]
    return loc


def split_coords(loc):
    """Split the location array into coordinate components.

    Parameters
    ----------
    loc : np.ndarray or array-like
        Location array with shape (4,) for [t, z, y, x] or (3,) for [t, y, x]

    Returns
    -------
    dict
        Dictionary with keys 't', 'z' (optional), 'y', 'x'
    """
    if len(loc) == 4:
        return {"t": int(loc[0]), "z": loc[1], "y": loc[2], "x": loc[3]}
    return {"t": int(loc[0]), "y": loc[1], "x": loc[2]}


def get_int_loc(loc):
    """Get the integer location by rounding coordinates.

    Parameters
    ----------
    loc : np.ndarray
        Location array with floating point coordinates

    Returns
    -------
    np.ndarray
        Integer location array suitable for array indexing
    """
    return np.round(loc).astype(int)


def get_src_tgt_idx(points_symbols):
    """Get the source and target indices from point symbols.

    Finds the indices of the source (disc) and target (ring) points
    in the symbols array.

    Parameters
    ----------
    points_symbols : np.ndarray or list
        Array of point symbols, should contain "disc" and "ring"

    Returns
    -------
    tuple[int | None, int | None]
        Tuple of (source_index, target_index). Returns (None, None) if
        either symbol is missing.
    """
    # TODO: make disc/ring etc. configurable
    (src_idx,) = np.where(points_symbols == "disc")
    (tgt_idx,) = np.where(points_symbols == "ring")
    if len(src_idx) == 0 or len(tgt_idx) == 0:
        show_info(
            "Missing source disc or target ring. Did you change symbols? Resetting edge."
        )
        return None, None
    src_idx = int(src_idx[0])
    tgt_idx = int(tgt_idx[0])
    return src_idx, tgt_idx
