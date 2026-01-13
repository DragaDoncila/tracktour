"""Tests for TrackAnnotator utility functions."""

import numpy as np

from tracktour._napari.track_annotator.utils import (
    get_int_loc,
    get_loc_array,
    get_loc_dict,
    get_region_center,
    get_src_tgt_idx,
    split_coords,
)


class TestGetRegionCenter:
    """Tests for get_region_center function."""

    def test_simple_midpoint(self):
        """Test midpoint calculation with simple values."""
        loc1 = np.array([0.0, 0.0, 0.0])
        loc2 = np.array([10.0, 10.0, 10.0])
        result = get_region_center(loc1, loc2)
        expected = np.array([5.0, 5.0, 5.0])
        np.testing.assert_array_equal(result, expected)

    def test_negative_coordinates(self):
        """Test midpoint with negative coordinates."""
        loc1 = np.array([-5.0, -10.0])
        loc2 = np.array([5.0, 10.0])
        result = get_region_center(loc1, loc2)
        expected = np.array([0.0, 0.0])
        np.testing.assert_array_equal(result, expected)

    def test_floating_point_result(self):
        """Test that result can be floating point."""
        loc1 = np.array([1.0, 2.0, 3.0])
        loc2 = np.array([2.0, 3.0, 4.0])
        result = get_region_center(loc1, loc2)
        expected = np.array([1.5, 2.5, 3.5])
        np.testing.assert_array_equal(result, expected)


class TestGetLocArray:
    """Tests for get_loc_array function."""

    def test_2d_node(self):
        """Test extraction with 2D node (t, y, x)."""
        node_info = {"t": 5, "y": 10.5, "x": 20.3}
        result = get_loc_array(node_info)
        expected = np.array([5, 10.5, 20.3])
        np.testing.assert_array_equal(result, expected)

    def test_3d_node(self):
        """Test extraction with 3D node (t, z, y, x)."""
        node_info = {"t": 3, "z": 7.2, "y": 15.1, "x": 25.8}
        result = get_loc_array(node_info)
        expected = np.array([3, 7.2, 15.1, 25.8])
        np.testing.assert_array_equal(result, expected)

    def test_order_is_tzyx(self):
        """Test that order is always t, z, y, x."""
        node_info = {"x": 1, "y": 2, "z": 3, "t": 4}
        result = get_loc_array(node_info)
        expected = np.array([4, 3, 2, 1])  # t, z, y, x
        np.testing.assert_array_equal(result, expected)


class TestGetLocDict:
    """Tests for get_loc_dict function."""

    def test_2d_node(self):
        """Test extraction with 2D node."""
        node_info = {"t": 5, "y": 10.5, "x": 20.3, "other_attr": "ignored"}
        result = get_loc_dict(node_info)
        expected = {"t": 5, "y": 10.5, "x": 20.3}
        assert result == expected

    def test_3d_node(self):
        """Test extraction with 3D node."""
        node_info = {"t": 3, "z": 7.2, "y": 15.1, "x": 25.8}
        result = get_loc_dict(node_info)
        expected = {"t": 3, "z": 7.2, "y": 15.1, "x": 25.8}
        assert result == expected

    def test_only_location_keys_extracted(self):
        """Test that only location keys are in result."""
        node_info = {
            "t": 1,
            "y": 2,
            "x": 3,
            "track_id": 42,
            "label": 5,
            "other": "data",
        }
        result = get_loc_dict(node_info)
        assert set(result.keys()) == {"t", "y", "x"}


class TestSplitCoords:
    """Tests for split_coords function."""

    def test_2d_location(self):
        """Test splitting 2D location array."""
        loc = [5.0, 10.5, 20.3]
        result = split_coords(loc)
        expected = {"t": 5, "y": 10.5, "x": 20.3}
        assert result == expected

    def test_3d_location(self):
        """Test splitting 3D location array."""
        loc = [3.0, 7.2, 15.1, 25.8]
        result = split_coords(loc)
        expected = {"t": 3, "z": 7.2, "y": 15.1, "x": 25.8}
        assert result == expected

    def test_time_converted_to_int(self):
        """Test that time coordinate is converted to int."""
        loc = [3.7, 7.2, 15.1, 25.8]
        result = split_coords(loc)
        assert isinstance(result["t"], int)
        assert result["t"] == 3

    def test_numpy_array_input(self):
        """Test that numpy arrays work as input."""
        loc = np.array([5.0, 10.5, 20.3])
        result = split_coords(loc)
        expected = {"t": 5, "y": 10.5, "x": 20.3}
        assert result == expected


class TestGetIntLoc:
    """Tests for get_int_loc function."""

    def test_rounds_coordinates(self):
        """Test that coordinates are rounded."""
        loc = np.array([1.4, 2.5, 3.6])
        result = get_int_loc(loc)
        expected = np.array([1, 2, 4])
        np.testing.assert_array_equal(result, expected)

    def test_result_is_int_dtype(self):
        """Test that result has integer dtype."""
        loc = np.array([1.4, 2.5, 3.6])
        result = get_int_loc(loc)
        assert np.issubdtype(result.dtype, np.integer)

    def test_negative_values(self):
        """Test rounding of negative values."""
        loc = np.array([-1.4, -2.5, -3.6])
        result = get_int_loc(loc)
        expected = np.array([-1, -2, -4])
        np.testing.assert_array_equal(result, expected)

    def test_already_int_values(self):
        """Test that integer values pass through correctly."""
        loc = np.array([1.0, 2.0, 3.0])
        result = get_int_loc(loc)
        expected = np.array([1, 2, 3])
        np.testing.assert_array_equal(result, expected)


class TestGetSrcTgtIdx:
    """Tests for get_src_tgt_idx function."""

    def test_finds_disc_and_ring(self):
        """Test finding source (disc) and target (ring) indices."""
        symbols = np.array(["disc", "ring"])
        src_idx, tgt_idx = get_src_tgt_idx(symbols)
        assert src_idx == 0
        assert tgt_idx == 1

    def test_reversed_order(self):
        """Test with ring before disc."""
        symbols = np.array(["ring", "disc"])
        src_idx, tgt_idx = get_src_tgt_idx(symbols)
        assert src_idx == 1
        assert tgt_idx == 0

    def test_missing_disc_returns_none(self):
        """Test that missing disc returns None."""
        symbols = np.array(["ring", "square"])
        src_idx, tgt_idx = get_src_tgt_idx(symbols)
        assert src_idx is None
        assert tgt_idx is None

    def test_missing_ring_returns_none(self):
        """Test that missing ring returns None."""
        symbols = np.array(["disc", "square"])
        src_idx, tgt_idx = get_src_tgt_idx(symbols)
        assert src_idx is None
        assert tgt_idx is None
