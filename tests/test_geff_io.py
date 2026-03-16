"""Tests for GEFF I/O (write_solution_geff, write_candidate_geff, read_geff)."""

import numpy as np
import pandas as pd
import pytest

from tracktour import (
    read_candidate_geff,
    read_geff,
    write_candidate_geff,
    write_solution_geff,
)
from tracktour._tracker import Tracked, VirtualVertices

# ---------------------------------------------------------------------------
# write_solution_geff
# ---------------------------------------------------------------------------


def test_solution_geff_round_trip_has_correct_nodes(synthetic_tracked, tmp_path):
    write_solution_geff(synthetic_tracked, tmp_path / "sol.zarr")
    g, _ = read_geff(tmp_path / "sol.zarr")
    assert set(g.nodes()) == set(synthetic_tracked.tracked_detections.index)


def test_solution_geff_round_trip_has_correct_edges(synthetic_tracked, tmp_path):
    write_solution_geff(synthetic_tracked, tmp_path / "sol.zarr")
    g, _ = read_geff(tmp_path / "sol.zarr")
    expected = set(
        zip(synthetic_tracked.tracked_edges["u"], synthetic_tracked.tracked_edges["v"])
    )
    assert set(g.edges()) == expected


def test_solution_geff_node_coords_preserved(synthetic_tracked, tmp_path):
    write_solution_geff(synthetic_tracked, tmp_path / "sol.zarr")
    g, _ = read_geff(tmp_path / "sol.zarr")
    det = synthetic_tracked.tracked_detections
    frame_key = synthetic_tracked.frame_key
    for node_id, attrs in g.nodes(data=True):
        assert attrs[frame_key] == pytest.approx(det.loc[node_id, frame_key])
        for k in synthetic_tracked.location_keys:
            assert attrs[k] == pytest.approx(det.loc[node_id, k])


def test_solution_geff_edge_cost_and_flow_preserved(synthetic_tracked, tmp_path):
    write_solution_geff(synthetic_tracked, tmp_path / "sol.zarr")
    g, _ = read_geff(tmp_path / "sol.zarr")
    edges_df = synthetic_tracked.tracked_edges.set_index(["u", "v"])
    for u, v, attrs in g.edges(data=True):
        assert attrs["flow"] == pytest.approx(edges_df.loc[(u, v), "flow"])
        assert attrs["cost"] == pytest.approx(edges_df.loc[(u, v), "cost"])


def test_solution_geff_contains_no_virtual_nodes(synthetic_tracked, tmp_path):
    write_solution_geff(synthetic_tracked, tmp_path / "sol.zarr")
    g, _ = read_geff(tmp_path / "sol.zarr")
    virtual_ids = {v.value for v in VirtualVertices}
    assert not any(n in virtual_ids for n in g.nodes())


def test_solution_geff_extra_feature_column_survives_round_trip(
    synthetic_tracked, tmp_path
):
    synthetic_tracked.tracked_edges["synthetic_feature"] = np.linspace(
        0, 1, len(synthetic_tracked.tracked_edges)
    )
    write_solution_geff(synthetic_tracked, tmp_path / "sol.zarr")
    g, _ = read_geff(tmp_path / "sol.zarr")
    edges_df = synthetic_tracked.tracked_edges.set_index(["u", "v"])
    for u, v, attrs in g.edges(data=True):
        assert attrs["synthetic_feature"] == pytest.approx(
            edges_df.loc[(u, v), "synthetic_feature"]
        )


def test_solution_geff_axis_names_stored_in_metadata(synthetic_tracked, tmp_path):
    write_solution_geff(synthetic_tracked, tmp_path / "sol.zarr")
    _, meta = read_geff(tmp_path / "sol.zarr")
    expected = [synthetic_tracked.frame_key] + list(synthetic_tracked.location_keys)
    assert [ax.name for ax in meta.axes] == expected


def test_solution_geff_overwrite_false_raises(synthetic_tracked, tmp_path):
    path = tmp_path / "sol.zarr"
    write_solution_geff(synthetic_tracked, path)
    with pytest.raises(FileExistsError):
        write_solution_geff(synthetic_tracked, path, overwrite=False)


def test_solution_geff_overwrite_true_replaces_existing(synthetic_tracked, tmp_path):
    path = tmp_path / "sol.zarr"
    write_solution_geff(synthetic_tracked, path)
    write_solution_geff(synthetic_tracked, path, overwrite=True)
    g, _ = read_geff(path)
    assert len(g.nodes()) > 0


def test_solution_geff_convenience_method_on_tracked(synthetic_tracked, tmp_path):
    path = tmp_path / "sol.zarr"
    synthetic_tracked.write_solution_geff(path)
    g, _ = read_geff(path)
    assert set(g.nodes()) == set(synthetic_tracked.tracked_detections.index)


# ---------------------------------------------------------------------------
# write_candidate_geff
# ---------------------------------------------------------------------------


def test_candidate_geff_contains_all_real_nodes(synthetic_tracked, tmp_path):
    write_candidate_geff(synthetic_tracked, tmp_path / "cand.zarr")
    g, _ = read_candidate_geff(tmp_path / "cand.zarr")
    real_ids = set(
        synthetic_tracked.all_vertices[synthetic_tracked.all_vertices.index >= 0].index
    )
    assert real_ids.issubset(set(g.nodes()))


def test_candidate_geff_contains_all_virtual_nodes(synthetic_tracked, tmp_path):
    write_candidate_geff(synthetic_tracked, tmp_path / "cand.zarr")
    g, _ = read_candidate_geff(tmp_path / "cand.zarr")
    virtual_ids = {v.value for v in VirtualVertices}
    assert virtual_ids.issubset(set(g.nodes()))


def test_candidate_geff_virtual_nodes_have_placeholder_coords(
    synthetic_tracked, tmp_path
):
    write_candidate_geff(synthetic_tracked, tmp_path / "cand.zarr")
    g, _ = read_candidate_geff(tmp_path / "cand.zarr")
    virtual_ids = {v.value for v in VirtualVertices}
    for node_id in virtual_ids:
        attrs = g.nodes[node_id]
        assert attrs[synthetic_tracked.frame_key] == -1
        for k in synthetic_tracked.location_keys:
            assert attrs[k] == -1


def test_candidate_geff_round_trip_has_correct_edges(synthetic_tracked, tmp_path):
    write_candidate_geff(synthetic_tracked, tmp_path / "cand.zarr")
    g, _ = read_candidate_geff(tmp_path / "cand.zarr")
    expected = set(
        zip(
            synthetic_tracked.all_edges["u"].astype(int),
            synthetic_tracked.all_edges["v"].astype(int),
        )
    )
    assert set(g.edges()) == expected


def test_candidate_geff_edge_cost_and_flow_preserved(synthetic_tracked, tmp_path):
    write_candidate_geff(synthetic_tracked, tmp_path / "cand.zarr")
    g, _ = read_candidate_geff(tmp_path / "cand.zarr")
    all_edges = synthetic_tracked.all_edges
    for u, v, attrs in g.edges(data=True):
        row = all_edges[(all_edges.u == u) & (all_edges.v == v)].iloc[0]
        assert attrs["cost"] == pytest.approx(row["cost"])
        assert attrs["flow"] == pytest.approx(row["flow"])


def test_candidate_geff_internal_index_column_excluded(synthetic_tracked, tmp_path):
    write_candidate_geff(synthetic_tracked, tmp_path / "cand.zarr")
    g, _ = read_candidate_geff(tmp_path / "cand.zarr")
    for _, _, attrs in g.edges(data=True):
        assert "index" not in attrs


def test_candidate_geff_extra_feature_column_survives_round_trip(
    synthetic_tracked, tmp_path
):
    synthetic_tracked.all_edges["synthetic_feature"] = np.linspace(
        0, 1, len(synthetic_tracked.all_edges)
    )
    write_candidate_geff(synthetic_tracked, tmp_path / "cand.zarr")
    g, _ = read_candidate_geff(tmp_path / "cand.zarr")
    all_edges = synthetic_tracked.all_edges
    for u, v, attrs in g.edges(data=True):
        row = all_edges[(all_edges.u == u) & (all_edges.v == v)].iloc[0]
        assert attrs["synthetic_feature"] == pytest.approx(row["synthetic_feature"])


def test_candidate_geff_raises_without_debug_data(tmp_path):
    # Build a non-debug Tracked directly (no all_edges/all_vertices)
    tracked = Tracked(
        tracked_edges=pd.DataFrame({"u": [0], "v": [1], "flow": [1], "cost": [0.1]}),
        tracked_detections=pd.DataFrame(
            {"t": [0, 1], "y": [1.0, 2.0], "x": [1.0, 2.0]}
        ),
        frame_key="t",
        location_keys=["y", "x"],
        value_key="label",
    )
    with pytest.raises(ValueError, match="DEBUG_MODE"):
        write_candidate_geff(tracked, tmp_path / "cand.zarr")


def test_candidate_geff_convenience_method_on_tracked(synthetic_tracked, tmp_path):
    path = tmp_path / "cand.zarr"
    synthetic_tracked.write_candidate_geff(path)
    g, _ = read_candidate_geff(path)
    virtual_ids = {v.value for v in VirtualVertices}
    assert virtual_ids.issubset(set(g.nodes()))


# ---------------------------------------------------------------------------
# Axis names with different dimensionalities (no Gurobi needed)
# ---------------------------------------------------------------------------


def _make_tracked_with_location_keys(location_keys):
    """Build a minimal synthetic Tracked with the given location keys."""
    n_dims = len(location_keys)
    coords = {k: [float(i)] * 2 for i, k in enumerate(location_keys)}
    tracked_detections = pd.DataFrame({"t": [0, 1], **coords})
    tracked_edges = pd.DataFrame({"u": [0], "v": [1], "flow": [1], "cost": [0.1]})

    v_ids = [v.value for v in VirtualVertices]
    v_coords = {k: [-1.0] * len(v_ids) for k in location_keys}
    all_vertices = pd.concat(
        [
            tracked_detections,
            pd.DataFrame(
                {
                    "t": [-1] * len(v_ids),
                    **{k: [-1.0] * len(v_ids) for k in location_keys},
                },
                index=v_ids,
            ),
        ]
    )
    all_edges = pd.DataFrame(
        {
            "u": [0, -2, 0],
            "v": [1, 1, -4],
            "cost": [0.1, 0.5, 0.0],
            "flow": [1, 0, 0],
            "distance": [1.0, -1.0, -1.0],
            "capacity": [2, 1, 2],
        }
    ).reset_index()

    return Tracked(
        tracked_edges=tracked_edges,
        tracked_detections=tracked_detections,
        frame_key="t",
        location_keys=list(location_keys),
        value_key="label",
        all_edges=all_edges,
        all_vertices=all_vertices,
    )


@pytest.mark.parametrize("location_keys", [("y", "x"), ("z", "y", "x")])
def test_solution_geff_axis_names_correct_for_dimensionality(location_keys, tmp_path):
    tracked = _make_tracked_with_location_keys(location_keys)
    write_solution_geff(tracked, tmp_path / "sol.zarr")
    _, meta = read_geff(tmp_path / "sol.zarr")
    assert [ax.name for ax in meta.axes] == ["t"] + list(location_keys)


@pytest.mark.parametrize("location_keys", [("y", "x"), ("z", "y", "x")])
def test_candidate_geff_axis_names_correct_for_dimensionality(location_keys, tmp_path):
    tracked = _make_tracked_with_location_keys(location_keys)
    write_candidate_geff(tracked, tmp_path / "cand.zarr")
    _, meta = read_candidate_geff(tmp_path / "cand.zarr")
    assert [ax.name for ax in meta.axes] == ["t"] + list(location_keys)
