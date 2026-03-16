"""Tests for edge feature assignment functions."""

import numpy as np
import pytest
from scipy.special import softmax as scipy_softmax

from tracktour import (
    assign_all_features,
    assign_migration_features,
    assign_probability_features,
    assign_sensitivity_features,
)

# ---------------------------------------------------------------------------
# assign_migration_features
# ---------------------------------------------------------------------------


def test_migration_features_returns_expected_column_names(synthetic_tracked):
    cols = assign_migration_features(synthetic_tracked.all_edges)
    assert cols == ["distance", "chosen_neighbour_rank"]


def test_migration_features_closest_neighbour_has_rank_zero(synthetic_tracked):
    assign_migration_features(synthetic_tracked.all_edges)
    migration = synthetic_tracked.all_edges[
        (synthetic_tracked.all_edges.u >= 0) & (synthetic_tracked.all_edges.v >= 0)
    ]
    for u, group in migration.groupby("u"):
        closest_idx = group["distance"].idxmin()
        assert group.loc[closest_idx, "chosen_neighbour_rank"] == 0


def test_migration_features_rank_matches_distance_order(synthetic_tracked):
    assign_migration_features(synthetic_tracked.all_edges)
    migration = synthetic_tracked.all_edges[
        (synthetic_tracked.all_edges.u >= 0) & (synthetic_tracked.all_edges.v >= 0)
    ]
    for u, group in migration.groupby("u"):
        sorted_by_dist = group.sort_values("distance")
        assert list(sorted_by_dist["chosen_neighbour_rank"]) == list(
            range(len(sorted_by_dist))
        )


def test_migration_features_virtual_edges_unchanged(synthetic_tracked):
    assign_migration_features(synthetic_tracked.all_edges)
    virtual = synthetic_tracked.all_edges[synthetic_tracked.all_edges.u < 0]
    assert (virtual["chosen_neighbour_rank"] == -1).all()


# ---------------------------------------------------------------------------
# assign_probability_features
# ---------------------------------------------------------------------------


def test_probability_features_returns_expected_column_names(synthetic_tracked):
    cols = assign_probability_features(synthetic_tracked.all_edges)
    assert cols == ["softmax", "softmax_entropy", "parental_softmax"]


def test_probability_features_softmax_sums_to_one_per_target(synthetic_tracked):
    assign_probability_features(synthetic_tracked.all_edges)
    incoming = synthetic_tracked.all_edges[
        (synthetic_tracked.all_edges.v >= 0)
        & ((synthetic_tracked.all_edges.u == -2) | (synthetic_tracked.all_edges.u >= 0))
    ]
    for v, group in incoming.groupby("v"):
        assert group["softmax"].sum() == pytest.approx(1.0)


def test_probability_features_entropy_is_non_negative(synthetic_tracked):
    assign_probability_features(synthetic_tracked.all_edges)
    incoming = synthetic_tracked.all_edges[
        (synthetic_tracked.all_edges.v >= 0)
        & ((synthetic_tracked.all_edges.u == -2) | (synthetic_tracked.all_edges.u >= 0))
    ]
    assert (incoming["softmax_entropy"] >= 0).all()


def test_probability_features_exit_edges_stay_minus_one(synthetic_tracked):
    assign_probability_features(synthetic_tracked.all_edges)
    exit_edges = synthetic_tracked.all_edges[synthetic_tracked.all_edges.v < 0]
    assert (exit_edges["softmax"] == -1.0).all()
    assert (exit_edges["softmax_entropy"] == -1.0).all()
    assert (exit_edges["parental_softmax"] == -1.0).all()


def test_probability_features_softmax_values_match_scipy(synthetic_tracked):
    assign_probability_features(synthetic_tracked.all_edges)
    incoming = synthetic_tracked.all_edges[
        (synthetic_tracked.all_edges.v >= 0)
        & ((synthetic_tracked.all_edges.u == -2) | (synthetic_tracked.all_edges.u >= 0))
    ]
    for v, group in incoming.groupby("v"):
        expected = scipy_softmax(-group["cost"].values)
        np.testing.assert_allclose(group["softmax"].values, expected)


def test_probability_features_parental_softmax_app_is_one_minus_migration_sum(
    synthetic_tracked,
):
    assign_probability_features(synthetic_tracked.all_edges)
    incoming = synthetic_tracked.all_edges[
        (synthetic_tracked.all_edges.v >= 0)
        & ((synthetic_tracked.all_edges.u == -2) | (synthetic_tracked.all_edges.u >= 0))
    ]
    for v, group in incoming.groupby("v"):
        app = group[group.u == -2]
        mig = group[group.u >= 0]
        if len(app) > 0:
            assert app["parental_softmax"].sum() == pytest.approx(
                1 - mig["parental_softmax"].sum()
            )


# ---------------------------------------------------------------------------
# assign_sensitivity_features
# ---------------------------------------------------------------------------


def test_sensitivity_features_returns_expected_column_names(
    synthetic_tracked, mock_gurobi_model
):
    model = mock_gurobi_model(synthetic_tracked.all_edges)
    cols = assign_sensitivity_features(synthetic_tracked.all_edges, model)
    assert cols == ["sa_obj_low", "sa_obj_up", "sensitivity_diff"]


def test_sensitivity_features_non_selected_edge_uses_sa_obj_low(
    synthetic_tracked, mock_gurobi_model
):
    all_edges = synthetic_tracked.all_edges
    model = mock_gurobi_model(all_edges)
    assign_sensitivity_features(all_edges, model)
    non_selected = all_edges[all_edges.flow == 0]
    for row in non_selected.itertuples():
        expected = abs(row.cost - row.sa_obj_low)
        assert row.sensitivity_diff == pytest.approx(expected)


def test_sensitivity_features_selected_edge_uses_sa_obj_up(
    synthetic_tracked, mock_gurobi_model
):
    all_edges = synthetic_tracked.all_edges
    model = mock_gurobi_model(all_edges)
    assign_sensitivity_features(all_edges, model)
    selected = all_edges[all_edges.flow > 0]
    for row in selected.itertuples():
        expected = abs(row.cost - row.sa_obj_up)
        assert row.sensitivity_diff == pytest.approx(expected)


# ---------------------------------------------------------------------------
# assign_all_features
# ---------------------------------------------------------------------------


def test_assign_all_features_adds_all_expected_columns(
    synthetic_tracked, mock_gurobi_model
):
    all_edges = synthetic_tracked.all_edges
    model = mock_gurobi_model(all_edges)
    cols = assign_all_features(all_edges, model)
    expected = {
        "distance",
        "chosen_neighbour_rank",
        "sa_obj_low",
        "sa_obj_up",
        "sensitivity_diff",
        "softmax",
        "softmax_entropy",
        "parental_softmax",
    }
    assert set(cols) == expected
    assert expected.issubset(set(all_edges.columns))


# ---------------------------------------------------------------------------
# Tracked.assign_features
# ---------------------------------------------------------------------------


def test_tracked_assign_features_raises_without_debug_mode(synthetic_tracked):
    import pytest

    # synthetic_tracked has all_edges; simulate missing by passing a plain Tracked
    from tracktour._tracker import Tracked

    no_debug = Tracked(
        tracked_edges=synthetic_tracked.tracked_edges,
        tracked_detections=synthetic_tracked.tracked_detections,
        frame_key=synthetic_tracked.frame_key,
        location_keys=synthetic_tracked.location_keys,
        value_key=synthetic_tracked.value_key,
    )
    with pytest.raises(ValueError, match="DEBUG_MODE"):
        no_debug.assign_features()


def test_tracked_assign_features_adds_all_columns_with_model(
    synthetic_tracked, mock_gurobi_model
):
    synthetic_tracked.model = mock_gurobi_model(synthetic_tracked.all_edges)
    cols = synthetic_tracked.assign_features()
    expected = {
        "distance",
        "chosen_neighbour_rank",
        "sa_obj_low",
        "sa_obj_up",
        "sensitivity_diff",
        "softmax",
        "softmax_entropy",
        "parental_softmax",
    }
    assert set(cols) == expected
    assert expected.issubset(set(synthetic_tracked.all_edges.columns))


def test_tracked_assign_features_warns_and_skips_sensitivity_without_model(
    synthetic_tracked,
):
    cols = synthetic_tracked.assign_features()
    sensitivity_cols = {"sa_obj_low", "sa_obj_up", "sensitivity_diff"}
    assert not sensitivity_cols.intersection(set(cols))


def test_tracked_assign_features_without_model_still_adds_probability_and_migration(
    synthetic_tracked,
):
    cols = synthetic_tracked.assign_features()
    expected = {
        "distance",
        "chosen_neighbour_rank",
        "softmax",
        "softmax_entropy",
        "parental_softmax",
    }
    assert expected.issubset(set(cols))
