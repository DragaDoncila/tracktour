"""Edge feature assignment for the candidate graph.

These features power the D-UCB bandit sampler in the TrackAnnotator.
All functions operate on the `all_edges` DataFrame produced by `Tracker.solve()`
in DEBUG_MODE and mutate it in-place, adding new columns.

Note: `assign_sensitivity_features` requires a live Gurobi model object and
must therefore be called immediately after solving, before the model is
discarded.
"""

import re

import numpy as np
from scipy.special import softmax as scipy_softmax

_FLOW_VAR_RE = re.compile(r"flow[\[(](\d+)")


def assign_migration_features(all_edges):
    """Add migration-based features to all_edges.

    For each migration edge (u >= 0, v >= 0), records the rank of the target
    among all candidate neighbours of the source sorted by distance.

    Parameters
    ----------
    all_edges : pd.DataFrame
        Candidate edge dataframe from Tracker (DEBUG_MODE). Modified in-place.

    Returns
    -------
    list[str]
        Names of columns added: ['distance', 'chosen_neighbour_rank']
    """
    all_edges["chosen_neighbour_rank"] = -1
    migration_mask = (all_edges.u >= 0) & (all_edges.v >= 0)
    ranks = (
        all_edges.loc[migration_mask]
        .groupby("u")["distance"]
        .rank(method="first")
        .astype(int)
        - 1
    )
    all_edges.loc[migration_mask, "chosen_neighbour_rank"] = ranks
    return ["distance", "chosen_neighbour_rank"]


def assign_sensitivity_features(all_edges, model):
    """Add sensitivity analysis features to all_edges from the solved Gurobi model.

    For each edge, records how much its cost would need to change before the
    solver's decision (flow=0 or flow>0) flips. Requires the live Gurobi model
    and must be called immediately after solving.

    The model interface expected is:
      - model.getVars() returns an iterable of variable objects
      - each variable has varName (str), X (float), SAObjLow (float), SAObjUp (float)
      - varName format: "flow(edge_idx, src, dst)"

    Parameters
    ----------
    all_edges : pd.DataFrame
        Candidate edge dataframe from Tracker (DEBUG_MODE). Modified in-place.
    model : gurobipy.Model or compatible mock
        The solved model.

    Returns
    -------
    list[str]
        Names of columns added: ['sa_obj_low', 'sa_obj_up', 'sensitivity_diff']
    """
    # Collect Gurobi variable attributes in one tight loop with no DataFrame access.
    # Row-level all_edges.loc inside a loop is expensive (creates a Series per call).
    idx_list = []
    sa_low_list = []
    sa_up_list = []
    x_list = []
    for var in model.getVars():
        idx_list.append(int(_FLOW_VAR_RE.match(var.varName).group(1)))
        sa_low_list.append(var.SAObjLow)
        sa_up_list.append(var.SAObjUp)
        x_list.append(var.X)

    idx_arr = np.array(idx_list)
    sa_low_arr = np.array(sa_low_list)
    sa_up_arr = np.array(sa_up_list)
    x_arr = np.array(x_list)

    costs = all_edges.loc[idx_arr, "cost"].values
    sens_diff = np.where(
        x_arr == 0, np.abs(costs - sa_low_arr), np.abs(costs - sa_up_arr)
    )

    sa_obj_low = np.full(len(all_edges), np.nan)
    sa_obj_up = np.full(len(all_edges), np.nan)
    sens_diffs = np.full(len(all_edges), np.nan)
    sa_obj_low[idx_arr] = sa_low_arr
    sa_obj_up[idx_arr] = sa_up_arr
    sens_diffs[idx_arr] = sens_diff

    all_edges["sa_obj_low"] = sa_obj_low
    all_edges["sa_obj_up"] = sa_obj_up
    all_edges["sensitivity_diff"] = sens_diffs

    return ["sa_obj_low", "sa_obj_up", "sensitivity_diff"]


def assign_probability_features(all_edges):
    """Add softmax probability features to all_edges.

    For each real target node, computes a softmax distribution over its
    incoming candidate edges (including appearance, u == -2). Also computes
    the entropy of this distribution and a parental softmax that accounts for
    the appearance edge explicitly.

    Exit edges (v < 0) receive -1 for all probability features.

    Softmax and softmax entropy features were first described in:
    https://doi.org/10.48550/arXiv.2503.09244

    Parental softmax was first described in:
    https://doi.org/10.1007/978-3-031-73116-7_27

    Parameters
    ----------
    all_edges : pd.DataFrame
        Candidate edge dataframe from Tracker (DEBUG_MODE). Modified in-place.

    Returns
    -------
    list[str]
        Names of columns added: ['softmax', 'softmax_entropy', 'parental_softmax']
    """
    all_edges["softmax"] = -1.0
    all_edges["softmax_entropy"] = -1.0
    all_edges["parental_softmax"] = -1.0

    # include appearance (u == -2) and migration edges (u >= 0), exclude exit edges
    incoming = all_edges[
        (all_edges.v >= 0) & ((all_edges.u == -2) | (all_edges.u >= 0))
    ]
    for _, v_edges in incoming.groupby("v"):
        softmax_dist = scipy_softmax(-v_edges.cost)
        all_edges.loc[v_edges.index, "softmax"] = softmax_dist
        entropy = -np.sum(softmax_dist * np.log(softmax_dist))
        all_edges.loc[v_edges.index, "softmax_entropy"] = entropy

        v_edges_no_app = v_edges[v_edges.u >= 0]
        v_edges_app = v_edges[v_edges.u == -2]
        exp_sum = np.sum(np.exp(-v_edges_no_app.cost))
        parental_softmax = np.exp(-v_edges_no_app.cost) / (1 + exp_sum)
        all_edges.loc[v_edges_no_app.index, "parental_softmax"] = parental_softmax
        all_edges.loc[v_edges_app.index, "parental_softmax"] = 1 - np.sum(
            parental_softmax
        )

    return ["softmax", "softmax_entropy", "parental_softmax"]


def assign_all_features(all_edges, model):
    """Compute and assign all available edge features.

    Convenience wrapper calling all three feature assignment functions.
    Requires a live Gurobi model (DEBUG_MODE only).

    Parameters
    ----------
    all_edges : pd.DataFrame
        Candidate edge dataframe from Tracker (DEBUG_MODE). Modified in-place.
    model : gurobipy.Model or compatible mock
        The solved model.

    Returns
    -------
    list[str]
        Names of all columns added.
    """
    cols = []
    cols += assign_migration_features(all_edges)
    cols += assign_sensitivity_features(all_edges, model)
    cols += assign_probability_features(all_edges)
    return cols
