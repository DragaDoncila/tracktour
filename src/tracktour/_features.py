"""Edge feature assignment for the candidate graph.

These features power the D-UCB bandit sampler in the TrackAnnotator.
All functions operate on the `all_edges` DataFrame produced by `Tracker.solve()`
in DEBUG_MODE and mutate it in-place, adding new columns.

Note: `assign_sensitivity_features` requires a live Gurobi model object and
must therefore be called immediately after solving, before the model is
discarded.
"""

import numpy as np
from scipy.special import softmax as scipy_softmax


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

    migration_edges = all_edges[(all_edges.u >= 0) & (all_edges.v >= 0)]
    for _, group in migration_edges.groupby("u"):
        sorted_group = group.sort_values(by="distance").reset_index()
        for i, row in enumerate(sorted_group.itertuples()):
            all_edges.loc[row.index, "chosen_neighbour_rank"] = i

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
    sa_obj_low = [None] * len(all_edges)
    sa_obj_up = [None] * len(all_edges)
    sens_diffs = [None] * len(all_edges)

    for var in model.getVars():
        edg_idx, _src, _dst = eval(var.varName.lstrip("flow"))
        edg_row = all_edges.loc[edg_idx]
        if var.X == 0:
            sens_diff = abs(edg_row["cost"] - var.SAObjLow)
        else:
            sens_diff = abs(edg_row["cost"] - var.SAObjUp)
        sa_obj_low[edg_idx] = var.SAObjLow
        sa_obj_up[edg_idx] = var.SAObjUp
        sens_diffs[edg_idx] = sens_diff

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
