import warnings

import numpy as np

try:
    from traccuracy.matchers._compute_overlap import get_labels_with_overlap
except ModuleNotFoundError:
    warnings.warn(
        UserWarning(
            "traccuracy not installed, cannot use oracle without it. If you are not using the oracle, ignore this! Oracle will be removed from tracktour in future versions."
        )
    )


def store_solution_on_graph(opt_model, graph):
    sol_vars = opt_model.getVars()
    v_info = [
        v.VarName.lstrip("flow[").rstrip("]").split(",") + [v.X] for v in sol_vars
    ]
    v_dict = {
        int(eid): {
            "var_name": var_name,
            "src_id": int(src_id),
            "target_id": int(target_id),
            "flow": float(flow),
        }
        for eid, var_name, src_id, src_label, target_id, target_label, flow in v_info
        if float(flow) > 0
    }

    # store the correct flow on each graph edge
    graph._g.es["flow"] = 0
    graph._g.es.select(list(v_dict.keys()))["flow"] = [
        v_dict[eid]["flow"] for eid in v_dict.keys()
    ]


def get_gt_match_vertices(coords, gt_coords, sol_ims, gt_ims, v_id, label_key="label"):
    # get mask of problem blob
    problem_info = coords.loc[[v_id], [label_key, "t"]]
    problem_label = problem_info[label_key].values[0]
    problem_t = problem_info["t"].values[0]
    if (ct := len(problem_info)) > 1:
        raise ValueError(
            f"Solution label {problem_label} appears {ct} times in frame {problem_t}."
        )
    # we're only interested in overlaps with this vertex
    only_problem_v_mask = sol_ims[problem_t] == problem_label
    gt_frame = gt_ims[problem_t]
    gt_ov_labels, _, _ = get_labels_with_overlap(gt_frame, only_problem_v_mask)
    gt_v_ids = []
    for label in gt_ov_labels:
        row = gt_coords[(gt_coords.label == label) & (gt_coords.t == problem_t)]
        if (ct := len(row)) > 1:
            raise ValueError(
                f"GT label {label} appears {ct} times in frame {problem_t}."
            )
        vid = row.index.values[0]
        gt_v_ids.append(vid)
    # some of these gt vertices might overlap with **other** vertices beyond this one
    # we need to filter those out.
    all_but_problem_v_mask = sol_ims[problem_t] * np.logical_not(
        only_problem_v_mask
    ).astype(int)
    # we only found one gt vertex, or there's only one solution vertex in this frame
    if len(gt_v_ids) == 1 or all_but_problem_v_mask.max() == 0:
        return gt_v_ids
    real_overlaps = filter_other_overlaps(
        gt_v_ids, all_but_problem_v_mask, gt_frame, gt_coords
    )
    return real_overlaps


def filter_other_overlaps(gt_v_ids, sol_frame, gt_frame, gt_coords):
    real_overlaps = []
    for gt_v in gt_v_ids:
        v_label = gt_coords.loc[[gt_v], ["label"]].values[0]
        only_gt_v_mask = gt_frame == v_label
        gt_overlaps, sol_overlaps, _ = get_labels_with_overlap(
            only_gt_v_mask, sol_frame
        )
        # no bounding box overlaps, we can return
        if not len(sol_overlaps):
            real_overlaps.append(gt_v)
        else:
            # check pixel by pixel overlaps
            if not has_overlapping_sol_vertex(
                sol_overlaps, gt_overlaps, only_gt_v_mask, sol_frame
            ):
                real_overlaps.append(gt_v)
    return real_overlaps


def has_overlapping_sol_vertex(sol_overlaps, gt_overlaps, gt_frame, sol_frame):
    for i in range(len(gt_overlaps)):
        gt_label = gt_overlaps[i]
        sol_label = sol_overlaps[i]
        gt_blob = gt_frame == gt_label
        comp_blob = sol_frame == sol_label
        if blobs_intersect(gt_blob, comp_blob):
            return True
    return False


def blobs_intersect(gt_blob, comp_blob):
    intersection = np.logical_and(gt_blob, comp_blob)
    return np.sum(intersection) > 0


def get_gt_unmatched_vertices_near_parent(
    coords, gt_coords, sol_ims, gt_ims, v_id, v_parents, dist, label_key="label"
):
    import numpy as np
    from scipy.spatial import KDTree
    from traccuracy.matchers._compute_overlap import get_labels_with_overlap

    problem_row = coords.loc[[v_id]]
    problem_t = problem_row["t"].values[0]
    cols = ["y", "x"]
    if "z" in coords.columns:
        cols = ["z", "y", "x"]
    parent_rows = coords.loc[v_parents]
    parent_coords = parent_rows[cols].values

    # build kdt from gt frame
    gt_frame_coords = gt_coords[gt_coords["t"] == problem_t][cols]
    coord_indices, *coord_tuples = zip(*list(gt_frame_coords.itertuples(name=None)))
    coord_tuples = np.asarray(list(zip(*coord_tuples)))
    coord_indices = np.asarray(coord_indices)

    # get nearby vertices close to both parents of v
    gt_tree = KDTree(coord_tuples)
    nearby = [
        n_index
        for n_list in gt_tree.query_ball_point(parent_coords, dist, return_sorted=True)
        for n_index in n_list
    ]
    potential_unmatched = coord_indices[nearby]
    unmatched = []
    problem_frame = sol_ims[problem_t]
    # check if they don't overlap with any solution vertices i.e. they are a fn
    for v in potential_unmatched:
        v_label = gt_coords.loc[[v], ["label"]].values[0]
        mask = gt_ims[problem_t] == v_label
        _, sol_overlaps, _ = get_labels_with_overlap(mask, problem_frame)
        if not len(sol_overlaps) and v not in unmatched:
            unmatched.append(v)
    return unmatched


def get_oracle(merge_node_ids, sol_graph, coords, gt_coords, sol_ims, gt_ims):
    last_label = 0
    last_index = 0
    v_info = None
    oracle = {}
    identified_gt_vs = set()
    for i in merge_node_ids:
        gt_matched = get_gt_match_vertices(coords, gt_coords, sol_ims, gt_ims, i)
        parent_ids = [
            v
            for v in sol_graph._g.neighbors(i, mode="in")
            if sol_graph._g.es[sol_graph._g.get_eid(v, i)]["flow"] > 0
            and not sol_graph._is_virtual_node(sol_graph._g.vs[v])
        ]
        gt_unmatched = get_gt_unmatched_vertices_near_parent(
            coords, gt_coords, sol_ims, gt_ims, i, parent_ids, 50
        )
        problem_v = coords.loc[[i]]
        problem_coords = tuple(problem_v[sol_graph.spatial_cols].values[0])
        # we don't want to "reuse" a vertex we have already found
        gt_matched = list(filter(lambda v: v not in identified_gt_vs, gt_matched))
        gt_unmatched = list(filter(lambda v: v not in identified_gt_vs, gt_unmatched))

        # we couldn't find a match for this vertex at all, we should just delete it
        if not len(gt_matched) and not len(gt_unmatched):
            decision = "delete"
        # we've only found one vertex nearby, it's v itself
        elif len(gt_matched) + len(gt_unmatched) == 1:
            decision = "terminate"
        # more than one "true" vertex overlaps v, a vertex should be introduced
        elif len(gt_matched) > 1:
            # closest match is `v`, second closest gets introduced
            distances_to_v = [
                np.linalg.norm(
                    np.asarray(problem_coords)
                    - np.asarray(gt_coords.loc[[v], sol_graph.spatial_cols].values[0])
                )
                for v in gt_matched
            ]
            second_closest = gt_matched[np.argsort(distances_to_v)[1]]
            v_info = gt_coords.loc[second_closest]
            decision = "introduce"
            identified_gt_vs.add(second_closest)
        # we didn't find >1 overlap, but we've found an unmatched GT vertex nearby
        elif len(gt_unmatched):
            # we just take the closest
            v_id = gt_unmatched[0]
            v_info = gt_coords.loc[v_id]
            decision = "introduce"
            identified_gt_vs.add(v_id)

        if v_info is not None:
            if last_label == 0:
                next_label = coords["label"].max() + 1
                # hypervertices...
                if max(coords.index.values) > sol_graph.division.index:
                    new_index = max(coords.index.values) + 1
                else:
                    new_index = max(coords.index.values) + 5
            else:
                next_label = last_label + 1
                new_index = last_index + 1

            last_label = next_label
            last_index = new_index

        oracle[i] = {
            "decision": decision,
            "v_info": None
            if v_info is None
            else (
                int(new_index),
                list(v_info[["t", *sol_graph.spatial_cols]]) + [int(next_label)],
            ),
            "parent": None,
        }
        v_info = None
    return oracle


def mask_new_vertices(introduce_info, sol_ims, gt_ims):
    for intro_t, coords, new_label in introduce_info.values():
        gt_frame = gt_ims[intro_t]
        int_coords = tuple(int(coord) for coord in coords)
        mask = gt_frame == gt_frame[int_coords]
        sol_ims[intro_t][mask] = new_label
