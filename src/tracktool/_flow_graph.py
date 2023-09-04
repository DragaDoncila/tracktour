from functools import partial
import math
import re
from typing import List, Tuple

import numpy as np
import networkx as nx
import igraph

from scipy.spatial import KDTree
import pandas as pd
import gurobipy as gp
import time

from ._costs import euclidean_cost_func, dist_to_edge_cost_func, closest_neighbour_child_cost

try:
    from napari.utils import progress as tqdm
except ImportError:
    from tqdm import tqdm
    
class FlowGraph:
    APPEARANCE_EDGE_REGEX = re.compile(r"e_a_[0-9]+\.[0-9]+")
    EXIT_EDGE_REGEX = re.compile(r"e_[0-9]+\.[0-9]+_t")
    DIVISION_EDGE_REGEX = re.compile(r"e_d_[0-9]+\.[0-9]+")
    MIGRATION_EDGE_REGEX = re.compile(r"e_[0-9]+\.[0-9]+_[0-9]+\.[0-9]+")

    def __init__(
        self,
        im_dim: Tuple[int],
        coords: "pandas.DataFrame",
        n_neighbours: int = 10,
        min_t=0,
        max_t=None,
        pixel_vals: List[int] = None,
        migration_only: bool = False,
    ) -> None:
        """Generate a FlowGraph from the coordinates with given pixel values

        Coords should be an list or numpy ndarray of nD point coordinates
        corresponding to identified objects within an nD image. Pixel vals
        is optional and should be of same length as coords. Time
        is assumed to be the first dimension of each coordinate.

        Parameters
        ----------
        im_dim : List[Tuple[int], Tuple[int]]
            top left and bottom right of frame bounding box (across all frames)
        coords : DataFrame
            DataFrame with columns 't', 'y', 'x' and optionally 'z' of
            blob center coordinates for which to solve
        n_neighbours: int
            Number of neighbours to consider for cell migration in next frame, by
            default 10. If fewer neighbours are present, all neighbours will be
            considered.
        min_t: int, optional
            smallest frame number in the image. If missing, will be determined
            from min value of first coordinate of each object in coords
        max_t: int, optional
            largest frame number in the image. If missing, will be determined
            from max value of first coordinate of each object in coords.
        pixel_vals : List[int], optional
            pixel value of each object at coordinate, by default None
        migration_only: bool, optional
            Whether the model ignores divisions or not, by default False
        """
        self.min_t = min_t or coords["t"].min()
        self.max_t = max_t or coords["t"].max()
        self.t = self.max_t - self.min_t + 1
        self.k = n_neighbours
        self.im_dim = im_dim
        self.migration_only = migration_only
        self.spatial_cols = ["y", "x"]
        if "z" in coords.columns:
            self.spatial_cols.insert(0, "z")
        if pixel_vals is None and "label" in coords.columns:
            pixel_vals = coords["label"].tolist()
        self._g = self._init_nodes(coords, pixel_vals)
        self._kdt_dict = self._build_trees()
        self._init_edges()

    def _build_trees(self):
        """Build dictionary of t -> kd tree for all vertices in all frames."""
        kd_dict = {}

        for t in tqdm(
            range(self.min_t, self.max_t + 1), total=self.t, desc="Building kD trees"
        ):
            tree, indices = self._build_tree_at_t(t)
            kd_dict[t] = {"tree": tree, "indices": indices}
        return kd_dict

    def _build_tree_at_t(self, t):
        """Get tree and vertex indices for a given t.

        Args:
            t (int): frame to build tree for 

        Returns:
            Tuple(kdtree, np.ndarray): tree and vertex indices for tree coordinates
        """
        frame_vertices = self._g.vs(t=t)
        frame_indices = np.asarray([v.index for v in frame_vertices])
        frame_coords = frame_vertices["coords"]
        new_tree = KDTree(frame_coords)
        return new_tree, frame_indices

    def _init_nodes(self, coords, pixel_vals=None):
        """Create igraph from coords and pixel_vals with len(coords) vertices.

        Parameters
        ----------
        coords : DataFrame
            DataFrame with columns 't', 'y', 'x' and optionally 'z' of
            blob center coordinates for which to solve
        pixel_vals : List[int], optional
            List of integer values for each node, by default None
        """
        n = len(coords)
        if not pixel_vals:
            pixel_vals = np.arange(n, dtype=np.uint16)
        pixel_vals = pd.Series(pixel_vals)

        coords_numpy = coords[self.spatial_cols].to_numpy()
        false_arr = np.broadcast_to(False, n)
        times = coords["t"]
        all_attrs_dict = {
            "label": times.astype(str).str.cat(pixel_vals.astype(str), sep="_"),
            "coords": coords_numpy,
            "pixel_value": pixel_vals,
            "t": times.to_numpy(dtype=np.int32),
            "is_source": false_arr,
            "is_target": false_arr,
            "is_appearance": false_arr,
            "is_division": false_arr,
        }
        g = igraph.Graph(directed=True)
        g.add_vertices(n, all_attrs_dict)

        self.source = g.add_vertex(
            name="source",
            label="source",
            coords=np.asarray((0, 0)),
            pixel_value=0,
            t=-1,
            is_source=True,
            is_target=False,
            is_appearance=False,
            is_division=False,
        )

        self.appearance = g.add_vertex(
            name="appearance",
            label="appearance",
            coords=np.asarray((0, 0)),
            pixel_value=0,
            t=-1,
            is_source=False,
            is_target=False,
            is_appearance=True,
            is_division=False,
        )

        self.target = g.add_vertex(
            name="target",
            label="target",
            coords=np.asarray((0, 0)),
            pixel_value=0,
            t=self.max_t + 1,  # max frame index is max_t
            is_source=False,
            is_target=True,
            is_appearance=False,
            is_division=False,
        )

        if not self.migration_only:
            self.division = g.add_vertex(
                name="division",
                label="division",
                # TODO: will break for 4d maybe
                coords=np.asarray((0, 0)),
                pixel_value=0,
                t=-1,
                is_source=False,
                is_target=False,
                is_appearance=False,
                is_division=True,
            )

        return g

    def _init_edges(self):
        self._init_appear_exit_edges()
        self._init_migration_division_edges()

    def _init_appear_exit_edges(self):
        """Connect appearance to all vertices, and all vertices to target.

        Cost for appearance is 0 for nodes in the first frame,
        and proportional to the distance of the node from a box
        edge for remaining frames.
        Cost for exit is 0 for nodes in the final frame,
        and proportional to distance of the node from closest
        box edge for remaining frames.
        """
        self._g.add_edge(self.source, self.appearance, cost=0, var_name="e_sa", label=0)

        real_nodes = self._g.vs(lambda v: not self._is_virtual_node(v))
        real_node_coords = np.asarray([v["coords"] for v in real_nodes])
        cost_func = partial(dist_to_edge_cost_func, self.im_dim)
        real_node_costs = np.apply_along_axis(cost_func, 1, real_node_coords)

        var_names_app = []
        costs_app = []
        edges_app = []
        labels_app = []

        var_names_target = []
        costs_target = []
        edges_target = []
        labels_target = []

        for i, v in tqdm(
            enumerate(real_nodes),
            desc="Building enter/exit edges",
            total=len(real_nodes),
        ):
            # first frame should be able to appear at no extra cost
            if v["t"] == self.min_t:
                cost_app = 0
                cost_target = real_node_costs[i]
            # final frame should flow into exit at no extra cost
            elif v["t"] == self.max_t:
                cost_app = real_node_costs[i]
                cost_target = 0
            else:
                cost_app = cost_target = real_node_costs[i]

            var_names_app.append(f"e_a_{v['t']}.{v['pixel_value']}")
            costs_app.append(cost_app)
            edges_app.append((self.appearance.index, v.index))
            labels_app.append(str(cost_app)[:4])

            var_names_target.append(f"e_{v['t']}.{v['pixel_value']}_t")
            costs_target.append(cost_target)
            edges_target.append((v.index, self.target.index))
            labels_target.append(str(cost_target)[:4])

        app_attrs = {"label": labels_app, "var_name": var_names_app, "cost": costs_app}
        target_attrs = {
            "label": labels_target,
            "var_name": var_names_target,
            "cost": costs_target,
        }

        self._g.add_edges(edges_app, attributes=app_attrs)
        self._g.add_edges(edges_target, attributes=target_attrs)

    def _init_migration_division_edges(self):
        """Connect all pairs vertices in frames 0..n-1 to 1..n.

        Cost is computed using the migration cost function.
        """
        edges = []
        var_names = []
        all_costs = []
        labels = []

        if not self.migration_only:
            self._g.add_edge(
                self.source, self.division, cost=0, var_name="e_sd", label=0
            )

        for source_t in tqdm(
            range(self.min_t, self.max_t),
            desc=f"Building frame-to-frame edges",
            total=self.t - 1,
        ):
            dest_t = source_t + 1
            new_mig_edges, new_div_edges = self._get_frame_edges(source_t, dest_t)
            for edge_dict in [new_mig_edges, new_div_edges]:
                edges.extend(edge_dict['edges'])
                var_names.extend(edge_dict['var_names'])
                all_costs.extend(edge_dict['costs'])
                labels.extend(edge_dict['labels'])

        all_attrs = {
            "var_name": var_names,
            "cost": all_costs,
            "label": labels,
        }
        self._g.add_edges(edges, attributes=all_attrs)

    def _is_virtual_node(self, v):
        return (
            v["is_source"] or v["is_appearance"] or v["is_division"] or v["is_target"]
        )

    def _may_divide(self, v):
        return not self._is_virtual_node(v) and self.min_t <= v["t"] < self.max_t

    def _bounded_edge(self, e):
        var_name = e["var_name"]
        return "e_sd" not in var_name and "e_sa" not in var_name

    def _get_var_sum_str(self, var_names, neg=""):
        var_sum = ""
        for edge_name in var_names:
            var_sum += f"{neg}{edge_name} + "
        var_sum = var_sum.rstrip(" +")
        return var_sum

    def _get_objective_string(self):
        var_names = self._g.es["var_name"]
        edge_costs = self._g.es["cost"]
        obj_str = "Minimize\n\t"
        for i in range(len(var_names)):
            var_cost_str = f"{edge_costs[i]} {var_names[i]} + "
            obj_str += var_cost_str
        obj_str = obj_str.rstrip(" +")
        obj_str += "\n"
        return obj_str

    def _get_incident_edges(self, node):
        incoming_indices = self._g.incident(node, "in")
        incoming = self._g.es.select(incoming_indices)
        outgoing_indices = self._g.incident(node, "out")
        outgoing = self._g.es.select(outgoing_indices)
        return incoming, outgoing

    def _get_flow_constraints(self):
        # out of source and into target
        source_outgoing_names = self._get_incident_edges(self.source)[1]["var_name"]
        target_incoming_names = self._get_incident_edges(self.target)[0]["var_name"]
        source_outgoing_sum = self._get_var_sum_str(source_outgoing_names)
        target_incoming_sum = self._get_var_sum_str(target_incoming_names, neg="-")
        network_capacity_str = (
            f"\tflow_all: {source_outgoing_sum} + {target_incoming_sum} = 0\n"
        )

        # division & appearance
        appearance_incoming, appearance_outgoing = self._get_incident_edges(
            self.appearance
        )
        appearance_incoming_sum = self._get_var_sum_str(appearance_incoming["var_name"])
        appearance_outgoing_sum = self._get_var_sum_str(
            appearance_outgoing["var_name"], neg="-"
        )
        virtual_capacity_str = (
            f"\tflow_app: {appearance_incoming_sum} + {appearance_outgoing_sum} = 0\n"
        )

        if not self.migration_only:
            division_incoming, division_outgoing = self._get_incident_edges(
                self.division
            )
            division_incoming_sum = self._get_var_sum_str(division_incoming["var_name"])
            division_outgoing_sum = self._get_var_sum_str(
                division_outgoing["var_name"], neg="-"
            )
            virtual_capacity_str += (
                f"\tflow_div: {division_incoming_sum} + {division_outgoing_sum} = 0\n"
            )

        # inner nodes
        inner_node_str = ""
        for t in range(self.min_t, self.max_t + 1):
            t_nodes = self._g.vs.select(t=t)
            for i, node in enumerate(t_nodes):
                incoming_edges, outgoing_edges = self._get_incident_edges(node)
                incoming_names = incoming_edges["var_name"]
                outgoing_names = outgoing_edges["var_name"]

                incoming_sum = self._get_var_sum_str(incoming_names)
                outgoing_sum = self._get_var_sum_str(outgoing_names, neg="-")
                inner_node_str += (
                    f"\tflow_{t}.{i}: {incoming_sum} + {outgoing_sum} = 0\n"
                )
                inner_node_str += f"\tforced_{t}.{i}: {incoming_sum} >= 1\n"
            inner_node_str += "\n"
        flow_const = f"\\Total network\n{network_capacity_str}\n\\Virtual nodes\n{virtual_capacity_str}\n\\Inner nodes\n{inner_node_str}"
        return flow_const

    def _get_division_constraints(self):
        """Constrain conditions required for division to occur.

        1. We must have flow from appearance or migration before we have flow
            from division
        """
        div_str = "\\Division constraints\n"
        potential_parents = self._g.vs(self._may_divide)
        for i, v in enumerate(potential_parents):
            incoming, outgoing = self._get_incident_edges(v)
            div_edge = incoming(lambda e: "e_d" in e["var_name"])[0]["var_name"]
            other_incoming_edges = incoming(lambda e: "e_d" not in e["var_name"])
            incoming_sum = self._get_var_sum_str(
                [e["var_name"] for e in other_incoming_edges]
            )

            # must have appearance or immigration before we divide
            div_str += f"\tdiv_{i}: {incoming_sum} - {div_edge} >= 0\n"

        return div_str

    def _get_constraints_string(self):
        cons_str = "Subject To\n"
        cons_str += self._get_flow_constraints()
        if not self.migration_only:
            cons_str += self._get_division_constraints()
        # cons_str += self._get_flow()
        return cons_str

    def _get_bounds_string(self):
        bounds_str = "Bounds\n"

        for edge in self._g.es(self._bounded_edge):
            bounds_str += f'\t0 <= {edge["var_name"]} <= 1\n'
        return bounds_str

    def _to_lp(self, path):
        obj_str = self._get_objective_string()
        constraints_str = self._get_constraints_string()
        bounds_str = self._get_bounds_string()

        total_str = f"{obj_str}\n{constraints_str}\n{bounds_str}"
        with open(path, "w") as f:
            f.write(total_str)

    def _to_gurobi_model(self):
        edge_costs = list(self._g.es["cost"])
        src_dest_info = gp.tuplelist([
            (
                e.index,
                e["var_name"],
                self._g.vs[e.source].index,
                self._g.vs[e.source]["label"],
                self._g.vs[e.target].index,
                self._g.vs[e.target]["label"],
            )
            for e in self._g.es
        ])
        bounds = [math.inf if 'e_s' in e['var_name'] else 1 for e in self._g.es]
        cost_dict = dict(zip(src_dest_info, edge_costs))
        m = gp.Model("tracks")
        flow = m.addVars(src_dest_info, obj=cost_dict, lb=0, ub=bounds, name="flow")

        # flow out of source into target
        src_edges = flow.select('*', '*', '*', 'source', '*', '*')
        target_edges = flow.select('*', '*', '*', '*', '*', 'target')
        m.addConstr(
            sum(src_edges) == sum(target_edges)
        )

        # flow appearance & division
        app_in = flow.select('*', '*', '*', '*', '*', 'appearance')
        app_out = flow.select('*', '*', '*', 'appearance', '*', '*')
        m.addConstr(
            sum(app_in) == sum(app_out)
        )
        if not self.migration_only:
            div_in = flow.select('*', '*', '*', '*', '*', 'division')
            div_out = flow.select('*', '*', '*', 'division', '*', '*')
            m.addConstr(
                sum(div_in) == sum(div_out)
            )

        # flow inner nodes
        for t in range(self.min_t, self.max_t + 1):
            t_nodes = self._g.vs.select(t=t)
            for i, node in enumerate(t_nodes):
                node_id = node.index
                node_out = flow.select('*', '*', node_id, '*', '*', '*')
                node_in = flow.select('*', '*', '*', '*', node_id, '*')
                m.addConstr(
                    sum(node_in) == sum(node_out), f'migration_{i}'
                )
                m.addConstr(sum(node_in) >= 1, f'forced_{i}')

                if not self.migration_only:
                    div_edge = flow.select('*', '*', '*', 'division', node_id, '*')
                    m.addConstr(sum(node_in) - sum(div_edge) >= sum(div_edge), f'div_{i}')

        return m, flow
    
    def solve(self, solver='gurobi'):
        if solver != 'gurobi':
            raise NotImplementedError("We don't yet support a non-gurobi solver")
        m, flow_info = self._to_gurobi_model()
        m.optimize()
        if m.Status == 2:
            self.store_solution(m)
        else:
            raise RuntimeError(f"Couldn't solve model. Model status: {m.Status}.")
        

    def save_flow_info(self, coords):
        coords['in-app'] = 0
        coords['in-div'] = 0
        coords['in-mig'] = 0
        coords['out-mig'] = 0
        coords['out-target'] = 0        

        sol_edges = self._g.es.select(flow_gt=0)
        for e in tqdm(sol_edges, desc='Summing flow'):
            flow = e['flow']
            src_id = e.source
            src = self._g.vs[src_id]
            dest_id = e.target
            dest = self._g.vs[dest_id]

            # migration edge
            if not self._is_virtual_node(src) and not self._is_virtual_node(dest):
                coords.loc[[dest_id], ['in-mig']] += flow
                coords.loc[[src_id], ['out-mig']] += flow
            elif src['is_division']:
                coords.loc[[dest_id], ['in-div']] += flow
            elif src['is_appearance']:
                coords.loc[[dest_id], ['in-app']] += flow
            elif dest['is_target']:
                coords.loc[[src_id], ['out-target']] += flow


    def introduce_vertex(self, new_vid, t, coords, new_label, add_hyperedges=False):
        node_coords = np.asarray(coords)
        v = self._g.add_vertex(
                    label=f'{t}_{new_label}',
                    coords=node_coords,
                    pixel_value=new_label,
                    t=t,
                    is_source=False,
                    is_target=False,
                    is_appearance=False,
                    is_division=False,
                )
        assert v.index == new_vid, f"New index was supposed to be {new_vid} but is {v.index}."
        if add_hyperedges:
            app = target = dist_to_edge_cost_func(self.im_dim, node_coords)
            if t == self.min_t:
                app = 0
            elif t == self.max_t:
                target = 0
            # add appearance to v
            self._g.add_edge(self.appearance, v, var_name=f"e_a_{v['t']}.{v['pixel_value']}", cost=app, label=str(app)[:5])
            # add v to target
            self._g.add_edge(v, self.target, var_name=f"e_{v['t']}.{v['pixel_value']}_t", cost=target, label=str(target)[:5])

    def introduce_vertices(self, oracle):
        """Introduce vertices from oracle into graph.

        This includes rebuilding the kdtrees for affected frames and
        recomputing edges between frames (t-1)->(t) and
        (t) -> (t+1) for each affected t.

        Args:
            oracle Dict[int, tuple]: dictionary of info about new vertices
        """
        affected_ts = set()
        for new_vid, v_info in oracle.items():
            t, coords, new_label = v_info
            self.introduce_vertex(new_vid, t, coords, new_label, add_hyperedges=True)
            affected_ts.add(t)
        affected_ts = list(sorted(affected_ts))
        for t in affected_ts:
            tree, indices = self._build_tree_at_t(t)
            self._kdt_dict[t] = {
                "tree": tree, 
                "indices": indices
            }
        all_pairs = []
        for i, t in enumerate(affected_ts):
            if not i:
                pairs = [(t-1, t), (t, t+1)]
            elif t - affected_ts[i-1] > 1:
                pairs = [(t-1, t), (t, t+1)]
            else:
                pairs = [(t, t+1)]
            for pair in pairs:
                t1 = pair[0]
                t2 = pair[1]
                if t1 < 0:
                    continue
                if t2 > self.max_t:
                    continue
                all_pairs.append(pair)

        start_time = time.time()
        for pair in all_pairs:
            # delete current migration & division edges
            self._clear_frame(pair[0], pair[1])
        for pair in tqdm(all_pairs, desc='Rebuilding frames'):
            # recompute
            self._rebuild_frame_edges(pair[0], pair[1])
        duration = time.time() - start_time
        return duration, len(all_pairs)
    
    def _clear_frame(self, source_t, dest_t):
        # delete all migration and division edges
        frame_mig_edges = self._g.es.select(lambda e: self._g.vs[e.source]['t'] == source_t and self._g.vs[e.target]['t'] == dest_t)
        self._g.delete_edges(frame_mig_edges)
        frame_div_edges = self._g.es.select(lambda e: self._g.vs[e.source]['is_division'] and self._g.vs[e.target]['t'] == source_t)
        self._g.delete_edges(frame_div_edges)

    def _rebuild_frame_edges(self, source_t, dest_t):
        # get edges for this frame
        mig_edges, div_edges = self._get_frame_edges(source_t, dest_t)
        all_attrs = {
            "var_name": mig_edges['var_names'] + div_edges['var_names'],
            "cost": mig_edges['costs'] + div_edges['costs'],
            "label": mig_edges['labels'] + div_edges['labels'],
        }
        self._g.add_edges(mig_edges['edges'] + div_edges['edges'], attributes=all_attrs)

    def _get_frame_edges(self, source_t, dest_t):
        source_nodes = self._g.vs(t=source_t)
        source_coords = np.asarray(source_nodes["coords"])
        if len(source_coords) == 0:
            raise ValueError(f"No coords found for frame {source_t}.")
        dest_tree = self._kdt_dict[dest_t]["tree"]
        if dest_tree.n == 0:
            raise ValueError(f"kD-tree for frame {dest_t} contains no coordinates.")
        k = self.k if dest_tree.n > self.k else dest_tree.n
        dest_distances, dest_indices = dest_tree.query(
            source_coords, 
            k=k if k > 1 else [k]
        )            
        frame_edges = []
        frame_var_names = []
        frame_labels = []
        frame_costs = []

        frame_div_edges = []
        frame_div_var_names = []
        frame_div_costs = []
        frame_div_labels = []
        for i, src in enumerate(source_nodes):
            dest_vertex_indices = self._kdt_dict[dest_t]["indices"][dest_indices[i]]
            dest_vertices = self._g.vs[list(dest_vertex_indices)]
            # We're relying on these indices not changing partway through construction.
            np.testing.assert_allclose(
                dest_tree.data[dest_indices[i]],
                [v["coords"] for v in dest_vertices],
            )

            current_edges = [
                (src.index, dest_index) for dest_index in dest_vertex_indices
            ]
            current_costs = dest_distances[i]
            current_var_names = [
                f"e_{src['t']}.{src['pixel_value']}_{dest['t']}.{dest['pixel_value']}"
                for dest in dest_vertices
            ]
            current_labels = [str(cost)[:5] for cost in current_costs]

            frame_edges.extend(current_edges)
            frame_var_names.extend(current_var_names)
            frame_labels.extend(current_labels)
            frame_costs.extend(current_costs)

            if self.migration_only:
                continue

            division_edge = (self.division.index, src.index)
            cost_div = closest_neighbour_child_cost(
                src["coords"], dest_tree.data[dest_indices[i]]
            )
            var_name_div = f"e_d_{src['t']}.{src['pixel_value']}"
            label_div = str(cost_div)[:5]

            frame_div_edges.append(division_edge)
            frame_div_costs.append(cost_div)
            frame_div_var_names.append(var_name_div)
            frame_div_labels.append(label_div)
        mig_edges = {
            'edges': frame_edges,
            'var_names': frame_var_names,
            'costs': frame_costs,
            'labels': frame_labels
        }
        div_edges = {
            'edges': frame_div_edges,
            'var_names': frame_div_var_names,
            'costs': frame_div_costs,
            'labels': frame_div_labels            
        }
        return mig_edges, div_edges

    def add_edge(self, u, v, is_fixed=False):
        u_node = self._g.vs[u]
        v_node = self._g.vs[v]
        if is_fixed:
            cost = 0
        else:
            cost = euclidean_cost_func(u_node, v_node)
        var_name = f"e_{u_node['t']}.{u_node['pixel_value']}_{v_node['t']}.{v_node['pixel_value']}"
        label = str(cost)[:5]
        self._g.add_edge(u_node, v_node, cost=cost, var_name=var_name, label=label)

    def store_solution(self, opt_model):
        start = time.time()
        sol_vars = opt_model.getVars()
        v_info = [v.VarName.lstrip('flow[').rstrip(']').split(',') + [v.X] for v in sol_vars]
        v_dict = {int(eid): {
            'var_name': var_name,
            'src_id': int(src_id),
            'target_id': int(target_id),
            'flow': float(flow)
        } for eid, var_name, src_id, src_label, target_id, target_label, flow in v_info if float(flow) > 0}

        # store the correct flow on each graph edge
        self._g.es['flow'] = 0
        self._g.es.select(list(v_dict.keys()))['flow'] = [v_dict[eid]['flow'] for eid in v_dict.keys()]
        duration = time.time() - start
        return duration

    def convert_sol_igraph_to_nx(self):
        for v in self._g.vs:
            v['y'] = v['coords'][0]
            v['x'] = v['coords'][1]
            for attr_name in self._g.vertex_attributes():
                if isinstance(v[attr_name], np.bool_):
                    v[attr_name] = int(v[attr_name])
                elif v[attr_name] is None:
                    v[attr_name] = 0
        for e in self._g.es:
            for attr_name in self._g.edge_attributes():
                if e[attr_name] is None:
                    e[attr_name] = 0
        del(self._g.vs['coords'])
        del(self._g.vs['name'])
        del(self._g.vs['label'])

        del(self._g.es['label'])
        nx_g = self._g.to_networkx(create_using=nx.DiGraph)
        return nx_g
    
    def get_coords_df(self):
        coords = self._g.get_vertex_dataframe()
        split_cols = tuple(zip(*list(coords['coords'].values)))
        for i, col in enumerate(self.spatial_cols):
            coords[col] = split_cols[i]
        coords.drop(columns=['label', 'name', 'coords'], inplace=True)
        coords.rename({'pixel_value': 'label'}, axis=1, inplace=True)
        return coords

