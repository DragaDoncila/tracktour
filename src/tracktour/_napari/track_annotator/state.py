"""Annotation state management for TrackAnnotator.

This module encapsulates all state tracking for track annotation,
ensuring that graph attributes and cached sets stay synchronized.

The annotation system works with two separate ID spaces:
- Original graph IDs: Node/edge IDs from the solution being annotated
- GT graph IDs: Node/edge IDs in the ground truth graph being constructed

TP nodes exist in both graphs (but may have different IDs).
FN nodes only exist in GT graph.
FP nodes only exist in original graph.
"""

import networkx as nx

# Graph attribute constants
FP_EDGE_ATTR = "tracktour_annotated_fp"
FN_EDGE_ATTR = "tracktour_annotated_fn"
TP_EDGE_ATTR = "tracktour_annotated_tp"

FP_NODE_ATTR = "tracktour_annotated_fp"
FN_NODE_ATTR = "tracktour_annotated_fn"
TP_NODE_VOTES = "tracktour_annotated_tp_votes"


class AnnotationState:
    """Encapsulates annotation state with efficient, synchronized access.

    This class maintains both the graph representations and cached sets
    for fast lookup. All mutations go through methods that update both
    atomically, preventing the sets from getting out of sync with the
    graph attributes.

    Parameters
    ----------
    original_graph : nx.DiGraph
        The original solution graph to be annotated

    Attributes
    ----------
    original_graph : nx.DiGraph
        The original solution graph (source of truth for TP/FP)
    gt_graph : nx.DiGraph
        The ground truth graph being constructed
    """

    def __init__(self, original_graph: nx.DiGraph):
        self.original_graph = original_graph
        self.gt_graph = nx.DiGraph()

        # Private cached sets for O(1) lookups
        # These are implementation details - external code should use properties

        # IDs reference nodes/edges in original_graph
        self._tp_objects: set[int] = set()
        self._fp_objects: set[int] = set()
        self._tp_edges: set[tuple[int, int]] = set()
        self._fp_edges: set[tuple[int, int]] = set()

        # IDs reference nodes/edges in gt_graph
        self._fn_objects: set[int] = set()
        self._fn_edges: set[tuple[int, int]] = set()

    # Read-only properties (return copies to prevent external modification)

    @property
    def tp_objects(self) -> set[int]:
        """Set of true positive object IDs in original graph."""
        return self._tp_objects.copy()

    @property
    def fp_objects(self) -> set[int]:
        """Set of false positive object IDs in original graph."""
        return self._fp_objects.copy()

    @property
    def fn_objects(self) -> set[int]:
        """Set of false negative object IDs in ground truth graph."""
        return self._fn_objects.copy()

    @property
    def tp_edges(self) -> set[tuple[int, int]]:
        """Set of true positive edges in original graph."""
        return self._tp_edges.copy()

    @property
    def fp_edges(self) -> set[tuple[int, int]]:
        """Set of false positive edges in original graph."""
        return self._fp_edges.copy()

    @property
    def fn_edges(self) -> set[tuple[int, int]]:
        """Set of false negative edges in ground truth graph."""
        return self._fn_edges.copy()

    # Node mutation methods

    def add_or_update_tp_node(
        self,
        orig_node_id: int,
        gt_node_id: int,
        location_attrs: dict,
    ) -> None:
        """Mark a node as TP and add/update it in the GT graph.

        If the GT node already exists (same orig_idx), increments its vote count.
        Otherwise creates a new GT node.

        Parameters
        ----------
        orig_node_id : int
            Node ID in the original graph
        gt_node_id : int
            Node ID to use in the GT graph
        location_attrs : dict
            Location attributes (must include t, y, x, optionally z)
        """
        # Mark as TP in original graph tracking
        self._tp_objects.add(orig_node_id)

        # Add or update in GT graph
        if gt_node_id in self.gt_graph.nodes:
            # Already exists, increment vote count
            current_votes = self.gt_graph.nodes[gt_node_id].get(TP_NODE_VOTES, 0)
            self.gt_graph.nodes[gt_node_id][TP_NODE_VOTES] = current_votes + 1
        else:
            # New node, create it with attributes
            attrs = {
                "orig_idx": orig_node_id,
                TP_NODE_VOTES: 1,
                **location_attrs,
            }
            self.gt_graph.add_node(gt_node_id, **attrs)

    def remove_tp_node(self, orig_node_id: int, gt_node_id: int) -> None:
        """Remove TP marking from a node, decrementing votes in GT graph.

        If votes reach zero, removes the node from GT graph entirely.

        Parameters
        ----------
        orig_node_id : int
            Node ID in original graph
        gt_node_id : int
            Node ID in GT graph
        """
        # Remove from original graph tracking
        self._tp_objects.discard(orig_node_id)

        # Decrement votes in GT graph
        if gt_node_id not in self.gt_graph.nodes:
            return

        current_votes = self.gt_graph.nodes[gt_node_id].get(TP_NODE_VOTES, 0)
        new_votes = current_votes - 1

        if new_votes > 0:
            self.gt_graph.nodes[gt_node_id][TP_NODE_VOTES] = new_votes
        else:
            # No more votes, remove the node
            self.gt_graph.remove_node(gt_node_id)

    def add_fp_node(self, orig_node_id: int) -> None:
        """Mark a node as false positive in the original graph.

        Parameters
        ----------
        orig_node_id : int
            Node ID in original graph
        """
        self.original_graph.nodes[orig_node_id][FP_NODE_ATTR] = True
        self._fp_objects.add(orig_node_id)

    def remove_fp_node(self, orig_node_id: int) -> None:
        """Remove false positive marking from a node.

        Parameters
        ----------
        orig_node_id : int
            Node ID in original graph
        """
        self.original_graph.nodes[orig_node_id].pop(FP_NODE_ATTR, None)
        self._fp_objects.discard(orig_node_id)

    def add_fn_node(self, gt_node_id: int, location_attrs: dict) -> None:
        """Add a false negative node to the GT graph.

        FN nodes are detections that should exist but don't in the original graph.

        Parameters
        ----------
        gt_node_id : int
            Node ID for the GT graph
        location_attrs : dict
            Location attributes (must include t, y, x, optionally z)
        """
        attrs = {
            "orig_idx": -1,  # No corresponding original node
            FN_NODE_ATTR: True,
            **location_attrs,
        }
        self.gt_graph.add_node(gt_node_id, **attrs)
        self._fn_objects.add(gt_node_id)

    def remove_fn_node(self, gt_node_id: int) -> None:
        """Remove a false negative node from the GT graph.

        Parameters
        ----------
        gt_node_id : int
            Node ID in GT graph
        """
        if self.gt_graph.has_node(gt_node_id):
            self.gt_graph.remove_node(gt_node_id)
        self._fn_objects.discard(gt_node_id)

    # Edge mutation methods

    def add_tp_edge(
        self,
        orig_edge: tuple[int, int],
        gt_edge: tuple[int, int],
    ) -> None:
        """Mark an edge as true positive in both graphs.

        Parameters
        ----------
        orig_edge : tuple[int, int]
            Edge in the original graph
        gt_edge : tuple[int, int]
            Corresponding edge in the GT graph
        """
        # Update original graph
        self.original_graph.edges[orig_edge][TP_EDGE_ATTR] = True
        self.original_graph.edges[orig_edge]["gt_src"] = gt_edge[0]
        self.original_graph.edges[orig_edge]["gt_tgt"] = gt_edge[1]

        # Update GT graph
        self.gt_graph.add_edge(gt_edge[0], gt_edge[1], **{TP_EDGE_ATTR: True})

        # Update cached set
        self._tp_edges.add(orig_edge)

    def remove_tp_edge(
        self,
        orig_edge: tuple[int, int],
        gt_edge: tuple[int, int],
    ) -> None:
        """Remove true positive marking from an edge.

        Parameters
        ----------
        orig_edge : tuple[int, int]
            Edge in the original graph
        gt_edge : tuple[int, int]
            Corresponding edge in the GT graph
        """
        # Update original graph
        self.original_graph.edges[orig_edge].pop(TP_EDGE_ATTR, None)
        self.original_graph.edges[orig_edge].pop("gt_src", None)
        self.original_graph.edges[orig_edge].pop("gt_tgt", None)

        # Update GT graph
        if self.gt_graph.has_edge(*gt_edge):
            self.gt_graph.remove_edge(*gt_edge)

        # Update cached set
        self._tp_edges.discard(orig_edge)

    def add_fp_edge(self, orig_edge: tuple[int, int]) -> None:
        """Mark an edge as false positive in the original graph.

        Parameters
        ----------
        orig_edge : tuple[int, int]
            Edge in original graph
        """
        self.original_graph.edges[orig_edge][FP_EDGE_ATTR] = True
        self._fp_edges.add(orig_edge)

    def remove_fp_edge(self, orig_edge: tuple[int, int]) -> None:
        """Remove false positive marking from an edge.

        Parameters
        ----------
        orig_edge : tuple[int, int]
            Edge in original graph
        """
        self.original_graph.edges[orig_edge].pop(FP_EDGE_ATTR, None)
        self._fp_edges.discard(orig_edge)

    def add_fn_edge(self, gt_edge: tuple[int, int]) -> None:
        """Add a false negative edge to the ground truth graph.

        Parameters
        ----------
        gt_edge : tuple[int, int]
            Edge in GT graph
        """
        # Add edge to GT graph with FN attribute
        if not self.gt_graph.has_edge(*gt_edge):
            self.gt_graph.add_edge(*gt_edge, **{FN_EDGE_ATTR: True})
        else:
            self.gt_graph.edges[gt_edge][FN_EDGE_ATTR] = True

        self._fn_edges.add(gt_edge)

    def remove_fn_edge(self, gt_edge: tuple[int, int]) -> None:
        """Remove a false negative edge from the ground truth graph.

        Parameters
        ----------
        gt_edge : tuple[int, int]
            Edge in GT graph
        """
        if self.gt_graph.has_edge(*gt_edge):
            self.gt_graph.remove_edge(*gt_edge)

        self._fn_edges.discard(gt_edge)

    # Metadata methods for original graph edges

    def set_edge_metadata(
        self,
        edge: tuple[int, int],
        key: str,
        value: any,
    ) -> None:
        """Set metadata on an edge in the original graph.

        Used for storing state like 'seen', 'src_present', 'tgt_present', etc.

        Parameters
        ----------
        edge : tuple[int, int]
            Edge in original graph
        key : str
            Attribute key
        value : any
            Attribute value
        """
        self.original_graph.edges[edge][key] = value

    def get_edge_metadata(self, edge: tuple[int, int], key: str, default=None) -> any:
        """Get metadata from an edge in the original graph.

        Parameters
        ----------
        edge : tuple[int, int]
            Edge in original graph
        key : str
            Attribute key
        default : any, optional
            Default value if key not found

        Returns
        -------
        any
            The attribute value or default
        """
        return self.original_graph.edges[edge].get(key, default)

    def remove_edge_metadata(self, edge: tuple[int, int], key: str) -> None:
        """Remove metadata from an edge.

        Parameters
        ----------
        edge : tuple[int, int]
            Edge in original graph
        key : str
            Attribute key to remove
        """
        self.original_graph.edges[edge].pop(key, None)
