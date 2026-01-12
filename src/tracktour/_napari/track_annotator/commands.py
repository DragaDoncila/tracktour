"""Command pattern implementation for track annotation.

Each command encapsulates a single edge annotation operation and knows how
to both execute and undo itself. Commands are tracked per-edge, allowing
users to reset individual edges to their original state.
"""

from abc import ABC, abstractmethod
from typing import Optional

from .state import AnnotationState


class EdgeAnnotationCommand(ABC):
    """Abstract base class for annotation commands.

    Each command must implement execute() and undo() methods that
    operate on an AnnotationState.
    """

    @abstractmethod
    def execute(self, state: AnnotationState) -> None:
        """Execute the command, modifying the state.

        Parameters
        ----------
        state : AnnotationState
            The annotation state to modify
        """
        pass

    @abstractmethod
    def undo(self, state: AnnotationState) -> None:
        """Undo the command, reverting the state.

        Parameters
        ----------
        state : AnnotationState
            The annotation state to revert
        """
        pass


class MarkEdgeTPCommand(EdgeAnnotationCommand):
    """Command to mark an edge and its nodes as true positive.

    This represents the case where both points in an edge are unchanged
    from the original positions.
    """

    def __init__(
        self,
        orig_edge: tuple[int, int],
        gt_edge: tuple[int, int],
        src_location: dict,
        tgt_location: dict,
    ):
        """Initialize the command.

        Parameters
        ----------
        orig_edge : tuple[int, int]
            Edge in original graph (orig_src_id, orig_tgt_id)
        gt_edge : tuple[int, int]
            Corresponding edge in GT graph (gt_src_id, gt_tgt_id)
        src_location : dict
            Source node location attributes
        tgt_location : dict
            Target node location attributes
        """
        self.orig_edge = orig_edge
        self.gt_edge = gt_edge
        self.src_location = src_location
        self.tgt_location = tgt_location

    def execute(self, state: AnnotationState) -> None:
        """Mark edge and both nodes as TP."""
        # Add TP nodes
        state.add_or_update_tp_node(
            self.orig_edge[0], self.gt_edge[0], self.src_location
        )
        state.add_or_update_tp_node(
            self.orig_edge[1], self.gt_edge[1], self.tgt_location
        )

        # Add TP edge
        state.add_tp_edge(self.orig_edge, self.gt_edge)

        # Mark edge as seen
        state.set_edge_metadata(self.orig_edge, "seen", True)

    def undo(self, state: AnnotationState) -> None:
        """Remove TP markings."""
        # Remove TP edge
        state.remove_tp_edge(self.orig_edge, self.gt_edge)

        # Remove TP nodes (decrements votes)
        state.remove_tp_node(self.orig_edge[0], self.gt_edge[0])
        state.remove_tp_node(self.orig_edge[1], self.gt_edge[1])

        # Remove seen marker
        state.remove_edge_metadata(self.orig_edge, "seen")


class MarkEdgeFPWithCorrectionCommand(EdgeAnnotationCommand):
    """Command to mark an edge as FP and add the corrected FN edge.

    This represents the case where one or both points were moved,
    indicating the original edge was wrong but there is a corrected edge.
    """

    def __init__(
        self,
        orig_edge: tuple[int, int],
        gt_edge: tuple[int, int],
        src_location: dict,
        tgt_location: dict,
        src_is_fn: bool,
        tgt_is_fn: bool,
        src_orig_id: Optional[int],
        tgt_orig_id: Optional[int],
    ):
        """Initialize the command.

        Parameters
        ----------
        orig_edge : tuple[int, int]
            Edge in original graph to mark as FP
        gt_edge : tuple[int, int]
            Corrected edge in GT graph
        src_location : dict
            Source node location attributes
        tgt_location : dict
            Target node location attributes
        src_is_fn : bool
            Whether source is a false negative (new detection)
        tgt_is_fn : bool
            Whether target is a false negative
        src_orig_id : int or None
            Original node ID for source (if TP), None if FN
        tgt_orig_id : int or None
            Original node ID for target (if TP), None if FN
        """
        self.orig_edge = orig_edge
        self.gt_edge = gt_edge
        self.src_location = src_location
        self.tgt_location = tgt_location
        self.src_is_fn = src_is_fn
        self.tgt_is_fn = tgt_is_fn
        self.src_orig_id = src_orig_id
        self.tgt_orig_id = tgt_orig_id

    def execute(self, state: AnnotationState) -> None:
        """Mark original edge as FP, add corrected GT edge and nodes."""
        # Mark original edge as FP
        state.add_fp_edge(self.orig_edge)

        # Add source node
        if self.src_is_fn:
            state.add_fn_node(self.gt_edge[0], self.src_location)
        else:
            state.add_or_update_tp_node(
                self.src_orig_id, self.gt_edge[0], self.src_location
            )

        # Add target node
        if self.tgt_is_fn:
            state.add_fn_node(self.gt_edge[1], self.tgt_location)
        else:
            state.add_or_update_tp_node(
                self.tgt_orig_id, self.gt_edge[1], self.tgt_location
            )

        # Add FN edge in GT graph
        state.add_fn_edge(self.gt_edge)

        # Store GT edge info in original graph
        state.set_edge_metadata(self.orig_edge, "gt_src", self.gt_edge[0])
        state.set_edge_metadata(self.orig_edge, "gt_tgt", self.gt_edge[1])
        state.set_edge_metadata(self.orig_edge, "seen", True)

    def undo(self, state: AnnotationState) -> None:
        """Remove FP and FN markings."""
        # Remove FN edge
        state.remove_fn_edge(self.gt_edge)

        # Remove nodes
        if self.src_is_fn:
            state.remove_fn_node(self.gt_edge[0])
        else:
            state.remove_tp_node(self.src_orig_id, self.gt_edge[0])

        if self.tgt_is_fn:
            state.remove_fn_node(self.gt_edge[1])
        else:
            state.remove_tp_node(self.tgt_orig_id, self.gt_edge[1])

        # Remove FP edge
        state.remove_fp_edge(self.orig_edge)

        # Remove metadata
        state.remove_edge_metadata(self.orig_edge, "gt_src")
        state.remove_edge_metadata(self.orig_edge, "gt_tgt")
        state.remove_edge_metadata(self.orig_edge, "seen")


class MarkEdgeFPWithSingleNodeCommand(EdgeAnnotationCommand):
    """Command to mark an edge as FP with only one node remaining as TP.

    This represents deleting one endpoint of an edge.
    """

    def __init__(
        self,
        orig_edge: tuple[int, int],
        gt_node_id: int,
        remaining_orig_id: int,
        location: dict,
        src_present: bool,
    ):
        """Initialize the command.

        Parameters
        ----------
        orig_edge : tuple[int, int]
            Edge in original graph
        gt_node_id : int
            GT node ID for the remaining node
        remaining_orig_id : int
            Original node ID of the remaining node
        location : dict
            Location of the remaining node
        src_present : bool
            True if source remains, False if target remains
        """
        self.orig_edge = orig_edge
        self.gt_node_id = gt_node_id
        self.remaining_orig_id = remaining_orig_id
        self.location = location
        self.src_present = src_present

    def execute(self, state: AnnotationState) -> None:
        """Mark edge as FP and add single TP node."""
        # Mark edge as FP
        state.add_fp_edge(self.orig_edge)

        # Add the remaining node as TP
        state.add_or_update_tp_node(
            self.remaining_orig_id, self.gt_node_id, self.location
        )

        # Store metadata about which node remains
        state.set_edge_metadata(self.orig_edge, "src_present", self.src_present)
        state.set_edge_metadata(self.orig_edge, "tgt_present", not self.src_present)
        state.set_edge_metadata(self.orig_edge, "seen", True)

    def undo(self, state: AnnotationState) -> None:
        """Remove FP marking and TP node."""
        # Remove TP node first
        state.remove_tp_node(self.remaining_orig_id, self.gt_node_id)

        # Remove FP edge
        state.remove_fp_edge(self.orig_edge)

        # Remove metadata
        state.remove_edge_metadata(self.orig_edge, "src_present")
        state.remove_edge_metadata(self.orig_edge, "tgt_present")
        state.remove_edge_metadata(self.orig_edge, "seen")


class MarkEdgeFPCommand(EdgeAnnotationCommand):
    """Command to mark an edge as false positive with no corrections.

    This represents completely deleting an edge (both endpoints removed).
    """

    def __init__(self, orig_edge: tuple[int, int]):
        """Initialize the command.

        Parameters
        ----------
        orig_edge : tuple[int, int]
            Edge in original graph
        """
        self.orig_edge = orig_edge

    def execute(self, state: AnnotationState) -> None:
        """Mark edge as FP."""
        state.add_fp_edge(self.orig_edge)
        state.set_edge_metadata(self.orig_edge, "seen", True)

    def undo(self, state: AnnotationState) -> None:
        """Remove FP marking."""
        state.remove_fp_edge(self.orig_edge)
        state.remove_edge_metadata(self.orig_edge, "seen")


class MarkNodeFPCommand(EdgeAnnotationCommand):
    """Command to mark a node as false positive.

    This is typically used for orphan nodes (nodes with no valid edges).
    """

    def __init__(self, node_id: int):
        """Initialize the command.

        Parameters
        ----------
        node_id : int
            Node ID in original graph
        """
        self.node_id = node_id

    def execute(self, state: AnnotationState) -> None:
        """Mark node as FP."""
        state.add_fp_node(self.node_id)

    def undo(self, state: AnnotationState) -> None:
        """Remove FP marking."""
        state.remove_fp_node(self.node_id)


class CompositeCommand(EdgeAnnotationCommand):
    """Command that executes multiple sub-commands as a single operation.

    Useful for operations that need to perform multiple actions atomically,
    like marking an edge FP and also marking orphan nodes FP.
    """

    def __init__(self, commands: list[EdgeAnnotationCommand]):
        """Initialize with a list of commands.

        Parameters
        ----------
        commands : list[EdgeAnnotationCommand]
            Commands to execute in order
        """
        self.commands = commands

    def execute(self, state: AnnotationState) -> None:
        """Execute all sub-commands in order."""
        for cmd in self.commands:
            cmd.execute(state)

    def undo(self, state: AnnotationState) -> None:
        """Undo all sub-commands in reverse order."""
        for cmd in reversed(self.commands):
            cmd.undo(state)
