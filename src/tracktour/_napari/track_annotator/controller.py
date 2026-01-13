"""Controller for managing track annotation operations.

The controller handles validation, command creation, and maintains
a mapping of edges to their annotation commands for per-edge undo.
"""

from typing import Union

from .commands import EdgeAnnotationCommand
from .state import AnnotationState


class AnnotationController:
    """Manages track annotation operations with per-edge undo support.

    The controller maintains a mapping from edges to their annotation
    commands, allowing individual edges to be reset to their original
    state independent of other edges.

    Parameters
    ----------
    state : AnnotationState
        The annotation state to manage

    Attributes
    ----------
    state : AnnotationState
        The annotation state
    edge_commands : dict
        Mapping from edge tuples to their annotation commands
    """

    def __init__(self, state: AnnotationState):
        self.state = state
        # Map edges to their current annotation command
        self.edge_commands: dict[tuple[int, int], EdgeAnnotationCommand] = {}

    def annotate_edge(
        self,
        edge: tuple[int, int],
        command: EdgeAnnotationCommand,
    ) -> None:
        """Annotate an edge with the given command.

        If the edge was previously annotated, undoes the old annotation
        before applying the new one.

        Parameters
        ----------
        edge : tuple[int, int]
            Edge to annotate
        command : EdgeAnnotationCommand
            Command to execute
        """
        # If edge was previously annotated, undo it first
        if edge in self.edge_commands:
            self.edge_commands[edge].undo(self.state)

        # Execute the new command
        command.execute(self.state)
        self.edge_commands[edge] = command

    def reset_edge_to_original(self, edge: tuple[int, int]) -> bool:
        """Reset a specific edge to its original state.

        Parameters
        ----------
        edge : tuple[int, int]
            Edge to reset

        Returns
        -------
        bool
            True if the edge was reset, False if it was never annotated
        """
        if edge in self.edge_commands:
            self.edge_commands[edge].undo(self.state)
            del self.edge_commands[edge]
            return True
        return False

    def is_edge_annotated(self, edge: tuple[int, int]) -> bool:
        """Check if an edge has been annotated.

        Parameters
        ----------
        edge : tuple[int, int]
            Edge to check

        Returns
        -------
        bool
            True if the edge has been annotated
        """
        return edge in self.edge_commands

    def get_edge_command(
        self, edge: tuple[int, int]
    ) -> Union[EdgeAnnotationCommand, None]:
        """Get the command associated with an edge.

        Parameters
        ----------
        edge : tuple[int, int]
            Edge to query

        Returns
        -------
        EdgeAnnotationCommand or None
            The command for this edge, or None if not annotated
        """
        return self.edge_commands.get(edge)

    def get_annotated_edges(self) -> set[tuple[int, int]]:
        """Get all edges that have been annotated.

        Returns
        -------
        set[tuple[int, int]]
            Set of annotated edges
        """
        return set(self.edge_commands.keys())

    def clear_all_annotations(self) -> None:
        """Clear all annotations and reset to initial state.

        This undoes all edge annotations and clears the command history.
        """
        # Undo all commands
        for command in self.edge_commands.values():
            command.undo(self.state)

        # Clear the mapping
        self.edge_commands.clear()
