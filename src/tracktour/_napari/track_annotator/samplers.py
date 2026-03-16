"""Edge sampling strategies for the TrackAnnotator widget."""

import random
from abc import ABC, abstractmethod
from typing import Optional


class EdgeSampler(ABC):
    """Abstract base class for edge sampling strategies."""

    @abstractmethod
    def current(self) -> tuple[int, int]:
        """Return the current edge."""
        pass

    @abstractmethod
    def next(self) -> Optional[tuple[int, int]]:
        """Move to and return the next edge, or None if at end."""
        pass

    @abstractmethod
    def previous(self) -> Optional[tuple[int, int]]:
        """Move to and return the previous edge, or None if at start."""
        pass

    @abstractmethod
    def at_start(self) -> bool:
        """Return True if at the first edge."""
        pass

    @abstractmethod
    def at_end(self) -> bool:
        """Return True if at the last edge."""
        pass

    @abstractmethod
    def total_count(self) -> int:
        """Return the total number of edges."""
        pass

    @abstractmethod
    def current_index(self) -> int:
        """Return the current position index."""
        pass

    def provide_reward(self, reward: float) -> None:
        """Provide feedback after annotation. Override for adaptive samplers."""
        pass


class RandomEdgeSampler(EdgeSampler):
    """Samples edges in random order with optional seeded shuffle."""

    def __init__(self, edges: list[tuple[int, int]], seed: Optional[int] = None):
        self._edges: list[tuple[int, int]] = [(int(e[0]), int(e[1])) for e in edges]
        if seed is not None:
            random.seed(seed)
        random.shuffle(self._edges)
        self._idx = 0

    def current(self) -> tuple[int, int]:
        return self._edges[self._idx]

    def next(self) -> Optional[tuple[int, int]]:
        if self._idx < len(self._edges) - 1:
            self._idx += 1
            return self.current()
        return None

    def previous(self) -> Optional[tuple[int, int]]:
        if self._idx > 0:
            self._idx -= 1
            return self.current()
        return None

    def at_start(self) -> bool:
        return self._idx == 0

    def at_end(self) -> bool:
        return self._idx >= len(self._edges) - 1

    def total_count(self) -> int:
        return len(self._edges)

    def current_index(self) -> int:
        return self._idx
