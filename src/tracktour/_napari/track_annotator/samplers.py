"""Edge sampling strategies for the TrackAnnotator widget."""

import math
import random
from abc import ABC, abstractmethod
from typing import Optional

import networkx as nx


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

    def __init__(self, nxg: nx.DiGraph, seed: Optional[int] = None):
        self._edges: list[tuple[int, int]] = list(nxg.edges())
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


class TrajectoryEdgeSampler(EdgeSampler):
    """Samples edges in trajectory order via depth-first traversal of the solution graph.

    Starts from randomly ordered root nodes (in-degree 0). Follows each
    trajectory from root to leaf, visiting every edge along the way before
    moving to the next trajectory. At division nodes (two children), one
    branch is followed immediately while the other is deferred to a stack and
    visited after the current branch is exhausted.

    The full traversal order is determined once at construction time.

    Parameters
    ----------
    nxg : nx.DiGraph
        The solution graph. Nodes are detection IDs; edges are (parent, child).
    seed : int, optional
        Random seed for root ordering and division child ordering.
    """

    def __init__(self, nxg: nx.DiGraph, seed: Optional[int] = None):
        rng = random.Random(seed)
        self._edges = self._build_order(nxg, rng)
        self._idx = 0

    @staticmethod
    def _build_order(nxg: nx.DiGraph, rng: random.Random) -> list[tuple[int, int]]:
        roots = [n for n in nxg.nodes if nxg.in_degree(n) == 0]
        rng.shuffle(roots)

        order = []
        # Stack holds (parent, child) pairs. Roots have parent=None.
        # Reversed so the first root is on top.
        stack = [(None, root) for root in reversed(roots)]

        while stack:
            parent, current = stack.pop()
            if parent is not None:
                # Deferred branch — append its incoming edge before walking forward.
                order.append((parent, current))

            # Walk forward along this trajectory until a leaf.
            while True:
                children = list(nxg.successors(current))
                if not children:
                    break  # leaf — trajectory complete
                rng.shuffle(children)
                order.append((current, children[0]))
                # Defer all other children
                for deferred in children[1:]:
                    stack.append((current, deferred))
                current = children[0]

        return order

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


class DUCBEdgeSampler(EdgeSampler):
    """Samples edges adaptively using a D-UCB on given features as bandit arms.

    Each bandit arm corresponds to one feature column. At each step, the arm
    with the highest UCB score is selected and its top-ranked unvisited edge
    is returned. Rewards from the user (via ``provide_reward``) update the
    selected arm's state.

    Bandit hyperparameters (B, epsilon, gamma) are fixed to the values from
    the reference implementation and are not configurable. Gamma is computed
    from the number of edges: ``1 - 1 / (4 * sqrt(2 * n))`` (as described
    in the original DUCB paper):
    https://doi.org/10.48550/arXiv.0805.3415

    Navigating backwards with ``previous()`` replays the history without
    re-running the bandit. Calling ``next()`` doesn't run the bandit unless
    we're at the end of the visited edges.

    Parameters
    ----------
    nxg : nx.DiGraph
        Solution graph. Each edge must carry the feature attributes named in
        ``bandit_arms``.
    bandit_arms : dict[str, bool]
        Mapping of edge-attribute name → ascending flag.
        ``ascending=True`` means smaller values are sorted first (i.e. small
        feature value = more likely to be an error).
    """

    _B = 2.0
    _EPSILON = 0.5

    def __init__(
        self,
        nxg: nx.DiGraph,
        bandit_arms: dict[str, bool],
    ):
        if not bandit_arms:
            raise ValueError("bandit_arms must contain at least one arm.")

        edges = list(nxg.edges())
        self._total = len(edges)
        self._gamma = 1 - 1 / (4 * math.sqrt(2 * self._total))

        # Per-arm sorted list of (u, v) tuples
        self._arm_sorted: dict[str, list[tuple[int, int]]] = {}
        self._arm_ptr: dict[str, int] = {}
        for arm, ascending in bandit_arms.items():
            self._arm_sorted[arm] = sorted(
                edges,
                key=lambda e, a=arm: nxg.edges[e][a],
                reverse=not ascending,
            )
            self._arm_ptr[arm] = 0

        self.discounted_arm_played: dict[str, float] = {arm: 0.0 for arm in bandit_arms}
        self.discounted_arm_rewards: dict[str, float] = {
            arm: 0.0 for arm in bandit_arms
        }
        # Unweighted selection count per arm — used only to ensure every arm
        # is tried at least once before UCB scores are compared.
        self._arm_play_count: dict[str, int] = {arm: 0 for arm in bandit_arms}

        # Navigation state
        self._visited: set[tuple[int, int]] = set()
        # History entries: [(u, v), arm, reward] — mutable lists so reward can be updated.
        # reward starts as None until provide_reward is called for that position.
        self._history: list[list] = []
        self._hist_pos: int = -1
        # Set when a reward is changed retroactively (at a non-frontier position).
        # Triggers a full bandit-state recompute before the next frontier pick.
        self._history_dirty: bool = False

        # Initialise by picking the first edge
        self._pick_next_and_append()
        self._hist_pos = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _next_unvisited(self, arm: str) -> Optional[tuple[int, int]]:
        """Return the next unvisited edge for *arm*, or None."""
        sorted_list = self._arm_sorted[arm]
        ptr = self._arm_ptr[arm]
        while ptr < len(sorted_list):
            edge = sorted_list[ptr]
            ptr += 1
            if edge not in self._visited:
                self._arm_ptr[arm] = ptr
                return edge
        self._arm_ptr[arm] = ptr
        return None

    def _select_arm(self) -> Optional[str]:
        """Select the arm with the highest UCB score.

        Ensures each arm is tried at least once before UCB scores are
        compared. After the init phase, arms with discounted play count
        below 1 are treated as needing exploration (UCB = ∞).

        Returns None if all edges have been visited.
        """
        if len(self._visited) >= self._total:
            return None

        # Init: play each arm at least once before UCB kicks in
        for arm in self._arm_sorted:
            if self._arm_play_count[arm] == 0:
                return arm

        eta_t = sum(self.discounted_arm_played.values())
        best_arm = None
        best_ucb = -math.inf

        for arm in self._arm_sorted:
            n = self.discounted_arm_played[arm]
            if n < 1:
                # Discounted below 1 — confidence bound is infinite (source: eta_t/Nt < 1)
                ucb = math.inf
            else:
                ucb = self.discounted_arm_rewards[arm] / n + self._B * math.sqrt(
                    self._EPSILON * math.log(eta_t) / n
                )
            if ucb > best_ucb:
                best_ucb = ucb
                best_arm = arm
        return best_arm

    def _apply_discount(self) -> None:
        for arm in self.discounted_arm_played:
            self.discounted_arm_played[arm] *= self._gamma
            self.discounted_arm_rewards[arm] *= self._gamma

    def _recompute_bandit_state(self) -> None:
        """Recompute discounted_arm_played and discounted_arm_rewards from history.

        Called when a retroactive reward change has dirtied the running totals.
        O(H) where H is the number of history entries.

        Each history entry at position i has been discounted by gamma once per
        subsequent round, so its contribution is gamma^(H - 1 - i).
        """
        H = len(self._history)
        for arm in self.discounted_arm_played:
            self.discounted_arm_played[arm] = 0.0
            self.discounted_arm_rewards[arm] = 0.0
        for i, (_, arm, reward) in enumerate(self._history):
            discount = self._gamma ** (H - 1 - i)
            self.discounted_arm_played[arm] += discount
            if reward is not None:
                self.discounted_arm_rewards[arm] += reward * discount

    def _pick_next_and_append(self) -> bool:
        """Run one UCB round, append the chosen edge to history.

        Returns True if an edge was appended, False if all edges are exhausted.
        """
        if self._history_dirty:
            self._recompute_bandit_state()
            self._history_dirty = False
        self._apply_discount()
        arm = self._select_arm()
        if arm is None:
            return False
        edge = self._next_unvisited(arm)
        if edge is None:
            return False
        self.discounted_arm_played[arm] += 1.0
        self._arm_play_count[arm] += 1
        self._visited.add(edge)
        self._history.append([edge, arm, None])
        return True

    # ------------------------------------------------------------------
    # EdgeSampler interface
    # ------------------------------------------------------------------

    def current(self) -> tuple[int, int]:
        return self._history[self._hist_pos][0]

    def next(self) -> Optional[tuple[int, int]]:
        if self._hist_pos < len(self._history) - 1:
            # Replay: advance through already-computed history
            self._hist_pos += 1
            return self.current()
        # At frontier: run UCB to pick the next edge
        if not self._pick_next_and_append():
            return None
        self._hist_pos += 1
        return self.current()

    def previous(self) -> Optional[tuple[int, int]]:
        if self._hist_pos > 0:
            self._hist_pos -= 1
            return self.current()
        return None

    def at_start(self) -> bool:
        return self._hist_pos == 0

    def at_end(self) -> bool:
        return (
            self._hist_pos == len(self._history) - 1
            and len(self._visited) >= self._total
        )

    def total_count(self) -> int:
        return self._total

    def current_index(self) -> int:
        return self._hist_pos

    def provide_reward(self, reward: float) -> None:
        """Record the reward for the current edge and update bandit state.

        At the frontier this is an O(1) update. At a previously visited
        position, if the reward has changed, the dirty flag is set so the
        full bandit state is recomputed (O(H)) before the next frontier pick.

        Parameters
        ----------
        reward : float
            Reward signal for the current edge. Typically 1.0 if the edge
            is an error and 0.0 if it is correct.
        """
        entry = self._history[self._hist_pos]
        _, arm, old_reward = entry
        if old_reward == reward:
            return
        entry[2] = reward
        at_frontier = self._hist_pos == len(self._history) - 1
        if at_frontier:
            # O(1) delta: subtract old contribution (0 if never set) and add new
            previous = old_reward if old_reward is not None else 0.0
            self.discounted_arm_rewards[arm] += reward - previous
        else:
            self._history_dirty = True
