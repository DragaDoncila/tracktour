"""Edge sampling strategies for the TrackAnnotator widget."""

import math
import random
from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd


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


class DUCBEdgeSampler(EdgeSampler):
    """Samples edges adaptively using a D-UCB on given features as bandit arms.

    Each bandit arm corresponds to one feature column. At each step, the arm
    with the highest UCB score is selected and its top-ranked unvisited edge
    is returned. Rewards from the user (via ``provide_reward``) update the
    selected arm's state. A discount factor ``gamma`` down-weights older
    observations, allowing the sampler to adapt as annotation progresses.

    Navigating backwards with ``previous()`` replays the history without
    re-running the bandit. Calling ``next()`` doesn't run the bandit unless
    we're at the end of the visited edges.

    Parameters
    ----------
    edges_df : pd.DataFrame
        Solution edges with feature columns. Must have ``u`` and ``v`` columns.
    bandit_arms : dict[str, bool]
        Mapping of feature column name → ascending flag.
        ``ascending=True`` means smaller values are sorted first (i.e. small
        feature value = more likely to be an error).
    B : float
        Exploration coefficient. Higher values favour less-played arms.
    epsilon : float
        Scales the confidence bound width.
    gamma : float
        Discount factor in (0, 1]. ``1.0`` means no discounting.
    """

    def __init__(
        self,
        edges_df: pd.DataFrame,
        bandit_arms: dict[str, bool],
        B: float = 1.0,
        epsilon: float = 2.0,
        gamma: float = 1.0,
    ):
        if not bandit_arms:
            raise ValueError("bandit_arms must contain at least one arm.")

        self._B = B
        self._epsilon = epsilon
        self._gamma = gamma
        self._total = len(edges_df)

        # Per-arm sorted list of DataFrame indices (not reset integer positions)
        self._arm_sorted: dict[str, list] = {}
        self._arm_ptr: dict[str, int] = {}
        for arm, ascending in bandit_arms.items():
            self._arm_sorted[arm] = list(
                edges_df.sort_values(arm, ascending=ascending).index
            )
            self._arm_ptr[arm] = 0

        # Bandit state: discounted play count (N, updated at selection) and
        # discounted reward sum (S, updated in provide_reward).
        self.discounted_arm_played: dict[str, float] = {arm: 0.0 for arm in bandit_arms}
        self.discounted_arm_rewards: dict[str, float] = {
            arm: 0.0 for arm in bandit_arms
        }
        # Unweighted selection count per arm — used only to ensure every arm
        # is tried at least once before UCB scores are compared.
        self._arm_play_count: dict[str, int] = {arm: 0 for arm in bandit_arms}

        # Build lookup from df index → (u, v)
        self._edge_lookup: dict = {
            row.Index: (int(row.u), int(row.v)) for row in edges_df.itertuples()
        }

        # Navigation state
        self._visited: set = set()
        # History entries: (df_index, arm_name_or_None)
        self._history: list[tuple] = []
        self._hist_pos: int = -1

        # Initialise by picking the first edge
        self._pick_next_and_append()
        self._hist_pos = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _next_unvisited(self, arm: str) -> Optional[int]:
        """Return the df-index of the next unvisited edge for *arm*, or None."""
        sorted_list = self._arm_sorted[arm]
        ptr = self._arm_ptr[arm]
        while ptr < len(sorted_list):
            idx = sorted_list[ptr]
            ptr += 1
            if idx not in self._visited:
                self._arm_ptr[arm] = ptr
                return idx
        self._arm_ptr[arm] = ptr
        return None

    def _select_arm(self) -> Optional[str]:
        """Select the arm with the highest UCB score.

        Ensures each arm is tried at least once before UCB scores are
        compared. After the init phase, arms with discounted play count
        below 1 are treated as needing exploration (UCB = ∞), matching
        the behaviour of the reference D-UCB implementation.

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
                    self._epsilon * math.log(eta_t) / n
                )
            if ucb > best_ucb:
                best_ucb = ucb
                best_arm = arm
        return best_arm

    def _apply_discount(self) -> None:
        for arm in self.discounted_arm_played:
            self.discounted_arm_played[arm] *= self._gamma
            self.discounted_arm_rewards[arm] *= self._gamma

    def _pick_next_and_append(self) -> bool:
        """Run one UCB round, append the chosen edge to history.

        Returns True if an edge was appended, False if all edges are exhausted.
        """
        self._apply_discount()
        arm = self._select_arm()
        if arm is None:
            return False
        df_idx = self._next_unvisited(arm)
        if df_idx is None:
            return False
        self.discounted_arm_played[arm] += 1.0
        self._arm_play_count[arm] += 1
        self._visited.add(df_idx)
        self._history.append((df_idx, arm))
        return True

    # ------------------------------------------------------------------
    # EdgeSampler interface
    # ------------------------------------------------------------------

    def current(self) -> tuple[int, int]:
        df_idx, _ = self._history[self._hist_pos]
        return self._edge_lookup[df_idx]

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
        """Update the reward sum for the arm that produced the current edge.

        Parameters
        ----------
        reward : float
            Reward signal for the current edge. Typically 1.0 if the edge
            is an error and 0.0 if it is correct.
        """
        _, arm = self._history[self._hist_pos]
        if arm is not None:
            self.discounted_arm_rewards[arm] += reward
