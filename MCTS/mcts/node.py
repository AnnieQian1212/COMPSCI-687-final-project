import numpy as np
from typing import Optional, Dict, Any, List


class MCTSNode:
    """Node in the Monte Carlo Tree Search tree."""

    def __init__(
        self,
        state: Optional[Any] = None,
        parent: Optional['MCTSNode'] = None,
        action: Optional[int] = None,
        n_actions: int = 2
    ):
        """
        Initialize an MCTS node.

        Args:
            state: Environment state (for stochastic environments)
            parent: Parent node
            action: Action taken to reach this node from parent
            n_actions: Number of possible actions
        """
        self.state = state
        self.parent = parent
        self.action = action
        self.n_actions = n_actions

        self.children: Dict[int, 'MCTSNode'] = {}
        self.visits: int = 0
        self.total_reward: float = 0.0

        # Track which actions haven't been tried yet
        self.untried_actions: List[int] = list(range(n_actions))

    @property
    def is_fully_expanded(self) -> bool:
        """Check if all actions have been tried from this node."""
        return len(self.untried_actions) == 0

    @property
    def is_terminal(self) -> bool:
        """Check if this is a terminal node (no children possible)."""
        return self.n_actions == 0

    @property
    def q_value(self) -> float:
        """Get the average reward (Q-value) for this node."""
        if self.visits == 0:
            return 0.0
        return self.total_reward / self.visits

    def ucb1(self, exploration_c: float = 1.41) -> float:
        """
        Calculate the UCB1 value for this node.

        UCB(n) = Q(n)/N(n) + c * sqrt(ln(N_parent) / N(n))

        Args:
            exploration_c: Exploration constant

        Returns:
            UCB1 value
        """
        if self.visits == 0:
            return float('inf')

        if self.parent is None:
            return self.q_value

        exploitation = self.q_value
        exploration = exploration_c * np.sqrt(
            np.log(self.parent.visits) / self.visits
        )

        return exploitation + exploration

    def select_child(self, exploration_c: float = 1.41) -> 'MCTSNode':
        """
        Select the child with the highest UCB1 value.

        Args:
            exploration_c: Exploration constant

        Returns:
            Child node with highest UCB1 value
        """
        return max(
            self.children.values(),
            key=lambda child: child.ucb1(exploration_c)
        )

    def expand(self, action: int, state: Optional[Any] = None) -> 'MCTSNode':
        """
        Expand the tree by adding a child node for the given action.

        Args:
            action: Action to expand
            state: State after taking the action (for stochastic envs)

        Returns:
            Newly created child node
        """
        if action in self.children:
            raise ValueError(f"Action {action} already expanded")

        if action in self.untried_actions:
            self.untried_actions.remove(action)

        child = MCTSNode(
            state=state,
            parent=self,
            action=action,
            n_actions=self.n_actions
        )
        self.children[action] = child

        return child

    def update(self, reward: float) -> None:
        """
        Update this node's statistics after a simulation.

        Args:
            reward: Reward from the simulation
        """
        self.visits += 1
        self.total_reward += reward

    def get_action_sequence(self) -> List[int]:
        """
        Get the sequence of actions from the root to this node.

        Returns:
            List of actions from root to this node
        """
        actions = []
        node = self
        while node.parent is not None:
            actions.append(node.action)
            node = node.parent
        return list(reversed(actions))

    def __repr__(self) -> str:
        return (
            f"MCTSNode(action={self.action}, "
            f"visits={self.visits}, "
            f"q_value={self.q_value:.3f})"
        )
