import numpy as np
import gymnasium as gym
from typing import Optional, Tuple, List, Any
from copy import deepcopy

from .node import MCTSNode


class MCTSAgent:
    """Monte Carlo Tree Search Agent for Gymnasium environments."""

    def __init__(
        self,
        env: gym.Env,
        n_simulations: int = 100,
        exploration_c: float = 1.41,
        max_rollout_depth: int = 500,
        seed: Optional[int] = None
    ):
        """
        Initialize the MCTS agent.

        Args:
            env: Gymnasium environment
            n_simulations: Number of MCTS iterations per decision
            exploration_c: UCB exploration constant
            max_rollout_depth: Maximum depth for rollout simulations
            seed: Random seed for reproducibility
        """
        self.env = env
        self.n_simulations = n_simulations
        self.exploration_c = exploration_c
        self.max_rollout_depth = max_rollout_depth

        # Get action space size
        if isinstance(env.action_space, gym.spaces.Discrete):
            self.n_actions = env.action_space.n
        else:
            raise ValueError("Only discrete action spaces are supported")

        # Set random seed
        self.rng = np.random.default_rng(seed)

    def select_action(
        self,
        root_state: Any,
        is_deterministic: bool = True
    ) -> int:
        """
        Select the best action using MCTS.

        Args:
            root_state: Current environment state
            is_deterministic: Whether the environment is deterministic

        Returns:
            Best action to take
        """
        # Create root node
        root = MCTSNode(state=root_state, n_actions=self.n_actions)

        # Run MCTS simulations
        for _ in range(self.n_simulations):
            if is_deterministic:
                self._simulate_deterministic(root, root_state)
            else:
                self._simulate_stochastic(root)

        # Select best action (most visited child)
        if not root.children:
            return self.rng.integers(self.n_actions)

        best_action = max(
            root.children.keys(),
            key=lambda a: root.children[a].visits
        )

        return best_action

    def _simulate_deterministic(
        self,
        root: MCTSNode,
        root_state: Any
    ) -> None:
        """
        Run one MCTS simulation for deterministic environments.
        Uses action-sequence replay approach.

        Args:
            root: Root node of the tree
            root_state: Root state to replay from
        """
        node = root

        # Phase 1: Selection - traverse tree using UCB1
        while node.is_fully_expanded and node.children:
            node = node.select_child(self.exploration_c)

        # Get action sequence to reach current node
        action_sequence = node.get_action_sequence()

        # Replay to reach the current node's state
        self.env.unwrapped.state = root_state.copy()
        done = False
        total_reward = 0.0

        for action in action_sequence:
            _, reward, terminated, truncated, _ = self.env.step(action)
            total_reward += reward
            done = terminated or truncated
            if done:
                break

        # Phase 2: Expansion - add a new child node
        if not done and not node.is_fully_expanded:
            action = self.rng.choice(node.untried_actions)
            _, reward, terminated, truncated, _ = self.env.step(action)
            total_reward += reward
            done = terminated or truncated
            node = node.expand(action)

        # Phase 3: Simulation/Rollout - random policy until terminal
        if not done:
            rollout_reward = self._rollout()
            total_reward += rollout_reward

        # Phase 4: Backpropagation - update statistics
        self._backpropagate(node, total_reward)

    def _simulate_stochastic(self, root: MCTSNode) -> None:
        """
        Run one MCTS simulation for stochastic environments.
        Uses state-storing approach.

        Args:
            root: Root node of the tree (contains initial state)
        """
        node = root

        # Phase 1: Selection
        while node.is_fully_expanded and node.children:
            node = node.select_child(self.exploration_c)

        # Create a fresh environment for simulation
        sim_env = deepcopy(self.env)

        # Set state if available (for Blackjack and similar envs)
        if node.state is not None:
            self._set_env_state(sim_env, node.state)

        # Phase 2: Expansion
        done = False
        total_reward = 0.0

        if not node.is_fully_expanded:
            action = self.rng.choice(node.untried_actions)
            obs, reward, terminated, truncated, _ = sim_env.step(action)
            total_reward += reward
            done = terminated or truncated

            # Store the new state
            new_state = self._get_env_state(sim_env, obs)
            node = node.expand(action, state=new_state)

        # Phase 3: Simulation/Rollout
        if not done:
            rollout_reward = self._rollout_env(sim_env)
            total_reward += rollout_reward

        # Phase 4: Backpropagation
        self._backpropagate(node, total_reward)

    def _rollout(self) -> float:
        """
        Perform a random rollout from the current environment state.

        Returns:
            Total reward from the rollout
        """
        total_reward = 0.0
        done = False
        depth = 0

        while not done and depth < self.max_rollout_depth:
            action = self.rng.integers(self.n_actions)
            _, reward, terminated, truncated, _ = self.env.step(action)
            total_reward += reward
            done = terminated or truncated
            depth += 1

        return total_reward

    def _rollout_env(self, env: gym.Env) -> float:
        """
        Perform a random rollout using a given environment instance.

        Args:
            env: Environment to use for rollout

        Returns:
            Total reward from the rollout
        """
        total_reward = 0.0
        done = False
        depth = 0

        while not done and depth < self.max_rollout_depth:
            action = self.rng.integers(self.n_actions)
            _, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
            depth += 1

        return total_reward

    def _backpropagate(self, node: MCTSNode, reward: float) -> None:
        """
        Backpropagate the reward up the tree.

        Args:
            node: Node to start backpropagation from
            reward: Reward to backpropagate
        """
        while node is not None:
            node.update(reward)
            node = node.parent

    def _get_env_state(self, env: gym.Env, obs: Any) -> Any:
        """
        Get the state from an environment for storage.

        Args:
            env: Environment
            obs: Current observation

        Returns:
            State representation for storage
        """
        # For Blackjack, the observation is the state
        return obs

    def _set_env_state(self, env: gym.Env, state: Any) -> None:
        """
        Set the environment to a given state.

        Args:
            env: Environment to modify
            state: State to set
        """
        # For Blackjack, we need to manipulate internal state
        # This is environment-specific
        if hasattr(env, 'unwrapped'):
            unwrapped = env.unwrapped
            if hasattr(unwrapped, 'player') and len(state) >= 3:
                # Blackjack state: (player_sum, dealer_showing, usable_ace)
                pass  # Blackjack doesn't support state setting easily


class CartPoleMCTSAgent(MCTSAgent):
    """MCTS Agent optimized for CartPole environment."""

    def _reset_env_state(self, state: np.ndarray, elapsed_steps: int = 0) -> None:
        """Reset environment to a specific state."""
        self.env.unwrapped.state = state.copy()
        self.env.unwrapped.steps_beyond_terminated = None
        # Reset the TimeLimit wrapper's step counter
        if hasattr(self.env, '_elapsed_steps'):
            self.env._elapsed_steps = elapsed_steps

    def select_action(self, observation: np.ndarray) -> int:
        """
        Select action for CartPole.

        Args:
            observation: Current observation

        Returns:
            Best action
        """
        # Save the current state and step count
        state = self.env.unwrapped.state
        root_state = np.array(state) if isinstance(state, tuple) else state.copy()
        elapsed_steps = getattr(self.env, '_elapsed_steps', 0)

        # Create root node
        root = MCTSNode(state=root_state, n_actions=self.n_actions)

        # Run MCTS simulations
        for _ in range(self.n_simulations):
            self._simulate_cartpole(root, root_state, elapsed_steps)

        # Restore the environment state after all simulations
        self._reset_env_state(root_state, elapsed_steps)

        # Select best action (most visited child)
        if not root.children:
            return self.rng.integers(self.n_actions)

        best_action = max(
            root.children.keys(),
            key=lambda a: root.children[a].visits
        )

        return best_action

    def _simulate_cartpole(
        self,
        root: MCTSNode,
        root_state: np.ndarray,
        elapsed_steps: int
    ) -> None:
        """
        Run one MCTS simulation for CartPole.
        Uses action-sequence replay approach.

        Args:
            root: Root node of the tree
            root_state: Root state to replay from
            elapsed_steps: Current step count to restore
        """
        node = root

        # Phase 1: Selection - traverse tree using UCB1
        while node.is_fully_expanded and node.children:
            node = node.select_child(self.exploration_c)

        # Get action sequence to reach current node
        action_sequence = node.get_action_sequence()

        # Reset env to root state and replay action sequence
        self._reset_env_state(root_state, elapsed_steps)
        done = False
        total_reward = 0.0

        for action in action_sequence:
            _, reward, terminated, truncated, _ = self.env.step(action)
            total_reward += reward
            done = terminated or truncated
            if done:
                break

        # Phase 2: Expansion - add a new child node
        if not done and not node.is_fully_expanded:
            action = self.rng.choice(node.untried_actions)
            _, reward, terminated, truncated, _ = self.env.step(action)
            total_reward += reward
            done = terminated or truncated
            node = node.expand(action)

        # Phase 3: Simulation/Rollout - random policy until terminal
        if not done:
            rollout_reward = self._rollout()
            total_reward += rollout_reward

        # Phase 4: Backpropagation - update statistics
        self._backpropagate(node, total_reward)


class BlackjackMCTSAgent(MCTSAgent):
    """MCTS Agent optimized for Blackjack environment."""

    def __init__(
        self,
        env: gym.Env,
        n_simulations: int = 100,
        exploration_c: float = 1.41,
        seed: Optional[int] = None
    ):
        """Initialize Blackjack MCTS agent."""
        super().__init__(
            env=env,
            n_simulations=n_simulations,
            exploration_c=exploration_c,
            max_rollout_depth=10,  # Blackjack games are short
            seed=seed
        )

    def select_action(self, observation: Tuple) -> int:
        """
        Select action for Blackjack.

        Args:
            observation: Current observation (player_sum, dealer_showing, usable_ace)

        Returns:
            Best action (0=stick, 1=hit)
        """
        # For Blackjack, we use a simplified approach:
        # Run simulations from the current state using fresh environments

        root = MCTSNode(state=observation, n_actions=self.n_actions)

        for _ in range(self.n_simulations):
            self._simulate_blackjack(root, observation)

        if not root.children:
            return self.rng.integers(self.n_actions)

        best_action = max(
            root.children.keys(),
            key=lambda a: root.children[a].visits
        )

        return best_action

    def _simulate_blackjack(
        self,
        root: MCTSNode,
        root_obs: Tuple
    ) -> None:
        """
        Run one MCTS simulation for Blackjack.

        Since Blackjack is stochastic and has short episodes,
        we use a flat MCTS approach focusing on the first decision.

        Args:
            root: Root node
            root_obs: Root observation
        """
        node = root

        # Selection
        while node.is_fully_expanded and node.children:
            node = node.select_child(self.exploration_c)

        # Expansion and Simulation combined for Blackjack
        # We estimate value through multiple random games
        if not node.is_fully_expanded:
            action = self.rng.choice(node.untried_actions)
            node = node.expand(action)

            # Estimate value by simulating from root with this action
            total_reward = self._estimate_action_value(root_obs, action)
        else:
            # Leaf node - estimate its value
            action_sequence = node.get_action_sequence()
            total_reward = self._simulate_game(root_obs, action_sequence)

        # Backpropagation
        self._backpropagate(node, total_reward)

    def _estimate_action_value(
        self,
        obs: Tuple,
        first_action: int
    ) -> float:
        """
        Estimate the value of taking an action from the given observation.

        Args:
            obs: Current observation
            first_action: Action to evaluate

        Returns:
            Estimated reward
        """
        # Create a new environment and play out the game
        sim_env = gym.make('Blackjack-v1')
        sim_env.reset()

        # Set the initial state to match our observation
        player_sum, dealer_showing, usable_ace = obs
        sim_env.unwrapped.player = [player_sum] if player_sum <= 21 else [10, player_sum - 10]
        sim_env.unwrapped.dealer = [dealer_showing]

        # Take the first action
        _, reward, terminated, truncated, _ = sim_env.step(first_action)

        # If game continues, play randomly
        while not (terminated or truncated):
            action = self.rng.integers(self.n_actions)
            _, reward, terminated, truncated, _ = sim_env.step(action)

        sim_env.close()
        return reward

    def _simulate_game(
        self,
        obs: Tuple,
        action_sequence: List[int]
    ) -> float:
        """
        Simulate a game following an action sequence then playing randomly.

        Args:
            obs: Initial observation
            action_sequence: Actions to take first

        Returns:
            Final reward
        """
        sim_env = gym.make('Blackjack-v1')
        sim_env.reset()

        # Set initial state
        player_sum, dealer_showing, usable_ace = obs
        sim_env.unwrapped.player = [player_sum] if player_sum <= 21 else [10, player_sum - 10]
        sim_env.unwrapped.dealer = [dealer_showing]

        reward = 0
        terminated = False
        truncated = False

        # Follow action sequence
        for action in action_sequence:
            if terminated or truncated:
                break
            _, reward, terminated, truncated, _ = sim_env.step(action)

        # Random rollout
        while not (terminated or truncated):
            action = self.rng.integers(self.n_actions)
            _, reward, terminated, truncated, _ = sim_env.step(action)

        sim_env.close()
        return reward
