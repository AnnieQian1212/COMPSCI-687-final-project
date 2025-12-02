"""
Environment wrappers for OpenAI Gym's Blackjack and CartPole environments.
"""

import gymnasium as gym
import numpy as np


def create_blackjack_env():
    """
    Create Blackjack environment.
    
    Observation Space:
        - player's current sum (1-31)
        - dealer's face up card (1-10)
        - whether player has usable ace (0 or 1)
    
    Action Space:
        - 0: stick (stop receiving cards)
        - 1: hit (receive another card)
    
    Returns:
        gym.Env: Blackjack environment
    """
    env = gym.make('Blackjack-v1', natural=False, sab=False)
    return env


def create_cartpole_env():
    """
    Create CartPole environment.
    
    Observation Space (4-dimensional continuous):
        - Cart Position (-4.8 to 4.8)
        - Cart Velocity (-Inf to Inf)
        - Pole Angle (-0.418 rad to 0.418 rad)
        - Pole Angular Velocity (-Inf to Inf)
    
    Action Space:
        - 0: push cart to the left
        - 1: push cart to the right
    
    Returns:
        gym.Env: CartPole environment
    """
    env = gym.make('CartPole-v1')
    return env


def get_state_dim(env_name):
    """
    Get state dimension for different environments.
    
    Args:
        env_name: Name of the environment ('blackjack' or 'cartpole')
    
    Returns:
        int: State dimension
    """
    if env_name.lower() == 'blackjack':
        return 3  # (player_sum, dealer_card, usable_ace)
    elif env_name.lower() == 'cartpole':
        return 4  # (position, velocity, angle, angular_velocity)
    else:
        raise ValueError(f"Unknown environment: {env_name}")


def get_action_dim(env_name):
    """
    Get action dimension for different environments.
    
    Args:
        env_name: Name of the environment ('blackjack' or 'cartpole')
    
    Returns:
        int: Action dimension
    """
    if env_name.lower() == 'blackjack':
        return 2  # stick or hit
    elif env_name.lower() == 'cartpole':
        return 2  # left or right
    else:
        raise ValueError(f"Unknown environment: {env_name}")


def preprocess_blackjack_state(state):
    """
    Preprocess Blackjack state for neural network input.
    
    Args:
        state: Tuple of (player_sum, dealer_card, usable_ace)
    
    Returns:
        np.ndarray: Normalized state vector
    """
    player_sum, dealer_card, usable_ace = state
    # Normalize: player_sum [1,31] -> [0,1], dealer_card [1,10] -> [0,1]
    normalized_state = np.array([
        player_sum / 31.0,
        dealer_card / 10.0,
        float(usable_ace)
    ], dtype=np.float32)
    return normalized_state


def preprocess_cartpole_state(state):
    """
    Preprocess CartPole state for neural network input.
    
    Args:
        state: Array of [position, velocity, angle, angular_velocity]
    
    Returns:
        np.ndarray: Normalized state vector
    """
    # Simple normalization for CartPole
    state = np.array(state, dtype=np.float32)
    # Approximate normalization based on typical ranges
    normalized_state = np.array([
        state[0] / 4.8,           # position
        np.clip(state[1], -3, 3) / 3.0,  # velocity
        state[2] / 0.418,         # angle
        np.clip(state[3], -3, 3) / 3.0   # angular velocity
    ], dtype=np.float32)
    return normalized_state


if __name__ == "__main__":
    # Test environments
    print("Testing Blackjack environment...")
    blackjack_env = create_blackjack_env()
    state, info = blackjack_env.reset()
    print(f"  Initial state: {state}")
    print(f"  Preprocessed: {preprocess_blackjack_state(state)}")
    print(f"  Action space: {blackjack_env.action_space}")
    
    print("\nTesting CartPole environment...")
    cartpole_env = create_cartpole_env()
    state, info = cartpole_env.reset()
    print(f"  Initial state: {state}")
    print(f"  Preprocessed: {preprocess_cartpole_state(state)}")
    print(f"  Action space: {cartpole_env.action_space}")
    
    blackjack_env.close()
    cartpole_env.close()
    print("\nEnvironments tested successfully!")
