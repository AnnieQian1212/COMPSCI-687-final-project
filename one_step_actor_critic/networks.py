"""
Neural network architectures for Actor and Critic in One-Step Actor-Critic.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class ActorNetwork(nn.Module):
    """
    Actor network that outputs action probabilities (policy).
    
    Architecture:
        Input -> Hidden1 -> Hidden2 -> Output (softmax)
    """
    
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        """
        Initialize Actor network.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dim: Number of hidden units
        """
        super(ActorNetwork, self).__init__()
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, state):
        """
        Forward pass to get action probabilities.
        
        Args:
            state: State tensor of shape (batch_size, state_dim)
        
        Returns:
            action_probs: Action probabilities of shape (batch_size, action_dim)
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        action_logits = self.fc3(x)
        action_probs = F.softmax(action_logits, dim=-1)
        return action_probs
    
    def get_action(self, state):
        """
        Sample action from policy and return action with log probability.
        
        Args:
            state: State tensor
        
        Returns:
            action: Sampled action
            log_prob: Log probability of the action
        """
        action_probs = self.forward(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob
    
    def get_log_prob(self, state, action):
        """
        Get log probability of taking action in state.
        
        Args:
            state: State tensor
            action: Action tensor
        
        Returns:
            log_prob: Log probability of the action
        """
        action_probs = self.forward(state)
        dist = Categorical(action_probs)
        return dist.log_prob(action)


class CriticNetwork(nn.Module):
    """
    Critic network that estimates state value V(s).
    
    Architecture:
        Input -> Hidden1 -> Hidden2 -> Output (scalar value)
    """
    
    def __init__(self, state_dim, hidden_dim=128):
        """
        Initialize Critic network.
        
        Args:
            state_dim: Dimension of state space
            hidden_dim: Number of hidden units
        """
        super(CriticNetwork, self).__init__()
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
    def forward(self, state):
        """
        Forward pass to get state value.
        
        Args:
            state: State tensor of shape (batch_size, state_dim)
        
        Returns:
            value: State value of shape (batch_size, 1)
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        value = self.fc3(x)
        return value


class ActorCriticNetwork(nn.Module):
    """
    Combined Actor-Critic network with shared feature layers.
    
    This is an alternative architecture where actor and critic 
    share the same feature extraction layers.
    
    Architecture:
        Input -> Shared Hidden -> Actor Head (action probs)
                              -> Critic Head (value)
    """
    
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        """
        Initialize combined Actor-Critic network.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dim: Number of hidden units
        """
        super(ActorCriticNetwork, self).__init__()
        
        # Shared feature layers
        self.shared_fc1 = nn.Linear(state_dim, hidden_dim)
        self.shared_fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        # Actor head
        self.actor_head = nn.Linear(hidden_dim, action_dim)
        
        # Critic head
        self.critic_head = nn.Linear(hidden_dim, 1)
        
    def forward(self, state):
        """
        Forward pass to get both action probabilities and state value.
        
        Args:
            state: State tensor of shape (batch_size, state_dim)
        
        Returns:
            action_probs: Action probabilities
            value: State value
        """
        # Shared features
        x = F.relu(self.shared_fc1(state))
        x = F.relu(self.shared_fc2(x))
        
        # Actor output
        action_logits = self.actor_head(x)
        action_probs = F.softmax(action_logits, dim=-1)
        
        # Critic output
        value = self.critic_head(x)
        
        return action_probs, value
    
    def get_action(self, state):
        """
        Sample action from policy.
        
        Args:
            state: State tensor
        
        Returns:
            action: Sampled action
            log_prob: Log probability of the action
            value: State value
        """
        action_probs, value = self.forward(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, value


if __name__ == "__main__":
    # Test networks
    print("Testing Actor network...")
    actor = ActorNetwork(state_dim=4, action_dim=2)
    state = torch.randn(1, 4)
    action_probs = actor(state)
    print(f"  Action probs shape: {action_probs.shape}")
    print(f"  Action probs: {action_probs}")
    
    print("\nTesting Critic network...")
    critic = CriticNetwork(state_dim=4)
    value = critic(state)
    print(f"  Value shape: {value.shape}")
    print(f"  Value: {value}")
    
    print("\nTesting Combined Actor-Critic network...")
    ac_net = ActorCriticNetwork(state_dim=4, action_dim=2)
    action_probs, value = ac_net(state)
    print(f"  Action probs: {action_probs}")
    print(f"  Value: {value}")
    
    print("\nNetworks tested successfully!")
