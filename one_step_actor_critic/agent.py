"""
One-Step Actor-Critic Agent Implementation.

This implements the One-Step Actor-Critic algorithm as described in 
Sutton & Barto's "Reinforcement Learning: An Introduction" (Chapter 13).

Algorithm:
    1. Initialize actor π(a|s; θ) and critic V(s; w)
    2. For each episode:
        a. Initialize state S
        b. For each step:
            i.   Sample action A ~ π(·|S; θ)
            ii.  Take action A, observe R, S'
            iii. Compute TD error: δ = R + γV(S'; w) - V(S; w)
            iv.  Update critic: w ← w + α_w * δ * ∇V(S; w)
            v.   Update actor: θ ← θ + α_θ * δ * ∇log π(A|S; θ)
            vi.  S ← S'
"""

import numpy as np
import torch
import torch.optim as optim
from networks import ActorNetwork, CriticNetwork, ActorCriticNetwork


class OneStepActorCritic:
    """
    One-Step Actor-Critic agent with separate actor and critic networks.
    """
    
    def __init__(
        self,
        state_dim,
        action_dim,
        actor_lr=1e-3,
        critic_lr=1e-3,
        gamma=0.99,
        hidden_dim=128,
        device='cpu'
    ):
        """
        Initialize One-Step Actor-Critic agent.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            actor_lr: Learning rate for actor
            critic_lr: Learning rate for critic
            gamma: Discount factor
            hidden_dim: Hidden layer dimension
            device: Device to use ('cpu' or 'cuda')
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.device = device
        
        # Initialize networks
        self.actor = ActorNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.critic = CriticNetwork(state_dim, hidden_dim).to(device)
        
        # Initialize optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        # For tracking I (discount factor for policy gradient)
        self.I = 1.0
        
    def select_action(self, state):
        """
        Select action using current policy.
        
        Args:
            state: Current state (numpy array)
        
        Returns:
            action: Selected action (int)
            log_prob: Log probability of the action (tensor)
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action, log_prob = self.actor.get_action(state_tensor)
        return action.item(), log_prob
    
    def update(self, state, action, reward, next_state, done, log_prob):
        """
        Perform one-step update for actor and critic.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
            log_prob: Log probability of the action
        
        Returns:
            actor_loss: Actor loss value
            critic_loss: Critic loss value
            td_error: TD error
        """
        # Convert to tensors
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
        reward_tensor = torch.FloatTensor([reward]).to(self.device)
        
        # Compute current value and next value
        current_value = self.critic(state_tensor)
        next_value = self.critic(next_state_tensor)
        
        # Compute TD target and TD error
        # δ = R + γV(S') - V(S)
        if done:
            td_target = reward_tensor
        else:
            td_target = reward_tensor + self.gamma * next_value.detach()
        
        td_error = td_target - current_value
        
        # Update critic: minimize TD error
        critic_loss = td_error.pow(2).mean()
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Update actor: policy gradient with TD error as advantage
        # θ ← θ + α * I * δ * ∇log π(A|S; θ)
        actor_loss = -self.I * log_prob * td_error.detach()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Update I for next step
        self.I *= self.gamma
        
        return actor_loss.item(), critic_loss.item(), td_error.item()
    
    def reset_episode(self):
        """Reset episode-specific variables."""
        self.I = 1.0
    
    def save(self, path):
        """Save model parameters."""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
        }, path)
    
    def load(self, path):
        """Load model parameters."""
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])


class OneStepActorCriticShared:
    """
    One-Step Actor-Critic agent with shared network architecture.
    """
    
    def __init__(
        self,
        state_dim,
        action_dim,
        lr=1e-3,
        gamma=0.99,
        hidden_dim=128,
        entropy_coef=0.01,
        value_coef=0.5,
        device='cpu'
    ):
        """
        Initialize One-Step Actor-Critic agent with shared network.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            lr: Learning rate
            gamma: Discount factor
            hidden_dim: Hidden layer dimension
            entropy_coef: Entropy bonus coefficient
            value_coef: Value loss coefficient
            device: Device to use
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.device = device
        
        # Initialize shared network
        self.network = ActorCriticNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        
        self.I = 1.0
        
    def select_action(self, state):
        """
        Select action using current policy.
        
        Args:
            state: Current state (numpy array)
        
        Returns:
            action: Selected action (int)
            log_prob: Log probability of the action
            value: State value
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action, log_prob, value = self.network.get_action(state_tensor)
        return action.item(), log_prob, value
    
    def update(self, state, action, reward, next_state, done, log_prob, value):
        """
        Perform one-step update.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
            log_prob: Log probability of the action
            value: State value
        
        Returns:
            total_loss: Total loss value
            td_error: TD error
        """
        # Get next state value
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
        _, next_value = self.network(next_state_tensor)
        
        # Compute TD target and error
        if done:
            td_target = torch.FloatTensor([reward]).to(self.device)
        else:
            td_target = reward + self.gamma * next_value.detach()
        
        td_error = td_target - value
        
        # Compute losses
        critic_loss = td_error.pow(2).mean()
        actor_loss = -self.I * log_prob * td_error.detach()
        
        # Compute entropy for exploration bonus
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action_probs, _ = self.network(state_tensor)
        entropy = -(action_probs * torch.log(action_probs + 1e-10)).sum()
        
        # Total loss
        total_loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy
        
        # Update network
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        # Update I
        self.I *= self.gamma
        
        return total_loss.item(), td_error.item()
    
    def reset_episode(self):
        """Reset episode-specific variables."""
        self.I = 1.0
    
    def save(self, path):
        """Save model parameters."""
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
    
    def load(self, path):
        """Load model parameters."""
        checkpoint = torch.load(path)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


if __name__ == "__main__":
    # Test agent
    print("Testing OneStepActorCritic agent...")
    agent = OneStepActorCritic(state_dim=4, action_dim=2)
    
    # Simulate one step
    state = np.random.randn(4).astype(np.float32)
    action, log_prob = agent.select_action(state)
    next_state = np.random.randn(4).astype(np.float32)
    reward = 1.0
    done = False
    
    actor_loss, critic_loss, td_error = agent.update(
        state, action, reward, next_state, done, log_prob
    )
    
    print(f"  Action: {action}")
    print(f"  Actor loss: {actor_loss:.4f}")
    print(f"  Critic loss: {critic_loss:.4f}")
    print(f"  TD error: {td_error:.4f}")
    
    print("\nAgent tested successfully!")
