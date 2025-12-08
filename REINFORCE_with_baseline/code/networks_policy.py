import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CartPolicyNet(nn.Module):
    # CartPole state bounds
    MIN_POSITION = -2.4
    MAX_POSITION = 2.4
    MIN_VELOCITY = -4.0
    MAX_VELOCITY = 4.0
    MIN_ANGLE = -0.2095  # ~12 degrees
    MAX_ANGLE = 0.2095
    MIN_ANGULAR_VELOCITY = -2.0
    MAX_ANGULAR_VELOCITY = 2.0

    def __init__(self, neurons_per_layer):
        super().__init__()
        layers = []
        n_inputs  = 4 # Number of inputs  is 4 because the state is s=[Cart position, Cart velocity, Pole angle, Pole angular velocity]
        n_outputs = 2 # Number of outputs is 2 because the network outputs one score value ("preference") for each action [left, right]
        self.temperature = 0.1

        last = n_inputs 
        for h in neurons_per_layer:
            layers.append(nn.Linear(last, h))
            layers.append(nn.Tanh())
            last = h
        layers.append(nn.Linear(last, n_outputs))
        self.net = nn.Sequential(*layers)

        # Precompute constants needed to later perform [-1, 1] normalization when given an unnormalized state as input

        self.pos_mid = 0.5 * (self.MIN_POSITION + self.MAX_POSITION)  
        self.pos_half = 0.5 * (self.MAX_POSITION - self.MIN_POSITION) 
        self.vel_mid = 0.5 * (self.MIN_VELOCITY + self.MAX_VELOCITY) 
        self.vel_half = 0.5 * (self.MAX_VELOCITY - self.MIN_VELOCITY) 
        self.angle_mid = 0.5 * (self.MIN_ANGLE + self.MAX_ANGLE)  
        self.angle_half = 0.5 * (self.MAX_ANGLE - self.MIN_ANGLE)  
        self.angular_vel_mid = 0.5 * (self.MIN_ANGULAR_VELOCITY + self.MAX_ANGULAR_VELOCITY)  
        self.angular_vel_half = 0.5 * (self.MAX_ANGULAR_VELOCITY - self.MIN_ANGULAR_VELOCITY)  
    
    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        pos = x[..., 0]
        vel = x[..., 1]
        angle = x[..., 2]
        angular_vel = x[..., 3]
        pos_n = (pos - self.pos_mid) / self.pos_half
        vel_n = (vel - self.vel_mid) / self.vel_half
        angle_n = (angle - self.angle_mid) / self.angle_half
        angular_vel_n = (angular_vel - self.angular_vel_mid) / self.angular_vel_half
        return torch.stack([pos_n, vel_n, angle_n, angular_vel_n], dim=-1)
    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_n = self._normalize(x)
        return self.net(x_n)
    
    @torch.no_grad()
    def act(self, state_np: np.ndarray) -> float:
        x = torch.from_numpy(state_np.astype(np.float32)).unsqueeze(0)  # (1,2)
        logits = self.forward(x).squeeze(0)  # (3,)
        probs = F.softmax(logits / self.temperature, dim=-1).cpu().numpy()
        probs = np.clip(probs, 1e-8, 1.0)
        probs = probs / probs.sum()
        action_idx = int(np.random.choice(2, p=probs))
        a_map = (-1.0, 1.0)
        return a_map[action_idx]
    
    def get_policy_parameters(self) -> np.ndarray:
        with torch.no_grad():
            return torch.cat([p.view(-1) for p in self.parameters()]).cpu().numpy().copy()

    def set_policy_parameters(self, theta: np.ndarray):
        theta = np.asarray(theta, dtype=np.float32)
        idx = 0
        with torch.no_grad():
            for p in self.parameters():
                numel = p.numel()
                block = torch.from_numpy(theta[idx:idx+numel]).view_as(p)
                p.copy_(block)
                idx += numel
        assert idx == theta.size, "Length of vector 'theta' does not match number of policy parameters"

    def get_ln_pi_grad(self, state_np:  np.ndarray, action: float) -> np.ndarray:
        state_tensor = torch.from_numpy(state_np.astype(np.float32)).unsqueeze(0)  # (1,4)
        action_idx = 0 if action < 0 else 1
        action_tensor = torch.tensor([action_idx], dtype=torch.long)  # (1,)

        logits = self.forward(state_tensor)  # (1,2)
        log_probs = F.log_softmax(logits / self.temperature, dim=-1)
        log_pi_a = log_probs.gather(1, action_tensor.unsqueeze(1)).squeeze(1)  # (1,)

        self.zero_grad()
        log_pi_a.backward()

        grad_list = []
        for p in self.parameters():
            grad_list.append(p.grad.view(-1))
        grad_vector = torch.cat(grad_list).cpu().numpy().copy()  # (num_params,)

        return grad_vector

