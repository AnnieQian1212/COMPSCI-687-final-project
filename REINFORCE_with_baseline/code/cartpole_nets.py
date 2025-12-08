import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# CartPole state bounds
MIN_POSITION = -2.4
MAX_POSITION = 2.4
MIN_VELOCITY = -3.0
MAX_VELOCITY = 3.0
MIN_ANGLE = -0.25  # ~12 degrees
MAX_ANGLE = 0.25
MIN_ANGULAR_VELOCITY = -3.0
MAX_ANGULAR_VELOCITY = 3.0


class CartNet(nn.Module):
    def __init__(self, neurons_per_layer,n_inputs, n_outputs):
        super().__init__()
        layers = []

        last = n_inputs 
        for h in neurons_per_layer:
            layers.append(nn.Linear(last, h))
            layers.append(nn.Tanh())
            last = h
        layers.append(nn.Linear(last, n_outputs))
        self.net = nn.Sequential(*layers)

        # Precompute constants needed to later perform [-1, 1] normalization when given an unnormalized state as input

        self.pos_mid = 0.5 * (MIN_POSITION + MAX_POSITION)  
        self.pos_half = 0.5 * (MAX_POSITION - MIN_POSITION) 
        self.vel_mid = 0.5 * (MIN_VELOCITY + MAX_VELOCITY) 
        self.vel_half = 0.5 * (MAX_VELOCITY - MIN_VELOCITY) 
        self.angle_mid = 0.5 * (MIN_ANGLE + MAX_ANGLE)  
        self.angle_half = 0.5 * (MAX_ANGLE - MIN_ANGLE)  
        self.angular_vel_mid = 0.5 * (MIN_ANGULAR_VELOCITY + MAX_ANGULAR_VELOCITY)  
        self.angular_vel_half = 0.5 * (MAX_ANGULAR_VELOCITY - MIN_ANGULAR_VELOCITY)
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
    def get_parameters(self) -> np.ndarray:
        with torch.no_grad():
            return torch.cat([p.view(-1) for p in self.parameters()]).cpu().numpy().copy()

    def set_parameters(self, theta: np.ndarray):
        theta = np.asarray(theta, dtype=np.float32)
        idx = 0
        with torch.no_grad():
            for p in self.parameters():
                numel = p.numel()
                block = torch.from_numpy(theta[idx:idx+numel]).view_as(p)
                p.copy_(block)
                idx += numel
        assert idx == theta.size, "Length of vector does not match number of parameters"

class CartPolicyNet(CartNet):


    def __init__(self, neurons_per_layer):
        
        
        n_inputs  = 4 # Number of inputs  is 4 because the state is s=[Cart position, Cart velocity, Pole angle, Pole angular velocity]
        n_outputs = 2 # Number of outputs is 2 because the network outputs one score value ("preference") for each action [left, right]
        super().__init__(neurons_per_layer, n_inputs, n_outputs)
        self.temperature = 1
    
    @torch.no_grad()
    def act(self, state_np: np.ndarray) -> float:
        x = torch.from_numpy(state_np.astype(np.float32)).unsqueeze(0)  # (1,2)
        logits = self.forward(x).squeeze(0)  # (3,)
        probs = F.softmax(logits / self.temperature, dim=-1).cpu().numpy()
        probs = np.clip(probs, 1e-8, 1.0)
        probs = probs / probs.sum()
        action_idx = int(np.random.choice(2, p=probs))
        a_map = (0, 1)
        return a_map[action_idx]
    
    
    def get_ln_pi_grad(self, state_np:  np.ndarray, action: float) -> np.ndarray:
        state_tensor = torch.from_numpy(state_np.astype(np.float32)).unsqueeze(0)  # (1,4)
        action_tensor = torch.tensor([action], dtype=torch.long)  # (1,)

        logits = self.forward(state_tensor)  # (1,2)
        log_probs = F.log_softmax(logits / self.temperature, dim=-1)
        log_pi_a = log_probs.gather(1, action_tensor.unsqueeze(1)).squeeze(1)  # (1,)

        self.zero_grad()
        log_pi_a.backward()

        grad_list = []
        for p in self.parameters():
            if p.grad is not None:
                grad_list.append(p.grad.view(-1))
            else:
                grad_list.append(torch.zeros(p.numel()))
        grad_vector = torch.cat(grad_list).cpu().numpy().copy()  # (num_params,)

        return grad_vector

class CartValueNet(CartNet):

    def __init__(self, neurons_per_layer):
        n_inputs  = 4 # Number of inputs  is 4 because the state is s=[Cart position, Cart velocity, Pole angle, Pole angular velocity]
        n_outputs = 1 # Number of outputs is 1 because the network outputs a single value estimate for the input state
        super().__init__(neurons_per_layer, n_inputs, n_outputs)

    @torch.no_grad()
    def predict(self, state_np: np.ndarray) -> float:
        x = torch.from_numpy(state_np.astype(np.float32)).unsqueeze(0)  # (1,4)
        value = self.forward(x).squeeze(0).item()  # scalar
        return value
    
    def get_gradients(self, state_np:  np.ndarray) -> np.ndarray:
        state_tensor = torch.from_numpy(state_np.astype(np.float32)).unsqueeze(0)  # (1,4)

        value = self.forward(state_tensor).squeeze(0)  # scalar

        self.zero_grad()
        value.backward()

        grad_list = []
        for p in self.parameters():
            grad_list.append(p.grad.view(-1))
        grad_vector = torch.cat(grad_list).cpu().numpy().copy()  # (num_params,)

        return grad_vector