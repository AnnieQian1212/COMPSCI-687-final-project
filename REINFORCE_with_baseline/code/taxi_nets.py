# taxi_nets.py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class TaxiNet(nn.Module):
    def __init__(self, neurons_per_layer, n_inputs, n_outputs):
        super().__init__()
        layers = []

        last = n_inputs 
        for h in neurons_per_layer:
            layers.append(nn.Linear(last, h))
            layers.append(nn.ReLU())  # ReLU often better than Tanh for discrete envs
            last = h
        layers.append(nn.Linear(last, n_outputs))
        self.net = nn.Sequential(*layers)

    def _decode_state(self, state_int: int) -> torch.Tensor:
        
        state = state_int
        destination = state % 4
        state = state // 4
        passenger_loc = state % 5
        state = state // 5
        taxi_col = state % 5
        taxi_row = state // 5
        
        
        features = torch.tensor([
            taxi_row / 4.0,      # 0-4 → 0-1
            taxi_col / 4.0,      # 0-4 → 0-1
            passenger_loc / 4.0, # 0-4 → 0-1
            destination / 3.0    # 0-3 → 0-1
        ], dtype=torch.float32)
        
        return features
    
    def forward(self, state_int: int) -> torch.Tensor:
        x = self._decode_state(state_int)
        return self.net(x)
    
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


class TaxiPolicyNet(TaxiNet):
    def __init__(self, neurons_per_layer):
        n_inputs = 4   # [taxi_row, taxi_col, passenger_loc, destination]
        n_outputs = 6  # 6 actions
        super().__init__(neurons_per_layer, n_inputs, n_outputs)
        self.temperature = 1.0
    
    @torch.no_grad()
    def act(self, state_int: int) -> int:
        logits = self.forward(state_int)
        probs = F.softmax(logits / self.temperature, dim=-1).cpu().numpy()
        probs = np.clip(probs, 1e-8, 1.0)
        probs = probs / probs.sum()
        action = int(np.random.choice(6, p=probs))
        return action
    
    def get_ln_pi_grad(self, state_int: int, action: int) -> np.ndarray:
        logits = self.forward(state_int)
        log_probs = F.log_softmax(logits / self.temperature, dim=-1)
        log_pi_a = log_probs[action]

        self.zero_grad()
        log_pi_a.backward()
        
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

        grad_list = []
        for p in self.parameters():
            if p.grad is not None:
                grad_list.append(p.grad.view(-1))
            else:
                grad_list.append(torch.zeros(p.numel()))
        grad_vector = torch.cat(grad_list).cpu().numpy().copy()
        return grad_vector


class TaxiValueNet(TaxiNet):
    def __init__(self, neurons_per_layer):
        n_inputs = 4   # [taxi_row, taxi_col, passenger_loc, destination]
        n_outputs = 1
        super().__init__(neurons_per_layer, n_inputs, n_outputs)

    @torch.no_grad()
    def predict(self, state_int: int) -> float:
        value = self.forward(state_int).item()
        return value
    
    def get_gradients(self, state_int: int) -> np.ndarray:
        value = self.forward(state_int)

        self.zero_grad()
        value.backward()
        
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

        grad_list = []
        for p in self.parameters():
            if p.grad is not None:
                grad_list.append(p.grad.view(-1))
            else:
                grad_list.append(torch.zeros(p.numel()))
        grad_vector = torch.cat(grad_list).cpu().numpy().copy()
        return grad_vector