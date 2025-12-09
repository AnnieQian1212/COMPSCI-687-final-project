import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Blackjack state bounds for normalization
PLAYER_MIN, PLAYER_MAX = 4.0, 21.0
DEALER_MIN, DEALER_MAX = 1.0, 10.0
ACE_MIN, ACE_MAX = 0.0, 1.0


class BlackNet(nn.Module):
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

    
    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        player = x[..., 0]
        dealer = x[..., 1]
        usable_ace = x[..., 2]
        player_n = 2 * (player - PLAYER_MIN) / (PLAYER_MAX - PLAYER_MIN) - 1
        dealer_n = 2 * (dealer - DEALER_MIN) / (DEALER_MAX - DEALER_MIN) - 1
        ace_n = 2 * (usable_ace - ACE_MIN) / (ACE_MAX - ACE_MIN) - 1
        return torch.stack([player_n, dealer_n, ace_n], dim=-1)
    

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

class BlackPolicyNet(BlackNet):


    def __init__(self, neurons_per_layer):
        
        
        n_inputs  = 3 
        n_outputs = 2 
        super().__init__(neurons_per_layer, n_inputs, n_outputs)
        self.temperature = 1
    
    @torch.no_grad()
    def act(self, state_np: np.ndarray) -> float:
        state_np = np.array(state_np, dtype=np.float32)  

        x = torch.from_numpy(state_np.astype(np.float32)).unsqueeze(0)  # (1,3)
        logits = self.forward(x).squeeze(0)  # (3,)
        probs = F.softmax(logits / self.temperature, dim=-1).cpu().numpy()
        probs = np.clip(probs, 1e-8, 1.0)
        probs = probs / probs.sum()
        action_idx = int(np.random.choice(2, p=probs))
        a_map = (0, 1)
        return a_map[action_idx]
    
    
    def get_ln_pi_grad(self, state_np:  np.ndarray, action: float) -> np.ndarray:
        state_np = np.array(state_np, dtype=np.float32)  

        state_tensor = torch.from_numpy(state_np.astype(np.float32)).unsqueeze(0)  
        action_tensor = torch.tensor([action], dtype=torch.long) 

        logits = self.forward(state_tensor)  
        log_probs = F.log_softmax(logits / self.temperature, dim=-1)
        log_pi_a = log_probs.gather(1, action_tensor.unsqueeze(1)).squeeze(1)  

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

class BlackValueNet(BlackNet):

    def __init__(self, neurons_per_layer):
        n_inputs  = 3
        n_outputs = 1 
        super().__init__(neurons_per_layer, n_inputs, n_outputs)

    @torch.no_grad()
    def predict(self, state_np: np.ndarray) -> float:
        state_np = np.array(state_np, dtype=np.float32)  

        x = torch.from_numpy(state_np.astype(np.float32)).unsqueeze(0)  
        value = self.forward(x).squeeze(0).item()  
        return value
    
    def get_gradients(self, state_np:  np.ndarray) -> np.ndarray:
        state_np = np.array(state_np, dtype=np.float32)  
        state_tensor = torch.from_numpy(state_np.astype(np.float32)).unsqueeze(0) 

        value = self.forward(state_tensor).squeeze(0)  

        self.zero_grad()
        value.backward()

        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

        grad_list = []
        for p in self.parameters():
            grad_list.append(p.grad.view(-1))
        grad_vector = torch.cat(grad_list).cpu().numpy().copy() 

        return grad_vector