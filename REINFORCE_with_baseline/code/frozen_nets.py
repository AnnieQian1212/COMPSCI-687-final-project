import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



class FrozenNet(nn.Module):
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

    
    def _normalize(self, i: int) -> torch.Tensor:
        x = torch.zeros(16, dtype=torch.float32)
        x[int(i)] = 1.0
        return x
    

    def forward(self, x: int) -> torch.Tensor:
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

class FrozenPolicyNet(FrozenNet):


    def __init__(self, neurons_per_layer):
        
        
        n_inputs  = 16
        n_outputs = 4
        super().__init__(neurons_per_layer, n_inputs, n_outputs)
        self.temperature = 1
    
    @torch.no_grad()
    def act(self, state_i: int) -> int:
        state_i = np.array(state_i, dtype=np.float32)  

        x = torch.from_numpy(state_i.astype(np.float32)).unsqueeze(0) 
        logits = self.forward(x).squeeze(0)  # (3,)
        probs = F.softmax(logits / self.temperature, dim=-1).cpu().numpy()
        probs = np.clip(probs, 1e-8, 1.0)
        probs = probs / probs.sum()
        action_idx = int(np.random.choice(4, p=probs))
        a_map = (0,1,2,3)
        return a_map[action_idx]
    
    
    def get_ln_pi_grad(self, state_i:  int, action: float) -> np.ndarray:

        logits = self.forward(state_i)  
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
        grad_vector = torch.cat(grad_list).cpu().numpy().copy()  # (num_params,)

        return grad_vector

class FrozenValueNet(FrozenNet):

    def __init__(self, neurons_per_layer):
        n_inputs  = 16
        n_outputs = 1 # Number of outputs is 1 because the network outputs a single value estimate for the input state
        super().__init__(neurons_per_layer, n_inputs, n_outputs)

    @torch.no_grad()
    def predict(self, state_i: int) -> float:
        state_i = int(state_i)  

       
        value = self.forward(state_i).item()  # scalar
        return value
    
    def get_gradients(self, state_i: int) -> np.ndarray:
        value = self.forward(state_i)  # scalar

        self.zero_grad()
        value.backward()

        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

        grad_list = []
        for p in self.parameters():
            grad_list.append(p.grad.view(-1))
        grad_vector = torch.cat(grad_list).cpu().numpy().copy()  # (num_params,)

        return grad_vector