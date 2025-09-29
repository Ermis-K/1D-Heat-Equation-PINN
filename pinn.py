import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

class PINN(nn.Module):
    def __init__(self, n_input, n_output, n_hidden, n_layers, X_for_norm, activation='tanh'):
        super().__init__()
        acts = {
            'tanh': nn.Tanh,
            'gelu': nn.GELU,
            'swish': nn.SiLU,
            'sigmoid': nn.Sigmoid,
        }
        act = acts[activation.lower()]
        layers = []

        self.register_buffer('x_mu',  X_for_norm.mean(0, keepdim=True))
        self.register_buffer('x_std', X_for_norm.std (0, keepdim=True))

        def make_layer(in_dim, out_dim):
            L = nn.Linear(in_dim, out_dim)
            L = weight_norm(L, dim=0)
            with torch.no_grad():
                nn.init.normal_(L.weight)
            return L

        layers += [make_layer(n_input, n_hidden), act()]
        for _ in range(n_layers - 1):
            layers += [make_layer(n_hidden, n_hidden), act()]
        layers += [make_layer(n_hidden, n_output)]
        self.net = nn.Sequential(*layers)

    def forward(self, X: torch.Tensor):
        Z = (X - self.x_mu) / (self.x_std + 1e-8)
        return self.net(Z)
