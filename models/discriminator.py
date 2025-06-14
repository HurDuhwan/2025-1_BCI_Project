import torch
import torch.nn as nn

class DomainDiscriminator(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.hidden_dim = int(in_dim/2)
        self.net = nn.Sequential(
            nn.Linear(in_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
            # nn.ReLU(),
        )
    def forward(self, x):
        return self.net(x)

