"""
Standard Unconstrained Deep Learning Baseline (Pure Data-Driven GRU)
"""

import torch
import torch.nn as nn

class StandardWearableGRU(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64, num_layers=2, output_dim=1):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )

    def forward(self, x):
        # x shape: [batch, window_size, input_dim]
        out, _ = self.gru(x)
        # Take the last time step embedding
        last_step = out[:, -1, :]
        pred = self.fc(last_step)
        return pred

if __name__ == "__main__":
    m = StandardWearableGRU()
    dummy_input = torch.randn(8, 30, 3)
    out = m(dummy_input)
    print("Baseline GRU Output shape:", out.shape)
