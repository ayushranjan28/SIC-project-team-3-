"""
Baseline Neural Network Architectures for Wearable Physiological Forecasting.

Implements pure data-driven benchmarks:
1. Standard LSTM Forecaster (Long Short-Term Memory)
2. Standard GRU Forecaster (Gated Recurrent Unit)
3. Standard 1D CNN Forecaster (Temporal Convolutional Network)

All baselines take input sequence X: [batch, window_size, features]
and output heart rate forecast: [batch, 1] without physics constraints.
"""

import torch
import torch.nn as nn

class StandardLSTMForecaster(nn.Module):
    """Multi-layer LSTM Forecaster."""
    def __init__(self, input_dim: int = 4, hidden_dim: int = 48, num_layers: int = 2, output_dim: int = 1, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 24),
            nn.SiLU(),
            nn.Linear(24, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, W, C]
        out, _ = self.lstm(x)
        last = out[:, -1, :] # [B, hidden_dim]
        pred = self.head(last)
        return pred

class StandardGRUForecaster(nn.Module):
    """Multi-layer GRU Forecaster."""
    def __init__(self, input_dim: int = 4, hidden_dim: int = 48, num_layers: int = 2, output_dim: int = 1, dropout: float = 0.1):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 24),
            nn.SiLU(),
            nn.Linear(24, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        last = out[:, -1, :]
        pred = self.head(last)
        return pred

class StandardCNN1DForecaster(nn.Module):
    """1D Temporal Convolutional Forecaster."""
    def __init__(self, input_dim: int = 4, hidden_dim: int = 48, output_dim: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(input_dim, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.SiLU(),
            nn.Conv1d(32, hidden_dim, kernel_size=3, padding=1, dilation=2),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1, dilation=4),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 24),
            nn.SiLU(),
            nn.Linear(24, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, W, C] -> transpose to [B, C, W]
        x_trans = x.transpose(1, 2)
        feats = self.net(x_trans).squeeze(-1) # [B, hidden_dim]
        pred = self.head(feats)
        return pred

if __name__ == "__main__":
    dummy_x = torch.randn(8, 30, 4)
    lstm = StandardLSTMForecaster()
    gru = StandardGRUForecaster()
    cnn = StandardCNN1DForecaster()
    
    print("LSTM params:", sum(p.numel() for p in lstm.parameters()), "out:", lstm(dummy_x).shape)
    print("GRU params:", sum(p.numel() for p in gru.parameters()), "out:", gru(dummy_x).shape)
    print("CNN params:", sum(p.numel() for p in cnn.parameters()), "out:", cnn(dummy_x).shape)

