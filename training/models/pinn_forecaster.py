"""
Physics-Informed Neural Network (PINN) for Wearable Physiological Forecasting.

Architecture:
1. Multi-scale 1D Convolutional feature extractor (denoises optical PPG and IMU motion).
2. Recurrent GRU temporal aggregator (captures multi-timescale cardiovascular dynamics).
3. Dual-Head Output:
   - Trajectory Head: Predicted future heart rate mean \hat{HR}(t + H) [BPM]
   - Uncertainty Head: Heteroscedastic predictive variance \sigma^2(t + H) [BPM^2]
4. Biophysical Parameter Layer:
   - Learnable recovery time constant tau > 0 [seconds]
   - Learnable workload response gain beta > 0 [BPM/workload-sec]
   - Supports both global and participant-personalized parameters (Exp F)
   - Supports asymmetric exercise/recovery parameters.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class PINNWearableForecaster(nn.Module):
    def __init__(
        self,
        input_dim: int = 4,
        hidden_dim: int = 48,
        num_layers: int = 2,
        output_dim: int = 1,
        dropout: float = 0.1,
        num_subjects: Optional[int] = None,
        asymmetric_tau: bool = False
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_subjects = num_subjects
        self.asymmetric_tau = asymmetric_tau

        # 1D Convolutional Denoising & Feature Extraction
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.SiLU(),
            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.SiLU()
        )

        # Recurrent Temporal Aggregation
        self.gru = nn.GRU(
            input_size=32,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # Head 1: Future HR Mean Trajectory
        self.fc_mean = nn.Sequential(
            nn.Linear(hidden_dim, 24),
            nn.SiLU(),
            nn.Linear(24, output_dim)
        )

        # Head 2: Predictive Uncertainty Variance (Heteroscedastic Aleatoric Variance)
        self.fc_variance = nn.Sequential(
            nn.Linear(hidden_dim, 24),
            nn.SiLU(),
            nn.Linear(24, output_dim)
        )

        # Global Biophysical Parameters: tau (recovery seconds) and beta (workload gain)
        # Clinical human physiology: tau ~ 30-60s, beta ~ 10-25
        self.raw_tau = nn.Parameter(torch.tensor([3.8])) # softplus(3.8) + 15 ~ 59s
        self.raw_beta = nn.Parameter(torch.tensor([2.7])) # softplus(2.7) + 1 ~ 16.0
        
        # Asymmetric parameters (recovery tau vs exercise tau)
        if self.asymmetric_tau:
            self.raw_tau_rec = nn.Parameter(torch.tensor([3.5])) # recovery tau

        # Personalized Subject Parameters (Exp F)
        if num_subjects is not None and num_subjects > 0:
            # Residual offsets around global biophysical parameters
            self.subj_tau_offset = nn.Embedding(num_subjects, 1)
            self.subj_beta_offset = nn.Embedding(num_subjects, 1)
            nn.init.zeros_(self.subj_tau_offset.weight)
            nn.init.zeros_(self.subj_beta_offset.weight)
        else:
            self.subj_tau_offset = None
            self.subj_beta_offset = None

    def get_tau(self, subject_idx: Optional[torch.Tensor] = None, is_recovery: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Returns constrained tau > 15.0 seconds."""
        base_tau = F.softplus(self.raw_tau) + 15.0
        if self.asymmetric_tau and is_recovery is not None:
            tau_rec = F.softplus(self.raw_tau_rec) + 15.0
            base_tau = torch.where(is_recovery > 0.5, tau_rec, base_tau)
            
        if self.subj_tau_offset is not None and subject_idx is not None:
            offset = self.subj_tau_offset(subject_idx.squeeze(-1))
            return torch.clamp(base_tau + offset, min=15.0, max=120.0)
        return base_tau

    def get_beta(self, subject_idx: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Returns constrained beta > 1.0 BPM/workload-sec."""
        base_beta = F.softplus(self.raw_beta) + 1.0
        if self.subj_beta_offset is not None and subject_idx is not None:
            offset = self.subj_beta_offset(subject_idx.squeeze(-1))
            return torch.clamp(base_beta + offset, min=1.0, max=50.0)
        return base_beta

    @property
    def tau(self) -> torch.Tensor:
        return self.get_tau()

    @property
    def beta(self) -> torch.Tensor:
        return self.get_beta()

    def forward(
        self,
        x: torch.Tensor,
        subject_idx: Optional[torch.Tensor] = None,
        is_recovery: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        Args:
            x: Input tensor of shape [batch, window_size, features]
            subject_idx: Optional tensor of shape [batch, 1] for personalized parameters
            is_recovery: Optional tensor of shape [batch, 1] for asymmetric parameters
            
        Returns:
            hr_pred: [batch, 1] predicted future HR mean
            hr_var:  [batch, 1] predicted uncertainty variance (BPM^2)
            tau:     [batch, 1] or scalar biophysical recovery time constant
            beta:    [batch, 1] or scalar biophysical workload gain
        """
        # x: [B, W, C] -> transpose for Conv1d: [B, C, W]
        x_trans = x.transpose(1, 2)
        conv_feats = self.conv(x_trans)
        conv_feats = conv_feats.transpose(1, 2) # [B, W, 32]

        gru_out, _ = self.gru(conv_feats)
        last_hidden = gru_out[:, -1, :] # [B, hidden_dim]

        hr_pred = self.fc_mean(last_hidden)
        # Softplus ensures positive variance with epsilon stability floor
        hr_var = F.softplus(self.fc_variance(last_hidden)) + 1e-4

        tau = self.get_tau(subject_idx, is_recovery)
        beta = self.get_beta(subject_idx)

        return hr_pred, hr_var, tau, beta

if __name__ == "__main__":
    dummy_x = torch.randn(8, 30, 4)
    model = PINNWearableForecaster(input_dim=4, num_subjects=10)
    subjs = torch.randint(0, 10, (8, 1))
    pred, var, t_val, b_val = model(dummy_x, subjs)
    print("PINN Pred shape:", pred.shape)
    print("PINN Var shape:", var.shape)
    print("PINN Tau:", t_val.mean().item(), "Beta:", b_val.mean().item())
    print("PINN Params:", sum(p.numel() for p in model.parameters()))

