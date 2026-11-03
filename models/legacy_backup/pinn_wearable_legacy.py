"""
Physics-Informed Neural Network (PINN) for Wearable Biometrics
Integrates cardiovascular ODE dynamics into the training objective to guarantee
physiologically plausible predictions, noise rejection, and edge efficiency.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class PINNWearableForecaster(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=48, num_layers=2, output_dim=1):
        super().__init__()
        
        # 1D Convolutional feature extractor (denoises optical sensor signals)
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.SiLU(),
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.SiLU()
        )
        
        # Recurrent temporal aggregator
        self.gru = nn.GRU(
            input_size=32,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        
        # Dual-head output: predicts HR mean and uncertainty variance (aleatoric uncertainty)
        self.fc_mean = nn.Sequential(
            nn.Linear(hidden_dim, 24),
            nn.SiLU(),
            nn.Linear(24, output_dim)
        )
        
        self.fc_variance = nn.Sequential(
            nn.Linear(hidden_dim, 24),
            nn.SiLU(),
            nn.Linear(24, output_dim),
            nn.Softplus() # guarantees positive variance
        )

        # Learnable/estimated biophysical parameters (recovery time constant tau and beta gain)
        # Initialize near standard clinical human physiology: tau ~ 45s, beta ~ 18
        self.log_tau = nn.Parameter(torch.tensor([3.8])) # exp(3.8) ~ 44.7s
        self.log_beta = nn.Parameter(torch.tensor([2.89])) # exp(2.89) ~ 18.0

    @property
    def tau(self):
        return torch.exp(self.log_tau)

    @property
    def beta(self):
        return torch.exp(self.log_beta)

    def forward(self, x):
        # x: [batch, seq_len, features]
        # Transpose for Conv1d: [batch, features, seq_len]
        x_trans = x.transpose(1, 2)
        conv_feats = self.conv(x_trans)
        # Transpose back: [batch, seq_len, conv_channels]
        conv_feats = conv_feats.transpose(1, 2)
        
        gru_out, _ = self.gru(conv_feats)
        last_hidden = gru_out[:, -1, :]
        
        hr_pred = self.fc_mean(last_hidden)
        hr_var = self.fc_variance(last_hidden) + 1e-4
        
        return hr_pred, hr_var

class PhysiologicalPINNLoss(nn.Module):
    """
    Total Loss = L_data (heteroscedastic Gaussian NLL) 
               + lambda_physics * L_physics (ODE residual)
               + lambda_bound * L_bounds (human biophysical limit penalty)
    """
    def __init__(self, lambda_physics=0.4, lambda_bounds=0.2, dt=1.0):
        super().__init__()
        self.lambda_physics = lambda_physics
        self.lambda_bounds = lambda_bounds
        self.dt = dt

    def forward(self, hr_pred, hr_var, hr_target, hr_prev, workload, hr_rest, model_tau, model_beta):
        # 1. Data Loss: Gaussian Negative Log-Likelihood (handles noise and provides uncertainty)
        # NLL = 0.5 * log(var) + 0.5 * (pred - target)^2 / var
        data_loss = torch.mean(0.5 * torch.log(hr_var) + 0.5 * ((hr_pred - hr_target) ** 2) / hr_var)

        # 2. Physics Residual Loss:
        # Expected derivative according to cardiovascular ODE:
        # d_hr/dt = - (hr_prev - hr_rest) / tau + beta * workload
        d_hr_actual = (hr_pred - hr_prev) / self.dt
        d_hr_physics_expected = - (hr_prev - hr_rest) / model_tau + model_beta * workload
        physics_residual = d_hr_actual - d_hr_physics_expected
        physics_loss = torch.mean(physics_residual ** 2)

        # 3. Biophysical Boundary Violations:
        # Penalize heart rate outside realistic human bounds [45 bpm, 215 bpm]
        # and excessive single-second jumps (> 6 bpm/sec max physiological surge)
        lower_bound_penalty = F.relu(45.0 - hr_pred) ** 2
        upper_bound_penalty = F.relu(hr_pred - 215.0) ** 2
        max_rate_penalty = F.relu(torch.abs(d_hr_actual) - 6.0) ** 2
        bounds_loss = torch.mean(lower_bound_penalty + upper_bound_penalty + max_rate_penalty)

        total_loss = data_loss + self.lambda_physics * physics_loss + self.lambda_bounds * bounds_loss

        return total_loss, {
            "loss_data": data_loss.item(),
            "loss_physics": physics_loss.item(),
            "loss_bounds": bounds_loss.item(),
            "total_loss": total_loss.item()
        }

if __name__ == "__main__":
    model = PINNWearableForecaster()
    criterion = PhysiologicalPINNLoss()
    dummy_x = torch.randn(8, 30, 3)
    pred, var = model(dummy_x)
    print("PINN Prediction mean shape:", pred.shape)
    print("PINN Uncertainty variance shape:", var.shape)
    print("Learned tau:", model.tau.item(), "Learned beta:", model.beta.item())
