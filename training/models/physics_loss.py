"""
Cardiovascular Physics-Informed Loss Function.

Implements the multi-objective loss:
    L_total = L_data + lambda_physics * L_physics + lambda_bounds * L_bounds

Where:
    1. L_data: Heteroscedastic Gaussian Negative Log-Likelihood (or MSE for standard baselines)
       Optionally weighted by sensor quality index (SQI) for noise-artifact rejection (Exp G).
    2. L_physics: Mean squared residual of the cardiovascular governing differential equation:
           dHR/dt = -(1/tau) * (HR(t) - HR_rest) + beta * Workload(t)
       With exact time-scaled finite differences:
           dHR/dt \approx (HR_pred - HR_prev) / dt  [units: BPM/s]
    3. L_bounds: Soft penalties for violating biophysical limits:
           HR in [45.0, 215.0] BPM
           |dHR/dt| <= 6.0 BPM/s
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

class PhysiologicalPINNLoss(nn.Module):
    def __init__(
        self,
        lambda_physics: float = 0.4,
        lambda_bounds: float = 0.2,
        use_nll: bool = True,
        quality_weighted: bool = False,
        hr_min: float = 45.0,
        hr_max: float = 215.0,
        max_rate_bpm_per_sec: float = 6.0
    ):
        super().__init__()
        self.lambda_physics = lambda_physics
        self.lambda_bounds = lambda_bounds
        self.use_nll = use_nll
        self.quality_weighted = quality_weighted
        self.hr_min = hr_min
        self.hr_max = hr_max
        self.max_rate_bpm_per_sec = max_rate_bpm_per_sec

    def forward(
        self,
        hr_pred: torch.Tensor,
        hr_var: torch.Tensor,
        hr_target: torch.Tensor,
        hr_prev: torch.Tensor,
        workload: torch.Tensor,
        hr_rest: torch.Tensor,
        tau: torch.Tensor,
        beta: torch.Tensor,
        dt: torch.Tensor,
        sqi: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Computes the complete physics-informed multi-objective loss.
        """
        # Ensure dt has minimum value 0.1s to avoid divide by zero
        dt_clamped = torch.clamp(dt, min=0.1)

        # 1. Data Loss
        if self.use_nll:
            # Gaussian NLL: 0.5 * log(var) + 0.5 * (pred - target)^2 / var
            raw_nll = 0.5 * torch.log(hr_var) + 0.5 * ((hr_pred - hr_target) ** 2) / hr_var
            if self.quality_weighted and sqi is not None:
                # Modulate by SQI: clean signal = 1.0 weight; low SQI = lower data weight
                sqi_weight = torch.clamp(sqi, min=0.1, max=1.0)
                data_loss = torch.mean(sqi_weight * raw_nll)
            else:
                data_loss = torch.mean(raw_nll)
        else:
            diff_sq = (hr_pred - hr_target) ** 2
            if self.quality_weighted and sqi is not None:
                sqi_weight = torch.clamp(sqi, min=0.1, max=1.0)
                data_loss = torch.mean(sqi_weight * diff_sq)
            else:
                data_loss = torch.mean(diff_sq)

        # 2. Physics Residual Loss
        if self.lambda_physics > 0.0:
            # Time-scaled finite difference: (HR_pred - HR_prev) / dt
            d_hr_actual = (hr_pred - hr_prev) / dt_clamped
            # Expected rate from cardiovascular ODE:
            # dHR/dt = -(HR_prev - HR_rest) / tau + beta * Workload
            d_hr_physics = - (hr_prev - hr_rest) / tau + beta * workload
            physics_residual = d_hr_actual - d_hr_physics
            physics_loss = torch.mean(physics_residual ** 2)
            mean_residual_bpm_s = torch.mean(torch.abs(physics_residual)).item()
        else:
            d_hr_actual = (hr_pred - hr_prev) / dt_clamped
            physics_loss = torch.tensor(0.0, device=hr_pred.device)
            mean_residual_bpm_s = 0.0

        # 3. Biophysical Boundary Violations Loss
        if self.lambda_bounds > 0.0:
            lower_viol = F.relu(self.hr_min - hr_pred) ** 2
            upper_viol = F.relu(hr_pred - self.hr_max) ** 2
            rate_viol = F.relu(torch.abs(d_hr_actual) - self.max_rate_bpm_per_sec) ** 2
            bounds_loss = torch.mean(lower_viol + upper_viol + rate_viol)
        else:
            bounds_loss = torch.tensor(0.0, device=hr_pred.device)

        # Total combined objective
        total_loss = data_loss + self.lambda_physics * physics_loss + self.lambda_bounds * bounds_loss

        metrics = {
            "loss_total": total_loss.item(),
            "loss_data": data_loss.item(),
            "loss_physics": physics_loss.item(),
            "loss_bounds": bounds_loss.item(),
            "mean_physics_residual": mean_residual_bpm_s
        }

        return total_loss, metrics

if __name__ == "__main__":
    criterion = PhysiologicalPINNLoss()
    p = torch.tensor([[85.0]], requires_grad=True)
    v = torch.tensor([[4.0]], requires_grad=True)
    t = torch.tensor([[84.0]])
    pr = torch.tensor([[82.0]])
    wl = torch.tensor([[0.5]])
    rst = torch.tensor([[65.0]])
    tau = torch.tensor([[45.0]])
    beta = torch.tensor([[15.0]])
    dt = torch.tensor([[1.0]])
    loss, details = criterion(p, v, t, pr, wl, rst, tau, beta, dt)
    print("Loss:", loss.item(), details)

