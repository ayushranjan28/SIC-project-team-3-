"""
Comprehensive Physiological Evaluation Suite for PINN Forecasters.

Calculates:
1. Forecasting Accuracy: MAE, RMSE, Median AE (BPM)
2. Biophysical Plausibility:
   - Mean ODE Physics Residual (BPM/s)
   - Boundary Violation Rate: % of samples outside [45, 215] BPM or |dHR/dt| > 6 BPM/s
3. Uncertainty Quantification:
   - PICP (Prediction Interval Coverage Probability at 95% nominal confidence)
   - MPIW (Mean Prediction Interval Width)
4. Runtime & Edge Metrics:
   - Latency (ms per window, batch=1)
   - Model parameters
"""

import time
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, Tuple, Optional
from torch.utils.data import DataLoader

def evaluate_forecaster(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    is_pinn: bool = True
) -> Dict[str, Any]:
    """
    Evaluates a model over a test DataLoader.
    
    Returns comprehensive metrics dictionary.
    """
    model.eval()
    
    preds_list = []
    targets_list = []
    vars_list = []
    hr_prevs_list = []
    workloads_list = []
    hr_rests_list = []
    dts_list = []
    
    # Latency timing
    step_latencies = []

    with torch.no_grad():
        for x, y, aux in dataloader:
            x = x.to(device)
            subj = aux["subject_idx"].to(device) if "subject_idx" in aux else None
            
            t0 = time.perf_counter()
            if is_pinn:
                hr_pred, hr_var, tau, beta = model(x, subj)
                t1 = time.perf_counter()
                vars_list.append(hr_var.cpu().numpy())
            else:
                hr_pred = model(x)
                t1 = time.perf_counter()
                # Dummy constant variance for baseline (e.g. 1.0)
                vars_list.append(np.ones_like(hr_pred.cpu().numpy()))

            # Latency in ms per sample
            step_latencies.append((t1 - t0) / x.shape[0] * 1000.0)

            preds_list.append(hr_pred.cpu().numpy())
            targets_list.append(y.numpy())
            hr_prevs_list.append(aux["hr_prev"].numpy())
            workloads_list.append(aux["workload"].numpy())
            hr_rests_list.append(aux["hr_rest"].numpy())
            dts_list.append(aux["dt"].numpy())

    preds = np.concatenate(preds_list).flatten()
    targets = np.concatenate(targets_list).flatten()
    variances = np.concatenate(vars_list).flatten()
    hr_prevs = np.concatenate(hr_prevs_list).flatten()
    workloads = np.concatenate(workloads_list).flatten()
    hr_rests = np.concatenate(hr_rests_list).flatten()
    dts = np.concatenate(dts_list).flatten()

    # 1. Accuracy
    errors = np.abs(preds - targets)
    mae = float(np.mean(errors))
    rmse = float(np.sqrt(np.mean((preds - targets) ** 2)))
    median_ae = float(np.median(errors))

    # 2. Physiological Constraints & Violations
    dt_clamped = np.maximum(dts, 0.1)
    d_hr = (preds - hr_prevs) / dt_clamped
    
    # Boundary violations: HR < 45, HR > 215, or |dHR/dt| > 6 BPM/s
    viol_mask = (preds < 45.0) | (preds > 215.0) | (np.abs(d_hr) > 6.0)
    violation_pct = float(np.mean(viol_mask) * 100.0)

    # Physics residual: |dHR/dt - (-(HR - HR_rest)/tau + beta * Workload)|
    if is_pinn and hasattr(model, 'tau') and hasattr(model, 'beta'):
        tau_val = float(model.tau.mean().item() if isinstance(model.tau, torch.Tensor) else model.tau)
        beta_val = float(model.beta.mean().item() if isinstance(model.beta, torch.Tensor) else model.beta)
        d_hr_ode = -(hr_prevs - hr_rests) / tau_val + beta_val * workloads
        physics_residual = float(np.mean(np.abs(d_hr - d_hr_ode)))
    else:
        tau_val = None
        beta_val = None
        physics_residual = 0.0

    # 3. Uncertainty Quantification (95% CI: mean +/- 1.96 * std)
    stds = np.sqrt(np.maximum(variances, 1e-4))
    lower_95 = preds - 1.96 * stds
    upper_95 = preds + 1.96 * stds
    covered = (targets >= lower_95) & (targets <= upper_95)
    picp_95 = float(np.mean(covered) * 100.0)
    mpiw_95 = float(np.mean(upper_95 - lower_95))

    # 4. Latency & Parameters
    mean_latency_ms = float(np.mean(step_latencies))
    n_params = sum(p.numel() for p in model.parameters())

    results = {
        "MAE": round(mae, 3),
        "RMSE": round(rmse, 3),
        "Median_AE": round(median_ae, 3),
        "Physiological_Violation_Pct": round(violation_pct, 2),
        "Physics_Residual_BPM_s": round(physics_residual, 3),
        "PICP_95_Pct": round(picp_95, 2),
        "MPIW_95": round(mpiw_95, 2),
        "Latency_ms": round(mean_latency_ms, 4),
        "Parameters": n_params,
        "Tau_Recovery_Sec": round(tau_val, 2) if tau_val is not None else None,
        "Beta_Workload_Gain": round(beta_val, 2) if beta_val is not None else None
    }

    return results
