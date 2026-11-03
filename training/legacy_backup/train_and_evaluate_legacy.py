"""
Training and Benchmark Pipeline:
Compares Baseline GRU vs. Physics-Informed Neural Network (PINN)
Evaluates accuracy, noise-artifact robustness, and physiological consistency.
"""

import os
import time
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.wearable_dataset import generate_synthetic_wearable_sessions, WearableTimeSeriesDataset
from models.baseline_lstm import StandardWearableGRU
from models.pinn_wearable import PINNWearableForecaster, PhysiologicalPINNLoss

def train_models():
    os.makedirs("models/saved_models", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[+] Using device: {device}")

    # 1. Generate Training and Testing Sessions
    print("[1/5] Synthesizing continuous Galaxy Watch sessions...")
    df_train = generate_synthetic_wearable_sessions(n_sessions=40, session_len=250, noise_level=0.15)
    df_test = generate_synthetic_wearable_sessions(n_sessions=15, session_len=250, noise_level=0.25, motion_artifact_prob=0.12)

    train_ds = WearableTimeSeriesDataset(df_train, window_size=30, horizon=1)
    test_ds = WearableTimeSeriesDataset(df_test, window_size=30, horizon=1)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    print(f"[+] Train samples: {len(train_ds)}, Test samples: {len(test_ds)}")

    # 2. Train Standard Baseline GRU
    print("\n[2/5] Training Baseline Data-Driven GRU...")
    baseline = StandardWearableGRU(input_dim=3, hidden_dim=48).to(device)
    opt_base = torch.optim.Adam(baseline.parameters(), lr=1e-3)
    crit_base = nn.MSELoss()

    baseline.train()
    for epoch in range(1, 6):
        total_loss = 0.0
        for x, y, wl, rest in train_loader:
            x, y = x.to(device), y.to(device)
            opt_base.zero_grad()
            pred = baseline(x)
            loss = crit_base(pred, y)
            loss.backward()
            opt_base.step()
            total_loss += loss.item() * len(x)
        print(f"  Baseline Epoch {epoch}/5 | Loss: {total_loss / len(train_ds):.4f}")

    torch.save(baseline.state_dict(), "models/saved_models/baseline_gru.pt")
    print("[+] Baseline model saved.")

    # 3. Train Physics-Informed Neural Network (PINN)
    print("\n[3/5] Training Physics-Informed Neural Network (PINN)...")
    pinn = PINNWearableForecaster(input_dim=3, hidden_dim=48).to(device)
    opt_pinn = torch.optim.Adam(pinn.parameters(), lr=1e-3)
    crit_pinn = PhysiologicalPINNLoss(lambda_physics=0.45, lambda_bounds=0.25)

    pinn.train()
    for epoch in range(1, 6):
        epoch_stats = {"loss_data": 0, "loss_physics": 0, "loss_bounds": 0, "total": 0}
        for x, y, wl, rest in train_loader:
            x, y = x.to(device), y.to(device)
            wl, rest = wl.to(device), rest.to(device)
            # Previous observed HR from last step of input sequence (feature index 2 is hr_observed)
            hr_prev = x[:, -1, 2:3]

            opt_pinn.zero_grad()
            pred_mean, pred_var = pinn(x)
            loss, details = crit_pinn(
                pred_mean, pred_var, y, hr_prev, wl, rest,
                pinn.tau, pinn.beta
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(pinn.parameters(), 2.0)
            opt_pinn.step()

            for k in ["loss_data", "loss_physics", "loss_bounds"]:
                epoch_stats[k] += details[k] * len(x)
            epoch_stats["total"] += loss.item() * len(x)

        print(f"  PINN Epoch {epoch}/5 | Total: {epoch_stats['total']/len(train_ds):.4f} "
              f"(Data: {epoch_stats['loss_data']/len(train_ds):.3f}, "
              f"Phys: {epoch_stats['loss_physics']/len(train_ds):.3f}, "
              f"Bounds: {epoch_stats['loss_bounds']/len(train_ds):.3f}) "
              f"| Tau: {pinn.tau.item():.1f}s, Beta: {pinn.beta.item():.2f}")

    torch.save(pinn.state_dict(), "models/saved_models/pinn_model.pt")
    print("[+] PINN model saved.")

    # 4. Rigorous Head-to-Head Benchmark
    print("\n[4/5] Running Comparative Benchmark on Test Set...")
    baseline.eval()
    pinn.eval()

    base_preds, pinn_preds, pinn_vars, targets, hr_prevs = [], [], [], [], []

    # Measure Edge Latency
    start_time = time.time()
    with torch.no_grad():
        for x, y, wl, rest in test_loader:
            x = x.to(device)
            b_out = baseline(x).cpu().numpy()
            base_preds.append(b_out)
    base_latency = (time.time() - start_time) / len(test_ds) * 1000

    start_time = time.time()
    with torch.no_grad():
        for x, y, wl, rest in test_loader:
            x = x.to(device)
            p_mean, p_var = pinn(x)
            pinn_preds.append(p_mean.cpu().numpy())
            pinn_vars.append(p_var.cpu().numpy())
            targets.append(y.numpy())
            hr_prevs.append(x[:, -1, 2:3].cpu().numpy())
    pinn_latency = (time.time() - start_time) / len(test_ds) * 1000

    base_preds = np.vstack(base_preds).flatten()
    pinn_preds = np.vstack(pinn_preds).flatten()
    pinn_vars = np.vstack(pinn_vars).flatten()
    targets = np.vstack(targets).flatten()
    hr_prevs = np.vstack(hr_prevs).flatten()

    # Calculate Metrics
    base_rmse = np.sqrt(np.mean((base_preds - targets) ** 2))
    pinn_rmse = np.sqrt(np.mean((pinn_preds - targets) ** 2))

    base_mae = np.mean(np.abs(base_preds - targets))
    pinn_mae = np.mean(np.abs(pinn_preds - targets))

    # Physiological violation: single-second jump > 6.0 bpm/sec
    base_jump = np.abs(base_preds - hr_prevs)
    pinn_jump = np.abs(pinn_preds - hr_prevs)
    base_violations = np.mean(base_jump > 6.0) * 100.0
    pinn_violations = np.mean(pinn_jump > 6.0) * 100.0

    benchmark_results = {
        "Baseline_GRU": {
            "RMSE": round(float(base_rmse), 3),
            "MAE": round(float(base_mae), 3),
            "Physiological_Violations_Pct": round(float(base_violations), 2),
            "Latency_ms_per_step": round(float(base_latency), 3),
            "Parameters": sum(p.numel() for p in baseline.parameters())
        },
        "PINN_Model": {
            "RMSE": round(float(pinn_rmse), 3),
            "MAE": round(float(pinn_mae), 3),
            "Physiological_Violations_Pct": round(float(pinn_violations), 2),
            "Latency_ms_per_step": round(float(pinn_latency), 3),
            "Parameters": sum(p.numel() for p in pinn.parameters()),
            "Learned_Tau_Recovery_Sec": round(float(pinn.tau.item()), 2),
            "Learned_Beta_Exertion_Gain": round(float(pinn.beta.item()), 2)
        }
    }

    print("\n[5/5] Benchmark Results Summary:")
    print(json.dumps(benchmark_results, indent=2))

    with open("models/saved_models/benchmark_results.json", "w") as f:
        json.dump(benchmark_results, f, indent=2)

    # Save a sample test session for the interactive live dashboard
    sample_session = df_test[df_test["session_id"] == 0].copy()
    sample_session.to_csv("data/sample_session_demo.csv", index=False)
    print("\n[+] Done! Sample telemetry saved to 'data/sample_session_demo.csv'")

if __name__ == "__main__":
    train_models()
