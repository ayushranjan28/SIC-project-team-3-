"""
Wearable Edge Optimization and INT8 Quantization Benchmark.

Prepares the PINN Forecaster for edge deployment on Samsung Galaxy Watch
(wearOS / ARM Cortex-A55 / Exynos W930):
1. Applies PyTorch Post-Training Dynamic INT8 Quantization to linear and recurrent layers.
2. Evaluates model size compression (FP32 vs INT8 in KB).
3. Evaluates CPU inference latency (batch=1, mimicking real-time watch sensor interrupts).
4. Evaluates accuracy preservation (MAE / RMSE on test data).
5. Exports optimized edge artifacts.
"""

import os
import sys
import time
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from training.models.pinn_forecaster import PINNWearableForecaster
from training.preprocessing.signal_adapters import PulseTransitTimeAdapter
from training.preprocessing.dataset_builder import (
    split_sessions_participant_independent,
    build_windows_from_sessions
)
from training.evaluation.evaluator import evaluate_forecaster

EDGE_DIR = BASE_DIR / "training" / "edge"
MODELS_SAVED_DIR = BASE_DIR / "models" / "saved_models"

def run_edge_benchmark():
    print("=" * 60)
    print("SAMSUNG GALAXY WATCH PINN EDGE OPTIMIZATION & BENCHMARK")
    print("=" * 60)

    device = torch.device("cpu") # Wearable edge processor runs on CPU
    print(f"[+] Target Runtime: Edge CPU ({torch.get_num_threads()} worker threads)")

    # 1. Load FP32 Trained PINN Model
    model_fp32 = PINNWearableForecaster(input_dim=4, hidden_dim=48)
    ckpt_path = MODELS_SAVED_DIR / "pinn_model.pt"
    if not ckpt_path.exists():
        ckpt_path = BASE_DIR / "training" / "checkpoints" / "pinn_complete_best.pt"

    if not ckpt_path.exists():
        print(f"[!] Checkpoint not found at {ckpt_path}. Please train first.")
        return

    model_fp32.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model_fp32.eval()

    # Save FP32 file for size measurement
    fp32_path = EDGE_DIR / "pinn_fp32.pt"
    torch.save(model_fp32.state_dict(), fp32_path)
    fp32_size_kb = fp32_path.stat().st_size / 1024.0

    # 2. Dynamic INT8 Quantization (quantizes nn.Linear and nn.GRU weights to int8)
    print("\n[+] Applying Dynamic INT8 Quantization (Linear + Recurrent layers)...")
    model_int8 = torch.ao.quantization.quantize_dynamic(
        model_fp32,
        {nn.Linear, nn.GRU},
        dtype=torch.qint8
    )
    model_int8.eval()

    # Save INT8 model
    int8_path = EDGE_DIR / "pinn_int8_quantized.pt"
    torch.save(model_int8.state_dict(), int8_path)
    int8_size_kb = int8_path.stat().st_size / 1024.0

    # 3. Load Real Test Data
    print("\n[+] Loading Test Telemetry for Edge Accuracy Validation...")
    ptt_adapter = PulseTransitTimeAdapter(BASE_DIR / "datasets/extracted/Pulse_Transit_Time_PPG")
    ptt_sessions = ptt_adapter.load_sessions(max_subjects=6)
    _, _, test_sess = split_sessions_participant_independent(ptt_sessions, train_ratio=0.7, val_ratio=0.15)
    
    test_ds, _ = build_windows_from_sessions(test_sess, window_sec=30, horizon_sec=1, stride_sec=2)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)
    print(f"Loaded {len(test_ds)} test windows.")

    # 4. Accuracy Evaluation
    print("\n[+] Evaluating FP32 Test Accuracy...")
    metrics_fp32 = evaluate_forecaster(model_fp32, test_loader, device=device, is_pinn=True)
    
    print("\n[+] Evaluating INT8 Test Accuracy...")
    # Wrap int8 model forward
    metrics_int8 = evaluate_forecaster(model_int8, test_loader, device=device, is_pinn=True)

    # 5. Latency Benchmarking (1000 single-step real-time inferences)
    print("\n[+] Measuring Real-Time Single-Window Inference Latency (Batch Size = 1)...")
    dummy_input = torch.randn(1, 30, 4)
    
    # Warmup
    for _ in range(100):
        _ = model_fp32(dummy_input)
        _ = model_int8(dummy_input)

    n_iter = 1000
    # FP32 Latency
    t0 = time.perf_counter()
    for _ in range(n_iter):
        _ = model_fp32(dummy_input)
    fp32_latency_ms = (time.perf_counter() - t0) / n_iter * 1000.0

    # INT8 Latency
    t0 = time.perf_counter()
    for _ in range(n_iter):
        _ = model_int8(dummy_input)
    int8_latency_ms = (time.perf_counter() - t0) / n_iter * 1000.0

    # 6. Summary Report
    results = {
        "Target_Device": "Samsung Galaxy Watch / Wearable Edge CPU",
        "FP32_Model": {
            "File_Size_KB": round(fp32_size_kb, 2),
            "Inference_Latency_ms": round(fp32_latency_ms, 3),
            "Throughput_inferences_per_sec": round(1000.0 / fp32_latency_ms, 1),
            "MAE_BPM": metrics_fp32["MAE"],
            "RMSE_BPM": metrics_fp32["RMSE"],
            "Boundary_Violation_Pct": metrics_fp32["Physiological_Violation_Pct"]
        },
        "INT8_Quantized_Model": {
            "File_Size_KB": round(int8_size_kb, 2),
            "Inference_Latency_ms": round(int8_latency_ms, 3),
            "Throughput_inferences_per_sec": round(1000.0 / int8_latency_ms, 1),
            "MAE_BPM": metrics_int8["MAE"],
            "RMSE_BPM": metrics_int8["RMSE"],
            "Boundary_Violation_Pct": metrics_int8["Physiological_Violation_Pct"]
        },
        "Improvements": {
            "Size_Reduction_Pct": round((1.0 - int8_size_kb / fp32_size_kb) * 100.0, 1),
            "Latency_Speedup": round(fp32_latency_ms / max(int8_latency_ms, 1e-6), 2),
            "MAE_Delta_BPM": round(metrics_int8["MAE"] - metrics_fp32["MAE"], 4)
        }
    }

    report_path = EDGE_DIR / "edge_benchmark_summary.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print("EDGE BENCHMARK RESULTS SUMMARY:")
    print("=" * 60)
    print(json.dumps(results, indent=2))
    print(f"\n[+] Results saved to {report_path}")

if __name__ == "__main__":
    run_edge_benchmark()

