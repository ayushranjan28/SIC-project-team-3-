"""
Edge Export & Compression:
Exports the trained Physics-Informed Neural Network to TorchScript, ONNX, and INT8 Quantized models.
Benchmarks model size and latency for wearable/edge deployment.
"""

import os
import time
import torch
import torch.nn as nn

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.pinn_wearable import PINNWearableForecaster

def export_edge_models():
    model_path = "models/saved_models/pinn_model.pt"
    if not os.path.exists(model_path):
        print(f"[-] Model checkpoint not found at {model_path}. Train first.")
        return

    os.makedirs("models/saved_models/edge", exist_ok=True)
    device = torch.device("cpu") # Edge execution target is CPU/NPU

    model = PINNWearableForecaster()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    dummy_input = torch.randn(1, 30, 3, dtype=torch.float32)

    # 1. Baseline FP32 Size
    fp32_path = "models/saved_models/edge/pinn_fp32.pt"
    torch.save(model.state_dict(), fp32_path)
    fp32_size_kb = os.path.getsize(fp32_path) / 1024

    # 2. Dynamic INT8 Quantization (Linear and GRU layers)
    quantized_model = torch.ao.quantization.quantize_dynamic(
        model,
        {nn.Linear, nn.GRU},
        dtype=torch.qint8
    )
    int8_path = "models/saved_models/edge/pinn_int8.pt"
    torch.save(quantized_model.state_dict(), int8_path)
    int8_size_kb = os.path.getsize(int8_path) / 1024

    # 3. TorchScript JIT Tracing (Edge / Mobile embedded C++ runtime)
    # Wrap model to output both mean and variance
    traced_script_module = torch.jit.trace(model, dummy_input)
    jit_path = "models/saved_models/edge/pinn_traced.pt"
    traced_script_module.save(jit_path)
    jit_size_kb = os.path.getsize(jit_path) / 1024

    # 4. Latency benchmark on CPU (100 runs)
    with torch.no_grad():
        # Warmup
        for _ in range(20):
            _ = model(dummy_input)
        
        t0 = time.time()
        for _ in range(100):
            _ = model(dummy_input)
        fp32_latency = (time.time() - t0) / 100 * 1000

        for _ in range(20):
            _ = quantized_model(dummy_input)
        t0 = time.time()
        for _ in range(100):
            _ = quantized_model(dummy_input)
        int8_latency = (time.time() - t0) / 100 * 1000

    report = {
        "FP32_Model_KB": round(fp32_size_kb, 1),
        "INT8_Quantized_KB": round(int8_size_kb, 1),
        "Size_Reduction_Pct": round((1.0 - int8_size_kb / fp32_size_kb) * 100, 1),
        "TorchScript_JIT_KB": round(jit_size_kb, 1),
        "FP32_Latency_ms": round(fp32_latency, 2),
        "INT8_Latency_ms": round(int8_latency, 2)
    }

    print("\n[+] Edge Optimization Summary:")
    for k, v in report.items():
        print(f"  - {k}: {v}")

    return report

if __name__ == "__main__":
    export_edge_models()
