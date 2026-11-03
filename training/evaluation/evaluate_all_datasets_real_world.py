"""
Comprehensive Real-World Physiological Forecasting Evaluation Engine.
Evaluates the PINN Forecaster, Baseline GRU, Baseline LSTM, and Edge INT8 Quantized Model
across ALL extracted real-world wearable datasets and activity scenarios:

Datasets Analyzed:
1. PhysioNet Pulse Transit Time PPG Dataset (Postures: sitting, walking, running)
2. PhysioNet Wrist PPG During Exercise (19 subjects on treadmill & high-resistance bikes)
3. IEEE Signal Processing Cup 2015 (Treadmill running under intense optical motion)
4. PAMAP2 Physical Activity Monitoring (9 subjects across 12 daily protocol activities)

Activity Scenarios Analyzed:
- Resting & Basal State (W <= 0.05, HR ~ 60-75 BPM)
- Light Daily Living (W in 0.05-0.25, HR ~ 75-95 BPM)
- Moderate Aerobic Activity (W in 0.25-0.60, HR ~ 95-125 BPM)
- Vigorous High-Intensity Exercise (W > 0.60, HR ~ 125-165 BPM)
- Severe Optical Motion Artifacts (PPG SQI < 0.40)
- Nominal High-Quality Optical Sensing (PPG SQI >= 0.70)
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from training.preprocessing.signal_adapters import (
    PulseTransitTimeAdapter,
    WristPPGExerciseAdapter,
    IEEESPC2015Adapter,
    PAMAP2Adapter,
    PhysiologicalSession
)
from training.preprocessing.dataset_builder import build_windows_from_sessions
from training.models.pinn_forecaster import PINNWearableForecaster
from training.models.baselines import StandardGRUForecaster, StandardLSTMForecaster
from training.evaluation.evaluator import evaluate_forecaster

DEVICE = torch.device("cpu")
EVAL_DIR = BASE_DIR / "training" / "evaluation"
MODELS_DIR = BASE_DIR / "models" / "saved_models"
DATASETS_DIR = BASE_DIR / "datasets" / "extracted"

def load_trained_models():
    """Loads all saved models from disk."""
    models = {}
    
    # 1. PINN Model
    pinn_path = MODELS_DIR / "pinn_model.pt"
    if pinn_path.exists():
        pinn = PINNWearableForecaster(input_dim=4, hidden_dim=48)
        pinn.load_state_dict(torch.load(pinn_path, map_location=DEVICE, weights_only=True))
        pinn.eval()
        models["Proposed_PINN"] = (pinn, True)
    
    # 2. Baseline GRU
    gru_path = MODELS_DIR / "baseline_gru.pt"
    if gru_path.exists():
        gru = StandardGRUForecaster(input_dim=4, hidden_dim=48)
        gru.load_state_dict(torch.load(gru_path, map_location=DEVICE, weights_only=True))
        gru.eval()
        models["Baseline_GRU"] = (gru, False)
        
    # 3. Baseline LSTM
    lstm_path = MODELS_DIR / "baseline_lstm.pt"
    if lstm_path.exists():
        lstm = StandardLSTMForecaster(input_dim=4, hidden_dim=48)
        lstm.load_state_dict(torch.load(lstm_path, map_location=DEVICE, weights_only=True))
        lstm.eval()
        models["Baseline_LSTM"] = (lstm, False)

    # 4. Edge Quantized PINN
    edge_path = BASE_DIR / "training" / "edge" / "pinn_int8_quantized.pt"
    if edge_path.exists():
        try:
            edge_model = torch.jit.load(str(edge_path), map_location=DEVICE)
            edge_model.eval()
            models["Edge_Quantized_INT8"] = (edge_model, True)
        except Exception as e:
            print(f"[!] Note: Quantized model load: {e}")
            
    return models

def evaluate_scenario_subsets(
    model: nn.Module,
    all_sessions: List[PhysiologicalSession],
    is_pinn: bool = True
) -> Dict[str, Any]:
    """Evaluates performance sliced by real-world physical activity contexts."""
    # Build complete dataset windows
    ds, meta = build_windows_from_sessions(all_sessions, window_sec=30, horizon_sec=1, stride_sec=2)
    loader = DataLoader(ds, batch_size=128, shuffle=False)
    
    # Run inference and collect all step records
    model.eval()
    records = []
    
    with torch.no_grad():
        for x, y, aux in loader:
            x_dev = x.to(DEVICE)
            subj = aux["subject_idx"].to(DEVICE) if "subject_idx" in aux else None
            
            if is_pinn:
                out = model(x_dev, subj) if hasattr(model, 'fc_mean') else model(x_dev)
                hr_pred = out[0] if isinstance(out, tuple) else out
                hr_var = out[1] if isinstance(out, tuple) and len(out) > 1 else torch.ones_like(hr_pred)
            else:
                hr_pred = model(x_dev)
                hr_var = torch.ones_like(hr_pred)
                
            pred_arr = hr_pred.cpu().numpy().flatten()
            var_arr = hr_var.cpu().numpy().flatten()
            target_arr = y.numpy().flatten()
            hr_prev_arr = aux["hr_prev"].numpy().flatten()
            wl_arr = aux["workload"].numpy().flatten()
            sqi_arr = aux["sqi"].numpy().flatten() if "sqi" in aux else aux.get("ppg_sqi", torch.ones_like(y)).numpy().flatten()
            dt_arr = aux["dt"].numpy().flatten()
            
            for j in range(len(pred_arr)):
                records.append({
                    "pred": pred_arr[j],
                    "target": target_arr[j],
                    "var": var_arr[j],
                    "hr_prev": hr_prev_arr[j],
                    "workload": wl_arr[j],
                    "ppg_sqi": sqi_arr[j],
                    "dt": dt_arr[j]
                })
                
    df = pd.DataFrame(records)
    if df.empty:
        return {}
        
    # Calculate scenario masks
    scenarios = {
        "Resting_Baseline": df["workload"] <= 0.05,
        "Light_Daily_Living": (df["workload"] > 0.05) & (df["workload"] <= 0.25),
        "Moderate_Aerobic": (df["workload"] > 0.25) & (df["workload"] <= 0.60),
        "Vigorous_Exercise": df["workload"] > 0.60,
        "Motion_Artifact_Surge": df["ppg_sqi"] < 0.40,
        "Clean_Optical_PPG": df["ppg_sqi"] >= 0.70
    }
    
    scenario_metrics = {}
    for name, mask in scenarios.items():
        sub_df = df[mask]
        n_pts = len(sub_df)
        if n_pts == 0:
            continue
            
        err = np.abs(sub_df["pred"].values - sub_df["target"].values)
        mae = float(np.mean(err))
        rmse = float(np.sqrt(np.mean((sub_df["pred"].values - sub_df["target"].values)**2)))
        
        # Violations
        d_hr = np.abs(sub_df["pred"].values - sub_df["hr_prev"].values) / np.maximum(sub_df["dt"].values, 0.1)
        viol = float(np.mean((d_hr > 6.0) | (sub_df["pred"].values < 45.0) | (sub_df["pred"].values > 215.0)) * 100.0)
        
        # Uncertainty coverage
        stds = np.sqrt(np.maximum(sub_df["var"].values, 1e-4))
        lower = sub_df["pred"].values - 1.96 * stds
        upper = sub_df["pred"].values + 1.96 * stds
        covered = (sub_df["target"].values >= lower) & (sub_df["target"].values <= upper)
        picp = float(np.mean(covered) * 100.0) if is_pinn else None
        
        scenario_metrics[name] = {
            "samples": n_pts,
            "MAE": round(mae, 2),
            "RMSE": round(rmse, 2),
            "Violation_Pct": round(viol, 2),
            "PICP_95": round(picp, 1) if picp is not None else "N/A"
        }
        
    return scenario_metrics

def main():
    print("=" * 80)
    print("  COMPREHENSIVE REAL-WORLD PHYSIOLOGICAL DATASET EVALUATION ENGINE")
    print("=" * 80)
    
    models = load_trained_models()
    print(f"[+] Loaded {len(models)} models: {list(models.keys())}")
    
    # 1. Load All Datasets
    dataset_sessions = {}
    
    # Dataset A: Wrist PPG During Exercise
    wrist_dir = DATASETS_DIR / "Wrist_PPG_Exercise"
    if wrist_dir.exists():
        w_adapter = WristPPGExerciseAdapter(wrist_dir)
        w_sess = w_adapter.load_sessions()
        if w_sess:
            dataset_sessions["Wrist_PPG_Exercise (PhysioNet)"] = w_sess
            print(f"[+] Loaded Wrist PPG During Exercise: {len(w_sess)} sessions ({sum(s.duration_sec for s in w_sess)/60:.1f} mins)")

    # Dataset B: IEEE SPC 2015
    ieee_dir = DATASETS_DIR / "IEEE_SPC_2015"
    if ieee_dir.exists():
        i_adapter = IEEESPC2015Adapter(ieee_dir)
        i_sess = i_adapter.load_sessions()
        if i_sess:
            dataset_sessions["IEEE_SPC_2015 (Treadmill Sprint)"] = i_sess
            print(f"[+] Loaded IEEE SPC 2015: {len(i_sess)} sessions ({sum(s.duration_sec for s in i_sess)/60:.1f} mins)")

    # Dataset C: PAMAP2 Physical Activity Monitoring
    pamap_dir = DATASETS_DIR / "PAMAP2"
    if pamap_dir.exists():
        p_adapter = PAMAP2Adapter(pamap_dir)
        p_sess = p_adapter.load_sessions(max_subjects=5)
        if p_sess:
            dataset_sessions["PAMAP2 (12 Daily Activities)"] = p_sess
            print(f"[+] Loaded PAMAP2 Physical Activity: {len(p_sess)} sessions ({sum(s.duration_sec for s in p_sess)/60:.1f} mins)")

    # Dataset D: Pulse Transit Time PPG
    ptt_dir = DATASETS_DIR / "Pulse_Transit_Time_PPG"
    if ptt_dir.exists():
        ptt_adapter = PulseTransitTimeAdapter(ptt_dir)
        ptt_sess = ptt_adapter.load_sessions(max_subjects=5)
        if ptt_sess:
            dataset_sessions["Pulse_Transit_Time_PPG (Postures)"] = ptt_sess
            print(f"[+] Loaded Pulse Transit Time PPG: {len(ptt_sess)} sessions ({sum(s.duration_sec for s in ptt_sess)/60:.1f} mins)")

    # 2. Run Comprehensive Multi-Dataset Cross-Evaluation
    dataset_benchmarks = {}
    all_dataset_sessions = []
    for d_name, sessions in dataset_sessions.items():
        all_dataset_sessions.extend(sessions)
        
        print(f"\n---> Evaluating on: {d_name} ...")
        ds, _ = build_windows_from_sessions(sessions, window_sec=30, horizon_sec=1, stride_sec=2)
        loader = DataLoader(ds, batch_size=64, shuffle=False)
        print(f"     Total 1-second prediction windows: {len(ds)}")
        
        d_results = {}
        for m_name, (m_obj, is_pinn) in models.items():
            metrics = evaluate_forecaster(m_obj, loader, DEVICE, is_pinn=is_pinn)
            d_results[m_name] = metrics
            print(f"     {m_name:24s} | MAE: {metrics['MAE']:5.2f} BPM | RMSE: {metrics['RMSE']:5.2f} BPM | Violations: {metrics['Physiological_Violation_Pct']:4.1f}% | Latency: {metrics['Latency_ms']*100:.2f}ms")
            
        dataset_benchmarks[d_name] = d_results

    # 3. Real-World Physical Activity Scenario Breakdown
    print("\n" + "=" * 80)
    print("  EVALUATING REAL-WORLD ACTIVITY CONTEXT SCENARIOS")
    print("=" * 80)
    
    scenario_breakdown = {}
    for m_name, (m_obj, is_pinn) in models.items():
        s_res = evaluate_scenario_subsets(m_obj, all_dataset_sessions, is_pinn=is_pinn)
        scenario_breakdown[m_name] = s_res

    # Display scenario comparison table
    print("\n[+] Activity Scenario Breakdown (PINN vs Baseline GRU):")
    pinn_sc = scenario_breakdown.get("Proposed_PINN", {})
    gru_sc = scenario_breakdown.get("Baseline_GRU", {})
    
    scenario_rows = []
    for sc_name in pinn_sc.keys():
        p_data = pinn_sc.get(sc_name, {})
        g_data = gru_sc.get(sc_name, {})
        scenario_rows.append({
            "Scenario": sc_name.replace("_", " "),
            "Samples": p_data.get("samples", 0),
            "PINN_MAE": p_data.get("MAE", "--"),
            "GRU_MAE": g_data.get("MAE", "--"),
            "PINN_Violations": f"{p_data.get('Violation_Pct', 0)}%",
            "GRU_Violations": f"{g_data.get('Violation_Pct', 0)}%",
            "PINN_PICP_95": f"{p_data.get('PICP_95', '--')}%"
        })
        
    df_sc = pd.DataFrame(scenario_rows)
    print(df_sc.to_string(index=False))
    
    # 4. Save All Results to Evaluation & Saved Models Directory
    out_json = EVAL_DIR / "comprehensive_real_world_benchmarks.json"
    with open(out_json, "w") as f:
        json.dump({
            "Dataset_Evaluations": dataset_benchmarks,
            "Activity_Scenario_Breakdown": scenario_breakdown
        }, f, indent=2)
    print(f"\n[+] Saved detailed evaluation JSON to: {out_json}")
    
    # Save CSV
    df_sc.to_csv(EVAL_DIR / "real_world_scenarios_breakdown.csv", index=False)
    print(f"[+] Saved scenario CSV to: {EVAL_DIR / 'real_world_scenarios_breakdown.csv'}")

    # Update dashboard benchmark results so frontend displays real-world metrics
    dash_bench = MODELS_DIR / "benchmark_results.json"
    if dash_bench.exists():
        with open(dash_bench, "r") as f:
            existing = json.load(f)
        existing["Real_World_Datasets"] = dataset_benchmarks
        existing["Real_World_Scenarios"] = scenario_breakdown
        with open(dash_bench, "w") as f:
            json.dump(existing, f, indent=2)
        print(f"[+] Synced real-world benchmarks into dashboard: {dash_bench}")

if __name__ == "__main__":
    main()
