"""
Wearable Physiological Data Generator & Loader
Simulates photoplethysmography (PPG), Heart Rate (HR), and Tri-axial Accelerometer (ACC)
based on non-linear human cardiovascular recovery ODEs + real-world motion artifacts.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class PhysiologicalODEDynamics:
    """
    Standard autonomic cardiovascular exertion-recovery model:
    d(HR)/dt = -1/tau * (HR - HR_rest) + beta * Workload(t)
    """
    def __init__(self, hr_rest=65.0, hr_max=185.0, tau_recovery=45.0, beta_gain=18.0):
        self.hr_rest = hr_rest
        self.hr_max = hr_max
        self.tau = tau_recovery
        self.beta = beta_gain

    def step(self, hr_prev, workload, dt=1.0):
        # Forward Euler integration of the ODE
        d_hr = (- (hr_prev - self.hr_rest) / self.tau + self.beta * workload) * dt
        hr_new = np.clip(hr_prev + d_hr, self.hr_rest - 5, self.hr_max)
        return hr_new

def generate_synthetic_wearable_sessions(n_sessions=100, session_len=300, noise_level=0.15, motion_artifact_prob=0.08):
    """
    Generates realistic continuous Galaxy Watch telemetry sessions:
    - Workload: intervals of resting (0.0), walking (0.3), running (0.8), high-intensity sprint (1.0)
    - True HR: follows physiological ODE
    - Observed Noisy HR: corrupted by sensor drift, Gaussian noise, and severe optical motion artifacts
    """
    np.random.seed(42)
    sessions = []

    for s in range(n_sessions):
        # Vary individual physiological parameters per user
        hr_rest = np.random.uniform(58.0, 75.0)
        tau = np.random.uniform(35.0, 60.0)
        beta = np.random.uniform(15.0, 22.0)
        dynamics = PhysiologicalODEDynamics(hr_rest=hr_rest, tau_recovery=tau, beta_gain=beta)

        # Generate realistic activity profile with transitions
        workload = np.zeros(session_len)
        t = 0
        while t < session_len:
            block_len = np.random.randint(30, 80)
            level = np.random.choice([0.0, 0.25, 0.65, 0.95], p=[0.25, 0.35, 0.25, 0.15])
            workload[t:min(t+block_len, session_len)] = level
            t += block_len

        # Smooth transitions
        workload = np.convolve(workload, np.ones(5)/5, mode='same')

        # Compute ground-truth HR trajectory via biological ODE
        hr_true = np.zeros(session_len)
        current_hr = hr_rest
        for i in range(session_len):
            current_hr = dynamics.step(current_hr, workload[i], dt=1.0)
            hr_true[i] = current_hr

        # Generate observed sensor signal: Sensor Noise + Motion Artifacts
        noise = np.random.normal(0, noise_level * 5.0, session_len)
        hr_observed = hr_true.copy() + noise

        # Inject severe motion artifacts (optical sensor disconnect / wrist movement)
        artifact_mask = np.random.binomial(1, motion_artifact_prob, session_len).astype(bool)
        hr_observed[artifact_mask] += np.random.choice([-35.0, 45.0, -25.0, 50.0], size=np.sum(artifact_mask))
        hr_observed = np.clip(hr_observed, 40.0, 220.0)

        # Accelerometer magnitude correlated with workload + noise
        acc_mag = workload * 2.5 + np.random.normal(0, 0.2, session_len)
        acc_mag = np.clip(acc_mag, 0.0, 4.0)

        df = pd.DataFrame({
            "session_id": s,
            "timestep": np.arange(session_len),
            "workload": workload,
            "acc_mag": acc_mag,
            "hr_observed": hr_observed,
            "hr_true": hr_true,
            "artifact_flag": artifact_mask.astype(int),
            "user_hr_rest": hr_rest,
            "user_tau": tau
        })
        sessions.append(df)

    return pd.concat(sessions, ignore_index=True)

class WearableTimeSeriesDataset(Dataset):
    """
    Sliding window dataset for sequence-to-sequence or sequence-to-one forecasting.
    Input: [workload, acc_mag, hr_observed] over window W (e.g. 30 seconds)
    Target: hr_true for the next step, plus workload for physics evaluation.
    """
    def __init__(self, df, window_size=30, horizon=1):
        self.window_size = window_size
        self.horizon = horizon
        self.samples = []

        feature_cols = ['workload', 'acc_mag', 'hr_observed']
        
        for _, group in df.groupby('session_id'):
            features = group[feature_cols].values
            hr_true = group['hr_true'].values
            hr_rest = group['user_hr_rest'].values
            workload = group['workload'].values
            
            n = len(group)
            for i in range(n - window_size - horizon + 1):
                x = features[i : i + window_size]
                y_true = hr_true[i + window_size : i + window_size + horizon]
                wl_target = workload[i + window_size : i + window_size + horizon]
                rest = hr_rest[i + window_size]
                self.samples.append((
                    torch.tensor(x, dtype=torch.float32),
                    torch.tensor(y_true, dtype=torch.float32),
                    torch.tensor(wl_target, dtype=torch.float32),
                    torch.tensor(rest, dtype=torch.float32)
                ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

if __name__ == "__main__":
    print("Generating synthetic Galaxy Watch wearable dataset...")
    data = generate_synthetic_wearable_sessions(n_sessions=50, session_len=200)
    print(f"Generated {len(data)} total sensor frames across 50 sessions.")
    print("Sample:\n", data.head())
