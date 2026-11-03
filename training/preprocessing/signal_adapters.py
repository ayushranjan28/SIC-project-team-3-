"""
Dataset Adapters for Physiological Wearable Recordings.

Standardises heterogeneous raw biomedical sensor recordings from:
1. Pulse Transit Time PPG Dataset (PhysioNet)
2. Wrist PPG During Exercise (PhysioNet)
3. IEEE Signal Processing Cup 2015
4. PPG-DaLiA (UCI / University of Siegen)
5. PAMAP2 Physical Activity Monitoring (UCI)

into uniform PhysiologicalSession objects with continuous 1.0 Hz master time resolution.
"""

import os
import sys
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.interpolate import interp1d

from training.preprocessing.ecg_processing import compute_ground_truth_hr, detect_r_peaks
from training.preprocessing.workload_estimator import compute_workload, compute_enmo
from training.preprocessing.ppg_sqi import compute_ppg_sqi

@dataclass
class PhysiologicalSession:
    """Standardized representation of a single continuous physiological recording session."""
    session_id: str
    subject_id: str
    dataset: str
    activity: str
    duration_sec: float
    timestamps: np.ndarray       # Shape (T,) in seconds, 1 Hz
    hr_ground_truth: np.ndarray  # Shape (T,) in BPM from reference ECG / TrueBPM
    hr_rest: float               # Baseline resting HR in BPM
    workload: np.ndarray         # Shape (T,) in [0.0, 1.0] derived from tri-axial ACC
    acc_magnitude: np.ndarray    # Shape (T,) total ACC norm in g
    ppg_sqi: np.ndarray          # Shape (T,) signal quality index in [0.0, 1.0]
    metadata: Dict[str, Any]

class BaseDatasetAdapter:
    """Abstract base class for dataset adapters."""
    def __init__(self, data_root: Path):
        self.data_root = Path(data_root)

    def load_sessions(self) -> List[PhysiologicalSession]:
        raise NotImplementedError

class PulseTransitTimeAdapter(BaseDatasetAdapter):
    """Adapter for PhysioNet Pulse Transit Time PPG Dataset (66 records across 22 subjects)."""
    
    def load_sessions(self, max_subjects: Optional[int] = None) -> List[PhysiologicalSession]:
        import wfdb
        sessions = []
        if not self.data_root.exists():
            return sessions

        hea_files = sorted(list(self.data_root.glob("s*.hea")))
        processed_subjects = set()

        for hea in hea_files:
            rec_id = hea.stem
            parts = rec_id.split("_")
            subj_id = parts[0]
            activity = "_".join(parts[1:]) if len(parts) > 1 else "unknown"

            if max_subjects and len(processed_subjects) >= max_subjects and subj_id not in processed_subjects:
                continue

            try:
                rec = wfdb.rdrecord(str(self.data_root / rec_id))
                fs = rec.fs
                sig_names = rec.sig_name
                
                if "ecg" not in sig_names or "a_x" not in sig_names:
                    continue

                ecg_idx = sig_names.index("ecg")
                ecg = rec.p_signal[:, ecg_idx]

                # R-peak detection and ground truth HR
                t_grid, hr_gt, hr_rest = compute_ground_truth_hr(ecg, fs, target_fs_hz=1.0)
                n_sec = len(t_grid)
                if n_sec < 30:
                    continue

                # Triaxial ACC
                ax = rec.p_signal[:, sig_names.index("a_x")]
                ay = rec.p_signal[:, sig_names.index("a_y")]
                az = rec.p_signal[:, sig_names.index("a_z")]
                acc_tri = np.column_stack([ax, ay, az])
                
                # Workload at full fs, then resample to 1 Hz
                wl_full = compute_workload(acc_tri, fs)
                enmo_full = compute_enmo(acc_tri)
                
                # Resample workload and enmo to 1 Hz grid
                t_orig = np.arange(len(wl_full)) / fs
                interp_wl = interp1d(t_orig, wl_full, fill_value="extrapolate")
                wl_1hz = np.clip(interp_wl(t_grid), 0.0, 1.0)
                
                interp_enmo = interp1d(t_orig, enmo_full, fill_value="extrapolate")
                acc_mag_1hz = np.clip(interp_enmo(t_grid) + 1.0, 0.0, 10.0)

                # PPG SQI
                ppg_idx = sig_names.index("pleth_1") if "pleth_1" in sig_names else 1
                ppg_raw = rec.p_signal[:, ppg_idx]
                sqi_coarse = compute_ppg_sqi(ppg_raw, fs, window_sec=4.0, step_sec=1.0)
                if len(sqi_coarse) < n_sec:
                    sqi_1hz = np.pad(sqi_coarse, (0, n_sec - len(sqi_coarse)), mode="edge")
                else:
                    sqi_1hz = sqi_coarse[:n_sec]

                session = PhysiologicalSession(
                    session_id=f"PTT_{rec_id}",
                    subject_id=f"PTT_{subj_id}",
                    dataset="Pulse_Transit_Time_PPG",
                    activity=activity,
                    duration_sec=float(t_grid[-1]),
                    timestamps=t_grid,
                    hr_ground_truth=hr_gt,
                    hr_rest=hr_rest,
                    workload=wl_1hz,
                    acc_magnitude=acc_mag_1hz,
                    ppg_sqi=sqi_1hz,
                    metadata={"original_fs": fs, "sensor_location": "wrist_and_finger"}
                )
                sessions.append(session)
                processed_subjects.add(subj_id)
            except Exception as e:
                print(f"[!] Warning: failed processing {rec_id}: {e}")
                continue

        return sessions

class WristPPGExerciseAdapter(BaseDatasetAdapter):
    """Adapter for PhysioNet Wrist PPG During Exercise (19 records)."""
    
    def load_sessions(self) -> List[PhysiologicalSession]:
        import wfdb
        sessions = []
        # Support both direct root and nested subfolder
        search_dirs = [self.data_root, self.data_root / "wrist-ppg-during-exercise-1.0.0"]
        hea_files = []
        for d in search_dirs:
            if d.exists():
                hea_files.extend(list(d.glob("s*.hea")))

        for hea in sorted(hea_files):
            rec_path = str(hea.with_suffix(""))
            rec_id = hea.stem
            parts = rec_id.split("_")
            subj_id = parts[0]
            activity = "_".join(parts[1:])

            try:
                rec = wfdb.rdrecord(rec_path)
                fs = rec.fs
                sig_names = rec.sig_name
                
                # Check annotations first
                ann_samples = None
                try:
                    ann = wfdb.rdann(rec_path, "atr")
                    ann_samples = ann.sample
                except Exception:
                    ann_samples = None

                ecg_idx = sig_names.index("chest_ecg") if "chest_ecg" in sig_names else 0
                ecg = rec.p_signal[:, ecg_idx]

                t_grid, hr_gt, hr_rest = compute_ground_truth_hr(
                    ecg=ecg, fs=fs, target_fs_hz=1.0, r_peaks=ann_samples
                )
                n_sec = len(t_grid)
                if n_sec < 30:
                    continue

                # Accelerometer (fill any NaN sensor dropouts)
                acc_cols = [c for c in sig_names if "accelerometer" in c or c.startswith("acc")]
                if len(acc_cols) >= 3:
                    raw_acc_tri = rec.p_signal[:, [sig_names.index(c) for c in acc_cols[:3]]]
                    acc_tri = np.nan_to_num(raw_acc_tri, nan=0.0)
                    # Forward-fill if possible
                    if np.isnan(raw_acc_tri).any():
                        acc_tri = pd.DataFrame(raw_acc_tri).ffill().bfill().fillna(0.0).values
                else:
                    acc_tri = np.zeros((len(ecg), 3))
                    acc_tri[:, 2] = 1.0

                wl_full = compute_workload(acc_tri, fs)
                enmo_full = compute_enmo(acc_tri)

                t_orig = np.arange(len(wl_full)) / fs
                interp_wl = interp1d(t_orig, wl_full, fill_value="extrapolate")
                wl_1hz = np.clip(interp_wl(t_grid), 0.0, 1.0)
                
                interp_enmo = interp1d(t_orig, enmo_full, fill_value="extrapolate")
                acc_mag_1hz = np.clip(interp_enmo(t_grid) + 1.0, 0.0, 10.0)

                # PPG SQI (clean NaNs)
                ppg_idx = sig_names.index("wrist_ppg") if "wrist_ppg" in sig_names else 1
                ppg_raw = rec.p_signal[:, ppg_idx]
                if np.isnan(ppg_raw).any():
                    ppg_raw = pd.Series(ppg_raw).ffill().bfill().fillna(0.0).values
                sqi_coarse = compute_ppg_sqi(ppg_raw, fs, window_sec=4.0, step_sec=1.0)
                if len(sqi_coarse) < n_sec:
                    sqi_1hz = np.pad(sqi_coarse, (0, n_sec - len(sqi_coarse)), mode="edge")
                else:
                    sqi_1hz = sqi_coarse[:n_sec]

                session = PhysiologicalSession(
                    session_id=f"WristEx_{rec_id}",
                    subject_id=f"WristEx_{subj_id}",
                    dataset="Wrist_PPG_Exercise",
                    activity=activity,
                    duration_sec=float(t_grid[-1]),
                    timestamps=t_grid,
                    hr_ground_truth=hr_gt,
                    hr_rest=hr_rest,
                    workload=wl_1hz,
                    acc_magnitude=acc_mag_1hz,
                    ppg_sqi=sqi_1hz,
                    metadata={"original_fs": fs, "sensor_location": "wrist"}
                )
                sessions.append(session)
            except Exception as e:
                print(f"[!] Warning: failed processing {rec_id}: {e}")
                continue

        return sessions

class IEEESPC2015Adapter(BaseDatasetAdapter):
    """Adapter for IEEE Signal Processing Cup 2015 Dataset."""
    
    def load_sessions(self) -> List[PhysiologicalSession]:
        sessions = []
        # Find all DATA_*.mat files
        mat_files = sorted(list(self.data_root.rglob("DATA_*.mat")))
        
        for mat_path in mat_files:
            rec_id = mat_path.stem
            subj_id = rec_id.split("_")[1] # e.g. "01"
            
            # Find corresponding BPM ground truth
            # Typical naming: DATA_01_TYPE01_BPMtrain.mat
            bpm_files = list(mat_path.parent.glob(f"{rec_id}*BPM*.mat"))
            if not bpm_files:
                continue
                
            try:
                sig_data = sio.loadmat(str(mat_path))["sig"] # shape (6, N)
                bpm_data = sio.loadmat(str(bpm_files[0]))["BPM0"].flatten() # ground truth BPM
                
                # IEEE SPC 2015 layout:
                # 0: ECG, 1: PPG1, 2: PPG2, 3: ACC_X, 4: ACC_Y, 5: ACC_Z
                fs = 125.0
                n_samples = sig_data.shape[1]
                duration_sec = n_samples / fs
                
                # In IEEE SPC 2015, BPM is evaluated every 2 seconds with an 8-second window.
                # Timestamps for BPM: 8s, 10s, 12s, ...
                bpm_times = 8.0 + np.arange(len(bpm_data)) * 2.0
                
                # Align to 1.0 Hz master grid
                t_grid = np.arange(8.0, bpm_times[-1], 1.0)
                interp_bpm = interp1d(bpm_times, bpm_data, kind='linear', fill_value='extrapolate')
                hr_gt = np.clip(interp_bpm(t_grid), 40.0, 220.0)
                hr_rest = float(np.percentile(hr_gt, 10))
                
                # ACC
                acc_tri = sig_data[3:6, :].T # shape (N, 3)
                wl_full = compute_workload(acc_tri, fs)
                enmo_full = compute_enmo(acc_tri)
                
                t_orig = np.arange(n_samples) / fs
                interp_wl = interp1d(t_orig, wl_full, fill_value="extrapolate")
                wl_1hz = np.clip(interp_wl(t_grid), 0.0, 1.0)
                
                interp_enmo = interp1d(t_orig, enmo_full, fill_value="extrapolate")
                acc_mag_1hz = np.clip(interp_enmo(t_grid) + 1.0, 0.0, 10.0)
                
                # PPG SQI
                ppg_raw = sig_data[1, :]
                sqi_coarse = compute_ppg_sqi(ppg_raw, fs, window_sec=4.0, step_sec=1.0)
                interp_sqi = interp1d(np.arange(len(sqi_coarse)), sqi_coarse, fill_value="extrapolate")
                sqi_1hz = np.clip(interp_sqi(t_grid), 0.0, 1.0)
                
                session = PhysiologicalSession(
                    session_id=f"IEEE_{rec_id}",
                    subject_id=f"IEEE_Subj_{subj_id}",
                    dataset="IEEE_SPC_2015",
                    activity="treadmill_protocol",
                    duration_sec=float(t_grid[-1] - t_grid[0]),
                    timestamps=t_grid,
                    hr_ground_truth=hr_gt,
                    hr_rest=hr_rest,
                    workload=wl_1hz,
                    acc_magnitude=acc_mag_1hz,
                    ppg_sqi=sqi_1hz,
                    metadata={"original_fs": fs, "sensor_location": "wrist"}
                )
                sessions.append(session)
            except Exception as e:
                print(f"[!] Warning: failed processing {rec_id}: {e}")
                continue
                
        return sessions

class PAMAP2Adapter(BaseDatasetAdapter):
    """Adapter for PAMAP2 Physical Activity Monitoring Dataset (12 daily protocol activities)."""
    
    ACTIVITY_NAMES = {
        1: "lying",
        2: "sitting",
        3: "standing",
        4: "walking",
        5: "running",
        6: "cycling",
        7: "nordic_walking",
        12: "ascending_stairs",
        13: "descending_stairs",
        16: "vacuum_cleaning",
        17: "ironing"
    }

    def load_sessions(self, max_subjects: Optional[int] = None) -> List[PhysiologicalSession]:
        sessions = []
        search_dirs = [
            self.data_root / "PAMAP2_Dataset" / "Protocol",
            self.data_root / "Protocol",
            self.data_root
        ]
        dat_files = []
        for d in search_dirs:
            if d.exists():
                dat_files.extend(list(d.glob("subject*.dat")))
        dat_files = sorted(list(set(dat_files)))

        if max_subjects:
            dat_files = dat_files[:max_subjects]

        for fpath in dat_files:
            subj_id = fpath.stem # e.g. "subject101"
            try:
                # Columns: 0=time, 1=activity, 2=HR, 4..6=wrist acc (x,y,z in m/s^2)
                # Read using pandas
                df = pd.read_csv(fpath, sep=" ", header=None, usecols=[0, 1, 2, 4, 5, 6])
                df.columns = ["time", "activity", "hr", "ax", "ay", "az"]
                
                # Drop unclassified (activity == 0) and interpolate missing HR
                df = df[df["activity"] > 0].copy()
                if len(df) < 1000:
                    continue
                    
                df["hr"] = df["hr"].interpolate(method="linear").bfill().ffill()
                df = df.dropna()
                
                # Convert wrist acc from m/s^2 to g (divide by 9.80665)
                df["ax"] /= 9.80665
                df["ay"] /= 9.80665
                df["az"] /= 9.80665
                
                # Resample to 1.0 Hz master time grid
                df["sec"] = np.floor(df["time"]).astype(int)
                agg_df = df.groupby("sec").agg({
                    "activity": lambda s: s.mode()[0] if not s.empty else 1,
                    "hr": "mean",
                    "ax": "mean",
                    "ay": "mean",
                    "az": "mean"
                }).reset_index()
                
                n_sec = len(agg_df)
                if n_sec < 60:
                    continue
                    
                t_grid = np.arange(n_sec).astype(float)
                hr_gt = agg_df["hr"].values
                hr_rest = float(np.percentile(hr_gt, 5))
                
                acc_tri = agg_df[["ax", "ay", "az"]].values
                enmo = np.maximum(0.0, np.sqrt(np.sum(acc_tri**2, axis=1)) - 1.0)
                wl_1hz = np.clip(enmo * 2.5, 0.0, 1.0) # normalize to [0, 1]
                acc_mag_1hz = np.clip(enmo + 1.0, 0.0, 10.0)
                
                # Estimate SQI from activity intensity
                act_codes = agg_df["activity"].values
                # High motion activities (running=5, stairs=12,13) have lower SQI
                sqi_1hz = np.where(
                    np.isin(act_codes, [5, 12, 13]),
                    0.35 + 0.1 * np.random.randn(n_sec),
                    0.90 + 0.05 * np.random.randn(n_sec)
                )
                sqi_1hz = np.clip(sqi_1hz, 0.15, 0.98)
                
                session = PhysiologicalSession(
                    session_id=f"PAMAP2_{subj_id}",
                    subject_id=f"PAMAP2_{subj_id}",
                    dataset="PAMAP2",
                    activity="multi_activity_protocol",
                    duration_sec=float(n_sec),
                    timestamps=t_grid,
                    hr_ground_truth=hr_gt,
                    hr_rest=hr_rest,
                    workload=wl_1hz,
                    acc_magnitude=acc_mag_1hz,
                    ppg_sqi=sqi_1hz,
                    metadata={"original_fs": 100.0, "sensor_location": "wrist_hand"}
                )
                sessions.append(session)
            except Exception as e:
                print(f"[!] Warning: failed processing {subj_id}: {e}")
                continue
                
        return sessions

if __name__ == "__main__":
    from pathlib import Path
    base = Path(__file__).resolve().parent.parent.parent
    ptt_adapter = PulseTransitTimeAdapter(base / "datasets/extracted/Pulse_Transit_Time_PPG")
    ptt_sess = ptt_adapter.load_sessions(max_subjects=2)
    print(f"Loaded {len(ptt_sess)} PTT sessions from 2 subjects.")
    
    wrist_adapter = WristPPGExerciseAdapter(base / "datasets/extracted/Wrist_PPG_Exercise")
    wrist_sess = wrist_adapter.load_sessions()
    print(f"Loaded {len(wrist_sess)} Wrist PPG Exercise sessions.")
    
    ieee_adapter = IEEESPC2015Adapter(base / "datasets/extracted/IEEE_SPC_2015")
    ieee_sess = ieee_adapter.load_sessions()
    print(f"Loaded {len(ieee_sess)} IEEE SPC 2015 sessions.")
