"""
Temporal Sliding Window Dataset Builder and Participant-Independent Splitting.

Constructs non-leaking sequence windows from continuous PhysiologicalSession objects:
    Input X: [W, features] (e.g. 30 seconds of workload, acc, sqi, observed HR)
    Target y: Future HR(t + H) (e.g. H = 1s, 5s, 10s)
    Physics Context: hr_prev, workload, hr_rest, dt, subject_id, sqi

Guarantees 100% strict participant-independent train / val / test partitioning.
"""

import sys
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Tuple, Optional
from training.preprocessing.signal_adapters import PhysiologicalSession

class WearablePINNDataset(Dataset):
    """
    PyTorch Dataset for Wearable PINN forecasting.
    Features per time step:
        0: Workload(t) in [0.0, 1.0]
        1: Accelerometer Norm(t) in g
        2: PPG SQI(t) in [0.0, 1.0]
        3: Observed HR(t) in BPM
    """
    def __init__(
        self,
        features: np.ndarray,       # (N, W, 4)
        targets: np.ndarray,        # (N, 1)
        hr_prevs: np.ndarray,       # (N, 1)
        workloads: np.ndarray,      # (N, 1)
        hr_rests: np.ndarray,       # (N, 1)
        subject_indices: np.ndarray,# (N, 1)
        sqis: np.ndarray,           # (N, 1)
        dt: float = 1.0
    ):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)
        self.hr_prevs = torch.tensor(hr_prevs, dtype=torch.float32)
        self.workloads = torch.tensor(workloads, dtype=torch.float32)
        self.hr_rests = torch.tensor(hr_rests, dtype=torch.float32)
        self.subject_indices = torch.tensor(subject_indices, dtype=torch.long)
        self.sqis = torch.tensor(sqis, dtype=torch.float32)
        self.dt = dt

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        return (
            self.features[idx],
            self.targets[idx],
            {
                "hr_prev": self.hr_prevs[idx],
                "workload": self.workloads[idx],
                "hr_rest": self.hr_rests[idx],
                "subject_idx": self.subject_indices[idx],
                "sqi": self.sqis[idx],
                "dt": torch.tensor(self.dt, dtype=torch.float32)
            }
        )

def build_windows_from_sessions(
    sessions: List[PhysiologicalSession],
    window_sec: int = 30,
    horizon_sec: int = 1,
    stride_sec: int = 1,
    subject_map: Optional[Dict[str, int]] = None
) -> Tuple[WearablePINNDataset, Dict[str, int]]:
    """
    Extracts non-leaking temporal windows across sessions.
    
    Args:
        sessions: List of continuous PhysiologicalSession recordings.
        window_sec: Input history length in seconds (W = 30s).
        horizon_sec: Forecast horizon into the future (H = 1s, 5s, 10s).
        stride_sec: Stride step between windows (stride=1s for thorough sampling).
        subject_map: Dict mapping subject_id string to integer ID.
        
    Returns:
        WearablePINNDataset, subject_map
    """
    if subject_map is None:
        unique_subjects = sorted(list(set(s.subject_id for s in sessions)))
        subject_map = {sid: idx for idx, sid in enumerate(unique_subjects)}

    features_list = []
    targets_list = []
    hr_prevs_list = []
    workloads_list = []
    hr_rests_list = []
    subject_indices_list = []
    sqis_list = []

    for sess in sessions:
        T = len(sess.timestamps)
        # Required length: window_sec + horizon_sec
        if T < window_sec + horizon_sec:
            continue

        subj_idx = subject_map.get(sess.subject_id, 0)
        
        # Session signals
        hr = sess.hr_ground_truth
        wl = sess.workload
        acc = sess.acc_magnitude
        sqi = sess.ppg_sqi
        rest = sess.hr_rest

        # Combine into 4-channel matrix: (T, 4)
        sig_matrix = np.column_stack([wl, acc, sqi, hr])

        for start_t in range(0, T - window_sec - horizon_sec + 1, stride_sec):
            end_t = start_t + window_sec
            target_t = end_t + horizon_sec - 1

            # Input window: [window_sec, 4]
            x_win = sig_matrix[start_t:end_t, :]
            
            # Future target
            y_val = hr[target_t]
            
            # Auxiliary physics parameters at the boundary:
            # hr_prev is the most recently observed HR (end of input window)
            hr_prev = hr[end_t - 1]
            # average workload across the forecast horizon
            wl_horizon = np.mean(wl[end_t - 1:target_t + 1])
            sqi_horizon = np.mean(sqi[end_t - 1:target_t + 1])

            features_list.append(x_win)
            targets_list.append([y_val])
            hr_prevs_list.append([hr_prev])
            workloads_list.append([wl_horizon])
            hr_rests_list.append([rest])
            subject_indices_list.append([subj_idx])
            sqis_list.append([sqi_horizon])

    if len(features_list) == 0:
        raise ValueError("No windows generated! Check session lengths vs window/horizon parameters.")

    features_arr = np.array(features_list, dtype=np.float32)
    targets_arr = np.array(targets_list, dtype=np.float32)
    hr_prevs_arr = np.array(hr_prevs_list, dtype=np.float32)
    workloads_arr = np.array(workloads_list, dtype=np.float32)
    hr_rests_arr = np.array(hr_rests_list, dtype=np.float32)
    subject_indices_arr = np.array(subject_indices_list, dtype=np.int64)
    sqis_arr = np.array(sqis_list, dtype=np.float32)

    dataset = WearablePINNDataset(
        features=features_arr,
        targets=targets_arr,
        hr_prevs=hr_prevs_arr,
        workloads=workloads_arr,
        hr_rests=hr_rests_arr,
        subject_indices=subject_indices_arr,
        sqis=sqis_arr,
        dt=float(horizon_sec)
    )

    return dataset, subject_map

def split_sessions_participant_independent(
    sessions: List[PhysiologicalSession],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[List[PhysiologicalSession], List[PhysiologicalSession], List[PhysiologicalSession]]:
    """
    Partitions sessions into Train, Val, and Test subsets strictly by participant ID.
    Guarantees no participant's data appears across multiple splits.
    """
    rng = np.random.default_rng(seed)
    
    # Group sessions by subject
    subjects = sorted(list(set(s.subject_id for s in sessions)))
    shuffled_subjects = list(rng.permutation(subjects))
    
    n_total = len(shuffled_subjects)
    n_train = max(1, int(round(n_total * train_ratio)))
    n_val = max(1, int(round(n_total * val_ratio)))
    if n_train + n_val >= n_total:
        n_val = max(1, n_total - n_train - 1)
        
    train_subjs = set(shuffled_subjects[:n_train])
    val_subjs = set(shuffled_subjects[n_train:n_train + n_val])
    test_subjs = set(shuffled_subjects[n_train + n_val:])
    
    # If test_subjs empty due to small count, guarantee at least 1 subject in test
    if len(test_subjs) == 0 and len(train_subjs) > 1:
        moved = list(train_subjs)[-1]
        train_subjs.remove(moved)
        test_subjs.add(moved)

    train_sess = [s for s in sessions if s.subject_id in train_subjs]
    val_sess = [s for s in sessions if s.subject_id in val_subjs]
    test_sess = [s for s in sessions if s.subject_id in test_subjs]

    return train_sess, val_sess, test_sess

if __name__ == "__main__":
    from training.preprocessing.signal_adapters import PulseTransitTimeAdapter
    base = Path(__file__).resolve().parent.parent.parent
    ptt = PulseTransitTimeAdapter(base / "datasets/extracted/Pulse_Transit_Time_PPG")
    sess = ptt.load_sessions(max_subjects=4)
    tr_s, va_s, te_s = split_sessions_participant_independent(sess)
    print(f"Subjects split: train={len(tr_s)}, val={len(va_s)}, test={len(te_s)}")
    ds_train, s_map = build_windows_from_sessions(tr_s, window_sec=30, horizon_sec=1)
    print(f"Generated {len(ds_train)} training windows. Features shape: {ds_train.features.shape}")

