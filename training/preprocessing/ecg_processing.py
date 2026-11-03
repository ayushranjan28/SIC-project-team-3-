"""
Electrocardiogram (ECG) R-Peak Detection and Ground-Truth Heart Rate Extraction.

Detects ventricular depolarisation (QRS / R-peaks) in reference ECG signals,
computes instantaneous RR intervals, enforces physiological validity constraints,
and interpolates continuous, aligned ground-truth heart rate trajectories:
    HR(t) in BPM.
"""

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
from scipy.interpolate import interp1d

def filter_ecg(ecg: np.ndarray, fs: float) -> np.ndarray:
    """Bandpass filter ECG in QRS energy band (5 Hz to 25 Hz)."""
    nyq = 0.5 * fs
    low = 5.0 / nyq
    high = min(25.0 / nyq, 0.99)
    b, a = butter(2, [low, high], btype='band')
    return filtfilt(b, a, ecg)

def detect_r_peaks(ecg: np.ndarray, fs: float) -> np.ndarray:
    """
    Pan-Tompkins inspired R-peak detector.
    
    Args:
        ecg: 1D raw or filtered ECG signal.
        fs: Sampling frequency in Hz.
        
    Returns:
        1D array of sample indices corresponding to detected R-peaks.
    """
    filtered = filter_ecg(ecg, fs)
    # Five-point derivative
    diff = np.diff(filtered)
    # Squaring
    squared = diff ** 2
    # Moving window integrator (150 ms)
    win_len = int(0.150 * fs)
    kernel = np.ones(win_len) / win_len
    integrated = np.convolve(squared, kernel, mode='same')
    
    # Adaptive threshold: minimum peak distance of 0.30s (max 200 BPM)
    min_dist = int(0.30 * fs)
    threshold = 0.35 * np.percentile(integrated, 95)
    
    peaks, _ = find_peaks(integrated, distance=min_dist, height=threshold)
    
    # Refine peaks by finding local maxima on original ECG within +/- 50ms of integrated peak
    refine_win = int(0.05 * fs)
    refined_peaks = []
    n = len(ecg)
    for p in peaks:
        start_idx = max(0, p - refine_win)
        end_idx = min(n, p + refine_win)
        local_max = start_idx + np.argmax(ecg[start_idx:end_idx])
        refined_peaks.append(local_max)
        
    return np.array(sorted(list(set(refined_peaks))), dtype=int)

def compute_ground_truth_hr(
    ecg: np.ndarray,
    fs: float,
    target_fs_hz: float = 1.0,
    r_peaks: np.ndarray = None
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Computes uniform 1 Hz ground-truth HR trajectory from ECG or known R-peaks.
    
    Args:
        ecg: 1D ECG waveform (can be None if r_peaks provided).
        fs: Sampling rate of ECG in Hz.
        target_fs_hz: Output sampling rate (default 1.0 Hz).
        r_peaks: Pre-annotated R-peak sample indices (optional).
        
    Returns:
        timestamps: (T,) array of time in seconds.
        hr_bpm: (T,) array of ground-truth heart rate in BPM.
        hr_rest: estimated baseline resting HR (10th percentile).
    """
    if r_peaks is None:
        if ecg is None:
            raise ValueError("Must provide either ecg signal or r_peaks")
        r_peaks = detect_r_peaks(ecg, fs)
        
    if len(r_peaks) < 3:
        raise ValueError(f"Insufficient R-peaks detected: {len(r_peaks)}")
        
    peak_times = r_peaks / fs # in seconds
    rr_intervals = np.diff(peak_times) # in seconds
    
    # Filter non-physiological RR intervals: 0.27s (220 bpm) to 2.0s (30 bpm)
    valid_mask = (rr_intervals >= 0.27) & (rr_intervals <= 2.0)
    
    # Remove sudden single-beat jump artifacts (> 35% difference from neighboring beat)
    for i in range(1, len(rr_intervals) - 1):
        if valid_mask[i]:
            neighbor_median = np.median([rr_intervals[i-1], rr_intervals[i+1]])
            if abs(rr_intervals[i] - neighbor_median) / neighbor_median > 0.35:
                valid_mask[i] = False
                
    valid_rr = rr_intervals[valid_mask]
    valid_times = peak_times[1:][valid_mask]
    
    if len(valid_rr) < 3:
        raise ValueError("Too few valid RR intervals after physiological filtering.")
        
    valid_bpm = 60.0 / valid_rr
    
    # Build uniform time grid
    total_duration = peak_times[-1]
    uniform_times = np.arange(0.0, total_duration, 1.0 / target_fs_hz)
    
    # Linear interpolation with nearest boundary clamp
    interp_fn = interp1d(
        valid_times,
        valid_bpm,
        kind='linear',
        fill_value='extrapolate',
        assume_sorted=True
    )
    hr_bpm = interp_fn(uniform_times)
    hr_bpm = np.clip(hr_bpm, 40.0, 220.0)
    
    # Estimate resting HR from lowest 10th percentile of clean signal
    hr_rest = float(np.percentile(hr_bpm, 10))
    
    return uniform_times, hr_bpm, hr_rest

if __name__ == "__main__":
    fs = 250.0
    t = np.arange(0, 30, 1/fs)
    # Simulate ECG with QRS spikes at 1 Hz (60 bpm)
    sim_ecg = np.zeros_like(t)
    spike_indices = np.arange(int(fs), len(t), int(fs))
    for s in spike_indices:
        sim_ecg[s] = 2.0 # R peak
        sim_ecg[s-5:s+5] = np.sin(np.linspace(0, np.pi, 10)) * 2.0
    
    u_t, u_hr, r_hr = compute_ground_truth_hr(sim_ecg, fs)
    print(f"Detected uniform HR: mean={u_hr.mean():.1f} BPM, rest={r_hr:.1f} BPM")

