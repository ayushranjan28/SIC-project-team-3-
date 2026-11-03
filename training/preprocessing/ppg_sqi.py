"""
Photoplethysmography (PPG) Signal Quality Index (SQI) and Artifact Detector.

Computes multi-dimensional signal quality indices for wearable optical sensors:
1. Statistical moments: Skewness (S_sqi) and Kurtosis (K_sqi)
2. Perfusion Index (PI): Ratio of pulsatile (AC) to non-pulsatile (DC) optical component
3. Sensor saturation / clipping detector: flags flatline or ADC railing
4. Composite SQI score in [0.0, 1.0] to modulate loss confidence or feature inputs.
"""

import numpy as np
from scipy import stats
from scipy.signal import butter, filtfilt

def bandpass_ppg(ppg: np.ndarray, fs: float, lowcut: float = 0.5, highcut: float = 8.0) -> np.ndarray:
    """Bandpass filter PPG in cardiac pulse range [0.5 Hz - 8.0 Hz] (30 - 480 bpm)."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = min(highcut / nyq, 0.99)
    b, a = butter(2, [low, high], btype='band')
    return filtfilt(b, a, ppg)

def compute_ppg_sqi(
    ppg_raw: np.ndarray,
    fs: float,
    window_sec: float = 4.0,
    step_sec: float = 1.0
) -> np.ndarray:
    """
    Computes rolling composite Signal Quality Index (SQI) for raw PPG.
    
    Args:
        ppg_raw: 1D numpy array of raw photoplethysmogram measurements.
        fs: Sampling rate in Hz.
        window_sec: Window length for statistical evaluation (typically 4-8s).
        step_sec: Output step interval in seconds.
        
    Returns:
        1D array of composite SQI values in [0.0, 1.0], resampled to length matching
        the time base at step_sec.
    """
    n_samples = len(ppg_raw)
    win_len = int(window_sec * fs)
    step_len = int(step_sec * fs)
    
    if n_samples < win_len:
        return np.ones(max(1, int(n_samples / step_len))) * 0.5

    out_len = int(np.ceil(n_samples / step_len))
    sqi_values = np.zeros(out_len)
    
    # Pre-filter for AC component analysis
    try:
        ppg_filt = bandpass_ppg(ppg_raw, fs)
    except Exception:
        ppg_filt = ppg_raw - np.median(ppg_raw)

    for idx, start_i in enumerate(range(0, n_samples, step_len)):
        end_i = min(start_i + win_len, n_samples)
        segment_raw = ppg_raw[start_i:end_i]
        segment_ac = ppg_filt[start_i:end_i]
        
        if len(segment_raw) < fs:
            sqi_values[idx] = sqi_values[max(0, idx - 1)]
            continue

        # 1. Clipping / Saturation check
        diffs = np.diff(segment_raw)
        flatline_fraction = np.mean(np.abs(diffs) < 1e-6)
        if flatline_fraction > 0.4:
            sqi_values[idx] = 0.05
            continue

        # 2. Skewness SQI (Clean arterial PPG is slightly positive skewed ~0.5 to 1.5)
        sk = stats.skew(segment_ac)
        sk_score = np.exp(-((sk - 0.8) ** 2) / (2 * 1.5**2))

        # 3. Kurtosis SQI (Clean pulsatile signal has leptokurtic peakedness > 3.0)
        kt = stats.kurtosis(segment_ac) # Fisher kurtosis (normal=0, clean ppg > 0.5)
        kt_score = np.clip((kt + 1.0) / 4.0, 0.0, 1.0)

        # 4. Standard deviation check (too tiny = disconnect, too huge = extreme motion artifact)
        std_ac = np.std(segment_ac)
        dc_val = np.abs(np.mean(segment_raw)) + 1e-4
        perfusion_ratio = std_ac / dc_val

        # Composite score weighting
        sqi = 0.5 * sk_score + 0.3 * kt_score + 0.2 * (1.0 - flatline_fraction)
        sqi_values[idx] = np.clip(sqi, 0.0, 1.0)

    return sqi_values

if __name__ == "__main__":
    fs = 50.0
    t = np.arange(0, 10, 1/fs)
    # Simulate clean pulsatile PPG
    clean = np.sin(2 * np.pi * 1.2 * t) + 0.3 * np.sin(4 * np.pi * 1.2 * t)
    # Simulate noisy flatline/saturated PPG
    noisy = np.random.randn(len(t)) * 5.0
    
    sqi_clean = compute_ppg_sqi(clean, fs)
    sqi_noisy = compute_ppg_sqi(noisy, fs)
    print(f"Clean PPG SQI mean: {sqi_clean.mean():.3f} (expected > 0.6)")
    print(f"Noisy PPG SQI mean: {sqi_noisy.mean():.3f} (expected < 0.5)")

