"""
Workload Estimation Module for Wearable Inertial Measurements.

Translates raw tri-axial accelerometer (and gyroscope) signals into a normalized,
dimensionally consistent, physiologically grounded workload metric:
    Workload(t) in [0.0, 1.0]

Formulation:
    1. Accelerometer Euclidean Norm: r(t) = sqrt(a_x^2 + a_y^2 + a_z^2) [units: g]
    2. Euclidean Norm Minus One (ENMO): ENMO(t) = max(0, r(t) - 1.0 g)
    3. Dynamic rolling standard deviation: sigma_acc(t) over a 3-5 second window
    4. Combined physical exertion metric:
           E(t) = ENMO(t) + alpha * sigma_acc(t)
    5. Calibrated normalization against standard human activity intensity thresholds:
           Workload(t) = clip(E(t) / E_vigorous, 0.0, 1.0)
       Where E_vigorous corresponds to vigorous treadmill running (~1.5g dynamic acceleration).
"""

import numpy as np
from scipy.ndimage import uniform_filter1d

def compute_enmo(acc_triaxial: np.ndarray) -> np.ndarray:
    """
    Computes Euclidean Norm Minus One (ENMO) from tri-axial accelerometer signals.
    Automatically detects whether accelerometer is in m/s^2 (norm ~ 9.81) or g (norm ~ 1.0)
    and calibrates to standard earth gravity units (g).
    
    Args:
        acc_triaxial: Array of shape (N, 3).
        
    Returns:
        1D array of ENMO values in g, clipped to non-negative values.
    """
    if acc_triaxial.ndim != 2 or acc_triaxial.shape[1] != 3:
        raise ValueError(f"Expected acc_triaxial of shape (N, 3), got {acc_triaxial.shape}")
        
    norm = np.linalg.norm(acc_triaxial, axis=1) # Total acceleration
    
    # Auto-calibrate units: if average norm is > 5.0, signal is in m/s^2, convert to g
    mean_norm = np.nanmean(norm)
    if mean_norm > 4.5:
        norm = norm / 9.80665
        
    enmo = np.maximum(0.0, norm - 1.0)          # Dynamic acceleration component in g
    return enmo

def compute_workload(
    acc_triaxial: np.ndarray,
    sampling_rate_hz: float,
    window_sec: float = 3.0,
    vigorous_threshold_g: float = 1.2
) -> np.ndarray:
    """
    Estimates continuous metabolic/physical exertion Workload(t) in [0.0, 1.0].
    
    Args:
        acc_triaxial: (N, 3) triaxial accelerometer signal in units of g.
        sampling_rate_hz: Sampling frequency in Hertz.
        window_sec: Moving window duration in seconds for rolling variance/jerk.
        vigorous_threshold_g: Normalization ceiling corresponding to heavy exertion (1.0 workload).
        
    Returns:
        1D array of shape (N,) containing normalized Workload(t) in [0.0, 1.0].
    """
    enmo = compute_enmo(acc_triaxial)
    
    # Rolling standard deviation over window_sec
    window_size = max(1, int(window_sec * sampling_rate_hz))
    
    # Rolling mean of squared signal minus square of rolling mean
    mean_enmo = uniform_filter1d(enmo, size=window_size, mode='nearest')
    mean_sq_enmo = uniform_filter1d(enmo**2, size=window_size, mode='nearest')
    var_enmo = np.maximum(0.0, mean_sq_enmo - mean_enmo**2)
    std_enmo = np.sqrt(var_enmo)
    
    # Combined dynamic exertion score: mean dynamic level + movement variability
    exertion_raw = mean_enmo + 0.5 * std_enmo
    
    # Normalize to [0.0, 1.0] with smooth tanh/linear saturation
    workload = np.clip(exertion_raw / vigorous_threshold_g, 0.0, 1.0)
    
    return workload

if __name__ == "__main__":
    # Test with rest (static 1g) and vigorous movement (3g peak)
    fs = 50.0
    t = np.arange(0, 10, 1/fs)
    # 5 sec rest, 5 sec exercise
    acc_rest = np.zeros((len(t)//2, 3))
    acc_rest[:, 2] = 1.0 # 1g along Z
    
    acc_run = np.random.randn(len(t) - len(acc_rest), 3) * 0.8
    acc_run[:, 2] += 1.0
    
    acc_sim = np.vstack([acc_rest, acc_run])
    wl = compute_workload(acc_sim, fs)
    print(f"Rest workload mean: {wl[:len(t)//2].mean():.4f} (expected ~0.0)")
    print(f"Run workload mean: {wl[len(t)//2:].mean():.4f} (expected > 0.4)")
