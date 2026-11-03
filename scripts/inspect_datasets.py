"""
Signal inspection script for Samsung SIC PINN Wearable Physiological Forecasting Project.
Loads real physiological recordings in native formats (.mat, .dat/.hea, .pkl, .csv),
identifies channels, sampling frequencies, durations, units, annotations,
and generates representative multi-channel waveform plots.
"""

import os
import sys
import glob
from pathlib import Path
import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent
DATASETS_DIR = BASE_DIR / "datasets"
RAW_DIR = DATASETS_DIR / "raw"
EXTRACTED_DIR = DATASETS_DIR / "extracted"
PLOTS_DIR = DATASETS_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

try:
    import wfdb
except ImportError:
    wfdb = None

def inspect_ieee_spc_2015():
    print("\n==========================================")
    print("Inspecting Dataset 2: IEEE SPC 2015")
    print("==========================================")
    ieee_dir = EXTRACTED_DIR / "IEEE_SPC_2015"
    train_file = list(ieee_dir.rglob("DATA_01_TYPE01.mat"))
    
    if not train_file:
        print("DATA_01_TYPE01.mat not found.")
        return
        
    mat_path = str(train_file[0])
    mat_data = sio.loadmat(mat_path)
    sig = mat_data['sig']  # channels: 1=ECG, 2=PPG1, 3=PPG2, 4=ACC_X, 5=ACC_Y, 6=ACC_Z
    
    # Ground truth BPM (REF)
    ref_b = None
    if 'BPM0' in mat_data:
        ref_b = mat_data['BPM0']
    else:
        # Check if true BPM file exists
        true_f = list(ieee_dir.rglob("True_01_TYPE01.mat")) or list(ieee_dir.rglob("True_*.mat"))
        if true_f:
            tmat = sio.loadmat(str(true_f[0]))
            if 'BPM0' in tmat:
                ref_b = tmat['BPM0']
                
    fs = 125.0
    duration_sec = sig.shape[1] / fs
    print(f"File: {mat_path}")
    print(f"Format: MATLAB .mat (v7)")
    print(f"Channels: 6 (1: ECG Ref, 2: PPG Ch1, 3: PPG Ch2, 4: ACC X, 5: ACC Y, 6: ACC Z)")
    print(f"Sampling Frequency: {fs} Hz")
    print(f"Total Samples: {sig.shape[1]}")
    print(f"Signal Duration: {duration_sec:.2f} seconds ({duration_sec/60:.2f} minutes)")
    print(f"Missing Values: {np.isnan(sig).sum()} NaN values")
    
    # Plot first 10 seconds of data
    t = np.arange(int(10 * fs)) / fs
    ecg = sig[0, :int(10 * fs)]
    ppg1 = sig[1, :int(10 * fs)]
    ppg2 = sig[2, :int(10 * fs)]
    acc_x = sig[3, :int(10 * fs)]
    acc_y = sig[4, :int(10 * fs)]
    acc_z = sig[5, :int(10 * fs)]
    
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    axes[0].plot(t, ppg1, color='crimson', label='Wrist PPG (Ch 1)')
    axes[0].plot(t, ppg2, color='darkorange', alpha=0.7, label='Wrist PPG (Ch 2)')
    axes[0].set_ylabel('Amplitude (a.u.)')
    axes[0].set_title('IEEE SPC 2015: Representative Physiological & Kinematic Signals (Subject 1, Type 1)')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(t, ecg, color='darkblue', label='Reference ECG (Chest)')
    axes[1].set_ylabel('Voltage (mV/a.u.)')
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(t, acc_x, label='ACC X', color='forestgreen')
    axes[2].plot(t, acc_y, label='ACC Y', color='limegreen')
    axes[2].plot(t, acc_z, label='ACC Z', color='teal')
    axes[2].set_ylabel('Acceleration (g)')
    axes[2].legend(loc='upper right')
    axes[2].grid(True, alpha=0.3)
    
    # Bottom: Ground-truth Heart Rate (BPM) over time
    if ref_b is not None:
        ref_bpm = ref_b.flatten()
        # Each window is 8s with 2s step
        t_bpm = 8.0 + np.arange(len(ref_bpm)) * 2.0
        axes[3].plot(t_bpm, ref_bpm, 'o-', color='purple', label='Ground-Truth Reference HR (BPM)')
        axes[3].set_ylabel('HR (BPM)')
        axes[3].set_xlabel('Time (seconds)')
        axes[3].legend(loc='upper right')
        axes[3].grid(True, alpha=0.3)
        axes[3].set_xlim(0, max(t[-1], 60))
    else:
        axes[3].set_visible(False)
        axes[2].set_xlabel('Time (seconds)')
        
    plt.tight_layout()
    plot_file = PLOTS_DIR / "IEEE_SPC_2015_waveform.png"
    plt.savefig(plot_file, dpi=150)
    plt.close()
    print(f"Saved inspection plot to {plot_file}")

def inspect_wrist_ppg_exercise():
    print("\n==========================================")
    print("Inspecting Dataset 3: Wrist PPG During Exercise")
    print("==========================================")
    w_dir = EXTRACTED_DIR / "Wrist_PPG_Exercise"
    hea_files = list(w_dir.rglob("*.hea"))
    if not hea_files:
        print("No WFDB recordings found in Wrist PPG Exercise.")
        return
        
    rec_path = str(hea_files[0].with_suffix(''))
    print(f"Reading WFDB record: {rec_path}")
    
    if wfdb:
        record = wfdb.rdrecord(rec_path)
        sig = record.p_signal
        fields = record.__dict__
        fs = record.fs
        sig_len = record.sig_len
        duration_sec = sig_len / fs
        
        print(f"Format: PhysioNet WFDB (.hea / .dat)")
        print(f"Channels ({record.n_sig}): {record.sig_name}")
        print(f"Units: {record.units}")
        print(f"Sampling Frequency: {fs} Hz")
        print(f"Duration: {duration_sec:.2f} s ({duration_sec/60:.2f} min)")
        print(f"Missing Values: {np.isnan(sig).sum()} NaNs")
        
        # Read R-peak annotations if present
        ann = None
        try:
            ann = wfdb.rdann(rec_path, 'atr')
            print(f"Annotated R-peaks: {len(ann.sample)} beats detected")
        except Exception as e:
            print(f"No annotation file or error: {e}")
            
        # Plot 10 seconds of data
        n_samples = int(10 * fs)
        t = np.arange(n_samples) / fs
        
        fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
        # Find PPG channel index
        ppg_idx = next((i for i, name in enumerate(record.sig_name) if 'ppg' in name.lower()), 0)
        ecg_idx = next((i for i, name in enumerate(record.sig_name) if 'ecg' in name.lower()), 1)
        
        axes[0].plot(t, sig[:n_samples, ppg_idx], color='crimson', label=f'PPG ({record.sig_name[ppg_idx]})')
        axes[0].set_ylabel(record.units[ppg_idx])
        axes[0].set_title(f"Wrist PPG During Exercise: {record.record_name} (fs = {fs} Hz)")
        axes[0].legend(loc='upper right')
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(t, sig[:n_samples, ecg_idx], color='darkblue', label=f'ECG ({record.sig_name[ecg_idx]})')
        axes[1].set_ylabel(record.units[ecg_idx])
        if ann:
            # R-peaks in this segment
            r_mask = (ann.sample < n_samples)
            axes[1].plot(ann.sample[r_mask] / fs, sig[ann.sample[r_mask], ecg_idx], 'r*', label='Reference R-peaks')
        axes[1].legend(loc='upper right')
        axes[1].grid(True, alpha=0.3)
        
        # Accelerometer
        acc_indices = [i for i, name in enumerate(record.sig_name) if 'acc' in name.lower()]
        for idx in acc_indices[:3]:
            axes[2].plot(t, sig[:n_samples, idx], label=f'{record.sig_name[idx]} ({record.units[idx]})')
        axes[2].set_ylabel('Acceleration')
        axes[2].legend(loc='upper right')
        axes[2].grid(True, alpha=0.3)
        
        # Gyroscope
        gyro_indices = [i for i, name in enumerate(record.sig_name) if 'gyr' in name.lower()]
        for idx in gyro_indices[:3]:
            axes[3].plot(t, sig[:n_samples, idx], label=f'{record.sig_name[idx]} ({record.units[idx]})')
        axes[3].set_ylabel('Angular Vel.')
        axes[3].set_xlabel('Time (seconds)')
        axes[3].legend(loc='upper right')
        axes[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_file = PLOTS_DIR / "Wrist_PPG_Exercise_waveform.png"
        plt.savefig(plot_file, dpi=150)
        plt.close()
        print(f"Saved inspection plot to {plot_file}")

def inspect_pulse_transit_time_ppg():
    print("\n==========================================")
    print("Inspecting Dataset 6: Pulse Transit Time PPG")
    print("==========================================")
    raw_dir = DATASETS_DIR / "raw" / "Pulse_Transit_Time_PPG"
    hea_files = list(raw_dir.glob("*.hea"))
    if not hea_files:
        print("No .hea files found in Pulse_Transit_Time_PPG.")
        return
        
    rec_path = str(hea_files[0].with_suffix(''))
    print(f"Reading WFDB record: {rec_path}")
    
    if wfdb:
        record = wfdb.rdrecord(rec_path)
        sig = record.p_signal
        fs = record.fs
        sig_len = record.sig_len
        duration_sec = sig_len / fs
        
        print(f"Format: PhysioNet WFDB (.hea / .dat)")
        print(f"Channels ({record.n_sig}): {record.sig_name}")
        print(f"Units: {record.units}")
        print(f"Sampling Frequency: {fs} Hz")
        print(f"Duration: {duration_sec:.2f} s ({duration_sec/60:.2f} min)")
        print(f"Missing Values: {np.isnan(sig).sum()} NaNs")
        print(f"Comments / Metadata: {record.comments}")
        
        # Plot first 10 seconds
        n_samples = int(10 * fs)
        t = np.arange(n_samples) / fs
        
        fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
        # 1. PPG channels (pleth_1, pleth_2)
        p1_idx = record.sig_name.index('pleth_1') if 'pleth_1' in record.sig_name else 1
        p2_idx = record.sig_name.index('pleth_2') if 'pleth_2' in record.sig_name else 2
        axes[0].plot(t, sig[:n_samples, p1_idx], color='crimson', label=f'PPG Ch1 ({record.sig_name[p1_idx]})')
        axes[0].plot(t, sig[:n_samples, p2_idx], color='darkorange', alpha=0.7, label=f'PPG Ch2 ({record.sig_name[p2_idx]})')
        axes[0].set_ylabel('Amplitude (NU)')
        axes[0].set_title(f"Pulse Transit Time PPG Dataset: {record.record_name} (fs = {fs} Hz)")
        axes[0].legend(loc='upper right')
        axes[0].grid(True, alpha=0.3)
        
        # 2. ECG
        ecg_idx = record.sig_name.index('ecg') if 'ecg' in record.sig_name else 0
        axes[1].plot(t, sig[:n_samples, ecg_idx], color='darkblue', label=f'ECG ({record.units[ecg_idx]})')
        axes[1].set_ylabel('mV')
        axes[1].legend(loc='upper right')
        axes[1].grid(True, alpha=0.3)
        
        # 3. Accelerometer (a_x, a_y, a_z)
        ax_idx = record.sig_name.index('a_x') if 'a_x' in record.sig_name else 13
        ay_idx = record.sig_name.index('a_y') if 'a_y' in record.sig_name else 14
        az_idx = record.sig_name.index('a_z') if 'a_z' in record.sig_name else 15
        axes[2].plot(t, sig[:n_samples, ax_idx], label='ACC X', color='forestgreen')
        axes[2].plot(t, sig[:n_samples, ay_idx], label='ACC Y', color='limegreen')
        axes[2].plot(t, sig[:n_samples, az_idx], label='ACC Z', color='teal')
        axes[2].set_ylabel('Acceleration (g)')
        axes[2].legend(loc='upper right')
        axes[2].grid(True, alpha=0.3)
        
        # 4. Gyroscope (g_x, g_y, g_z)
        gx_idx = record.sig_name.index('g_x') if 'g_x' in record.sig_name else 16
        gy_idx = record.sig_name.index('g_y') if 'g_y' in record.sig_name else 17
        gz_idx = record.sig_name.index('g_z') if 'g_z' in record.sig_name else 18
        axes[3].plot(t, sig[:n_samples, gx_idx], label='GYRO X', color='rebeccapurple')
        axes[3].plot(t, sig[:n_samples, gy_idx], label='GYRO Y', color='mediumorchid')
        axes[3].plot(t, sig[:n_samples, gz_idx], label='GYRO Z', color='indigo')
        axes[3].set_ylabel('Ang. Vel (deg/s)')
        axes[3].set_xlabel('Time (seconds)')
        axes[3].legend(loc='upper right')
        axes[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_file = PLOTS_DIR / "Pulse_Transit_Time_PPG_waveform.png"
        plt.savefig(plot_file, dpi=150)
        plt.close()
        print(f"Saved inspection plot to {plot_file}")

def inspect_mimic_perform():
    print("\n==========================================")
    print("Inspecting Dataset 8: MIMIC PERform")
    print("==========================================")
    mimic_dir = EXTRACTED_DIR / "MIMIC_PERform"
    hea_files = list(mimic_dir.rglob("*.hea"))
    mat_files = list(RAW_DIR.glob("MIMIC_PERform/*.mat"))
    
    if hea_files and wfdb:
        rec_path = str(hea_files[0].with_suffix(''))
        print(f"Reading MIMIC PERform WFDB record: {rec_path}")
        record = wfdb.rdrecord(rec_path)
        sig = record.p_signal
        fs = record.fs
        sig_len = record.sig_len
        duration_sec = sig_len / fs
        
        print(f"Format: PhysioNet WFDB (.hea / .dat)")
        print(f"Channels ({record.n_sig}): {record.sig_name}")
        print(f"Units: {record.units}")
        print(f"Sampling Frequency: {fs} Hz")
        print(f"Duration: {duration_sec:.2f} s ({duration_sec/60:.2f} min)")
        print(f"Missing Values: {np.isnan(sig).sum()} NaNs")
        
        # Plot 10 seconds
        n_samples = int(10 * fs)
        t = np.arange(n_samples) / fs
        
        fig, axes = plt.subplots(record.n_sig, 1, figsize=(12, 2.5 * record.n_sig), sharex=True)
        if record.n_sig == 1:
            axes = [axes]
        for i, sname in enumerate(record.sig_name):
            axes[i].plot(t, sig[:n_samples, i], color='teal' if 'ppg' in sname.lower() else 'darkblue')
            axes[i].set_ylabel(record.units[i] if i < len(record.units) else 'a.u.')
            axes[i].set_title(f"Channel: {sname}")
            axes[i].grid(True, alpha=0.3)
        axes[-1].set_xlabel('Time (seconds)')
        plt.tight_layout()
        plot_file = PLOTS_DIR / "MIMIC_PERform_waveform.png"
        plt.savefig(plot_file, dpi=150)
        plt.close()
        print(f"Saved inspection plot to {plot_file}")
    elif mat_files:
        mpath = str(mat_files[0])
        print(f"Inspecting MAT file: {mpath}")
        mdata = sio.loadmat(mpath)
        print("Keys in MAT:", [k for k in mdata.keys() if not k.startswith('__')])

def inspect_pamap2():
    print("\n==========================================")
    print("Inspecting Dataset 5: PAMAP2")
    print("==========================================")
    pamap_dir = EXTRACTED_DIR / "PAMAP2"
    dat_files = list(pamap_dir.rglob("subject*.dat"))
    if not dat_files:
        print("No subject*.dat files found in PAMAP2.")
        return
        
    dat_path = str(dat_files[0])
    print(f"Reading PAMAP2 file: {dat_path}")
    # Load first 10,000 lines
    data = []
    with open(dat_path, "r") as f:
        for i, line in enumerate(f):
            if i >= 10000:
                break
            parts = [float(x) if x != 'NaN' else np.nan for x in line.strip().split()]
            if len(parts) >= 54:
                data.append(parts)
    data = np.array(data)
    
    # Columns:
    # 0: timestamp (s)
    # 1: activityID
    # 2: HR (bpm)
    # 4-6: wrist 3D acc 16g
    t = data[:, 0] - data[0, 0]
    act = data[:, 1]
    hr = data[:, 2]
    wrist_acc_x = data[:, 4]
    wrist_acc_y = data[:, 5]
    wrist_acc_z = data[:, 6]
    
    print(f"Format: Space-delimited ASCII .dat")
    print(f"Channels: 54 columns (Timestamp, ActivityID, Heart Rate, 3 IMUs with 3D ACC, GYRO, MAG, Temp)")
    print(f"Nominal IMU Sampling Rate: 100 Hz, HR sampling: ~9 Hz")
    print(f"Loaded samples: {len(data)}")
    print(f"Active activities in sample: {np.unique(act[~np.isnan(act)])}")
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    # 1. Heart rate
    valid_hr = ~np.isnan(hr)
    axes[0].plot(t[valid_hr], hr[valid_hr], color='crimson', label='Heart Rate (BPM)')
    axes[0].set_ylabel('HR (BPM)')
    axes[0].set_title(f"PAMAP2 Physical Activity Monitoring: {Path(dat_path).stem} (Multi-Modal Kinematics & HR)")
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)
    
    # 2. Activity ID
    axes[1].step(t, act, color='darkorange', where='post', label='Physical Activity Label ID')
    axes[1].set_ylabel('Activity ID')
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)
    
    # 3. Wrist ACC
    axes[2].plot(t, wrist_acc_x, label='Wrist ACC X', color='forestgreen')
    axes[2].plot(t, wrist_acc_y, label='Wrist ACC Y', color='limegreen')
    axes[2].plot(t, wrist_acc_z, label='Wrist ACC Z', color='teal')
    axes[2].set_ylabel('Acceleration (m/s²)')
    axes[2].set_xlabel('Time (seconds)')
    axes[2].legend(loc='upper right')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_file = PLOTS_DIR / "PAMAP2_waveform.png"
    plt.savefig(plot_file, dpi=150)
    plt.close()
    print(f"Saved inspection plot to {plot_file}")

def inspect_wesad():
    print("\n==========================================")
    print("Inspecting Dataset 4: WESAD")
    print("==========================================")
    import pickle
    wesad_dir = EXTRACTED_DIR / "WESAD"
    pkl_files = list(wesad_dir.rglob("S*.pkl"))
    if not pkl_files:
        print("No S*.pkl files found in WESAD.")
        return
        
    pkl_path = str(pkl_files[0])
    print(f"Loading WESAD pickle: {pkl_path}")
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
        
    wrist_bvp = data['signal']['wrist']['BVP'].flatten()
    wrist_acc = data['signal']['wrist']['ACC']
    chest_ecg = data['signal']['chest']['ECG'].flatten()
    chest_acc = data['signal']['chest']['ACC']
    labels = data['label'].flatten()
    
    fs_bvp = 64.0
    fs_ecg = 700.0
    
    print(f"Subject: {data.get('subject', Path(pkl_path).stem)}")
    print(f"Wrist BVP: {len(wrist_bvp)} samples at {fs_bvp} Hz ({len(wrist_bvp)/fs_bvp/60:.2f} min)")
    print(f"Chest ECG: {len(chest_ecg)} samples at {fs_ecg} Hz ({len(chest_ecg)/fs_ecg/60:.2f} min)")
    print(f"Affect Labels: {np.unique(labels)} (0=transient, 1=baseline, 2=stress, 3=amusement, 4=meditation)")
    
    # Plot 10 seconds of synchronized data
    t_bvp = np.arange(int(10 * fs_bvp)) / fs_bvp
    t_ecg = np.arange(int(10 * fs_ecg)) / fs_ecg
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(t_bvp, wrist_bvp[:int(10 * fs_bvp)], color='crimson', label=f'Wrist BVP (PPG) at {fs_bvp} Hz')
    axes[0].set_ylabel('Amplitude (a.u.)')
    axes[0].set_title(f"WESAD: Multi-Modal Wearable Stress and Affect (Subject {data.get('subject')})")
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(t_ecg, chest_ecg[:int(10 * fs_ecg)], color='darkblue', label=f'Chest ECG at {fs_ecg} Hz')
    axes[1].set_ylabel('mV')
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)
    
    # Wrist ACC
    fs_acc = 32.0
    t_acc = np.arange(int(10 * fs_acc)) / fs_acc
    axes[2].plot(t_acc, wrist_acc[:int(10 * fs_acc), 0], label='ACC X', color='forestgreen')
    axes[2].plot(t_acc, wrist_acc[:int(10 * fs_acc), 1], label='ACC Y', color='limegreen')
    axes[2].plot(t_acc, wrist_acc[:int(10 * fs_acc), 2], label='ACC Z', color='teal')
    axes[2].set_ylabel('ACC (g)')
    axes[2].set_xlabel('Time (seconds)')
    axes[2].legend(loc='upper right')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_file = PLOTS_DIR / "WESAD_waveform.png"
    plt.savefig(plot_file, dpi=150)
    plt.close()
    print(f"Saved inspection plot to {plot_file}")

def inspect_ppg_dalia():
    print("\n==========================================")
    print("Inspecting Dataset 1: PPG-DaLiA")
    print("==========================================")
    import pickle
    dalia_dir = EXTRACTED_DIR / "PPG_DaLiA"
    pkl_files = list(dalia_dir.rglob("S*.pkl"))
    if not pkl_files:
        print("No S*.pkl files found in PPG-DaLiA.")
        return
        
    pkl_path = str(pkl_files[0])
    print(f"Loading PPG-DaLiA pickle: {pkl_path}")
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
        
    wrist_bvp = data['signal']['wrist']['BVP'].flatten()
    wrist_acc = data['signal']['wrist']['ACC']
    chest_ecg = data['signal']['chest']['ECG'].flatten()
    activity = data['activity'].flatten()
    ref_hr = data['label'].flatten()  # Reference HR at 4-second intervals
    
    fs_bvp = 64.0
    fs_ecg = 700.0
    
    print(f"Subject: {Path(pkl_path).stem}")
    print(f"Wrist BVP: {len(wrist_bvp)} samples at {fs_bvp} Hz ({len(wrist_bvp)/fs_bvp/60:.2f} min)")
    print(f"Chest ECG: {len(chest_ecg)} samples at {fs_ecg} Hz ({len(chest_ecg)/fs_ecg/60:.2f} min)")
    print(f"Reference HR points: {len(ref_hr)}")
    print(f"Unique activities: {np.unique(activity)}")
    
    # Plot 10 seconds of waveform
    t_bvp = np.arange(int(10 * fs_bvp)) / fs_bvp
    t_ecg = np.arange(int(10 * fs_ecg)) / fs_ecg
    
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    axes[0].plot(t_bvp, wrist_bvp[:int(10 * fs_bvp)], color='crimson', label=f'Wrist BVP (PPG) at {fs_bvp} Hz')
    axes[0].set_ylabel('Amplitude (a.u.)')
    axes[0].set_title(f"PPG-DaLiA: Real-Life Physiological Forecasting Benchmark ({Path(pkl_path).stem})")
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(t_ecg, chest_ecg[:int(10 * fs_ecg)], color='darkblue', label=f'Reference ECG at {fs_ecg} Hz')
    axes[1].set_ylabel('mV')
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)
    
    # Wrist ACC
    fs_acc = 32.0
    t_acc = np.arange(int(10 * fs_acc)) / fs_acc
    axes[2].plot(t_acc, wrist_acc[:int(10 * fs_acc), 0], label='ACC X', color='forestgreen')
    axes[2].plot(t_acc, wrist_acc[:int(10 * fs_acc), 1], label='ACC Y', color='limegreen')
    axes[2].plot(t_acc, wrist_acc[:int(10 * fs_acc), 2], label='ACC Z', color='teal')
    axes[2].set_ylabel('ACC (g)')
    axes[2].legend(loc='upper right')
    axes[2].grid(True, alpha=0.3)
    
    # HR reference
    # 8-second window, 2-second shift
    t_hr = 8.0 + np.arange(min(len(ref_hr), 30)) * 2.0
    axes[3].plot(t_hr, ref_hr[:len(t_hr)], 's-', color='purple', label='Reference Ground-Truth HR (BPM)')
    axes[3].set_ylabel('HR (BPM)')
    axes[3].set_xlabel('Time (seconds)')
    axes[3].legend(loc='upper right')
    axes[3].grid(True, alpha=0.3)
    axes[3].set_xlim(0, max(t_bvp[-1], 20))
    
    plt.tight_layout()
    plot_file = PLOTS_DIR / "PPG_DaLiA_waveform.png"
    plt.savefig(plot_file, dpi=150)
    plt.close()
    print(f"Saved inspection plot to {plot_file}")

def main():
    print("Executing Full Physiological Signal Inspection Pipeline...")
    inspect_ieee_spc_2015()
    inspect_wrist_ppg_exercise()
    inspect_pulse_transit_time_ppg()
    inspect_mimic_perform()
    inspect_pamap2()
    inspect_wesad()
    inspect_ppg_dalia()

if __name__ == "__main__":
    main()


def main():
    print("Executing Signal Inspection Pipeline...")
    inspect_ieee_spc_2015()
    inspect_wrist_ppg_exercise()

if __name__ == "__main__":
    main()
