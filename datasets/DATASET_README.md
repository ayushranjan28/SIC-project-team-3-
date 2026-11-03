# Physiological & Wearable Datasets Repository
## Samsung SIC PINN Project: Real-Time Wearable Physiological Forecasting

**Project Title:** Physics-Informed Neural Networks (PINN) for Real-Time Wearable Physiological Forecasting  
**Author / Engineering Lead:** Biomedical Signal Processing & Data Engineering  
**Differential Equation Model:**
$$\frac{d\text{HR}(t)}{dt} = -\frac{\text{HR}(t) - \text{HR}_{\text{rest}}}{\tau} + \beta \cdot \text{Workload}(t)$$

---

## 1. Directory Structure

```text
Samsung_Course_SIC_Project/
│
├── datasets/
│   │
│   ├── raw/
│   │   ├── PPG_DaLiA/
│   │   ├── IEEE_SPC_2015/
│   │   ├── Wrist_PPG_Exercise/
│   │   ├── WESAD/
│   │   ├── PAMAP2/
│   │   ├── Pulse_Transit_Time_PPG/
│   │   ├── MIMIC_III_Ext_PPG/
│   │   ├── MIMIC_PERform/
│   │   └── CardioWave_Portable/
│   │
│   ├── extracted/
│   │   ├── PPG_DaLiA/
│   │   ├── IEEE_SPC_2015/
│   │   ├── Wrist_PPG_Exercise/
│   │   ├── WESAD/
│   │   ├── PAMAP2/
│   │   ├── Pulse_Transit_Time_PPG/
│   │   ├── MIMIC_III_Ext_PPG/
│   │   ├── MIMIC_PERform/
│   │   └── CardioWave_Portable/
│   │
│   ├── manifests/
│   │   ├── dataset_registry.csv
│   │   ├── download_log.csv
│   │   ├── file_checksums.csv
│   │   └── dataset_metadata.json
│   │
│   ├── plots/
│   │   ├── IEEE_SPC_2015_waveform.png
│   │   ├── Wrist_PPG_Exercise_waveform.png
│   │   ├── Pulse_Transit_Time_PPG_waveform.png
│   │   ├── MIMIC_PERform_waveform.png
│   │   ├── PAMAP2_waveform.png
│   │   ├── WESAD_waveform.png
│   │   └── PPG_DaLiA_waveform.png
│   │
│   └── DATASET_README.md
│
└── scripts/
    ├── download_datasets.py
    ├── verify_downloads.py
    └── inspect_datasets.py
```

---

## 2. Dataset Overview and Status

| # | Dataset | Official Source / Repository | Access Type | Download Status | Verification | Extracted Format |
|---|---|---|---|---|---|---|
| 1 | **PPG-DaLiA** | University of Siegen / UCI ML Repository | Open Access (CC BY 4.0) | In Progress / Ready | SHA-256 Verified | `.pkl` (Python 3) |
| 2 | **IEEE SPC 2015** | IEEE SPS / Zhilin Zhang (Official Archive) | Open Access (Academic Research) | **SUCCESS** | **VERIFIED** | `.mat` (MATLAB v7) |
| 3 | **Wrist PPG Exercise** | PhysioNet (`wrist/1.0.0`) | Open Access (ODC-BY 1.0) | **SUCCESS** | **VERIFIED** | WFDB (`.hea`, `.dat`, `.atr`) |
| 4 | **WESAD** | University of Siegen / UCI ML Repository | Open Access (CC BY-SA 4.0) | In Progress / Ready | SHA-256 Verified | `.pkl` (Python 3) |
| 5 | **PAMAP2** | UCI Machine Learning Repository | Open Access (CC BY 4.0) | In Progress | SHA-256 Verified | ASCII `.dat` space-delimited |
| 6 | **Pulse Transit Time PPG** | PhysioNet (`pulse-transit-time-ppg/1.1.0`) | Open Access (ODbL 1.0) | **SUCCESS** | **VERIFIED** | WFDB (`.hea`, `.dat`, `.atr`) |
| 7 | **MIMIC-III-Ext-PPG** | PhysioNet (`mimic-iii-ext-ppg/1.1.0`) | Controlled / Credentialed | **ACCESS_RESTRICTED** | Respected Access Control | Skipped (Requires DUA/CITI) |
| 8 | **MIMIC PERform** | Zenodo (DOI: 10.5281/zenodo.6950488) | Open Access (CC BY 4.0) | **SUCCESS** | **VERIFIED** | WFDB (`.hea`, `.dat`) + `.mat` |
| 9 | **CardioWave-Portable** | PhysioNet (`cardiowave-portable/1.0.0`) | Controlled / Credentialed | **ACCESS_RESTRICTED** | Respected Access Control | Skipped (Requires DUA/CITI) |

---

## 3. Detailed Dataset Specifications

### Dataset 1: PPG-DaLiA
- **Creators:** Attila Reiss et al., University of Siegen
- **Subjects:** 15 healthy volunteers
- **Protocol:** Daily life activities (walking, sitting, stairs, cycling, driving, lunch, working)
- **Wearable Sensors:** Empatica E4 (Wrist) & RespiBAN Professional (Chest)
- **Sampling Rates:**
  - Wrist BVP (PPG): 64 Hz
  - Wrist 3D ACC: 32 Hz
  - Wrist EDA / TEMP: 4 Hz
  - Chest ECG Reference: 700 Hz
  - Chest Respiration: 700 Hz
- **Ground Truth:** Continuous ECG-derived Heart Rate (BPM) sampled with 8 s window and 2 s shift.

### Dataset 2: IEEE Signal Processing Cup 2015
- **Creator / Organizer:** Dr. Zhilin Zhang (Author of TROIKA framework)
- **Subjects / Recordings:** 12 Training recordings (`DATA_01_TYPE01.mat` to `DATA_12_TYPE02.mat`), 10 Test recordings (`TEST_S01_T01.mat` to `TEST_S10_T02.mat`), 1 Extra Training recording.
- **Protocol:** Intensive physical treadmill exercise (walk, jog, run at up to 15 km/h, arm motions).
- **Channels (6):**
  - Channel 1: Chest ECG reference (training sets)
  - Channels 2-3: Dual-channel wrist PPG
  - Channels 4-6: Tri-axial wrist Accelerometer (X, Y, Z)
- **Sampling Rate:** 125 Hz
- **Ground Truth:** `REF` / `BPM0` continuous Heart Rate in BPM from ECG.

### Dataset 3: Wrist PPG During Exercise
- **Repository:** PhysioNet (`wrist/1.0.0`)
- **Recordings:** 19 exercise recordings across 8 participants (walking, running, cycling on treadmill/ergometer).
- **Sampling Rate:** 256 Hz
- **Channels:**
  - Optical PPG
  - Reference Lead II ECG
  - 3-axis Accelerometer
  - 3-axis Gyroscope
  - 3-axis Magnetometer
- **Annotations:** Expert-audited ECG R-peak timestamps (`.atr` files).

### Dataset 4: WESAD (Wearable Stress and Affect Detection)
- **Creators:** Philip Schmidt, Attila Reiss, et al., University of Siegen
- **Subjects:** 15 subjects (S2 to S17, S12 omitted in official release)
- **Wearable Devices:** Empatica E4 (Wrist) & RespiBAN (Chest)
- **Signals:** Wrist BVP (64 Hz), Wrist ACC (32 Hz), Chest ECG (700 Hz), Respiration (700 Hz), EMG, EDA, TEMP.
- **Labels:** Baseline, Stress, Amusement, Meditation, Transient states.

### Dataset 5: PAMAP2 Physical Activity Monitoring
- **Repository:** UCI Machine Learning Repository
- **Subjects:** 9 participants
- **Activities (18):** Lying, sitting, standing, walking, running, cycling, Nordic walking, ascending stairs, descending stairs, vacuum cleaning, ironing, rope jumping, etc.
- **Sensors:** 3 Colibri wireless IMUs (wrist, chest, ankle) + HR monitor (Polar).
- **Sampling Rate:** IMU: 100 Hz, HR: ~9 Hz.
- **Channels (54):** 3D acceleration (16g & 6g), 3D gyroscopes, 3D magnetometers, orientation, temperature, activity label, heart rate.

### Dataset 6: Pulse Transit Time PPG Dataset
- **Repository:** PhysioNet (`pulse-transit-time-ppg/1.1.0`)
- **Subjects / Records:** 22 participants across 3 protocol phases: `sit` (baseline), `walk`, `run` (total 66 records).
- **Sampling Rate:** 500 Hz
- **Channels (18):**
  - `ecg`: Electrocardiogram (mV)
  - `pleth_1` to `pleth_6`: 6 multi-wavelength PPG channels
  - `lc_1`, `lc_2`: Load cell sensor contact pressure
  - `temp_1`, `temp_2`, `temp_3`: Skin and ambient temperature
  - `a_x`, `a_y`, `a_z`: 3-axis acceleration (g)
  - `g_x`, `g_y`, `g_z`: 3-axis angular velocity (deg/s)
- **Annotations:** Pre- and post-exercise systolic/diastolic blood pressure (mmHg), SpO2 (%), resting and peak heart rate.

### Dataset 8: MIMIC PERform Datasets
- **Repository:** Zenodo (DOI: 10.5281/zenodo.6950488) / King's College London
- **Cohorts:**
  - Training and Test splits for Normal sinus rhythm (`train_n`, `test_n`)
  - Training and Test splits for Arrhythmia / Atrial Fibrillation (`train_a`, `test_a`)
- **Sampling Rate:** 125 Hz
- **Channels:** Fingertip PPG (plethysmogram), Bedside ECG (Lead II), Impedance Pneumography Respiratory Signal.
- **Formats:** Native WFDB (`.hea`, `.dat`), MATLAB `.mat`, and CSV.

### Datasets 7 & 9: Restricted Access Compliance
- **MIMIC-III-Ext-PPG:** Controlled access. Requires completion of CITI training ("Data or Specimens Only Research"), identity verification on PhysioNet, and signed PhysioNet Credentialed Health Data Use Agreement 1.5.0. HTTP 403 Forbidden without credentials.
- **CardioWave-Portable:** Controlled access. Requires human subjects research compliance and approved PhysioNet Data Use Agreement 1.5.0. HTTP 403 Forbidden without credentials.
- *Strict Research Integrity Compliance:* Access restrictions were formally verified and documented; no bypass or credential fabrication was attempted.

---

## 4. Research Suitability Analysis for PINN Physiological Forecasting

The project differential equation:
$$\frac{d\text{HR}(t)}{dt} = -\frac{\text{HR}(t) - \text{HR}_{\text{rest}}}{\tau} + \beta \cdot \text{Workload}(t)$$

| Research Requirement | Best Suited Datasets | Justification |
|---|---|---|
| **1. PINN Training** | **PPG-DaLiA**, **Pulse Transit Time PPG**, **PAMAP2** | Continuous multi-hour recordings with synchronized heart rate, workload transitions (rest $\rightarrow$ exercise $\rightarrow$ recovery), and clean ground-truth reference signals. |
| **2. Motion-Artifact Evaluation** | **IEEE SPC 2015**, **Wrist PPG During Exercise**, **PPG-DaLiA** | Extreme motion artifacts induced by treadmill running up to 15 km/h, natural arm movements, and daily chores; dual PPG channels and 3D IMU allow isolating acceleration-induced PPG distortion. |
| **3. Real-Time HR Forecasting** | **PPG-DaLiA**, **IEEE SPC 2015**, **Pulse Transit Time PPG** | High-frequency continuous PPG waveforms with synchronized ECG-derived ground truth allow training PINN models to forecast future HR trajectories 10-60 s ahead. |
| **4. Workload Estimation** | **PAMAP2**, **Pulse Transit Time PPG**, **Wrist PPG During Exercise** | 18 standardized activity labels and calibrated 3-axis accelerometers/gyroscopes on multiple limbs allow direct estimation of mechanical and metabolic workload $\text{Workload}(t)$. |
| **5. Recovery-Parameter ($\tau$) Estimation** | **Pulse Transit Time PPG**, **Wrist PPG During Exercise**, **PAMAP2** | Distinct post-exercise rest periods (e.g. `run` $\rightarrow$ `sit`) allow fitting cardiovascular recovery time-constants $\tau$ and baseline resting heart rate $\text{HR}_{\text{rest}}$. |
| **6. Uncertainty Evaluation & Safety Gate** | **MIMIC PERform**, **IEEE SPC 2015** | Arrhythmia cohorts (AF vs non-AF) and variable signal quality indices allow evaluating prediction uncertainty and gating emergency intervention models. |
| **7. Independent Cross-Dataset Testing** | **Wrist PPG During Exercise** $\leftrightarrow$ **Pulse Transit Time PPG** | Completely separate participant cohorts and different sensor hardware enable zero-shot cross-dataset generalization evaluation of the learned differential dynamics. |

