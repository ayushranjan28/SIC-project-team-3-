# Case File: Limited-Label Medical Image Classification with Active Learning + Edge-Ready Model

## 1. Project Identity & Pitch
- **Title**: Edge-Ready Active Learning for Limited-Label Medical Image Classification
- **Domain**: Edge AI / Healthcare / Computer Vision (Samsung SIC Project)
- **Target Users**: Point-of-care clinicians in resource-constrained rural clinics and mobile diagnostics.

## 2. Core Problem & Existing System
- **Problem**: Supervised medical vision models require tens of thousands of expert-annotated images (radiologist annotations are scarce and costly). High-parameter models (e.g., standard ResNet-50 / ViTs) cannot execute locally on low-power mobile or edge devices without transmitting sensitive patient scans to cloud servers.
- **Current Baseline**: Monolithic CNNs running on cloud GPUs trained on fully labeled datasets (ChestX-ray14, ISIC Skin Lesion), suffering from bandwidth latency, privacy regulations (HIPAA/GDPR), and prohibitively expensive dataset labeling.

## 3. Proposed Solution Architecture
- **Active Learning Loop**: An uncertainty sampling & diversity sampling engine (Entropy / Margin / BALD / Core-Set) that automatically identifies the top 10-15% most informative unlabeled medical images for human annotation, matching 95%+ of full-dataset accuracy.
- **Model Compression**: Post-Training Quantization (INT8) + structured pruning on MobileNetV3 / EfficientNet-Lite to compress model weights below 15MB.
- **Deployment & Interface**: An edge-compatible web client (FastAPI + HTML5/JS / ONNX Runtime Web / TFLite) allowing offline, sub-100ms local inference on standard laptop/mobile hardware with an interactive clinician labeling queue.

## 4. Key Metrics & Benchmarks
- **Label Efficiency**: Target AUC achieved with only 20% labeled data vs. 100% baseline.
- **Edge Performance**: Latency (<80ms on CPU), memory footprint (<20MB), and model accuracy retention (>98% of FP32 baseline).
- **Public Datasets**: MedMNIST (PneumoniaMNIST / DermaMNIST) or NIH ChestX-ray8 sample.

## 5. Potential Failure Modes (Points for the Prosecutor)
1. Active learning sample-selection latency could bottle-neck batch queries.
2. Quantization might degrade sensitivity on rare pathological edge cases.
3. Cold-start problem: insufficient initial labels can bias early active-learning acquisition loops.
