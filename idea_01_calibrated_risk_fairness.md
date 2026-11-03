# Case File: Calibrated Early Risk Prediction with Uncertainty & Fairness Constraints

## 1. Project Identity & Pitch
- **Title**: Clinician Decision-Support: Calibrated Early Disease Risk with Demographic Fairness Constraints
- **Domain**: Healthcare AI / Tabular & Clinical ML / Fair & Trustworthy AI (Samsung SIC Project)
- **Target Users**: Hospital triage doctors, preventative health clinicians, and healthcare administrators.

## 2. Core Problem & Existing System
- **Problem**: Standard clinical risk models (e.g., predicting cardiovascular disease, diabetes onset, or ICU sepsis) produce uncalibrated single-point probabilities (e.g., "72% risk") without statistical confidence bounds. Crucially, models trained on imbalanced clinical records systematically under-estimate risk for protected demographic groups (e.g., underrepresented ethnicities or age brackets), creating life-threatening clinical disparities.
- **Current Baseline**: Unconstrained Logistic Regression, XGBoost, or multilayer perceptrons optimized purely on AUROC or Log-Loss, offering zero uncertainty guarantees and failing demographic parity / equalized odds audits.

## 3. Proposed Solution Architecture
- **Calibration & Uncertainty**: Probability calibration via Isotonic Regression / Platt Scaling + Conformal Prediction intervals to output guaranteed 90% confidence bands `[Risk_Low, Risk_High]`.
- **Fairness Regularization**: Integration of Fairlearn / AIF360 adversarial debiasing or constraint-based threshold optimization ensuring equalized opportunity across demographic subsets.
- **Clinician Dashboard**: FastAPI backend with a modern web interface displaying:
  1. Patient calibrated risk score with error margins.
  2. Automated fairness compliance scorecards (Disparate Impact ratio, False Negative parity).
  3. Feature attribution flags (SHAP values) explaining top clinical drivers.

## 4. Key Metrics & Benchmarks
- **Calibration**: Expected Calibration Error (ECE < 0.05) and Brier Score.
- **Fairness**: Equalized Odds difference (<0.08) and Demographic Parity ratio (>0.80).
- **Public Datasets**: MIMIC-III / PhysioNet ICU Sepsis dataset, or Framingham / CDC BRFSS Heart Disease dataset.

## 5. Potential Failure Modes (Points for the Prosecutor)
1. Fairness-accuracy tradeoff: enforces strict parity constraints at the expense of overall global AUROC.
2. Clinicians might ignore uncertainty intervals if interface presentation is cluttered.
3. Conformal prediction intervals could widen excessively on out-of-distribution demographic clusters.
