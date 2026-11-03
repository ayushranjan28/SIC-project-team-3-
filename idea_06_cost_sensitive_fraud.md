# Case File: Cost-Sensitive Fraud Detection with Operator Feedback Loop

## 1. Project Identity & Pitch
- **Title**: Cost-Aware Financial Fraud Shield with Continuous Operator Active-Feedback
- **Domain**: Financial Fraud / Cost-Sensitive ML / MLOps Feedback Loops (Samsung SIC Project)
- **Target Users**: Financial crime risk investigators, anti-fraud operations centers, and compliance analysts.

## 2. Core Problem & Existing System
- **Problem**: Real-world fraud detection treats all classification errors equally, optimizing for standard F1-score or PR-AUC. In reality, a False Negative (missing a $50,000 transaction fraud) costs orders of magnitude more than a False Positive (a temporary alert on a $20 purchase). Furthermore, investigator feedback on flagged alerts is logged in silos and rarely updates the model dynamically.
- **Current Baseline**: Static imbalanced models (XGBoost, LightGBM, Isolation Forests) with fixed 0.5 probability thresholds, evaluated on symmetric loss functions without financial risk matrices.

## 3. Proposed Solution Architecture
- **Cost-Sensitive Objective**: Custom asymmetric utility loss incorporating dynamic dollar-weighted transaction costs and operator review overhead costs.
- **Active Operator Feedback Loop**: A stream-processing mechanism that ingests analyst verdicts (confirmed fraud vs. false alarm) into a continuous incremental learner or active-retraining buffer.
- **Operator Cockpit UI**: Modern web operations console displaying:
  1. Priority queue ranked by Expected Dollar Savings rather than raw probability.
  2. One-click investigator labeling and override telemetry.
  3. Real-time cost-savings dashboard comparing the cost-sensitive model against traditional symmetric models.

## 4. Key Metrics & Benchmarks
- **Financial Metric**: Total Financial Loss Saved ($) and Cost-Weighted Misclassification Loss.
- **Operational Metric**: Reduction in investigator alert fatigue (Precision @ Top 100 alerts).
- **Public Datasets**: Kaggle Credit Card Fraud Detection (European Cardholders) or IEEE-CIS Fraud Detection benchmark.

## 5. Potential Failure Modes (Points for the Prosecutor)
1. Feedback delay: genuine fraud chargebacks can take 30-90 days to confirm, delaying clean feedback.
2. Operator confirmation bias: investigators may only inspect what the model flags, creating a sampling bias loop.
3. Cold start on new fraud vectors not represented in historical cost matrices.
