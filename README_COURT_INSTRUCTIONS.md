# How to Run Your Project Ideas Through `/court` in Claude Code

This workspace now contains ready-to-test case files for your top candidate projects from the Samsung SIC PDF list.

---

## 1. Quick Installation (inside Claude Code)

Open **Claude Code** in this directory (`c:\Users\ayush\Desktop\Samsung Course(SIC project)`) and run:

```bash
https://github.com/alexyc9381/court-skill
Install this skill, then confirm /court works.
```

Or manually clone it if you prefer:
```bash
git clone https://github.com/alexyc9381/court-skill
cp -r court-skill/skills/court ~/.claude/skills/
```

---

## 2. Running the Trials

Because a trial with 12 sub-agents takes several minutes, you can run a **full 12-juror trial** or a **quick 6-juror trial (`--quick`)**.

### Test Idea #2: Limited-Label Medical Image + Edge AI (Recommended Top Pick)
In Claude Code, type:
```bash
/court --quick Review the project proposal in idea_02_active_learning_medical_edge.md. Put this idea on trial: evaluate if the active learning and edge quantization claim will work, identify technical flaws, and determine if it is suitable for a Samsung capstone.
```

### Test Idea #1: Calibrated Early Risk Prediction + Fairness
```bash
/court --quick Review the project proposal in idea_01_calibrated_risk_fairness.md. Put this idea on trial: can clinical fairness and calibrated probability intervals be reliably achieved together, and does it provide a compelling demo?
```

### Test Idea #8: Uncertainty-Aware Model with Safety Gate (Reject Option)
```bash
/court --quick Review the project proposal in idea_08_uncertainty_safety_gate.md. Put this idea on trial: will the selective classification reject-option genuinely prevent high-stakes failures or overwhelm human reviewers?
```

### Test Idea #6: Cost-Sensitive Fraud Detection + Feedback Loop
```bash
/court --quick Review the project proposal in idea_06_cost_sensitive_fraud.md. Put this idea on trial: is a custom asymmetric cost-matrix and live feedback loop technically rigorous and feasible for an SIC project?
```

---

## 3. How to Read the Court Output

After running, Claude Code will create a folder `.court/` in this directory and output:
- **`VERDICT.md`**: Contains the final vote (e.g., `NOT GUILTY, 10 to 2` or `GUILTY, 8 to 4`).
- **Individual Juror Rationales**: Feedback from the Engineer, Statistician, Investor, Customer, Designer, etc.
- **Conditions for Acquittal**: The exact changes, safeguards, and benchmarks you must add to guarantee project success before submitting to Samsung evaluators.
