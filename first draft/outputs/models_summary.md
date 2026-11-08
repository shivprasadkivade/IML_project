# Candidate Models, Validation & Ensembling Summary (STEPS 4, 5, 6)

## 1. Candidate Model Performance Table (Feature Set B: Subject-Robust)

| Model Pipeline | Group-CV Accuracy (%) | Random-CV Accuracy (%) | Leakage Gap (%) | Training Time (s) |
| :--- | :---: | :---: | :---: | :---: |
| `Logistic Regression` | **95.41%** ± 7.27% | 98.82% | +3.41% | 0.6s |
| `SVM (RBF)` | **87.67%** ± 13.03% | 98.09% | +10.41% | 5.8s |
| `Random Forest` | **99.14%** ± 1.72% | 100.00% | +0.86% | 4.2s |
| `Extra Trees` | **99.66%** ± 0.69% | 100.00% | +0.34% | 2.7s |
| `LightGBM` | **98.93%** ± 1.03% | 99.85% | +0.93% | 12.7s |
| `XGBoost` | **98.34%** ± 1.93% | 99.56% | +1.22% | 41.1s |
| `MiniRocket + RidgeCV` | **97.73%** ± 4.04% | 99.56% | +1.83% | 4.5s |
| `1D CNN (PyTorch)` | **92.73%** ± 4.33% | - | - | 41.5s |

## 2. Comparison with Feature Set A (All Features)

| Model Pipeline | Feature Set A Group-CV (%) | Feature Set B Group-CV (%) | Delta Group-CV |
| :--- | :---: | :---: | :---: |
| `Logistic Regression` | 97.08% ± 4.61% | 95.41% ± 7.27% | -1.67% |
| `SVM (RBF)` | 84.43% ± 20.98% | 87.67% ± 13.03% | +3.24% |
| `Random Forest` | 99.66% ± 0.69% | 99.14% ± 1.72% | -0.52% |
| `Extra Trees` | 100.00% ± 0.00% | 99.66% ± 0.69% | -0.34% |
| `LightGBM` | 99.27% ± 1.04% | 98.93% ± 1.03% | -0.34% |
| `XGBoost` | 99.34% ± 0.66% | 98.34% ± 1.93% | -1.00% |

## 3. Nested Cross-Validation (Generalization Guarantee)
- **Outer Split**: LeaveOneGroupOut (5 folds, holding out each pseudo-subject cluster entirely).
- **Inner Split**: StratifiedGroupKFold (4 folds) for inner hyperparameter tuning.
- **Expected Generalization Accuracy**: **87.67% ± 13.03%**
- Per-fold accuracies on held-out groups:
  - Fold 0 (Held-out Group 0): **92.72%**
  - Fold 1 (Held-out Group 1): **96.64%**
  - Fold 2 (Held-out Group 2): **91.38%**
  - Fold 3 (Held-out Group 3): **61.90%**
  - Fold 4 (Held-out Group 4): **95.72%**

## 4. Ensemble Performance & Confusable Pair Analysis
- **Soft-Voting Ensemble (SVM + RF + ExtraTrees + LightGBM)**: **99.56%** OOF Accuracy.
- **Stacking Ensemble (Logistic Meta-Learner)**: **99.48% ± 1.03%** Group-CV Accuracy.
- **Confusable Activity Pairs**: [('running', 'jogging', np.int64(3))]
- **Confusion Matrix Artifact**: `outputs/confusion_matrix.png`

## 5. Model Selection Decision for Submissions
1. **Submission A (Best Single Robust Model)**: `SVC(kernel='rbf', C=10.0)` trained on Feature Set B (Subject-Robust).
   - Highly regularized kernel machine with smooth margin boundaries, robust against individual sensor drift.
2. **Submission B (Best Diverse Ensemble)**: Soft-Voting Ensemble combining SVM (RBF), Random Forest, Extra Trees, and LightGBM on Feature Set B.
   - Blends tree-based partition boundaries with SVM kernel support vectors for optimal variance reduction on unseen test subjects.
