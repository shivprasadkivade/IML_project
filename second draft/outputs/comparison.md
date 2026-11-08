# Quantitative Pipeline Comparison: First Draft vs Second Draft (Scikit-Learn)

This document provides a side-by-side empirical verification of the models, cross-validation scores, generalization metrics, and test submissions between **First Draft** and **Second Draft (Scikit-Learn Re-implementation)**.

---

## 1. Candidate Model Performance Table (Group-CV & Random-CV)

All first-draft scores were directly extracted from `first draft/outputs/models_summary.md` (Table 1, lines 7–14) and `first draft/README.md` (Table 5, lines 83–90).
Second-draft scores were obtained from the complete pipeline run executed using `./second draft/.venv/bin/python second draft/train.py`.

| Model Architecture | Feature Set | First-Draft Group-CV (%) [Source: `first draft/outputs/models_summary.md`] | Second-Draft Group-CV (%) [Executed via scikit-learn pipeline] | Group-CV Delta (%) | Second-Draft Random-CV (%) | Second-Draft Subject Leakage Gap (%) | Second-Draft Train Time (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Extra Trees Classifier** | Set B (Robust) | **99.66% ± 0.69%** (line 10) | **99.66% ± 0.69%** | **0.00%** | 100.00% | +0.34% | 2.8s |
| **Random Forest** | Set B (Robust) | **99.14% ± 1.72%** (line 9) | **99.14% ± 1.72%** | **0.00%** | 100.00% | +0.86% | 4.4s |
| **LightGBM** | Set B (Robust) | **98.93% ± 1.03%** (line 11) | **98.93% ± 1.03%** | **0.00%** | 99.85% | +0.93% | 15.0s |
| **XGBoost** | Set B (Robust) | **98.34% ± 1.93%** (line 12) | **98.34% ± 1.93%** | **0.00%** | 99.56% | +1.22% | 47.2s |
| **Logistic Regression (C=0.1)** | Set B (Robust) | **95.41% ± 7.27%** (line 7) | **95.41% ± 7.27%** | **0.00%** | 98.82% | +3.41% | 0.6s |
| **SVM (RBF, C=10.0)** | Set B (Robust) | **87.67% ± 13.03%** (line 8) | **87.67% ± 13.03%** | **0.00%** | 98.09% | +10.41% | 5.8s |
| **MiniRocket + RidgeCV** | Raw Series `(N, 23, 250)` | **97.73% ± 4.04%** (line 13, `aeon` MiniRocket 10k kernels) | **91.18% ± 3.11%** (pure NumPy MiniRocket 2k kernels) | **-6.55%** | 93.67% | +2.49% | 18.0s |
| **Deep Neural Network** | Raw Series vs Engineered | **92.73% ± 4.33%** (line 14, PyTorch 1D CNN) | **90.83% ± 4.83%** (Scikit-Learn `MLPClassifier` on Set B) | **-1.90%** | 97.06% | +6.22% | 2.1s |

---

## 2. Ensembles, Nested CV & Confusable Pair Analysis

| Metric / Evaluation Scheme | First-Draft Score [Source: `first draft/outputs/models_summary.md`] | Second-Draft Score [Executed in Second Draft] | Discrepancy |
| :--- | :---: | :---: | :---: |
| **Soft-Voting Ensemble OOF Accuracy** (SVM + RF + ET + LightGBM) | **99.56%** (line 39) | **99.56%** | **0.00%** (Exact Match) |
| **Stacking Meta-Learner Group-CV** (Logistic Regression on OOF Probs) | **99.48% ± 1.03%** (line 40) | **99.48% ± 1.03%** | **0.00%** (Exact Match) |
| **Nested-CV Generalization Accuracy** (Outer LOGO, Inner StratifiedGroupKFold) | **87.67% ± 13.03%** (line 30) | **87.67% ± 13.03%** | **0.00%** (Exact Match) |
| - Outer Fold 0 (Held-out Group 0) | 92.72% (line 32) | 92.72% | 0.00% |
| - Outer Fold 1 (Held-out Group 1) | 96.64% (line 33) | 96.64% | 0.00% |
| - Outer Fold 2 (Held-out Group 2) | 91.38% (line 34) | 91.38% | 0.00% |
| - Outer Fold 3 (Held-out Group 3) | 61.90% (line 35) | 61.90% | 0.00% |
| - Outer Fold 4 (Held-out Group 4) | 95.72% (line 36) | 95.72% | 0.00% |
| **Confusable Activity Pairs Detected** | `[('running', 'jogging', 3)]` (line 41) | `[('running', 'jogging', 3)]` | Exact Match (3 samples only) |

---

## 3. Test Set Submission Agreement & Fidelity

Evaluated across all 403 unseen test sample windows in `test.csv`:

| Comparison Pair | Matching Rows | Agreement (%) | Status |
| :--- | :---: | :---: | :---: |
| **Submission A (First Draft) vs Submission A (Second Draft)** | **403 / 403** | **100.00%** | **PERFECT MATCH** |
| **Submission B (First Draft) vs Submission B (Second Draft)** | **403 / 403** | **100.00%** | **PERFECT MATCH** |
| **Submission A vs Submission B (Within Second Draft)** | **377 / 403** | **93.55%** | **IDENTICAL DIVERSITY** |
| **Submission A vs Submission B (Within First Draft)** | **377 / 403** | **93.55%** | **IDENTICAL DIVERSITY** |

---

## 4. Architectural Summary of Substituted Components

1. **PyTorch 1D CNN $\rightarrow$ Scikit-Learn `MLPClassifier`**:
   - *First Draft*: PyTorch `nn.Conv1d` with BatchNorm, Dropout, Cosine Annealing, trained on raw multi-channel series for 30 epochs (41.5s, 92.73% Group-CV).
   - *Second Draft*: `sklearn.neural_network.MLPClassifier` with 2 hidden layers (128, 64), ReLU activations, Adam solver, L2 penalty ($\alpha = 10^{-3}$), and native early stopping (`early_stopping=True, n_iter_no_change=10`) trained on the engineered Feature Set B (2.1s, 90.83% Group-CV).
   - *Analysis*: Hand-crafted domain features already capture temporal, spectral, and cross-channel dynamics. Both neural approaches confirm that non-linear multi-layer perceptrons achieve ~91–93% Group-CV, whereas randomized tree ensembles achieve **99.66% Group-CV**.
2. **`aeon` MiniRocket $\rightarrow$ Pure-NumPy `NumPyMiniRocket`**:
   - *First Draft*: `from aeon.transformations.collection.convolution_based import MiniRocket` using external C/Numba compilation of 10,000 dilated kernels (97.73% Group-CV).
   - *Second Draft*: Pure NumPy MiniRocket transformer subclassing `BaseEstimator` and `TransformerMixin` with 2,000 kernels computing Positive Proportion of Values (PPV) (91.18% Group-CV). Eliminates all external non-sklearn package dependencies.
3. **Core Submission Models (Submission A & B)**:
   - **Zero substitution needed**: In `first draft`, both Submission A (`ExtraTreesClassifier`) and Submission B (`VotingClassifier` of SVM, Random Forest, Extra Trees, and LightGBM) were already scikit-learn / LightGBM models!
   - Consequently, the second draft generates **100.00% identical predictions** on all 403 test samples.
