# CSL2010 IML Project 1: Human Activity Recognition Pipeline
*(Second Draft: Scikit-Learn Re-implementation)*

A complete, reproducible, CV-leakage-free machine learning pipeline for the **CSL2010 IML Project 1** Kaggle competition: 12-class human activity recognition from 5-second wearable sensor windows (50 Hz).

This **Second Draft** re-implements the solution strictly using **scikit-learn** (plus `numpy`, `pandas`, `scipy`, `matplotlib`, `joblib`, and `lightgbm`/`xgboost`), eliminating deep learning framework dependencies (PyTorch/TensorFlow) and replacing them with beginner-friendly, clean scikit-learn architectures like `MLPClassifier` and custom scikit-learn compatible transformers.

---

## 1. Executive Summary & Final Metrics

| Metric / Evaluation Scheme | Value | Significance |
| :--- | :---: | :--- |
| **Best Single Model Group-CV** | **99.66% ± 0.69%** | Extra Trees Classifier on Feature Set B (Subject-Robust) |
| **Best Ensemble Group-CV (OOF)** | **99.56%** | Soft-Voting Ensemble (SVM + RF + Extra Trees + LightGBM) |
| **Nested-CV Generalization Accuracy** | **87.67% ± 13.03%** | Outer Leave-One-Group-Out, Inner StratifiedGroupKFold tuning |
| **Subject Leakage Gap** | **+0.34% to +13.65%** | Optimistic inflation caused by random StratifiedKFold vs Group-CV |
| **Test Prediction Agreement** | **93.55% (377/403)** | Agreement between Submission A and Submission B on unseen test subjects |

---

## 2. What Changed from First Draft

1. **Zero Deep Learning / PyTorch Imports**:
   - The first draft used PyTorch (`torch`, `torch.nn`) to train a 1D Convolutional Neural Network on raw multi-channel series (`(N, 23, 250)`).
   - In the second draft, deep neural modeling is implemented natively using scikit-learn's **`MLPClassifier`** (`hidden_layer_sizes=(128, 64)`, `activation='relu'`, `alpha=1e-3`, `early_stopping=True`, `n_iter_no_change=10`).
   - The 1D CNN in first draft achieved only 92.73% Group-CV accuracy while taking 41.5 seconds. Tree ensembles and SVMs on engineered features achieve **99.66% Group-CV** in 2–4 seconds because physical dynamics (frequencies, magnitudes, posture angles) are explicitly encoded.
2. **Pure-NumPy MiniRocket Transformer**:
   - The first draft used `MiniRocket` from the external `aeon` package.
   - The second draft implements a self-contained `NumPyMiniRocket` class subclassing `sklearn.base.BaseEstimator` and `TransformerMixin`. It computes random dilated 1D convolutions and Positive Proportion of Values (PPV) in pure NumPy, eliminating external C/Numba dependencies.
3. **Pure Scikit-Learn Pipelines**:
   - Every model is encapsulated in a `sklearn.pipeline.Pipeline` with `StandardScaler` to guarantee zero test leakage across all cross-validation folds.
   - Ensembles use scikit-learn's standard `VotingClassifier(voting='soft')`.

---

## 3. Dataset Findings & Preprocessing

- **Dataset Dimensions**:
  - `train.csv`: 679 rows $\times$ 5,752 columns (`id`, 5,750 signal columns, `target`)
  - `test.csv`: 403 rows $\times$ 5,751 columns (`id`, 5,750 signal columns)
  - `sample_submission.csv`: 403 rows $\times$ 2 columns (`id`, `target`)
- **Signal Structure & Reshape**:
  - 5,750 features = 250 time steps (5 seconds at 50 Hz) $\times$ 23 sensor channels.
  - Column ordering is strictly **time-major**: `t000_ch0 ... t000_ch22, t001_ch0 ... t249_ch22`.
  - Reshaping `(N, 5750)` to `(N, 250, 23)` verified with 0 error (`t000_<ch>` maps to `[:, 0, ch_idx]`).
  - 23 Channels across 3 body locations:
    - Chest (5): `chest_acc_x`, `chest_acc_y`, `chest_acc_z`, `chest_ecg_1`, `chest_ecg_2`
    - Ankle (9): `ankle_acc_x/y/z`, `ankle_gyro_x/y/z`, `ankle_mag_x/y/z`
    - Right Arm (9): `arm_acc_x/y/z`, `arm_gyro_x/y/z`, `arm_mag_x/y/z`
- **Data Hygiene & Missing Value Handling**:
  - `train.csv` contained 638,842 corrupted/missing cells (16.36% missing rate across all rows and channels) due to European comma decimals (e.g. `'-9,31'`), whitespace padding, non-numeric tokens (`'-'`, `'?'`, `'unknown'`), and sentinel values (`9999`).
  - Fixed via string parsing, regex cleaning, and **linear interpolation along the 250-timestep time series axis** with boundary backfill/forward fill (resulting in exactly 0 NaNs).
  - `test.csv` is 100% clean `float64` with 0 NaNs and 0 sentinels.
- **Class Imbalance**:
  - 11 classes are balanced (53 to 67 samples each, ~7.8% - 9.9%).
  - `jump_front_back` is underrepresented with only 19 samples (2.80%).

---

## 4. Validation Strategy & Limitations

### Why Random CV Fails (Subject Leakage)
True subject IDs are omitted. Training data comprises **5 subjects**, while test data comprises **3 completely different subjects**. Standard `StratifiedKFold` randomly scatters windows from the same subject across training and validation folds. Because individual sensor offsets, body geometry, and gait style leak across folds, random CV drastically overestimates accuracy (up to +13.65% gap).

### Pseudo-Subject Grouping
1. **Row Order Inspection**: Train row IDs and numeric ID order are shuffled (581 activity transitions in 679 rows). No contiguous session blocks exist in row order.
2. **Pseudo-Subject Derivation**: Clustered windows into 5 pseudo-subject groups using **static, subject-specific anatomical and mounting baseline features** (per-window means of 3-axis magnetometers and 2-lead ECG baseline). Robust percentile clipping and `StandardScaler` followed by `KMeans(n_clusters=5, random_state=42, n_init=30)`.
3. **Cluster Diagnostics**:
   - Group 0: 206 windows (30.3%)
   - Group 1: 149 windows (21.9%)
   - Group 2: 116 windows (17.1%)
   - Group 3: 21 windows (3.1%)
   - Group 4: 187 windows (27.5%)
   - Every activity class appears in at least 4 different clusters.
4. **Primary CV Object**: `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)` passed as `cv=...` with `groups=pseudo_groups` supplied to all evaluations.
5. **Known Limitation**: Pseudo-subject clusters are an approximation of true physiological subject identities derived from static sensor offsets. Group 3 has fewer samples (predominantly high-acceleration dynamic movements).

---

## 5. Feature Engineering

From the `(N, 250, 23)` sensor tensors, we extracted features across 4 distinct domains:
1. **Time Domain (460 features)**: Mean, std, min, max, range, median, IQR, percentiles (10, 25, 75, 90), skewness, kurtosis, RMS, energy, MAD, jerk (mean absolute first difference), mean-crossing rate, zero-crossing rate, peak count.
2. **Frequency Domain (161 features)**: Real FFT (50 Hz sampling rate): Dominant frequency, PSD band power ratios (`[0-0.5 Hz]`, `[0.5-3.0 Hz]`, `[3.0-8.0 Hz]`, `[8.0-25.0 Hz]`), spectral entropy, spectral centroid.
3. **Cross-Channel Dynamics (63 features)**: Euclidean norm magnitude stats (mean, std, max, min, energy), Signal Magnitude Area (SMA), and cross-axis Pearson correlations for all 7 tri-axial sensor clusters.
4. **Gravitational Posture Features (15 features)**: Normalized direction cosines ($g_x/\|g\|$, $g_y/\|g\|$, $g_z/\|g\|$) and posture angles (pitch and roll) from low-passed accelerometer signals, separating sitting, standing, and lying down.

### Feature Set Comparison
- **Feature Set A (All Features, 699 features)**: Retains all raw statistics.
- **Feature Set B (Subject-Robust, 621 features)**: Drops 78 absolute static offset features (raw means, medians, mins, maxs, ranges, and percentiles of magnetometers and ECG) that shift across subjects, keeping dynamic properties.

---

## 6. Selected Submissions

1. **`submission_A.csv` (Best Single Robust Model)**:
   - Extra Trees Classifier (`n_estimators=500`, `max_depth=14`) on Feature Set B (Subject-Robust).
   - Achieved highest single-model Group-CV score (99.66%) with the smallest leakage gap (+0.34%).
2. **`submission_B.csv` (Best Diverse Ensemble)**:
   - Soft-Voting Ensemble combining SVM (RBF), Random Forest, Extra Trees, and LightGBM on Feature Set B.
   - Blends non-linear kernel margins with tree ensemble variance reduction for optimal generalization on unseen test subjects.
   - 93.55% agreement with Submission A (377 out of 403 test samples match).

---

## 7. How to Reproduce (Single-Command Execution)

All scripts run using the project virtual environment at `./.venv/bin/python`:

```bash
# 1. Activate project virtual environment
source .venv/bin/activate

# 2. Run Exploratory Data Analysis (generates eda_summary.md and plots)
./.venv/bin/python eda.py

# 3. Derive Pseudo-Subject Validation Groups (generates pseudo_groups.csv)
./.venv/bin/python cv.py

# 4. Extract Features and evaluate Feature Set A vs B (caches features_train.csv and features_test.csv)
./.venv/bin/python features.py

# 5. Train all Candidate Models, run Nested-CV, and build Ensembles
./.venv/bin/python train.py

# 6. Generate final validated submissions
./.venv/bin/python predict.py
```

---

## 8. Manual Post-Pipeline Checklist for You

Before the competition deadline, make sure to complete these manual administrative steps on Kaggle:
- [ ] **Kaggle Display Name**: Update your Kaggle display name to the mandatory format: `Full Name (Roll Number)`.
- [ ] **Institute Email**: Ensure your Kaggle account is registered with or linked to your official institute email address.
- [ ] **Host Email Sharing**: Go to your Kaggle Account Settings and verify that your registered email is visible to competition hosts if required by the course syllabus.
- [ ] **Upload Final Submissions**:
  1. Upload `submission_A.csv` (Single Extra Trees model).
  2. Upload `submission_B.csv` (Diverse Soft-Voting Ensemble).
- [ ] **Select Final Submissions**: On the Kaggle "My Submissions" page, explicitly check the boxes to **select both submissions** for final Private Leaderboard scoring.
