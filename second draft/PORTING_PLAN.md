# Comprehensive Porting Plan: First Draft to Second Draft (Scikit-Learn)

## Overview & Context
This porting plan details the complete architectural and mathematical design of **CSL2010 IML Project 1** (12-class human activity recognition from 5-second wearable sensor windows at 50 Hz). The first draft was thoroughly inspected across all source files (`eda.py`, `cv.py`, `features.py`, `train.py`, `predict.py`, `README.md`, `viva_notes.md`, and all `outputs/`).

**Key Architectural Finding**:
Inspection confirms that `first draft` did not use TensorFlow or Keras. Instead, its deep learning / specialized modeling consisted of:
1. A **1D Convolutional Neural Network (PyTorch)** on raw multi-channel series (`(N, 23, 250)`).
2. **MiniRocket** (`aeon`) time-series convolutional feature transform combined with `RidgeClassifierCV`.
3. Gradient boosting libraries: **LightGBM** (`lightgbm`) and **XGBoost** (`xgboost`).
4. Core machine learning pipelines in **scikit-learn** (`StandardScaler`, `LogisticRegression`, `SVC`, `RandomForestClassifier`, `ExtraTreesClassifier`, `VotingClassifier`, `StratifiedGroupKFold`).

The second draft re-implements the complete pipeline strictly using **scikit-learn** (plus `numpy`, `pandas`, `scipy`, `matplotlib`, `joblib`, and `lightgbm`/`xgboost` as permitted by scope). Deep learning and external time-series components are replaced with readable, pure scikit-learn equivalents (such as `MLPClassifier` with early stopping and pure-NumPy MiniRocket transform).

---

## a) Data Loading, Cleaning & Reshaping
### 1. File Dimensions & Schema
- `train.csv`: 679 sample windows $\times$ 5,752 columns (`id`, 5,750 sensor signal columns, `target`).
- `test.csv`: 403 sample windows $\times$ 5,751 columns (`id`, 5,750 sensor signal columns).
- `sample_submission.csv`: 403 sample rows $\times$ 2 columns (`id`, `target`).
- 23 sensor channels across 3 body placements:
  - **Chest (5 channels)**: `chest_acc_x`, `chest_acc_y`, `chest_acc_z`, `chest_ecg_1`, `chest_ecg_2`
  - **Ankle (9 channels)**: `ankle_acc_x`, `ankle_acc_y`, `ankle_acc_z`, `ankle_gyro_x`, `ankle_gyro_y`, `ankle_gyro_z`, `ankle_mag_x`, `ankle_mag_y`, `ankle_mag_z`
  - **Arm (9 channels)**: `arm_acc_x`, `arm_acc_y`, `arm_acc_z`, `arm_gyro_x`, `arm_gyro_y`, `arm_gyro_z`, `arm_mag_x`, `arm_mag_y`, `arm_mag_z`
- 250 time steps per window: 5-second physical recording at $f_s = 50\text{ Hz}$. Total features per sample: $250 \times 23 = 5,750$.

### 2. Time-Major Column Ordering & Reshape Logic
- Columns follow strict time-major formatting: `t<000-249>_<channel_name>`.
- Flattened signal array `(N, 5750)` is reshaped to 3D tensor `(N, 250, 23)` where:
  - Axis 0 = Window index ($0 \dots N-1$).
  - Axis 1 = Timestep index ($0 \dots 249$).
  - Axis 2 = Sensor channel index ($0 \dots 22$).
- Verified by asserting `tensor[:, 0, ch_idx] == df['t000_' + ch]` with zero discrepancy.

### 3. Missing Value & Noise Cleaning Strategy
- **`train.csv`**: Contains 638,842 corrupted/missing cells (16.36% missing rate uniformly distributed across all channels and windows).
  - String corruption: European decimal commas (e.g. `'-9,31'`), whitespace padding, non-numeric sentinels (`'-'`, `'?'`, `'unknown'`).
  - Outlier sentinels: Synthetic extreme numbers (`9999` and `9999.0`).
  - Handling: String cleaning, regex comma-to-dot replacement, coercion of invalid tokens to `NaN`, setting values $\ge 9000$ to `NaN`.
  - Imputation: Linear interpolation along the 250-timestep time series axis (`series.interpolate(method='linear', limit_direction='both').fillna(0.0)`), leaving exactly **0 NaNs** and preserving physical waveform continuity.
- **`test.csv`**: 100% clean `float64` with 0 NaNs and 0 sentinels; directly reshaped to `(403, 250, 23)`.

---

## b) Feature Extraction Specification
Features are extracted from `(N, 250, 23)` across 4 distinct domains:

### 1. Time Domain (20 stats $\times$ 23 channels = 460 features)
For each channel window of length 250:
- Basic statistics: `mean`, `std`, `min`, `max`, `range`, `median`, `p10`, `p25`, `p75`, `p90`, `iqr`
- Distribution shape: `skewness` (`scipy.stats.skew`), `kurtosis` (`scipy.stats.kurtosis`), `mad` (mean absolute deviation from mean)
- Energy & exertion: `rms` ($\sqrt{\frac{1}{T}\sum x^2}$), `energy` ($\frac{1}{T}\sum x^2$)
- Dynamics & Cadence: `jerk` (mean absolute first difference $\frac{1}{T-1}\sum |x_{t+1}-x_t|$), `mean_cross_rate`, `zero_cross_rate`, `peak_count` (`scipy.signal.find_peaks`)

### 2. Frequency Domain (7 stats $\times$ 23 channels = 161 features)
Computed via Real FFT (`np.fft.rfft`) on centered signal:
- `dom_freq`: Frequency corresponding to maximum Power Spectral Density (excluding DC).
- PSD Band Power Ratios:
  - `band_0_05_ratio`: $[0.0, 0.5)\text{ Hz}$ (postural sway and slow drift)
  - `band_05_3_ratio`: $[0.5, 3.0)\text{ Hz}$ (human locomotion / walking / stairs)
  - `band_3_8_ratio`: $[3.0, 8.0)\text{ Hz}$ (jogging, running, rapid limb dynamics)
  - `band_8_25_ratio`: $[8.0, 25.0]\text{ Hz}$ (impact transients and tremors)
- `spectral_entropy`: Normalized Shannon entropy of the power spectrum.
- `spectral_centroid`: Spectral center of mass $\sum (f \cdot \text{PSD}) / \sum \text{PSD}$.

### 3. Cross-Channel Dynamics & Multi-Axial Coupling (7 sensors $\times$ 9 stats = 63 features)
Sensors: `chest_acc`, `ankle_acc`, `ankle_gyro`, `ankle_mag`, `arm_acc`, `arm_gyro`, `arm_mag`.
- Magnitude statistics: `mag_mean`, `mag_std`, `mag_max`, `mag_min`, `mag_energy` of Euclidean norm vector $\sqrt{x^2+y^2+z^2}$ (rotation-invariant).
- `sma`: Signal Magnitude Area $\frac{1}{T}\sum (|x| + |y| + |z|)$.
- Cross-axis Pearson correlations: `corr_xy`, `corr_xz`, `corr_yz`.

### 4. Gravitational Posture Angles (3 accelerometers $\times$ 5 stats = 15 features)
Sensors: `chest_acc`, `ankle_acc`, `arm_acc`.
- Normalized direction cosines: $g_x / \|g\|$, $g_y / \|g\|$, $g_z / \|g\|$.
- Posture angles: `pitch` ($\arctan2(x, \sqrt{y^2+z^2})$) and `roll` ($\arctan2(y, \sqrt{x^2+z^2})$) for clean separation of sitting, standing, and lying down.

### Total Feature Counts:
- **Feature Set A (All Features)**: $460 + 161 + 63 + 15 = 699\text{ features}$.
- **Feature Set B (Subject-Robust)**: $621\text{ features}$. Drops 78 static offset features (`_mean`, `_median`, `_min`, `_max`, `_range`, `_p10`, `_p25`, `_p75`, `_p90`) of `ankle_mag_*`, `arm_mag_*`, and `chest_ecg_*` that exhibit high train-to-test covariate shift due to subject sensor mounting and local magnetic variations.

---

## c) Validation Strategy & Pseudo-Subject Grouping
### 1. Subject Leakage Context
- Train windows come from **5 subjects**; Test windows come from **3 completely different subjects**.
- Standard `StratifiedKFold` leaks static sensor offsets and gait styles between train and validation folds, producing artificially inflated CV scores (+0.34% to +13.65% gap).

### 2. Pseudo-Subject Derivation
- File order inspection: IDs are permuted (581 activity transitions across 679 rows).
- Windows are clustered into 5 pseudo-subject groups using static sensor offsets:
  - Channels: `chest_ecg_1`, `chest_ecg_2`, `ankle_mag_x`, `ankle_mag_y`, `ankle_mag_z`, `arm_mag_x`, `arm_mag_y`, `arm_mag_z`.
  - Window means clipped to [1st, 99th] percentiles, standardized via `StandardScaler()`.
  - `KMeans(n_clusters=5, random_state=42, n_init=30)`.
- Verified cluster distribution:
  - Group 0: 206 windows (30.3%)
  - Group 1: 149 windows (21.9%)
  - Group 2: 116 windows (17.1%)
  - Group 3: 21 windows (3.1%)
  - Group 4: 187 windows (27.5%)
  - Every activity class appears in at least 4 clusters.

### 3. Primary Cross-Validators & Seeds
- Primary CV: `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)` with `groups=pseudo_groups`.
- Nested CV (generalization estimate): Outer `LeaveOneGroupOut()`, Inner `StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=42)`.
- Baseline Leakage Tracking: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`.
- Global seed: `SEED = 42`.

---

## d) Models, Hyperparameters & Ensembling Logic
### 1. Candidate Models on Feature Set B
1. **Logistic Regression**: `StandardScaler() -> LogisticRegression(C=0.1, max_iter=1000, random_state=42)`.
2. **SVM (RBF)**: `StandardScaler() -> SVC(C=10.0, gamma='scale', probability=True, random_state=42)`.
3. **Random Forest**: `StandardScaler() -> RandomForestClassifier(n_estimators=300, max_depth=12, random_state=42, n_jobs=-1)`.
4. **Extra Trees**: `StandardScaler() -> ExtraTreesClassifier(n_estimators=300, max_depth=12, random_state=42, n_jobs=-1)`.
5. **LightGBM**: `StandardScaler() -> LGBMClassifier(n_estimators=150, learning_rate=0.05, num_leaves=31, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0, random_state=42, verbose=-1, n_jobs=-1)`.
6. **XGBoost**: `StandardScaler() -> XGBClassifier(n_estimators=150, learning_rate=0.05, max_depth=5, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0, eval_metric='mlogloss', random_state=42, n_jobs=-1)`.

### 2. Time-Series / Neural Models
7. **MiniRocket + RidgeCV**: In first draft, `aeon` MiniRocket + `RidgeClassifierCV`. In second draft, implemented via a pure NumPy MiniRocket transformer + `RidgeClassifierCV(alphas=np.logspace(-2, 3, 10))` in a scikit-learn Pipeline.
8. **1D CNN Replacement**: In first draft, PyTorch `Conv1DHAR`. In second draft, replaced by scikit-learn's `MLPClassifier(hidden_layer_sizes=(128, 64), activation='relu', solver='adam', alpha=1e-3, batch_size=32, early_stopping=True, n_iter_no_change=10, max_iter=200, random_state=42)` within a `StandardScaler` Pipeline on the engineered features.

### 3. Ensembles & Submissions
- **Soft-Voting Ensemble**: Out-of-fold probability average of top diverse models (`SVM`, `RandomForest`, `ExtraTrees`, `LightGBM`).
- **Stacking Ensemble**: Meta-learner `LogisticRegression(C=0.5, max_iter=500, random_state=42)` fit on stacked out-of-fold probability vectors ($679 \times 48$).
- **Submission A (Best Single Model)**: `Pipeline([('scaler', StandardScaler()), ('clf', ExtraTreesClassifier(n_estimators=500, max_depth=14, random_state=42, n_jobs=-1))])` fit on Feature Set B.
- **Submission B (Best Diverse Ensemble)**: `VotingClassifier(voting='soft')` combining:
  - `SVM (RBF)`: `SVC(C=5.0, gamma='scale', probability=True, random_state=42)`
  - `Random Forest`: `RandomForestClassifier(n_estimators=400, max_depth=12, random_state=42)`
  - `Extra Trees`: `ExtraTreesClassifier(n_estimators=400, max_depth=14, random_state=42)`
  - `LightGBM`: `LGBMClassifier(n_estimators=200, learning_rate=0.04, num_leaves=31, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0, random_state=42)`

---

## e) Deep Learning / Non-Standard Component Replacement Table

| First Draft Component | Framework / Source | Proposed Second Draft Replacement | Scikit-Learn Class / Pipeline Implementation |
| :--- | :--- | :--- | :--- |
| **1D CNN (`Conv1DHAR`)** | PyTorch (`torch.nn.Conv1d`, `BatchNorm1d`, `ReLU`, `MaxPool1d`, `Linear`) | **Multilayer Perceptron (MLP)** with equivalent capacity, ReLU activation, L2 regularization, and early stopping on engineered features. | `sklearn.neural_network.MLPClassifier(hidden_layer_sizes=(128, 64), activation='relu', solver='adam', alpha=1e-3, batch_size=32, early_stopping=True, n_iter_no_change=10, max_iter=200, random_state=42)` |
| **PyTorch DataLoader & Batches** | `torch.utils.data.DataLoader(batch_size=32, shuffle=True)` | Native mini-batch SGD/Adam handling in `MLPClassifier`. | Handled natively by `batch_size=32` parameter in `MLPClassifier`. |
| **PyTorch Early Stopping & Cosine Annealing** | Custom loop tracking validation accuracy across epochs. | Native validation monitoring with patience in `MLPClassifier`. | `early_stopping=True, validation_fraction=0.15, n_iter_no_change=10`. |
| **MiniRocket Feature Extractor** | `aeon.transformations.collection.convolution_based.MiniRocket` | Pure NumPy MiniRocket-style random convolutional projection adhering to `BaseEstimator, TransformerMixin`. | Custom scikit-learn transformer `NumPyMiniRocket(num_kernels=2000, random_state=42)` in Pipeline with `StandardScaler` and `RidgeClassifierCV`. |
| **Fold-wise Channel Normalization** | Custom per-fold tensor mean/std calculation in PyTorch loop. | Standardized inside scikit-learn Pipeline with fold-only fitting. | `sklearn.preprocessing.StandardScaler()` inside `sklearn.pipeline.Pipeline`. |
| **Target Label Encoding** | `sklearn.preprocessing.LabelEncoder` | `LabelEncoder` preserved identically. | `sklearn.preprocessing.LabelEncoder()` |
| **Ensemble Soft Voting** | Custom numpy probability averaging | `sklearn.ensemble.VotingClassifier(voting='soft')` | Direct scikit-learn standard ensemble class. |

---

## f) Non-Exact Matches and Expected Effects
1. **PyTorch 1D CNN vs `MLPClassifier`**:
   - *Difference*: A 1D CNN extracts shift-invariant local temporal patterns from raw waveforms via learned convolutional kernels. `MLPClassifier` operates on the comprehensive 621 hand-crafted features (which already capture spectral, temporal, and multi-axial dynamics).
   - *Expected Effect*: Since hand-crafted features on Feature Set B proved substantially more accurate (99.66% vs 92.73% for the 1D CNN in first draft), `MLPClassifier` on engineered features is expected to achieve comparable or higher Group-CV accuracy (~96–98%) while training in seconds without GPU requirements.
2. **`aeon` MiniRocket vs Pure NumPy MiniRocket**:
   - *Difference*: `aeon` uses a C/Numba-compiled implementation of 10,000 kernels. The pure NumPy equivalent uses a subset of vectorized random kernels (e.g. 1,000–2,000 kernels computing positive proportion of values PPV).
   - *Expected Effect*: Pure NumPy execution without external C/Numba dependencies; achieves similar ~96–98% linear separability with `RidgeClassifierCV`.
3. **Core Submission Models (Submission A & B)**:
   - *Difference*: **NONE**. Both Submission A (`ExtraTreesClassifier`) and Submission B (`VotingClassifier` of SVM, RF, ET, LightGBM) in first draft were already scikit-learn / LightGBM models! Therefore, Submission A and Submission B are 100% architecturally identical between drafts.
